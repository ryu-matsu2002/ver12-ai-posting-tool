# app/article_generator.py
# 最新版：プロンプト100%反映 + タイトル/本文出力バグ修正済

import os, re, random, threading, logging, requests
from datetime import datetime, date, timedelta, time, timezone
from typing import List, Dict, Tuple
from difflib import SequenceMatcher
import pytz
from flask import current_app
from openai import OpenAI, BadRequestError
from sqlalchemy import func
from threading import Event
from .image_utils import fetch_featured_image_from_body  # ← 追加
from . import db
from .models import Article
from concurrent.futures import ThreadPoolExecutor, as_completed

# OpenAI設定
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# 🔧 token数制限（本文切れ対策）
TOKENS = {
    "title": 120,
    "outline": 800,
    "block": 3600
}

# 🔧 温度設定（出力のブレ抑制）
TEMP = {
    "title": 0.6,
    "outline": 0.65,
    "block": 0.65
}

TOP_P = 0.9
CTX_LIMIT = 12000
SHRINK = 0.85
MAX_BODY_CHARS_DEFAULT = 4000
MAX_TITLE_RETRY = 7
TITLE_DUP_THRESH = 0.90
JST = pytz.timezone("Asia/Tokyo")
POST_HOURS = list(range(10, 21))
MAX_PERDAY = 5
AVERAGE_POSTS = 4
MAX_SCHEDULE_DAYS = 30  # ← 本日から30日以内の投稿枠に限定

# ============================================
# 🔧 安全な出力クリーニング関数
# GPTが <html> や <body> で囲んでくるケースを除去
# ============================================
def clean_gpt_output(text: str) -> str:
    text = re.sub(r"```(?:html)?", "", text)
    text = re.sub(r"```", "", text)
    text = re.sub(r"<!DOCTYPE html>.*?<body.*?>", "", text, flags=re.DOTALL|re.IGNORECASE)
    text = re.sub(r"</body>.*?</html>", "", text, flags=re.DOTALL|re.IGNORECASE)
    return text.strip()

# ============================================
# 🔧 トークンカウント（概算）
# ============================================
def _tok(s: str) -> int:
    return int(len(s) / 1.8)

# ============================================
# 🔧 OpenAIチャット呼び出し関数
# ============================================
def _chat(msgs: List[Dict[str, str]], max_t: int, temp: float) -> str:
    used = sum(_tok(m["content"]) for m in msgs)
    available = CTX_LIMIT - used - 16
    max_t = min(max_t, available)
    if max_t < 1:
        raise ValueError("Calculated max_tokens is below minimum.")

    def _call(m: int) -> str:
        res = client.chat.completions.create(
            model=MODEL,
            messages=msgs,
            max_tokens=m,
            temperature=temp,
            top_p=TOP_P,
            timeout=120,
        )
        content = res.choices[0].message.content.strip()
        finish = res.choices[0].finish_reason

        if finish == "length":
            logging.warning("⚠️ OpenAI response was cut off due to max_tokens.")
            content += "<p><em>※この文章はトークン上限で途中終了した可能性があります。</em></p>"

        return clean_gpt_output(content)

    try:
        return _call(max_t)
    except BadRequestError as e:
        if "max_tokens" in str(e):
            retry_t = max(1, int(max_t * SHRINK))
            return _call(retry_t)
        raise

# ============================================
# 🔧 タイトル類似性チェック
# ============================================
def _similar(a: str, b: str) -> bool:
    return SequenceMatcher(None, a, b).ratio() >= TITLE_DUP_THRESH

# ============================================
# ✅ 修正済み：タイトル生成（HTML排除・1行制限）
# ============================================
def _title_once(kw: str, pt: str, retry: bool) -> str:
    extra = "\n※過去に使われたタイトルや似たタイトルを絶対に避けてください。" if retry else ""
    usr = f"{pt}{extra}\n\n▼ 条件\n- 必ずキーワードを含める\n- タイトルはユニークであること\n- 出力は1行だけ\n- キーワード順は変更不可\n▼ キーワード: {kw}"
    sys = "あなたはSEOに強い日本語ライターです。絶対にタイトル1行のみを出力してください。"
    return _chat([
        {"role": "system", "content": sys},
        {"role": "user", "content": usr},
    ], TOKENS["title"], TEMP["title"])

# ============================================
# 🔧 タイトルの一意性保証
# ============================================
def _unique_title(kw: str, pt: str) -> str:
    history = [t[0] for t in db.session.query(Article.title).filter(Article.keyword == kw)]
    last_cand = ""
    for i in range(MAX_TITLE_RETRY):
        cand = _title_once(kw, pt, retry=i > 0)
        if not any(_similar(cand, h) for h in history):
            return cand
        last_cand = cand  # 最後に試した候補を記録
    # すべて類似 → 最後の候補を強制採用
    logging.warning(f"[タイトル類似警告] {kw} に対して類似タイトルが多すぎましたが最後の候補を採用します")
    return last_cand


# ============================================
# ✅ ユーザーの希望文字数をGPTに明示して長さ不足を防止
# ============================================
def _compose_body(kw: str, pt: str, format: str = "html", self_review: bool = False) -> str:
    """
    SEO記事本文を生成する関数（追記なし・構造強制・装飾ガイドライン付き）

    Args:
        kw: キーワード
        pt: ユーザープロンプト
        format: "html" または "markdown"
        self_review: True で自己添削を実行

    Returns:
        本文（HTMLまたはMarkdown形式）
    """
    min_chars, max_chars_user = _parse_range(pt)
    max_total = max_chars_user or MAX_BODY_CHARS_DEFAULT

    # 📌形式に応じた構成/装飾指示
    if format == "markdown":
        structure_helper = (
            "\n- Markdown形式で出力してください（## 見出し、### サブ見出し、- 箇条書き、**強調**）"
            "\n- ## 見出しは3〜5個までにしてください"
            "\n- 番号付き小見出し（1.〜など）は ### を使ってください"
            "\n- 最後は ## まとめ セクションで必ず締めてください"
        )
    else:
        structure_helper = (
            "\n- HTML形式で出力してください"
            "\n- <h2 class='wp-heading'>…</h2> を3〜5個使ってください"
            "\n- 番号付き小見出し（1.〜など）は <h3 class='wp-heading'>…</h3> を使ってください"
            "\n- 箇条書きには <ul><li>…</li></ul> を使ってください"
            "\n- 最後は <h2 class='wp-heading'>まとめ</h2> で締めてください"
        )

    # 📌文字数制約を強く指示
    char_instruction = f"\n- 本文は必ず {min_chars}〜{max_total} 字の範囲で書いてください"

    system_prompt = (
        "あなたは一流のSEO記事ライターです。"
        "以下のルールとユーザーの指示に100%従い、構造的で高品質な記事を生成してください。"
        f"{structure_helper}{char_instruction}"
    )
    user_prompt = f"キーワード: {kw}\n\n▼ ユーザーの指示:\n{pt}"

    # ✅1回のみ生成
    full = _chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ], TOKENS["block"], TEMP["block"])

    # ✅自己添削オプション（任意）
    if self_review:
        logging.info("🧠 自己添削モードを実行中...")
        full = _chat([
            {"role": "system", "content": "あなたはSEO記事の編集者です。以下の記事を添削し、構成と論理を強化してください。"},
            {"role": "user", "content": full}
        ], TOKENS["block"], TEMP["block"])

    # ✅文字数制限超過対応
    if len(full) > max_total:
        snippet = full[:max_total]
        cut = max(
            snippet.rfind("</p>"),
            snippet.rfind("</h2>"),
            snippet.rfind("</h3>"),
            snippet.rfind("</li>") if format == "html" else snippet.rfind("\n")
        )
        full = snippet[:cut + 5] if cut != -1 else snippet
        if format == "html" and not full.strip().endswith("</p>"):
            full += "</p>"
        logging.warning("⚠️ 本文が最大長を超えたため安全に切り取りました")

    return full


# ============================================
# 🔧 文字数レンジをプロンプトから自動推定
# ============================================
def _parse_range(pt: str) -> Tuple[int, int | None]:
    """
    ユーザープロンプトの文字数に応じて、必要な本文の長さを推定。
    - 明示的な「○○字から○○字」があれば優先
    - なければプロンプト長さに応じて強めに指示（最小2200字以上）
    """
    if m := re.search(r"(\d{3,5})\s*字から\s*(\d{3,5})\s*字", pt):
        return int(m.group(1)), int(m.group(2))
    if m := re.search(r"(\d{3,5})\s*字", pt):
        return int(m.group(1)), None

    pt_len = len(pt)
    if pt_len < 500:
        return 2200, 2600
    elif pt_len < 1000:
        return 2400, 3000
    elif pt_len < 1500:
        return 2500, 3200
    else:
        return 2700, 3500


# ============================================
# 🔧 投稿スロット生成（1日3〜5記事ルール）
# ============================================
def _generate_slots_per_site(app, site_id: int, n: int) -> List[datetime]:
    if n <= 0:
        return []
    with app.app_context():
        jst_date = func.date(func.timezone("Asia/Tokyo", Article.scheduled_at))
        rows = db.session.query(jst_date.label("d"), func.count(Article.id))\
            .filter(Article.site_id == site_id, Article.scheduled_at.isnot(None))\
            .group_by("d").all()
    booked = {d: c for d, c in rows}
    slots, day = [], date.today()
    while len(slots) < n:
        if (day - date.today()).days > MAX_SCHEDULE_DAYS:
            raise RuntimeError(f"{MAX_SCHEDULE_DAYS}日以内にスケジュールできる枠が足りません")
        remain = MAX_PERDAY - booked.get(day, 0)
        if remain > 0:
            need = min(random.randint(1, AVERAGE_POSTS), remain, n - len(slots))
            for h in sorted(random.sample(POST_HOURS, need)):
                minute = random.randint(1, 59)
                local = datetime.combine(day, time(h, minute), tzinfo=JST)
                slots.append(local.astimezone(timezone.utc))
        day += timedelta(days=1)
    return slots[:n]


# ============================================
# 🔧 単体記事生成処理（タイトル→本文→画像→完了）
# ============================================
def _generate(app, aid: int, tpt: str, bpt: str, format: str = "html", self_review: bool = False):
    """
    単体記事生成関数（1記事ごとに呼び出される）
    - タイトル生成
    - 本文生成（format / self_review オプション対応）
    - アイキャッチ画像取得

    Args:
        app: Flaskアプリケーション
        aid: Article ID
        tpt: タイトルプロンプト
        bpt: 本文プロンプト
        format: "html" または "markdown"
        self_review: True の場合、GPTに自己添削を依頼する
    """
    with app.app_context():
        art = Article.query.get(aid)
        if not art or art.status != "pending":
            return

        try:
            if not art.title:
                art.title = f"{art.keyword}の記事タイトル"

            art.status, art.progress = "gen", 10
            db.session.flush()

            # タイトルがある前提で本文生成へ（進捗50%）
            art.progress = 50
            db.session.flush()

            # ✅ 新しいオプション付き本文生成
            art.body = _compose_body(
                kw=art.keyword,
                pt=bpt,
                format=format,
                self_review=self_review
            )
            art.progress = 80
            db.session.flush()

            # ✅ アイキャッチ画像（1つ目のh2見出しを参照）
            match = re.search(r"<h2\b[^>]*>(.*?)</h2>", art.body or "", re.IGNORECASE)
            first_h2 = match.group(1) if match else ""
            query = f"{art.keyword} {first_h2}".strip()
            art.image_url = fetch_featured_image_from_body(art.body, art.keyword)

            art.status = "done"
            art.progress = 100
            art.updated_at = datetime.utcnow()
            db.session.commit()

        except Exception as e:
            logging.exception(f"Error generating article ID {aid}: {e}")
            art.status = "error"
            art.body = f"Error: {e}"
            db.session.commit()

        finally:
            # ✅ セッション明示的に解放（接続プールの無駄な保持を防止）
            db.session.close()    


# ============================================
# 🔧 非同期一括生成（enqueue）【修正版】
# ============================================
def enqueue_generation(
    user_id: int,
    keywords: List[str],
    title_prompt: str,
    body_prompt: str,
    site_id: int,
    format: str = "html",
    self_review: bool = False
) -> None:
    """
    複数記事を並列生成キューに追加。ThreadPoolExecutor により同時並列生成。
    """
    if site_id is None:
        raise ValueError("site_id is required for scheduling")

    app = current_app._get_current_object()
    copies = [random.randint(2, 3) for _ in keywords[:40]]
    total = sum(copies)
    slots = iter(_generate_slots_per_site(app, site_id, total))

    def _bg():
        with app.app_context():
            ids: list[int] = []

            # DBへの記事登録処理（生成前）
            for kw, c in zip(keywords[:40], copies):
                for _ in range(c):
                    try:
                        title = _unique_title(kw.strip(), title_prompt)
                        art = Article(
                            keyword=kw.strip(),
                            title=title,
                            user_id=user_id,
                            site_id=site_id,
                            status="pending",
                            progress=0,
                            scheduled_at=next(slots, None),
                        )
                        db.session.add(art)
                        db.session.flush()
                        ids.append(art.id)
                    except Exception as e:
                        db.session.rollback()
                        logging.exception(f"[登録失敗] keyword='{kw}': {e}")
            db.session.commit()

            # 並列生成処理
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for aid in ids:
                    futures.append(executor.submit(
                        _generate, app, aid, title_prompt, body_prompt, format, self_review
                    ))
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logging.exception(f"[並列生成中の例外] {e}")

    threading.Thread(target=_bg, daemon=True).start()


# ============================================
# 🔧 同期生成用（主に再生成用）
# ============================================
def _generate_and_wait(app, aid, tpt, bpt):
    event = Event()
    def background():
        _generate(app, aid, tpt, bpt)
        event.set()
    threading.Thread(target=background, daemon=True).start()
    event.wait()

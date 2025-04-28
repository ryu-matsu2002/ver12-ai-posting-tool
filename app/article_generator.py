# ─────────────────────────────────────────────
# app/article_generator.py   – v9-dynlen (2025-05-XX)
# ─────────────────────────────────────────────
"""
● 文字数レンジを厳守し、途中で途切れない本文を生成する完全版
"""

from __future__ import annotations
import os, re, random, threading, logging
from datetime import datetime, date, time, timedelta, timezone
from typing import List, Dict, Tuple

from difflib import SequenceMatcher
import pytz
from flask import current_app
from openai import OpenAI, BadRequestError
from sqlalchemy import func

from . import db
from .models import Article
from .image_utils import fetch_featured_image

# ──────────────────────────────
# OpenAI 共通設定
# ──────────────────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
MODEL  = os.getenv("OPENAI_MODEL", "gpt-4-turbo")

TOKENS = {
    "title":   80,
    "outline": 400,
    # 1 ブロック最大 1 400 字 ≒ 780 token 程度を想定
    "block_max": 900,             # 動的設定用の上限値
}

TEMP = {"title": 0.4, "outline": 0.45, "block": 0.7}

CTX_LIMIT              = 4096
SHRINK                 = 0.8          # BadRequest 時のリトライ係数

MIN_BODY_CHARS_DEFAULT = 1800   # 下限
MAX_BODY_CHARS_DEFAULT = 2100   # 上限を 2100 に絞る
AVG_BLOCK_CHARS        = 500    # ブロック平均も 500 に
TOKENS["block_max"]    = 1000   # 余裕を持たせる
TITLE_DUP_THRESH       = 0.9
MAX_TITLE_RETRY        = 7

# ──────────────────────────────
# スケジュール設定（1 日 ≤5 本、端数分）
# ──────────────────────────────
JST         = pytz.timezone("Asia/Tokyo")

# 2 時間間隔でベースとなる “時” を固定（きり良い間隔は保つ）
BASE_HOURS  = [10, 12, 14, 16, 18]

# 0,15,30,45 を避けた “端数分” 候補
ODD_MINUTES = [3, 17, 23, 33, 37, 43, 47, 53]

MAX_PERDAY  = 5          # hard-limit
AVG_PERDAY  = 4          # あくまで「目安」— 超える場合もあります
# ──────────────────────────────
def _generate_slots(app, n: int) -> List[datetime]:
    """
    ・1 日あたり BASE_HOURS×ODD_MINUTES から最大 5 本割当て
    ・平均 4 本程度に収束（確率的）
    ・既存予約を考慮して空き枠に入れる
    """
    if n <= 0:
        return []

    # ① 既存予約を JST 日単位で集計
    with app.app_context():
        jst_date = func.date(func.timezone("Asia/Tokyo", Article.scheduled_at)).label("jst_date")
        rows = (
            db.session.query(jst_date, func.count(Article.id))
            .filter(Article.scheduled_at.isnot(None))
            .group_by(jst_date)
            .all()
        )
    booked_per_day: dict[date, int] = {row[0]: row[1] for row in rows}

    # ② スロット割当て
    slots: list[datetime] = []
    day = date.today() + timedelta(days=1)

    while len(slots) < n:
        already   = booked_per_day.get(day, 0)
        # その日の割当本数を決定（平均 4 本前後）
        available = min(MAX_PERDAY - already, n - len(slots))
        if available > 0:
            # 「出来れば 4 本」に寄せるため乱数で決定
            today_quota = min(
                available,
                random.choices(                # 4 本が最頻になるよう重み付け
                    population=[1, 2, 3, 4, 5],
                    weights   =[5, 15, 25, 35, 20],
                    k=1
                )[0]
            )

            # 空き時間帯からランダムに today_quota 件選択
            candidates = []
            for h in BASE_HOURS:
                for m in ODD_MINUTES:
                    local = datetime.combine(day, time(h, m), tzinfo=JST)
                    candidates.append(local)

            random.shuffle(candidates)
            for local in candidates[:today_quota]:
                slots.append(local.astimezone(timezone.utc))
                if len(slots) >= n:
                    break

        day += timedelta(days=1)

        # 安全ブレーク（12 か月先まで）
        if (day - date.today()).days > 365:
            raise RuntimeError("slot generation runaway")

    return slots[:n]


# ──────────────────────────────
# コンテンツ生成設定
# ──────────────────────────────
PERSONAS = [
    "節約志向の学生", "ビジネス渡航が多い会社員",
    "小さな子供連れファミリー", "リタイア後の移住者",
    "ペット同伴で移動する読者"
]

SAFE_SYS = (
    "あなたは一流の日本語 SEO ライターです。"
    "公序良俗に反する表現・誤情報・個人情報や差別的・政治的主張は禁止します。"
)

# ──────────────────────────────
# Chat API ラッパ
# ──────────────────────────────
def _tok_est(text: str) -> int:
    """ざっくり文字数→token 変換（日本語 1token≒1.8字）"""
    return int(len(text) / 1.8)

def _chat(msgs: List[Dict[str, str]], max_t: int, temp: float) -> str:
    prompt_tok = sum(_tok_est(m["content"]) for m in msgs)
    max_t = min(max_t, max(256, CTX_LIMIT - prompt_tok - 64))

    def _call(m: int) -> str:
        res = client.chat.completions.create(
            model=MODEL,
            messages=msgs,
            max_tokens=m,
            temperature=temp,
            timeout=120,
        )
        return res.choices[0].message.content.strip()

    try:
        return _call(max_t)
    except BadRequestError as e:
        if "max_tokens" in str(e):
            return _call(int(max_t * SHRINK))
        # ChatGPT が “” を返す／理由不明で例外にならない場合
    if not (_res := _call(int(max_t * 0.7))):
        raise RuntimeError("ChatGPT returned empty string twice")
    return _res

# ══════════════════════════════════════════════
# タイトル生成
# ══════════════════════════════════════════════
def _similar(a: str, b: str) -> bool:
    return SequenceMatcher(None, a, b).ratio() >= TITLE_DUP_THRESH

def _title_once(kw: str, pt: str, retry: bool) -> str:
    extra = "\n※既存と類似するため、異なる切り口にしてください。" if retry else ""
    usr   = f"{pt}{extra}\n\n▼ 条件\n- KW を含める\n- 末尾は？\n▼ KW: {kw}"
    sys   = SAFE_SYS + "Q&A形式タイトルを1行で返してください。"
    return _chat(
        [{"role":"system","content":sys},
         {"role":"user","content":usr}],
        TOKENS["title"], TEMP["title"]
    )

def _unique_title(kw: str, pt: str) -> str:
    # DB と既生成分を合わせて重複チェック
    history = [
        t[0] for t in db.session.query(Article.title)
                    .filter(Article.keyword==kw, Article.title.isnot(None))
    ]
    cand = ""
    for i in range(MAX_TITLE_RETRY):
        cand = _title_once(kw, pt, retry=(i>0))
        if not any(_similar(cand, h) for h in history):
            # DB に追加して次回も参照可能に
            history.append(cand)
            break
    return cand

# ══════════════════════════════════════════════
# アウトライン & 本文生成
# ══════════════════════════════════════════════
def _outline(kw: str, title: str, pt: str) -> str:
    sys = (
        SAFE_SYS
        + "必ず Markdown形式の##/###で6見出し以上の**小見出し（各20字以内）**のアウトラインを返してください。"
    )
    usr = f"{pt}\n\n▼ KW:{kw}\n▼ TITLE:{title}"
    return _chat(
        [{"role":"system","content":sys},
         {"role":"user","content":usr}],
        TOKENS["outline"], TEMP["outline"]
    )

def _parse_outline(raw: str) -> List[Tuple[str,List[str]]]:
    blocks, cur, subs = [], None, []
    for ln in raw.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("## "):
            if cur:
                blocks.append((cur, subs))
            cur, subs = s[3:], []
        elif s.startswith("### "):
            subs.append(s[4:])
        else:
            if cur:
                blocks.append((cur, subs))
            cur, subs = s, []
    if cur:
        blocks.append((cur, subs))
    return blocks

# ──────────────────────────────
# セクション生成（動的長さ）
# ──────────────────────────────
def _block_html(
    kw: str,
    h2: str,
    h3s: List[str],
    persona: str,
    pt: str,
    target_chars: int,
) -> str:
    low = int(target_chars * 0.9)
    high = int(target_chars * 1.1)
    h3_mark = "\n".join(f"### {h}" for h in h3s) if h3s else ""

    sys = (
        "あなたは一流の日本語 SEO ライターです。"
        f"以下の条件で <h2> セクションを HTML で生成してください。\n"
        f"- 小見出し (H2) は {h2}\n"
        f"- 本文は {low}〜{high} 字で完結させる\n"
        "- 構成: 結論→理由→具体例×3→要点まとめ\n"
        "- 具体例は <h3 class=\"wp-heading\"> で示す\n"
        "- H2/H3 には class=\"wp-heading\" を必ず付ける\n"
        f"- 視点: {persona}\n"
    )
    usr = (
        f"{pt}\n\n▼ キーワード: {kw}\n"
        f"▼ H2: {h2}\n"
        f"▼ H3 候補:\n{h3_mark}"
    )

    # 動的 token 上限
    max_tok = min(TOKENS["block_max"], _tok_est(str(target_chars)) + 150)
    return _chat(
        [{"role": "system", "content": sys},
         {"role": "user",   "content": usr}],
        max_tok,
        TEMP["block"]
    )

def _parse_range(pt: str) -> tuple[int, int | None]:
    """プロンプトから「○○字から△△字」or「○○字」パターンを抽出"""
    if m := re.search(r"(\d{3,5})\s*字から\s*(\d{3,5})\s*字", pt):
        return int(m.group(1)), int(m.group(2))
    if m := re.search(r"(\d{3,5})\s*字", pt):
        return int(m.group(1)), None
    return MIN_BODY_CHARS_DEFAULT, None

# ─────────────────────────────────────────────
# _compose_body  – 目標文字数を厳守して生成
# ─────────────────────────────────────────────
def _compose_body(kw: str, outline_raw: str, pt: str) -> str:
    # ① プロンプトから目標文字数レンジを取得
    def _range_in_pt(p: str) -> tuple[int, int | None]:
        if m := re.search(r"(\d{3,5})\s*字から\s*(\d{3,5})\s*字", p):
            return int(m.group(1)), int(m.group(2))
        if m := re.search(r"(\d{3,5})\s*字", p):
            return int(m.group(1)), None
        return MIN_BODY_CHARS_DEFAULT, None

    min_chars, max_chars_user = _range_in_pt(pt)
    max_total = max_chars_user or MAX_BODY_CHARS_DEFAULT

    # ② アウトラインを解析
    blocks = _parse_outline(outline_raw)
    if not blocks:
        raise RuntimeError("outline parse failed")

    # ③ 必要ブロック数を決定
    need_blocks = min(len(blocks), max(
        3,
        round(max_total / AVG_BLOCK_CHARS)
    ))
    blocks = blocks[:need_blocks]

    # ④ 各ブロックを順次生成（残り文字数で動的割当て）
    html_parts: List[str] = []
    remaining = max_total
    remaining_blocks = len(blocks)

    for h2, h3s in blocks:
        target = round(remaining / remaining_blocks)
        section = _block_html(
            kw, h2, h3s[:3], random.choice(PERSONAS), pt, target_chars=target
        )
        html_parts.append(section)

        # 更新
        remaining -= len(section)
        remaining_blocks -= 1

    body_html = "\n\n".join(html_parts)

    # ⑤ 下限不足ならまとめセクションで補完
    if len(body_html) < min_chars:
        add = (
            "<h2 class=\"wp-heading\">まとめ</h2>"
            "<p>この記事の要点を簡潔に振り返りました。</p>"
        )
        body_html += "\n\n" + add

    # ⑥ 上限超過時は安全に切り詰めて末尾に要約
    if len(body_html) > max_total:
        snippet = body_html[:max_total]
        cut = max(snippet.rfind("</p>"), snippet.rfind("</h2>"), snippet.rfind("</h3>"))
        snippet = snippet[:cut + 5] if cut != -1 else snippet
        snippet += (
            "\n\n<p><!-- notice: 長さ超過のため自動トリム。全文はプレビューで確認 --></p>"
        )
        body_html = snippet

    logging.debug("compose_body len=%s  (target≤%s)", len(body_html), max_total)
    return body_html




# ──────────────────────────────
# 生成ワーカー
# ──────────────────────────────
def _generate(app, aid: int, tpt: str, bpt: str):
    with app.app_context():
        art = Article.query.get(aid)
        if not art or art.status != "pending":
            return

        try:
            # 進捗: 開始
            art.status, art.progress = "gen", 10
            db.session.commit()

            # タイトル
            art.title = _unique_title(art.keyword, tpt)
            art.progress = 30
            db.session.commit()

            # アウトライン
            outline = _outline(art.keyword, art.title, bpt)
            art.progress = 50
            db.session.commit()

            # 本文
            art.body = _compose_body(art.keyword, outline, bpt)
            art.progress = 80
            db.session.commit()

            # アイキャッチ（失敗してもデフォルトを返すので例外にはしない）
            h2s   = re.findall(r"<h2\b[^>]*>(.*?)</h2>", art.body or "", re.I)[:2]
            query = " ".join(dict.fromkeys([art.keyword, art.title, *h2s]))
            art.image_url = fetch_featured_image(query)

            # 完了
            art.status, art.progress = "done", 100
            art.updated_at = datetime.utcnow()

        except Exception as e:
            logging.exception("記事生成失敗: %s", e)
            art.status, art.body = "error", f"Error: {e}"

        finally:
            # 生成成功／失敗にかかわらずセッションを確実に反映
            db.session.commit()


def enqueue_generation(
    user_id: int,
    keywords: List[str],
    title_prompt: str,
    body_prompt: str,
    site_id: int | None = None,
) -> None:
    """
    1 キーワードにつき 1〜3 本の記事を生成（上限 40 キーワード）。
    生成直後に scheduled_at を必ず埋める。
    """
    app = current_app._get_current_object()

    # 生成予定本数を先に算出して slot 確保
    copies_each_kw = [random.randint(1, 3) for _ in keywords[:40]]
    total_posts    = sum(copies_each_kw)
    slots_iter     = iter(_generate_slots(app, total_posts))

    def background():
        with app.app_context():
            article_ids: list[int] = []
            for kw, copies in zip(keywords[:40], copies_each_kw):
                for _ in range(copies):
                    art = Article(
                        keyword      = kw.strip(),
                        user_id      = user_id,
                        site_id      = site_id,
                        status       = "pending",
                        progress     = 0,
                        scheduled_at = next(slots_iter, None),  # ← 必ずセット
                    )
                    db.session.add(art)
                    db.session.flush()          # art.id を確定
                    article_ids.append(art.id)

            db.session.commit()

        # タイトル重複を避けるため直列生成
        for aid in article_ids:
            from .article_generator import _generate  # 遅延 import
            _generate(app, aid, title_prompt, body_prompt)

    threading.Thread(target=background, daemon=True).start()

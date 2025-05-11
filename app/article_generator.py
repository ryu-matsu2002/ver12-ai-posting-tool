# app/article_generator.py
# 修正版（Pixabay対応 + バグ修正済）

import os, re, random, threading, logging, requests
from datetime import datetime, date, timedelta, time, timezone
from typing import List, Dict, Tuple
from difflib import SequenceMatcher
import pytz
import re
from flask import current_app
from openai import OpenAI, BadRequestError
from sqlalchemy import func
from threading import Event
from .image_utils import fetch_featured_image
from . import db
from .models import Article

# OpenAI設定
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TOKENS = {"title": 120, "outline": 800, "block": 3000}
TEMP = {"title": 0.6, "outline": 0.65, "block": 0.7}
TOP_P = 0.9
CTX_LIMIT = 12000
SHRINK = 0.6
AVG_BLOCK_CHARS = 600
MIN_BODY_CHARS_DEFAULT = 1800
MAX_BODY_CHARS_DEFAULT = 4000
MAX_TITLE_RETRY = 7
TITLE_DUP_THRESH = 0.90

# スケジュール設定（JST）
JST = pytz.timezone("Asia/Tokyo")
POST_HOURS = list(range(10, 21))
MAX_PERDAY = 5
AVERAGE_POSTS = 4


def _generate_slots_per_site(app, site_id: int, n: int) -> List[datetime]:
    """
    特定のサイトごとに、1日3～5記事（平均4記事）ルールに従って投稿スロットを生成する。
    """
    if n <= 0:
        return []

    with app.app_context():
        # JST で日単位にすでに何件スケジュールされているか取得
        jst_date = func.date(func.timezone("Asia/Tokyo", Article.scheduled_at))
        rows = db.session.query(jst_date.label("d"), func.count(Article.id))\
            .filter(
                Article.site_id == site_id,
                Article.scheduled_at.isnot(None)
            ).group_by("d").all()

    # 日付ごとの予約数を dict に
    booked = {d: c for d, c in rows}
    slots = []
    day = date.today() + timedelta(days=1)

    while len(slots) < n:
        # その日の残り投稿枠を計算
        remain = MAX_PERDAY - booked.get(day, 0)
        if remain > 0:
            # ランダムで 1～5件、ただし remain/n に応じて調整
            need = min(random.randint(1, AVERAGE_POSTS), remain, n - len(slots))
            for h in sorted(random.sample(POST_HOURS, need)):
                minute = random.randint(1, 59)
                local = datetime.combine(day, time(h, minute), tzinfo=JST)
                slots.append(local.astimezone(timezone.utc))  # UTC に変換して保存
        day += timedelta(days=1)

        if (day - date.today()).days > 365:
            raise RuntimeError("slot generation runaway")

    current_app.logger.debug(f"Generated {n} slots for site {site_id}: {slots}")
    return slots[:n]


SAFE_SYS = "あなたは一流の日本語 SEO ライターです。SEOを意識した見出しや本文を構成し、読者にとって有益な情報を提供してください。"

def _tok(s: str) -> int:
    return int(len(s) / 1.8)

def clean_gpt_output(text: str) -> str:
    text = re.sub(r"```(?:html)?", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()


def _chat(msgs: List[Dict[str, str]], max_t: int, temp: float) -> str:
    used = sum(_tok(m["content"]) for m in msgs)
    available = CTX_LIMIT - used - 16
    max_t = min(max_t, available)
    if max_t < 1:
        logging.error(f"max_tokens below minimum: {max_t} (used: {used})")
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
        finish = res.choices[0].finish_reason
        content = res.choices[0].message.content.strip()

        # ✅ usageログの保護付き表示
        usage = getattr(res, "usage", None)
        if usage:
            logging.info(
                f"[ChatGPT] finish_reason={finish} | tokens: prompt={usage.prompt_tokens}, "
                f"completion={usage.completion_tokens}, total={usage.total_tokens}"
            )

        if finish == "length":
            logging.warning("⚠️ OpenAI response was cut off due to max_tokens.")
            content += "\n<p><em>※この文章はトークン上限で途中終了した可能性があります。</em></p>"

        content = re.sub(r"^```html\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"^```\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        content = clean_gpt_output(content)
        return content

    try:
        return _call(max_t)
    except BadRequestError as e:
        if "max_tokens" in str(e):
            retry_t = max(1, int(max_t * SHRINK))
            return _call(retry_t)
        raise



def _similar(a: str, b: str) -> bool:
    return SequenceMatcher(None, a, b).ratio() >= TITLE_DUP_THRESH

def _title_once(kw: str, pt: str, retry: bool) -> str:
    extra = "\n※過去に使われたタイトルや似たタイトルを絶対に避けてください。" if retry else ""
    usr = f"{pt}{extra}\n\n▼ 条件\n- 必ずキーワードを含める\n- タイトルはユニークであること\n▼ キーワード: {kw}"
    sys = SAFE_SYS + "魅力的な日本語タイトルを 1 行だけ返してください。"
    return _chat([
        {"role": "system", "content": sys},
        {"role": "user", "content": usr},
    ], TOKENS["title"], TEMP["title"])

def _unique_title(kw: str, pt: str) -> str:
    history = [t[0] for t in db.session.query(Article.title).filter(Article.keyword == kw)]
    for i in range(MAX_TITLE_RETRY):
        cand = _title_once(kw, pt, retry=i > 0)
        if not any(_similar(cand, h) for h in history):
            return cand
    logging.error(f"[タイトル生成失敗] keyword={kw}")
    raise ValueError(f"タイトル生成に失敗しました: {kw}")

# outline, body 作成（省略せずに続きが必要なら送信）


def _outline(kw: str, title: str, pt: str) -> str:
    sys = SAFE_SYS + "## / ### で見出しを生成し、記事の内容に合わせて柔軟に調整します。"
    usr = f"{pt}\n\n▼ KW: {kw}\n▼ TITLE: {title}"
    return _chat([
        {"role": "system", "content": sys},
        {"role": "user", "content": usr},
    ], TOKENS["outline"], TEMP["outline"])

def _parse_outline(raw: str) -> List[Tuple[str, List[str]]]:
    blocks, h2, h3s = [], None, []
    for ln in raw.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("## "):
            if h2:
                blocks.append((h2, h3s))
            h2, h3s = s[3:], []
        elif s.startswith("### "):
            h3s.append(s[4:])
        else:
            if h2:
                blocks.append((h2, h3s))
            h2, h3s = s, []
    if h2:
        blocks.append((h2, h3s))
    return blocks

def _block_html(kw: str, h2: str, h3s: List[str], persona: str, pt: str) -> str:
    h3_mark = "\n".join(f"### {h}" for h in h3s) if h3s else ""
    sys = (
        SAFE_SYS +
        "以下の条件で <h2> セクションを HTML 生成してください。\n"
        "- この H2 ブロックは 550〜750 字でまとめる\n"
        "- 小見出し(H2) は 15 字以内\n"
        "- 構成: 結論→理由→具体例×3→再結論\n"
        "- 具体例は <h3 class=\"wp-heading\"> で示す\n"
        f"- 視点: {persona}\n"
        "- <h2>/<h3> には class=\"wp-heading\" を付与"
    )
    usr = f"{pt}\n\n▼ キーワード: {kw}\n▼ H2: {h2}\n▼ H3 候補:\n{h3_mark}"
    return _chat([
        {"role": "system", "content": sys},
        {"role": "user", "content": usr},
    ], TOKENS["block"], TEMP["block"])

def _parse_range(pt: str) -> Tuple[int, int | None]:
    if m := re.search(r"(\d{3,5})\s*字から\s*(\d{3,5})\s*字", pt):
        return int(m.group(1)), int(m.group(2))
    if m := re.search(r"(\d{3,5})\s*字", pt):
        return int(m.group(1)), None

    pt_len = len(pt)
    if pt_len < 500:
        return 800, 1200
    elif pt_len < 1000:
        return 1200, 1800
    elif pt_len < 1500:
        return 1800, 2400
    else:
        return 2200, 3000


def _compose_body(kw: str, outline_raw: str, pt: str) -> str:
    min_chars, max_chars_user = _parse_range(pt)
    max_total = max_chars_user or MAX_BODY_CHARS_DEFAULT
    outline = _parse_outline(outline_raw)
    parts: List[str] = []

    for h2, h3s in outline:
        h2_short = (h2[:15] + "…") if len(h2) > 15 else h2
        h3s_limited = [h for h in h3s if len(h) <= 10][:3]
        block_html = _block_html(kw, h2_short, h3s_limited, "default_persona", pt)
        parts.append(block_html)

    # 🔰 まとめセクション生成
    summary_prompt_sys = (
        SAFE_SYS +
        "以下の本文を要約して、<h2 class=\"wp-heading\">まとめ</h2><p>～</p> を HTML で返してください。\n"
        "・最後に読了感があるように結論やおすすめなどで締めくくってください。"
    )
    summary_prompt_usr = "\n\n".join(parts) + "\n\n▼ 上記をまとめてください。"

    summary_html = _chat([
        {"role": "system", "content": summary_prompt_sys},
        {"role": "user", "content": summary_prompt_usr}
    ], TOKENS["block"], TEMP["block"]).strip()

    # 🔧 応答が <h2> から始まらなければ明示的に囲む
    if not summary_html.startswith("<h2"):
        summary_html = '<h2 class="wp-heading">まとめ</h2><p>' + summary_html + '</p>'

    full = "\n\n".join(parts + [summary_html])

    # 🔧 長すぎる場合は安全に切り取る
    if len(full) > max_total:
        snippet = full[:max_total]

        # 🔧 最後の <p>, <h2>, <h3> の終了タグ位置を探す
        cut = max(
            snippet.rfind("</p>"),
            snippet.rfind("</h2>"),
            snippet.rfind("</h3>")
        )

        # 🔧 安全にタグごとカット、それでも見つからなければそのまま切る
        full = snippet[:cut + 5] if cut != -1 else snippet

        # 🔧 不完全タグで終わってたら <p> で閉じる
        if not full.strip().endswith("</p>"):
            full += "</p>"

        logging.warning("⚠️ 本文が最大長を超えたため安全に切り取りました")

    logging.debug("compose_body len=%s (max=%s)", len(full), max_total)
    return full

def _parse_range(pt: str) -> Tuple[int, int | None]:
    """
    ユーザーの本文プロンプトの文字数に応じて、生成する本文の長さ（文字数）を調整する。
    ユーザーが「○○字から○○字」と明示した場合はその指定を優先。
    """
    # ユーザーが明示的に文字数範囲を指定している場合
    if m := re.search(r"(\d{3,5})\s*字から\s*(\d{3,5})\s*字", pt):
        return int(m.group(1)), int(m.group(2))
    if m := re.search(r"(\d{3,5})\s*字", pt):
        return int(m.group(1)), None

    # 🔧 自動調整（プロンプトの長さベース）
    pt_len = len(pt)

    if pt_len < 500:
        return 800, 1200
    elif pt_len < 1000:
        return 1200, 1800
    elif pt_len < 1500:
        return 1800, 2400
    else:
        return 2200, 3000



def _generate(app, aid: int, tpt: str, bpt: str):

    with app.app_context():
        art = Article.query.get(aid)
        if not art or art.status != "pending":
            return
        try:
            if not art.title:
                art.title = f"{art.keyword}の記事タイトル"
                logging.warning(f"Title was empty, setting default title: {art.title}")

            art.status, art.progress = "gen", 10
            db.session.flush()

            # ✅ STEP1: アウトライン生成
            outline = _outline(art.keyword, art.title, bpt)
            art.progress = 50
            db.session.flush()

            # ✅ STEP2: 本文生成
            art.body = _compose_body(art.keyword, outline, bpt)
            art.progress = 80
            db.session.flush()

            # ✅ STEP3: アイキャッチ画像取得（キーワード + h2 で精度強化）
            match = re.search(r"<h2\b[^>]*>(.*?)</h2>", art.body or "", re.IGNORECASE)
            first_h2 = match.group(1) if match else ""
            query = f"{art.keyword} {first_h2}".strip()
            art.image_url = fetch_featured_image(query)  # ✅ 1引数に統一

            # ✅ STEP4: 完了処理
            art.status = "done"
            art.progress = 100
            art.updated_at = datetime.utcnow()
            db.session.commit()

            logging.info(f"Completed article ID {aid} generation.")

        except Exception as e:
            logging.exception(f"Error generating article ID {aid}: {e}")
            art.status = "error"
            art.body = f"Error: {e}"
            db.session.commit()

        finally:
            db.session.commit()

def enqueue_generation(user_id: int,
                       keywords: List[str],
                       title_prompt: str,
                       body_prompt: str,
                       site_id: int) -> None:
    if site_id is None:
        raise ValueError("site_id is required for scheduling")

    app = current_app._get_current_object()
    copies = [random.randint(1, 3) for _ in keywords[:40]]
    total = sum(copies)

    # サイトごとのスケジュール生成関数を使用
    slots = iter(_generate_slots_per_site(app, site_id, total))

    def _bg():
        with app.app_context():
            ids: list[int] = []
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
                        db.session.commit()
                        ids.append(art.id)
                    except Exception as e:
                        db.session.rollback()
                        logging.exception(f"Error creating Article for keyword '{kw}': {e}")
            for aid in ids:
                _generate(app, aid, title_prompt, body_prompt)

    threading.Thread(target=_bg, daemon=True).start()




def _generate_and_wait(app, aid, tpt, bpt):
    event = Event()
    def background():
        _generate(app, aid, tpt, bpt)
        event.set()
    threading.Thread(target=background, daemon=True).start()
    event.wait()

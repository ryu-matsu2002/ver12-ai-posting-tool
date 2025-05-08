# app/article_generator.py
# 修正版（Pixabay対応 + バグ修正済）

import os, re, random, threading, logging, requests
from datetime import datetime, date, timedelta, time, timezone
from typing import List, Dict, Tuple
from difflib import SequenceMatcher
import pytz
from flask import current_app
from openai import OpenAI, BadRequestError
from sqlalchemy import func
from threading import Event

from . import db
from .models import Article

# OpenAI設定
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TOKENS = {"title": 80, "outline": 400, "block": 3000}
TEMP = {"title": 0.7, "outline": 0.7, "block": 0.7}
TOP_P = 0.95
CTX_LIMIT = 4096
SHRINK = 0.75
AVG_BLOCK_CHARS = 600
MIN_BODY_CHARS_DEFAULT = 1800
MAX_BODY_CHARS_DEFAULT = 3000
MAX_TITLE_RETRY = 7
TITLE_DUP_THRESH = 0.90

# スケジュール設定（JST）
JST = pytz.timezone("Asia/Tokyo")
POST_HOURS = list(range(10, 21))
MAX_PERDAY = 5
AVERAGE_POSTS = 4

# Pixabay APIキー
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY", "")

def fetch_featured_image(body_text: str, keyword: str) -> str | None:
    """Pixabay API から記事用画像を1枚取得"""
    query = keyword.strip() or "ブログ"
    url = "https://pixabay.com/api/"
    params = {
        "key": PIXABAY_API_KEY,
        "q": query,
        "image_type": "photo",
        "per_page": 5,
        "safesearch": "true",
        "lang": "ja"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data["hits"][0]["webformatURL"] if data.get("hits") else None
    except Exception as e:
        logging.error(f"Pixabay fetch error: {e}")
        return None

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

def _chat(msgs: List[Dict[str, str]], max_t: int, temp: float) -> str:
    used = sum(_tok(m["content"]) for m in msgs)
    max_t = min(max_t, CTX_LIMIT - used - 16)
    def _call(m: int) -> str:
        res = client.chat.completions.create(
            model=MODEL, messages=msgs,
            max_tokens=m, temperature=temp, top_p=TOP_P, timeout=120,
        )
        return res.choices[0].message.content.strip()
    try:
        return _call(max_t)
    except BadRequestError as e:
        if "max_tokens" in str(e):
            return _call(int(max_t * SHRINK))
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
    return cand

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
    return MIN_BODY_CHARS_DEFAULT, None

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
    summary_prompt_sys = SAFE_SYS + "以下の本文を要約して、<h2 class=\"wp-heading\">まとめ</h2><p>～</p> を HTML で返してください。"
    summary_prompt_usr = "\n\n".join(parts) + "\n\n▼ 上記をまとめてください。"
    summary_html = _chat([
        {"role":"system","content":summary_prompt_sys},
        {"role":"user","content":summary_prompt_usr}
    ], TOKENS["block"], TEMP["block"]).strip()
    if not summary_html.startswith("<h2"):
        summary_html = '<h2 class="wp-heading">まとめ</h2><p>' + summary_html + '</p>'
    full = "\n\n".join(parts + [summary_html])
    if len(full) > max_total:
        snippet = full[:max_total]
        cut = max(snippet.rfind("</p>"), snippet.rfind("</h2>"), snippet.rfind("</h3>"))
        full = snippet[:cut+5] if cut != -1 else snippet
    logging.debug("compose_body len=%s (max=%s)", len(full), max_total)
    return full

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
            outline = _outline(art.keyword, art.title, bpt)
            art.progress = 50
            db.session.flush()
            art.body = _compose_body(art.keyword, outline, bpt)
            art.progress = 80
            db.session.flush()
            match = re.search(r"<h2\\b[^>]*>(.*?)</h2>", art.body or "", re.IGNORECASE)
            first_h2 = match.group(1) if match else ""
            query = f"{art.keyword} {first_h2}".strip()
            art.image_url = fetch_featured_image(art.body or "", query)
            art.status, art.progress = "done", 100
            art.updated_at = datetime.utcnow()
            db.session.commit()
            logging.info(f"Completed article ID {aid} generation.")
        except Exception as e:
            logging.exception(f"Error generating article ID {aid}: {e}")
            art.status, art.body = "error", f"Error: {e}"
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

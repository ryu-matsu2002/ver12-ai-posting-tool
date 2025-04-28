# ─────────────────────────────────────────────
# app/article_generator.py   – v8 (2025-05-XX)
# ─────────────────────────────────────────────
"""
記事生成エンジン

◎ v8 主要変更
 1. GPT 応答長不足を自動リトライ (+temperature)
 2. 1 ブロック 850-1000 字生成に合わせて token 上限増加
 3. compose_body が最終長不足なら追加ブロックを自動生成
 4. fetch_featured_image 改良版に対応（クエリ多様化）
 5. 大量生成時でも slot が被らないようダブルチェック
"""

from __future__ import annotations
import os, re, random, logging, threading
from datetime import datetime, date, time, timedelta, timezone
from typing import List, Dict, Tuple

from difflib import SequenceMatcher
import pytz
from flask import current_app
from openai import OpenAI, BadRequestError, RateLimitError

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
    "block":   1800,   # ← 十分な余裕
}
TEMP = {"title": 0.4, "outline": 0.45, "block": 0.7}

CTX_LIMIT = 4096
SHRINK    = 0.75      # max_tokens エラー時に掛ける係数

# 文字数関連
AVG_BLOCK_CHARS        = 850   # 目標
MIN_BODY_CHARS_DEFAULT = 2200
MAX_BODY_CHARS_DEFAULT = 3000

MAX_TITLE_RETRY  = 7
TITLE_DUP_THRESH = 0.9

# スケジュール
JST        = pytz.timezone("Asia/Tokyo")
POST_HOURS = list(range(10, 21))   # JST 10-20
MAX_PERDAY = 5

# ──────────────────────────────
# Slot generator  (unchanged logic + double-check)
# ──────────────────────────────
def _generate_slots(app, n: int) -> List[datetime]:
    if n <= 0:
        return []

    with app.app_context():
        jst_date = func.date(func.timezone("Asia/Tokyo",
                                           Article.scheduled_at)).label("jst")
        rows = (db.session.query(jst_date, func.count(Article.id))
                .filter(Article.scheduled_at.isnot(None))
                .group_by(jst_date).all())
    booked: dict[date, int] = {r[0]: r[1] for r in rows}

    slots: list[datetime] = []
    day = date.today() + timedelta(days=1)
    while len(slots) < n:
        remain = MAX_PERDAY - booked.get(day, 0) - slots.count(day)
        if remain > 0:
            need = min(remain, n - len(slots))
            for h in sorted(random.sample(POST_HOURS, need)):
                minute = random.randint(1, 59)
                local  = datetime.combine(day, time(h, minute), tzinfo=JST)
                slots.append(local.astimezone(timezone.utc))
        day += timedelta(days=1)
    return slots

# ──────────────────────────────
# LLM helper
# ──────────────────────────────
def _tok(txt: str) -> int:
    return int(len(txt) * 0.45)

def _call_llm(msgs, m_tokens, temp) -> str:
    res = client.chat.completions.create(
        model=MODEL, messages=msgs,
        max_tokens=m_tokens,
        temperature=temp,
        timeout=120,
    )
    return res.choices[0].message.content.strip()

def _chat(msgs: List[Dict[str,str]], max_t: int, temp: float) -> str:
    prompt = sum(_tok(m["content"]) for m in msgs)
    max_t  = min(max_t, max(256, CTX_LIMIT - prompt - 64))
    try:
        txt = _call_llm(msgs, max_t, temp)
        # 応答が異常に短い場合リトライ (一度だけ)
        if len(txt) < 200:
            txt = _call_llm(msgs, max_t, temp + 0.1)
        return txt
    except (BadRequestError, RateLimitError) as e:
        logging.warning("LLM BadRequest/RateLimit: %s → shrink", e)
        return _call_llm(msgs, int(max_t * SHRINK), temp)

# ──────────────────────────────
# Title generation
# ──────────────────────────────
def _similar(a: str, b: str) -> bool:
    return SequenceMatcher(None, a, b).ratio() >= TITLE_DUP_THRESH

def _unique_title(kw: str, prompt: str) -> str:
    existing = [
        t[0] for t in db.session.query(Article.title)
        .filter(Article.keyword == kw, Article.title.isnot(None))
    ]
    for i in range(MAX_TITLE_RETRY):
        extra = "\n※既存と類似するため、切り口を変えてください" if i else ""
        usr   = f"{prompt}{extra}\n\n▼条件\n- KW を含む\n- 末尾は？\n▼KW:{kw}"
        sys   = "あなたはSEOに強いライター。" \
                "一行のQ&A形式タイトルのみ出力。"
        title = _chat(
            [{"role":"system","content":sys},
             {"role":"user","content":usr}],
            TOKENS["title"], TEMP["title"])
        if not any(_similar(title, e) for e in existing):
            return title
    return title  # 最後の案 (多少近くても採用)

# ──────────────────────────────
# Outline / body helpers
# ──────────────────────────────
PERSONAS = ["節約志向の学生","ビジネス渡航が多い会社員",
            "小さな子供連れファミリー","リタイア後の移住者",
            "ペット同伴で移動する読者"]

SAFE_SYS = "あなたは一流の日本語 SEO ライターです。公序良俗に反する表現は禁止。"

def _outline(kw: str, title: str, pt: str) -> str:
    sys = SAFE_SYS + "Markdown の ##/### で 6 見出し以上のアウトラインを返す。"
    usr = f"{pt}\n\nKW:{kw}\nTITLE:{title}"
    return _chat(
        [{"role":"system","content":sys},
         {"role":"user","content":usr}],
        TOKENS["outline"], TEMP["outline"])

def _parse_outline(raw: str) -> List[Tuple[str,List[str]]]:
    h2, h3s, blocks = None, [], []
    for ln in raw.splitlines():
        s = ln.strip()
        if s.startswith("## "):
            if h2: blocks.append((h2, h3s))
            h2, h3s = s[3:], []
        elif s.startswith("### "):
            h3s.append(s[4:])
    if h2: blocks.append((h2, h3s))
    return blocks

def _block_html(kw, h2, h3s, persona, pt) -> str:
    h3txt = "\n".join(f"### {h}" for h in h3s) if h3s else ""
    sys = SAFE_SYS + (
        "以下条件で H2 セクションを HTML 生成:\n"
        "- 850～1000 字で本文\n"
        "- 結論→理由→具体例×3→再結論\n"
        "- <h2>/<h3> に class=\"wp-heading\"\n"
        f"- Persona: {persona}"
    )
    usr = f"{pt}\n\nKW:{kw}\nH2:{h2}\nH3s:\n{h3txt}"
    return _chat(
        [{"role":"system","content":sys},
         {"role":"user","content":usr}],
        TOKENS["block"], TEMP["block"])

def _parse_range(pt: str) -> tuple[int,int|None]:
    if m := re.search(r"(\d{3,5})字から(\d{3,5})字", pt):
        return int(m.group(1)), int(m.group(2))
    if m := re.search(r"(\d{3,5})字", pt):
        return int(m.group(1)), None
    return MIN_BODY_CHARS_DEFAULT, None

def _compose_body(kw: str, outline_raw: str, pt: str) -> str:
    min_chars, max_chars_user = _parse_range(pt)
    max_total = max_chars_user or MAX_BODY_CHARS_DEFAULT

    blocks = _parse_outline(outline_raw)
    need   = max(3, min(len(blocks),
                 (max_total + AVG_BLOCK_CHARS - 1)//AVG_BLOCK_CHARS))
    blocks = blocks[:need]

    parts = []
    for h2, h3s in blocks:
        h2 = h2[:20] + ("…" if len(h2) > 20 else "")
        h3s = [h[:10] for h in h3s][:3]
        parts.append(_block_html(
            kw, h2, h3s, random.choice(PERSONAS), pt))

    html = "\n\n".join(parts)
    html = re.sub(r"<h([23])(?![^>]*wp-heading)>",
                  r'<h\1 class="wp-heading">', html)

    # 足りなければ追加ブロック自動生成
    while len(html) < min_chars and len(blocks) < len(_parse_outline(outline_raw)):
        idx = len(blocks)
        h2, h3s = _parse_outline(outline_raw)[idx]
        blocks.append((h2, h3s))
        html += "\n\n" + _block_html(kw, h2, h3s,
                                     random.choice(PERSONAS), pt)

    # 長過ぎ対策
    if len(html) > max_total:
        html = html[:max_total]
        cut = max(html.rfind("</p>"),
                  html.rfind("</h2>"),
                  html.rfind("</h3>"))
        html = html[:cut+5] if cut != -1 else html

    return html

# ──────────────────────────────
# Main generation
# ──────────────────────────────
def _generate(app, aid: int, tpt: str, bpt: str):
    with app.app_context():
        art = Article.query.get(aid)
        if not art or art.status != "pending":
            return
        try:
            art.status = "gen"; art.progress = 10; db.session.commit()
            art.title  = _unique_title(art.keyword, tpt)
            art.progress = 30; db.session.commit()

            outline = _outline(art.keyword, art.title, bpt)
            art.progress = 50; db.session.commit()

            art.body = _compose_body(art.keyword, outline, bpt)
            art.progress = 80; db.session.commit()

            # 画像クエリ: KW + TITLE + 最初の H2
            h2s = re.findall(r"<h2\b[^>]*>(.*?)</h2>", art.body or "", re.I)[:1]
            query = " ".join(dict.fromkeys([art.keyword, art.title, *h2s]))
            art.image_url = fetch_featured_image(query)

            art.status = "done"; art.progress = 100
            art.updated_at = datetime.utcnow()
            db.session.commit()
        except Exception as e:
            logging.exception("記事生成失敗: %s", e)
            art.status, art.body = "error", f"Error: {e}"
            db.session.commit()

# ──────────────────────────────
# Enqueue
# ──────────────────────────────
def enqueue_generation(user_id: int, keywords: List[str],
                       title_pt: str, body_pt: str,
                       site_id: int|None = None) -> None:
    app = current_app._get_current_object()
    copies = [random.randint(1,3) for _ in keywords[:40]]
    total  = sum(copies)
    slots  = iter(_generate_slots(app, total))

    def bg():
        with app.app_context():
            ids = []
            for kw, c in zip(keywords[:40], copies):
                for _ in range(c):
                    art = Article(
                        keyword=kw.strip(), user_id=user_id,
                        site_id=site_id, status="pending", progress=0,
                        scheduled_at=next(slots, None))
                    db.session.add(art); db.session.flush(); ids.append(art.id)
            db.session.commit()

        for aid in ids:
            _generate(app, aid, title_pt, body_pt)

    threading.Thread(target=bg, daemon=True).start()

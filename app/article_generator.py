# ─────────────────────────────────────────────
# app/article_generator.py   – v7-full+ (2025-05-XX)
# ─────────────────────────────────────────────
"""
● 記事生成 + 予約投稿時刻自動決定
  - タイトル重複判定
  - 本文見出し構成 + class付与
  - 最低字数 2,000 (プロンプト依存で可変)
  - 画像取得: 本文先頭 H2 or キーワード でクエリ強化
"""

from __future__ import annotations
import os, re, random, threading, logging, math
from datetime import datetime, timedelta, date, time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple

from difflib import SequenceMatcher
import pytz
from flask import current_app
from openai import OpenAI, BadRequestError

from . import db
from .models import Article
from .image_utils import fetch_featured_image

# ──────────────────────────────
# OpenAI 共通設定
# ──────────────────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
MODEL  = os.getenv("OPENAI_MODEL", "gpt-4-turbo")
TOKENS = {"title": 80, "outline": 400, "block": 950}
TEMP   = {"title": 0.40, "outline": 0.45, "block": 0.70}

CTX_LIMIT              = 4096
SHRINK                 = 0.75
MIN_BODY_CHARS_DEFAULT = 2_000   # ← デフォルト 2000字に
MAX_TITLE_RETRY        = 7
TITLE_DUP_THRESH       = 0.80

# ──────────────────────────────
# スケジュール設定
# ──────────────────────────────
JST        = pytz.timezone("Asia/Tokyo")
POST_HOURS = list(range(10, 21))
LAMBDA     = 4
MAX_PERDAY = 5

def _poisson_rand(lambda_: float = LAMBDA) -> int:
    L, k, p = math.exp(-lambda_), 0, 1.0
    while p > L:
        k += 1
        p *= random.random()
    return k or 1

def _generate_slots(n: int) -> List[datetime]:
    slots, cur = [], date.today()
    while len(slots) < n:
        cnt = min(MAX_PERDAY, _poisson_rand())
        hrs = random.sample(POST_HOURS, min(cnt, len(POST_HOURS)))
        for h in hrs:
            dt = datetime.combine(cur, time(hour=h), tzinfo=JST)
            slots.append(dt.astimezone(pytz.utc))
        cur += timedelta(days=1)
    return sorted(slots)[:n]

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
# Chat API wrapper
# ──────────────────────────────
def _tok(txt: str) -> int:
    return int(len(txt) * 0.45)

def _chat(msgs: List[Dict[str, str]], max_t: int, temp: float) -> str:
    prompt = sum(_tok(m["content"]) for m in msgs)
    max_t  = min(max_t, max(256, CTX_LIMIT - prompt - 64))
    def call(m: int) -> str:
        res = client.chat.completions.create(
            model=MODEL, messages=msgs,
            max_tokens=m, temperature=temp, timeout=120
        )
        return res.choices[0].message.content.strip()
    try:
        return call(max_t)
    except BadRequestError as e:
        if "max_tokens" in str(e):
            return call(int(max_t * SHRINK))
        raise

# ──────────────────────────────
# タイトル生成
# ──────────────────────────────
def _similar(a: str, b: str) -> bool:
    return SequenceMatcher(None, a, b).ratio() >= TITLE_DUP_THRESH

def _title_once(kw: str, pt: str, retry: bool) -> str:
    extra = "\n※既存と類似するため、異なる切り口にしてください。" if retry else ""
    usr   = f"{pt}{extra}\n\n▼ 条件\n- KW を含める\n- 末尾は？\n▼ KW: {kw}"
    sys   = SAFE_SYS + "Q&A形式タイトルを1行で返してください。"
    return _chat([{"role":"system","content":sys},
                  {"role":"user","content":usr}],
                 TOKENS["title"], TEMP["title"])

def _unique_title(kw: str, pt: str) -> str:
    history = [t[0] for t in db.session.query(Article.title)
                         .filter(Article.keyword==kw,
                                 Article.title.isnot(None))]
    for i in range(MAX_TITLE_RETRY):
        cand = _title_once(kw, pt, retry=(i>0))
        if not any(_similar(cand, h) for h in history):
            return cand
    return cand

# ──────────────────────────────
# アウトライン & 本文生成
# ──────────────────────────────
def _outline(kw: str, title: str, pt: str) -> str:
    sys = (
        SAFE_SYS
        + "必ず Markdown形式の##/###で6見出し以上の詳細アウトラインを返してください。"
    )
    usr = f"{pt}\n\n▼ KW:{kw}\n▼ TITLE:{title}"
    return _chat([{"role":"system","content":sys},
                  {"role":"user","content":usr}],
                 TOKENS["outline"], TEMP["outline"])

def _parse_outline(raw: str) -> List[Tuple[str,List[str]]]:
    blocks, cur, subs = [], None, []
    for ln in raw.splitlines():
        s = ln.strip()
        if not s: continue
        if s.startswith("## "):
            if cur: blocks.append((cur, subs))
            cur, subs = s[3:], []
        elif s.startswith("### "):
            subs.append(s[4:])
        else:
            if cur: blocks.append((cur, subs))
            cur, subs = s, []
    if cur: blocks.append((cur, subs))
    return blocks

def _block_html(kw: str, h2: str, h3s: List[str], persona: str, pt: str) -> str:
    h3txt = "\n".join(f"### {h}" for h in h3s) if h3s else ""
    sys   = (
        SAFE_SYS
        + "以下制約でH2セクションをHTML生成:\n"
          "- 600-800字\n"
          "- 結論→理由→具体例×3→再結論\n"
          "- 具体例は<h3 class=\"wp-heading\">で示す\n"
          f"- 視点:{persona}\n"
          "- <h2>/<h3>にclass=\"wp-heading\"付与"
    )
    usr = f"{pt}\n\n▼ KW:{kw}\n▼ H2:{h2}\n▼ H3s\n{h3txt}"
    return _chat([{"role":"system","content":sys},
                  {"role":"user","content":usr}],
                 TOKENS["block"], TEMP["block"])

def _compose_body(kw: str, outline: str, pt: str) -> str:
    # ① 「2500字から3000字」パターンを探す
    m_range = re.search(r"(\d{3,5})\s*字から\s*(\d{3,5})\s*字", pt)
    if m_range:
        min_chars, max_chars = map(int, m_range.groups())
    else:
        # 従来通り「○○字」で min_chars を決定、上限は None
        m = re.search(r"(\d{3,5})\s*字", pt)
        min_chars = int(m.group(1)) if m else MIN_BODY_CHARS_DEFAULT
        max_chars = None
    parts = [
        _block_html(
            kw, h2, h3s,
            random.choice(PERSONAS),
            pt
        )
        for h2, h3s in _parse_outline(outline)
    ]
    html = "\n\n".join(parts)
    html = re.sub(r"<h([23])(?![^>]*wp-heading)",
                  r'<h\1 class="wp-heading"', html)
    if len(html) < min_chars:
        html += '\n\n<h2 class="wp-heading">まとめ</h2><p>要点を整理しました。</p>'

    # ③ **新規：上限超過時に切り詰め**
    if max_chars and len(html) > max_chars:
        html = html[:max_chars]
        # HTML タグが中途半端に切れると不正になるので、
        # 最後の '</p>' まで戻すなどの処理を入れてください:
        last_p = html.rfind("</p>")
        if last_p != -1:
            html = html[:last_p+4]

    return html


# ──────────────────────────────
# 生成タスク
# ──────────────────────────────
def _generate(app, aid: int, tpt: str, bpt: str):
    with app.app_context():
        art = Article.query.get(aid)
        if not art or art.status != "pending":
            return
        try:
            art.status, art.progress = "gen", 10; db.session.commit()
            art.title   = _unique_title(art.keyword, tpt)
            art.progress = 30; db.session.commit()

            outline = _outline(art.keyword, art.title, bpt)
            art.progress = 50; db.session.commit()

            art.body    = _compose_body(art.keyword, outline, bpt)
            art.progress = 80; db.session.commit()

            # 画像取得: 本文先頭 H2 or キーワード で強化クエリ
            body_html = art.body or ""
            match = re.search(r"<h2\b[^>]*>(.*?)</h2>", body_html, re.IGNORECASE)
            query = match.group(1) if match else art.keyword
            art.image_url = fetch_featured_image(body_html, query or art.keyword)

            art.status, art.progress = "done", 100
            art.updated_at = datetime.utcnow()
        except Exception as e:
            logging.exception("記事生成失敗: %s", e)
            art.status, art.body = "error", f"Error: {e}"
        finally:
            db.session.commit()

# ──────────────────────────────
# enqueue_generation
# ──────────────────────────────
def enqueue_generation(
    user_id: int,
    keywords: List[str],
    title_prompt: str,
    body_prompt: str,
    site_id: int | None = None
) -> None:
    app = current_app._get_current_object()
    total = sum(random.randint(1,3) for _ in keywords[:40])
    slots = iter(_generate_slots(total))

    def bg():
        with app.app_context():
            ids: List[int] = []
            for kw in keywords[:40]:
                for _ in range(random.randint(1,3)):
                    art = Article(
                        keyword      = kw.strip(),
                        user_id      = user_id,
                        site_id      = site_id,
                        status       = "pending",
                        progress     = 0,
                        scheduled_at = next(slots, None)
                    )
                    db.session.add(art)
                    db.session.flush()
                    ids.append(art.id)
            db.session.commit()

        for aid in ids:
            _generate(app, aid, title_prompt, body_prompt)
                

    threading.Thread(target=bg, daemon=True).start()

# ─────────────────────────────────────────────
# app/article_generator.py   – v7-full (2025-04-XX)
# ─────────────────────────────────────────────
"""
● 記事生成 + 予約投稿時刻を自動決定する完全版
    ・タイトル重複判定: difflib.SequenceMatcher
    ・本文: H2 600-800 字×複数、<h2>/<h3> に class 付与保証
    ・ガードレール: 禁止事項を system prompt で明示
    ・スケジュール: 1 日あたり 1-5 本（ポアソン分布 λ=4）、
                    JST 10:00-20:59 のランダム時刻
"""

from __future__ import annotations
import os, random, threading, logging, re, uuid, math
from datetime import datetime, timedelta, date, time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple

from difflib import SequenceMatcher           # 重複判定
import pytz                                   # ★ pip install pytz
from flask import current_app
from openai import OpenAI, BadRequestError

from . import db
from .models import Article
from .image_utils import fetch_featured_image

# ──────────────────────────────
# OpenAI 共通設定
# ──────────────────────────────
client  = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
MODEL   = "gpt-4-turbo"
TOKENS  = {"title": 80, "outline": 400, "block": 950}
TEMP    = {"title": 0.40, "outline": 0.45, "block": 0.70}

CTX_LIMIT, SHRINK = 4096, 0.75
MIN_BODY_CHARS    = 3_000
MAX_TITLE_RETRY   = 7
TITLE_DUP_THRESH  = 0.80        # SequenceMatcher ratio (0-1)

# ──────────────────────────────
# スケジュール設定
# ──────────────────────────────
JST         = pytz.timezone("Asia/Tokyo")
POST_HOURS  = list(range(10, 21))      # 10-20 時
LAMBDA      = 4                        # 平均 4 本／日
MAX_PERDAY  = 5                        # 上限 5 本

def _poisson_rand(lambda_: float = LAMBDA) -> int:
    """疑似ポアソン乱数 (0 を除外して 1-∞)"""
    L, k, p = math.exp(-lambda_), 0, 1.0
    while p > L:
        k += 1
        p *= random.random()
    return k or 1

def _generate_slots(n: int) -> List[datetime]:
    """
    今日日付から必要数だけ UTC の日時スロットを返す
    """
    slots: List[datetime] = []
    cur = date.today()
    while len(slots) < n:
        count = min(MAX_PERDAY, _poisson_rand())
        hours = random.sample(POST_HOURS, min(count, len(POST_HOURS)))
        for h in hours:
            dt_local = datetime.combine(cur, time(hour=h), tzinfo=JST)
            slots.append(dt_local.astimezone(pytz.utc))
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

# ══════════════════════════════════════════════
# ChatCompletion ラッパ
# ══════════════════════════════════════════════
def _tok(txt: str) -> int:
    return int(len(txt) * 0.45)        # 日本語 1 文字≈0.45 token

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

# ══════════════════════════════════════════════
# タイトル生成
# ══════════════════════════════════════════════
def _similar(a: str, b: str) -> bool:
    return SequenceMatcher(None, a, b).ratio() >= TITLE_DUP_THRESH

def _title_once(kw: str, pt: str, retry: bool) -> str:
    extra = "\n※既存と類似するため、まったく異なる切り口にしてください。" if retry else ""
    usr = f"{pt}{extra}\n\n▼ 条件\n- KW を含める\n- 末尾は？\n▼ KW: {kw}"
    sys = SAFE_SYS + "条件を満たす Q&A 形式タイトルを 1 行のみ返す。"
    return _chat([{"role":"system","content":sys},
                  {"role":"user","content":usr}],
                 TOKENS["title"], TEMP["title"])

def _unique_title(kw: str, pt: str) -> str:
    bases = [t[0] for t in db.session.query(Article.title)
                       .filter(Article.keyword == kw,
                               Article.title.isnot(None))]
    cand = ""
    for i in range(MAX_TITLE_RETRY):
        cand = _title_once(kw, pt, i > 0)
        if not any(_similar(cand, b) for b in bases):
            break
    return cand

# ══════════════════════════════════════════════
# アウトライン & 本文
# ══════════════════════════════════════════════
def _outline(kw: str, title: str, pt: str) -> str:
    sys = SAFE_SYS + "H2/H3 構成で 6 見出し以上の詳細アウトラインを返す。"
    usr = f"{pt}\n\n▼ KW:{kw}\n▼ TITLE:{title}"
    return _chat([{"role":"system","content":sys},
                  {"role":"user","content":usr}],
                 TOKENS["outline"], TEMP["outline"])

def _parse_outline(raw: str) -> List[Tuple[str, List[str]]]:
    blocks, cur, h3 = [], None, []
    for ln in raw.splitlines():
        s = ln.strip()
        if s.startswith("## "):
            if cur: blocks.append((cur, h3))
            cur, h3 = s[3:], []
        elif s.startswith("### "):
            h3.append(s[4:])
    if cur: blocks.append((cur, h3))
    return blocks

def _block_html(kw: str, h2: str, h3s: List[str], persona: str, pt: str) -> str:
    h3txt = "\n".join(f"### {h}" for h in h3s) if h3s else ""
    sys = SAFE_SYS + (
        f"以下制約で H2 セクションを HTML で生成:\n"
        "- 600-800 字\n- 結論→理由→具体例×3→再結論\n"
        "- 具体例は <h3 class=\"wp-heading\"> で示す\n"
        f"- 視点: {persona}\n- <h2>/<h3> に class=\"wp-heading\" を付与"
    )
    usr = f"{pt}\n\n▼ KW:{kw}\n▼ H2:{h2}\n▼ H3s\n{h3txt}"
    return _chat([{"role":"system","content":sys},
                  {"role":"user","content":usr}],
                 TOKENS["block"], TEMP["block"])

def _compose_body(kw: str, outline: str, pt: str) -> str:
    parts = [_block_html(kw, h2, h3, random.choice(PERSONAS), pt)
             for h2, h3 in _parse_outline(outline)]
    html  = "\n\n".join(parts)
    # 見出しに class を付与漏れがあれば補完
    html  = re.sub(r"<h([23])(?![^>]*wp-heading)",
                   r'<h\1 class="wp-heading"', html)
    if len(html) < MIN_BODY_CHARS:
        html += '\n\n<h2 class="wp-heading">まとめ</h2><p>要点を整理しました。</p>'
    return html

# ══════════════════════════════════════════════
# 生成タスク
# ══════════════════════════════════════════════
def _generate(app, aid: int, tpt: str, bpt: str):
    with app.app_context():
        art = Article.query.get(aid)
        if not art or art.status != "pending":
            return
        try:
            art.status, art.progress = "gen", 10; db.session.commit()
            art.title   = _unique_title(art.keyword, tpt); art.progress = 30; db.session.commit()
            outline     = _outline(art.keyword, art.title, bpt); art.progress = 50; db.session.commit()
            art.body    = _compose_body(art.keyword, outline, bpt); art.progress = 80; db.session.commit()
            art.image_url = fetch_featured_image(art.body, art.keyword)
            art.status, art.progress = "done", 100
            art.updated_at = datetime.utcnow()
        except Exception as e:
            logging.exception("記事生成失敗: %s", e)
            art.status, art.body = "error", f"Error: {e}"
        finally:
            db.session.commit()

# ══════════════════════════════════════════════
# enqueue_generation  (予約時刻をセット)
# ══════════════════════════════════════════════
def enqueue_generation(user_id: int,
                       keywords: List[str],
                       title_prompt: str,
                       body_prompt: str,
                       site_id: int | None = None) -> None:
    app = current_app._get_current_object()

    # 生成本数を事前に算出し予約スロットを作成
    total = sum(random.randint(1, 3) for _ in keywords[:40])
    slots = iter(_generate_slots(total))

    def bg():
        with app.app_context():
            ids: List[int] = []
            for kw in keywords[:40]:
                for _ in range(random.randint(1, 3)):
                    art = Article(
                        keyword      = kw.strip(),
                        user_id      = user_id,
                        site_id      = site_id,
                        status       = "pending",
                        progress     = 0,
                        scheduled_at = next(slots, None)
                    )
                    db.session.add(art); db.session.flush(); ids.append(art.id)
            db.session.commit()
        with ThreadPoolExecutor(max_workers=3) as ex:
            for aid in ids:
                ex.submit(_generate, app, aid, title_prompt, body_prompt)

    threading.Thread(target=bg, daemon=True).start()

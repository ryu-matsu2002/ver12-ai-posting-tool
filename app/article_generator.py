# ─────────────────────────────────────────────
# app/article_generator.py   – v7-full+ (2025-05-XX)
# ─────────────────────────────────────────────
"""
● 記事生成 + 予約投稿時刻自動決定
  - タイトル重複判定
  - 本文見出し構成 + class付与
  - 文字数範囲（2500字〜3000字など）での下限・上限制御
  - 画像取得: 本文先頭 H2 + キーワード でクエリ強化
  - スケジュールは「翌日」以降のUTCスロット（JST 10-20時、それぞれ１時間ずつ）
"""

from __future__ import annotations
import os, re, random, threading, logging
from datetime import datetime, timedelta, date, time
from typing import List, Dict, Tuple

from difflib import SequenceMatcher
import pytz
from flask import current_app
from openai import OpenAI, BadRequestError

from . import db
from .models import Article
from .image_utils import fetch_featured_image
from sqlalchemy import func

# ──────────────────────────────
# OpenAI 共通設定
# ──────────────────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
MODEL  = os.getenv("OPENAI_MODEL", "gpt-4-turbo")
TOKENS = {"title": 80, "outline": 400, "block": 950}
TEMP   = {"title": 0.40, "outline": 0.45, "block": 0.70}

CTX_LIMIT              = 4096
SHRINK                 = 0.75
MIN_BODY_CHARS_DEFAULT = 2_000
MAX_BODY_CHARS_DEFAULT = 3_000
MAX_TITLE_RETRY        = 7
TITLE_DUP_THRESH = 0.90

# ──────────────────────────────
# スケジュール設定
# ──────────────────────────────
JST        = pytz.timezone("Asia/Tokyo")
POST_HOURS = list(range(10, 21))  # JST 10–20時
MAX_PERDAY = 5

def _generate_slots(app, n: int) -> List[datetime]:
    """
    既存予約を JST 日付単位で集計し、
    1 日あたり MAX_PERDAY (=5) 本まで埋めたら翌日に送る。
    """
    global MAX_PERDAY, POST_HOURS, JST         # IDE 警告回避用

    # ── ① 既存予約を DB から取得（JST 変換して date 抽出）
    with app.app_context():
        rows = (
            db.session.query(
                func.date(func.timezone('Asia/Tokyo', Article.scheduled_at)),
                func.count(Article.id)
            )
            .filter(Article.scheduled_at.isnot(None))
            .group_by(1)
            .all()
        )
    booked: dict[date, int] = {row[0]: row[1] for row in rows}

    # ── ② 空き枠に従って n 個のスロットを作る
    slots: list[datetime] = []
    day = date.today() + timedelta(days=1)      # 翌日スタート
    while len(slots) < n:
        remain_today = MAX_PERDAY - booked.get(day, 0)
        if remain_today > 0:
            need = min(remain_today, n - len(slots))
            for h in sorted(random.sample(POST_HOURS, need)):
                minute = random.randint(1, 59)
                local = datetime.combine(day, time(h, minute), tzinfo=JST)
                slots.append(local.astimezone(pytz.utc))
        day += timedelta(days=1)

    return slots

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
# Chat API wrapper
# ══════════════════════════════════════════════
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

def _block_html(
    kw: str, h2: str, h3s: List[str], persona: str, pt: str
) -> str:
    h3txt = "\n".join(f"### {h}" for h in h3s) if h3s else ""
    sys   = (
        SAFE_SYS
        + "以下制約でH2セクションをHTML生成:\n"
        + "- 記事全体は2000〜3000字以内になるように調整\n"
        + "- 小見出し（H2）は10字以内で簡潔に\n"
        + "- 600-800字で本文を生成\n"
        + "- 結論→理由→具体例×3→再結論\n"
        + "- 具体例は<h3 class=\"wp-heading\">で示す\n"
        + f"- 視点:{persona}\n"
        + "- <h2>/<h3>にclass=\"wp-heading\"付与"
    )
    usr   = f"{pt}\n\n▼ KW:{kw}\n▼ H2:{h2}\n▼ H3s\n{h3txt}"
    return _chat(
        [{"role":"system","content":sys},
         {"role":"user","content":usr}],
        TOKENS["block"], TEMP["block"]
    )

# ─────────────────────────────────────────────
# _compose_body (3ブロック / 2500字ターゲット版)
# ─────────────────────────────────────────────
def _compose_body(kw: str, outline_raw: str, pt: str) -> str:
    """ブロック数を目標字数から動的決定（2〜6 本）し、上限カットは行わない。"""

    # ── 字数レンジ取得（指定なし→2500〜∞）
    m = re.search(r"(\d{3,5})\s*字から\s*(\d{3,5})\s*字", pt)
    if m:
        min_chars, max_chars = map(int, m.groups())
    else:
        m2 = re.search(r"(\d{3,5})\s*字", pt)
        min_chars = int(m2.group(1)) if m2 else 2500
        max_chars = None                     # 上限なし

    # ── ブロック数を決める   目標 ≒ 平均(下限+上限)/700 字
    target      = (min_chars + (max_chars or min_chars)) // 2
    n_blocks    = max(2, min(6, round(target / 700)))    # 2〜6 本
    parsed      = _parse_outline(outline_raw)[:n_blocks]

    max_h2_len  = 20
    parts: list[str] = []
    for h2, h3s in parsed:
        if len(h2) > max_h2_len:
            h2 = h2[:max_h2_len] + "…"
        h3_f = [h for h in h3s if len(h) <= 10][:3]
        sec  = _block_html(kw, h2, h3_f, random.choice(PERSONAS), pt)

        # 未閉じ <p> があれば補完
        if sec.count("<p") != sec.count("</p>"):
            sec += "</p>"
        parts.append(sec)

    html = "\n\n".join(parts)
    html = re.sub(r"<h([23])(?![^>]*wp-heading)>", r'<h\1 class="wp-heading">', html)

    # まとめ追加（最小字数未満）
    if len(html) < min_chars:
        html += '\n\n<h2 class="wp-heading">まとめ</h2><p>要点を整理しました。</p>'

    # 重複行除去
    seen, uniq = set(), []
    for ln in html.splitlines():
        t = ln.strip()
        if not t or t not in seen:
            uniq.append(ln)
            if t:
                seen.add(t)

    return "\n".join(uniq)



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

            # 画像取得: タイトル＋先頭2つのH2を使ったクエリ
            h2s     = re.findall(r"<h2\b[^>]*>(.*?)</h2>", art.body or "", re.I)[:2]
            tokens  = [art.keyword, art.title, *h2s]
            query   = " ".join(dict.fromkeys(tokens))          # 重複除去で順序保持
            art.image_url = fetch_featured_image(query)

            art.status, art.progress = "done", 100
            art.updated_at = datetime.utcnow()
        except Exception as e:
            logging.exception("記事生成失敗: %s", e)
            art.status, art.body = "error", f"Error: {e}"
        finally:
            db.session.commit()

def enqueue_generation(
    user_id: int,
    keywords: List[str],
    title_prompt: str,
    body_prompt: str,
    site_id: int | None = None
) -> None:
    app = current_app._get_current_object()
    total = sum(random.randint(1, 3) for _ in keywords[:40])
    slots = iter(_generate_slots(app, total))

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

        # タイトル重複防止のため直列実行
        for aid in ids:
            _generate(app, aid, title_prompt, body_prompt)

    threading.Thread(target=bg, daemon=True).start()

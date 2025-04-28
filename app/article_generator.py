# ──────────────────────────────────────────────
# app/article_generator.py   – v10 (2025-05-XX)
# ──────────────────────────────────────────────
"""
● 記事生成 + 予約投稿時刻自動決定
  - タイトル重複判定
  - 本文見出し構成 + class 付与
  - 目標文字数レンジ（例: 2 000〜3 000 字）の厳守
  - 画像取得: キーワード + タイトル + 先頭 H2 をクエリ
  - スケジュールは翌日以降の JST 10-20 時 → UTC 変換
"""

from __future__ import annotations
import os, re, random, threading, logging
from datetime import datetime, date, time, timedelta, timezone
from typing import List, Dict, Tuple

from difflib import SequenceMatcher
import pytz
from flask import current_app
from openai import OpenAI, BadRequestError

from . import db
from .models import Article
from .image_utils import fetch_featured_image
from sqlalchemy import func
from threading import Event

# ──────────────────────────────
# OpenAI 共通設定
# ──────────────────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
MODEL  = os.getenv("OPENAI_MODEL", "gpt-4-turbo")

TOKENS = {
    "title":   80,
    "outline": 400,
    "block":   1600,     # 1 ブロック 550-750 字 ×3 でも途切れない
}

# UI (chat.openai.com) と合わせた温度 & top_p
TEMP   = {"title": 0.7, "outline": 0.7, "block": 0.7}
TOP_P  = 0.95

CTX_LIMIT              = 4096
SHRINK                 = 0.75            # BadRequest 時の max_tokens 割合

AVG_BLOCK_CHARS        = 600
MIN_BODY_CHARS_DEFAULT = 1_800
MAX_BODY_CHARS_DEFAULT = 3_000
MAX_TITLE_RETRY        = 7
TITLE_DUP_THRESH       = 0.90            # 類似度 0.9 以上を重複とみなす

# ──────────────────────────────
# スケジュール設定
# ──────────────────────────────
JST        = pytz.timezone("Asia/Tokyo")
POST_HOURS = list(range(10, 21))         # JST 10-20 時
MAX_PERDAY = 5

def _generate_slots(app, n: int) -> List[datetime]:
    """
    既存予約を JST 日単位で集計し MAX_PERDAY 以内で空きを割り当て。
    戻り値は必ず n 個の UTC datetime。
    """
    if n <= 0:
        return []

    with app.app_context():
        jst_date = func.date(func.timezone("Asia/Tokyo", Article.scheduled_at))
        rows = (
            db.session.query(jst_date.label("d"), func.count(Article.id))
            .filter(Article.scheduled_at.isnot(None))
            .group_by("d")
            .all()
        )
    booked: dict[date, int] = {d: c for d, c in rows}

    slots: list[datetime] = []
    day = date.today() + timedelta(days=1)

    while len(slots) < n:
        remain = MAX_PERDAY - booked.get(day, 0)
        if remain > 0:
            need = min(remain, n - len(slots))
            for h in sorted(random.sample(POST_HOURS, need)):
                minute = random.randint(1, 59)
                local  = datetime.combine(day, time(h, minute), tzinfo=JST)
                slots.append(local.astimezone(timezone.utc))
        day += timedelta(days=1)

        if (day - date.today()).days > 365:          # 無限ループ安全弁
            raise RuntimeError("slot generation runaway")

    return slots[:n]

# ──────────────────────────────
# コンテンツ生成設定
# ──────────────────────────────
PERSONAS = [
    "節約志向の学生", "ビジネス渡航が多い会社員",
    "小さな子供連れファミリー", "リタイア後の移住者",
    "ペット同伴で移動する読者",
]

SAFE_SYS = (
    "あなたは一流の日本語 SEO ライターです。"
    "公序良俗に反する表現・誤情報・個人情報や差別的・政治的主張は禁止します。"
    "SEOを意識した見出しや本文を構成し、読者にとって有益な情報を提供してください。"  # SEOに特化した指示
)

# ══════════════════════════════════════════════
# Chat API ラッパ
# ══════════════════════════════════════════════
def _tok(s: str) -> int:
    """日本語 1.8 字 ≒ 1 token で概算"""
    return int(len(s) / 1.8)

def _chat(msgs: List[Dict[str, str]], max_t: int, temp: float) -> str:
    used = sum(_tok(m["content"]) for m in msgs)
    max_t = min(max_t, CTX_LIMIT - used - 16)       # 余白 16 token

    def _call(m: int) -> str:
        res = client.chat.completions.create(
            model       = MODEL,
            messages    = msgs,
            max_tokens  = m,
            temperature = temp,
            top_p       = TOP_P,
            timeout     = 120,
        )
        return res.choices[0].message.content.strip()

    try:
        return _call(max_t)
    except BadRequestError as e:
        if "max_tokens" in str(e):
            return _call(int(max_t * SHRINK))  # max_tokens に合った値で再試行
        raise

# ══════════════════════════════════════════════
# タイトル生成
# ══════════════════════════════════════════════
def _similar(a: str, b: str) -> bool:
    return SequenceMatcher(None, a, b).ratio() >= TITLE_DUP_THRESH

def _title_once(kw: str, pt: str, retry: bool) -> str:
    extra = "\n※類似タイトルを避けてください。" if retry else ""
    usr   = f"{pt}{extra}\n\n▼ 条件\n- 必ずキーワードを含める\n▼ KW: {kw}"
    sys   = SAFE_SYS + "魅力的な日本語タイトルを 1 行だけ返してください。"
    return _chat(
        [{"role": "system", "content": sys},
         {"role": "user",   "content": usr}],
        TOKENS["title"], TEMP["title"]
    )

def _unique_title(kw: str, pt: str) -> str:
    history = [t[0] for t in db.session.query(Article.title)
                         .filter(Article.keyword == kw,
                                 Article.title.isnot(None))]
    cand = ""
    for i in range(MAX_TITLE_RETRY):
        cand = _title_once(kw, pt, retry=i > 0)
        if not any(_similar(cand, h) for h in history):
            history.append(cand)
            break
    return cand

# ══════════════════════════════════════════════
# アウトライン & 本文生成
# ══════════════════════════════════════════════
def _outline(kw: str, title: str, pt: str) -> str:
    sys = SAFE_SYS + "## / ### で 6〜8 個の見出しを Markdown で返してください。各 H2 は 15 字以内。"
    usr = f"{pt}\n\n▼ KW: {kw}\n▼ TITLE: {title}"
    return _chat(
        [{"role": "system", "content": sys},
         {"role": "user",   "content": usr}],
        TOKENS["outline"], TEMP["outline"]
    )

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

def _block_html(
    kw: str, h2: str, h3s: List[str], persona: str, pt: str
) -> str:
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
    return _chat(
        [{"role": "system", "content": sys},
         {"role": "user",   "content": usr}],
        TOKENS["block"], TEMP["block"]
    )

def _parse_range(pt: str) -> Tuple[int, int | None]:
    if m := re.search(r"(\d{3,5})\s*字から\s*(\d{3,5})\s*字", pt):
        return int(m.group(1)), int(m.group(2))
    if m := re.search(r"(\d{3,5})\s*字", pt):
        return int(m.group(1)), None
    return MIN_BODY_CHARS_DEFAULT, None

# ─────────────────────────────────────────────
# 本文組み立て
# ─────────────────────────────────────────────
def _compose_body(kw: str, outline_raw: str, pt: str) -> str:
    min_chars, max_chars_user = _parse_range(pt)
    max_total = max_chars_user or MAX_BODY_CHARS_DEFAULT

    outline = _parse_outline(outline_raw)
    need = max(3, min(len(outline),
                      (max_total + AVG_BLOCK_CHARS - 1) // AVG_BLOCK_CHARS))
    outline = outline[:need]

    parts: List[str] = []
    for h2, h3s in outline:
        h2  = (h2[:15] + "…") if len(h2) > 15 else h2
        h3s = [h for h in h3s if len(h) <= 10][:3]
        parts.append(
            _block_html(kw, h2, h3s, random.choice(PERSONAS), pt)
        )

    html = "\n\n".join(parts)
    html = re.sub(r"<h([23])(?![^>]*wp-heading)>",
                  r'<h\1 class="wp-heading">', html)

    if len(html) < min_chars:
        html += '\n\n<h2 class="wp-heading">まとめ</h2><p>この記事の要点を簡潔に振り返りました。</p>'

    if len(html) > max_total:
        snippet = html[:max_total]
        cut = max(snippet.rfind("</p>"),
                  snippet.rfind("</h2>"),
                  snippet.rfind("</h3>"))
        html = snippet[:cut + 5] if cut != -1 else snippet

    logging.debug("compose_body len=%s (max=%s)", len(html), max_total)
    return html

# ──────────────────────────────
# 生成ワーカー
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

            art.body = _compose_body(art.keyword, outline, bpt)
            art.progress = 80; db.session.commit()

            # 画像クエリ: keyword + title + 先頭 2 H2
            h2s   = re.findall(r"<h2\b[^>]*>(.*?)</h2>", art.body or "", re.I)[:2]
            query = " ".join(dict.fromkeys([art.keyword, art.title, *h2s]))
            url   = fetch_featured_image(query)
            if len(url.encode()) > 500:                 # DB 500byte 制限対策
                url = url.split("?", 1)[0]
                if len(url.encode()) > 500:
                    url = url.encode()[:497].decode("utf-8", "ignore") + "…"
            art.image_url = url

            art.status, art.progress = "done", 100
            art.updated_at = datetime.utcnow()
        except Exception as e:
            logging.exception("記事生成失敗: %s", e)
            art.status, art.body = "error", f"Error: {e}"
        finally:
            db.session.commit()

# ──────────────────────────────
# バックグラウンドエンキュー
# ──────────────────────────────
def enqueue_generation(
    user_id: int,
    keywords: List[str],
    title_prompt: str,
    body_prompt: str,
    site_id: int | None = None,
) -> None:
    app = current_app._get_current_object()

    copies = [random.randint(1, 3) for _ in keywords[:40]]
    total  = sum(copies)
    slots  = iter(_generate_slots(app, total))

    def background():
        with app.app_context():
            ids: list[int] = []
            for kw, c in zip(keywords[:40], copies):
                for _ in range(c):
                    art = Article(
                        keyword      = kw.strip(),
                        user_id      = user_id,
                        site_id      = site_id,
                        status       = "pending",
                        progress     = 0,
                        scheduled_at = next(slots, None),
                    )
                    db.session.add(art)
                    db.session.flush()
                    ids.append(art.id)
            db.session.commit()

        # タイトル重複を避けたいので直列生成
        for aid in ids:
            from .article_generator import _generate
            _generate_and_wait(app, aid, title_prompt, body_prompt)  # 完了するまで待機

    threading.Thread(target=background, daemon=True).start()

def _generate_and_wait(app, aid, tpt, bpt):
    """非同期タスクが完了するまで待機する"""
    event = Event()

    def background():
        _generate(app, aid, tpt, bpt)
        event.set()  # 完了したことを通知

    threading.Thread(target=background, daemon=True).start()
    event.wait()  # 完了するまで待機


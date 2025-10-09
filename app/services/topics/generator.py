# app/services/topics/generator.py
# Step1.5: アンカー＝タイトル、クリック時に本文生成／WP反映、アフィリリンク挿入、ユーザー特性注入、既存再利用フォールバック
from __future__ import annotations

import os, re, time, json, uuid, logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple

from flask import current_app
from sqlalchemy.exc import IntegrityError
from openai import OpenAI, BadRequestError

from app import db
from app.models import (
    TopicPrompt, TopicPage, TopicAnchorLog, TokenUsageLog,
)

# === OpenAI 設定（article_generator.py と同等） ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TOP_P = 0.9
CTX_LIMIT = 12000
SHRINK = 0.85

TOKENS = {
    "anchor": 160,     # アンカー＝タイトル
    "topic": 2200,     # 本文
}
TEMP = {
    "anchor": 0.4,
    "topic": 0.55,
}

# === 依存（未実装でも落ちないように動的 import） ===
def _get_affiliate_links(user_id: int, site_id: Optional[int], limit: int = 2) -> List[Dict[str, str]]:
    """
    AffiliateLink モデル（別途追加予定）からユーザー紐付けの提携リンクを取得。
    まだモデル未導入でも安全にスキップ。
    返却: [{"title": "...", "url": "...", "tags": "csv"}]
    """
    try:
        from app.models import AffiliateLink  # ← あとで作るモデル
        q = db.session.query(AffiliateLink).filter(AffiliateLink.user_id == user_id)
        if site_id:
            q = q.filter((AffiliateLink.site_id == site_id) | (AffiliateLink.site_id.is_(None)))
        q = q.order_by(AffiliateLink.priority.desc(), AffiliateLink.created_at.desc())
        rows = q.limit(limit * 3).all()
        return [
            {"title": r.title or r.program_name or "おすすめ", "url": r.url, "tags": r.tags or ""}
            for r in rows if r.url
        ][:limit]
    except Exception as e:
        # 未定義/未マイグレーションでも静かに降りる
        logging.info(f"[affiliate] skip (model missing or query failed): {e}")
        return []

def _topics_snapshot_for_user() -> Optional[Dict[str, Any]]:
    """
    Step2で Topics API を噛ませる入口。現時点では空 or 擬似。
    ブラウザ側で取得→クッキー/ヘッダ経由で渡す構成にしたら、
    そのJSONをそのまま渡してここに入れる想定。
    """
    return None

# === ユーティリティ ===
def _tok(s: str) -> int: return int(len(s) / 1.8)

def _strip_tags(s: str) -> str: return re.sub(r"<[^>]+>", "", s or "").strip()

def clean_gpt_output(text: str) -> str:
    text = re.sub(r"```(?:html|markdown)?", "", text)
    text = re.sub(r"```", "", text)
    text = re.sub(r"<!DOCTYPE html>.*?<body.*?>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</body>.*?</html>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()

def _chat(msgs: list[dict[str, str]], max_t: int, temp: float, *, user_id: Optional[int] = None) -> str:
    used = sum(_tok(m.get("content", "")) for m in msgs)
    available = CTX_LIMIT - used - 16
    max_t = max(1, min(max_t, available))
    def _call(m: int) -> str:
        res = client.chat.completions.create(
            model=MODEL, messages=msgs, max_tokens=m, temperature=temp, top_p=TOP_P, timeout=120,
        )
        try:
            if hasattr(res, "usage") and user_id:
                usage = res.usage
                log = TokenUsageLog(
                    user_id=user_id,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0),
                    completion_tokens=getattr(usage, "completion_tokens", 0),
                    total_tokens=getattr(usage, "total_tokens", 0),
                )
                db.session.add(log); db.session.commit()
        except Exception as e:
            logging.warning(f"[TokenUsageLog 保存失敗] {e}")
        content = res.choices[0].message.content.strip()
        if res.choices[0].finish_reason == "length":
            content += "\n\n※トークン上限で途中終了の可能性があります。"
        return clean_gpt_output(content)
    try:
        return _call(max_t)
    except BadRequestError as e:
        if "max_tokens" in str(e):
            return _call(max(1, int(max_t * SHRINK)))
        raise

def _slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\-]+", "-", s, flags=re.UNICODE)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return (s or "topic")[:180]

def _ensure_unique_slug(user_id: int, base: str) -> str:
    slug = _slugify(base)
    if not db.session.query(TopicPage.id).filter_by(user_id=user_id, slug=slug).first():
        return slug
    for _ in range(8):
        suf = uuid.uuid4().hex[:6]
        cand = f"{slug}-{suf}"
        if not db.session.query(TopicPage.id).filter_by(user_id=user_id, slug=cand).first():
            return cand
    return uuid.uuid4().hex

# === 返却型 ===
@dataclass
class AnchorItem: text: str; slug: str
@dataclass
class AnchorResult: top: AnchorItem; bottom: AnchorItem

# === あなたの正式プロンプト（本文） ===
OFFICIAL_TOPIC_PROMPT = (
    "あなたはSEOとアフィリエイトの専門家です。\n"
    "以下の条件を満たす「自然にアフィリエイト広告へ誘導する記事文」を作成してください。\n"
    "【条件】\n"
    "- タイトル（h2程度）と本文を必ず出力する。\n"
    "- タイトルは「～～～したい方はこちら」「～～～たい方はこちら」のようにセグメントして誘導するような文章で。\n"
    "- 本文構成は「問題提起と共感 → 解決策の提案 → さりげない後押し」。\n"
    "- 読み手の状況や気持ちに共感しながら会話調。広告感は一切出さない。\n"
    "- アフィリエイト広告は最後に「おすすめはこちら」とだけさらっと紹介（詳述は不要）。\n"
    "- 商品よりもまず相手の気持ちに寄り添う。\n"
    "- 「困っている親友」にそっと伝えるように気遣いとあたたかさのある敬語。\n"
    "- 最後は押し売り感のない“そっと背中を押す言葉”。\n"
    "- 本文は300〜500文字。\n"
    "【アフィリエイト広告について】\n"
    "入力された商品やサービスを最後に「おすすめはこちら」で紹介。\n"
    "【入力】\n"
    "・ユーザー特性: {user_traits}\n"
    "・元ページタイトル: {title}\n"
    "・元ページ要約: {summary}\n"
    "・アンカーテキスト（タイトルとして使用）: {anchor}\n"
    "・おすすめ商材候補（JSON）: {affiliates}\n"
    "【出力形式】\n"
    "【タイトル】\n{anchor}\n\n【本文】\n（ここに本文）\n"
    "※HTMLは使わずにテキストのみ。"
)

# === アンカー用（タイトルのみ返す派生プロンプト） ===
OFFICIAL_ANCHOR_PROMPT = (
    "あなたは日本語のウェブ編集者です。以下の入力に基づき、"
    "読者のセグメントを意識した『～～したい方はこちら』系の短い誘導タイトルを2つ作成してください。"
    "各24文字以内、広告臭さは避け、名詞止めを多用。\n"
    "【入力】\n"
    "・ユーザー特性: {user_traits}\n"
    "・元ページタイトル: {title}\n"
    "・要約: {summary}\n"
    "・商材候補: {affiliates}\n"
    "【出力（JSONのみ）】\n"
    '{"top":"…","bottom":"…"}'
)

# === アンカー生成（表示時） ===
def generate_anchor_texts(
    *, user_id: int, site_id: Optional[int], source_url: str,
    current_title: Optional[str] = None, page_summary: Optional[str] = None,
    anchor_prompt: Optional[str] = None, topic_prompt: Optional[TopicPrompt] = None,
    user_traits_json: Optional[Dict[str, Any]] = None,
) -> AnchorResult:
    traits = user_traits_json or _topics_snapshot_for_user() or {}
    affiliates = _get_affiliate_links(user_id, site_id, limit=2)
    ap = (anchor_prompt or OFFICIAL_ANCHOR_PROMPT).format(
        user_traits=json.dumps(traits, ensure_ascii=False),
        title=current_title or "",
        summary=(page_summary or "")[:800],
        affiliates=json.dumps(affiliates, ensure_ascii=False),
    )
    raw = _chat(
        [{"role": "system", "content": "JSONだけを返してください。"},
         {"role": "user", "content": ap}],
        TOKENS["anchor"], TEMP["anchor"], user_id=user_id
    )
    # JSONパース
    top_text = "詳しくはこちら"
    bottom_text = "関連の解説はこちら"
    try:
        data = json.loads(raw); top_text = (data.get("top") or top_text).strip(); bottom_text = (data.get("bottom") or bottom_text).strip()
    except Exception:
        logging.warning(f"[anchor-json] parse failed: {raw[:120]}...")

    base = current_title or source_url.split("/")[-1] or "topic"
    slug_top = _ensure_unique_slug(user_id, f"{base}-top")
    slug_bottom = _ensure_unique_slug(user_id, f"{base}-bottom")

    snap = traits or None
    for pos, text, slug in (("slot_top", top_text, slug_top), ("slot_bottom", bottom_text, slug_bottom)):
        db.session.add(TopicAnchorLog(
            user_id=user_id, site_id=site_id, page_id=None, source_url=source_url,
            position=pos, anchor_text=text, event="impression", latency_ms=None, topics_snapshot=snap
        ))
    db.session.commit()

    return AnchorResult(top=AnchorItem(text=top_text, slug=slug_top),
                        bottom=AnchorItem(text=bottom_text, slug=slug_bottom))

# === 類似ユーザー用の再利用候補検索（フォールバック） ===
def _find_reusable_page(user_id: int, site_id: Optional[int], traits: Dict[str, Any]) -> Optional[TopicPage]:
    """
    ユーザー特性が取れない／似たユーザーが過去にいた場合に再利用する既存Topicを探す。
    ここでは簡易に、同一site_idかつ直近作成順で返す（将来はタグやクラスタリングで厳密化）。
    """
    q = db.session.query(TopicPage).filter(TopicPage.user_id == user_id)
    if site_id: q = q.filter(TopicPage.site_id == site_id)
    q = q.order_by(TopicPage.created_at.desc())
    return q.first()

# === 本文生成（クリック瞬間）＋ アフィリ挿入 ＋ WP反映フック ===
def get_or_generate_topic_page(
    *, user_id: int, slug: str, site_id: Optional[int] = None, source_url: Optional[str] = None,
    clicked_anchor_text: Optional[str] = None, topic_prompt: Optional[TopicPrompt] = None,
    fallback_topic_prompt: Optional[str] = None, current_title: Optional[str] = None,
    page_summary: Optional[str] = None, user_traits_json: Optional[Dict[str, Any]] = None,
    publish_to_wp: bool = True,
) -> TopicPage:
    # 既存キャッシュ
    existing = db.session.query(TopicPage).filter_by(user_id=user_id, slug=slug).first()
    if existing: return existing

    traits = user_traits_json or _topics_snapshot_for_user() or {}
    # 特性が全くない場合は再利用候補を先に試す
    if not traits:
        reuse = _find_reusable_page(user_id, site_id, traits={})
        if reuse: return reuse

    # プロンプト決定
    if topic_prompt and topic_prompt.prompt is not None:
        payload = topic_prompt.prompt
        template = payload.get("template") if isinstance(payload, dict) else str(payload)
    else:
        template = (fallback_topic_prompt or OFFICIAL_TOPIC_PROMPT)

    affiliates = _get_affiliate_links(user_id, site_id, limit=2)
    filled = template.format(
        user_traits=json.dumps(traits, ensure_ascii=False),
        title=current_title or "",
        summary=(page_summary or "")[:1200],
        anchor=clicked_anchor_text or "",
        affiliates=json.dumps(affiliates, ensure_ascii=False),
    )

    # 生成
    sys = "あなたは日本語のSEOとアフィリエイトに詳しい編集者です。出力形式に厳密に従ってください。"
    start = time.perf_counter()
    out = _chat(
        [{"role": "system", "content": sys}, {"role": "user", "content": filled}],
        TOKENS["topic"], TEMP["topic"], user_id=user_id
    )
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    # タイトル＆本文抽出
    title = ""
    body = ""
    m1 = re.search(r"【タイトル】\s*(.+?)\s*【本文】", out, flags=re.DOTALL)
    m2 = re.search(r"【本文】\s*(.+)$", out, flags=re.DOTALL)
    if m1: title = _strip_tags(m1.group(1))
    if m2: body = m2.group(1).strip()
    # アンカーとズレたらアンカー優先
    if clicked_anchor_text: title = clicked_anchor_text

    # アフィリエイト挿入（自然な末尾ブロック）
    if affiliates:
        # 1件だけ自然に追加（複数あれば先頭を使用）
        a = affiliates[0]
        aff_block = f"\n\nおすすめはこちら：{a.get('title','おすすめ')}（{a.get('url','')}）"
        # 既に“おすすめはこちら”があれば重複を避ける
        if "おすすめはこちら" not in body:
            body = f"{body.rstrip()}\n{aff_block}"

    # 保存
    page = TopicPage(
        user_id=user_id, site_id=site_id, source_prompt_id=(topic_prompt.id if topic_prompt else None),
        slug=slug, title=(title or "トピック解説")[:255],
        description=None, body=body, topics_json=traits or None,
        meta={"source_url": source_url, "clicked_anchor": clicked_anchor_text, "affiliates_used": affiliates},
        generated_ms=elapsed_ms,
    )
    db.session.add(page)
    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        page = db.session.query(TopicPage).filter_by(user_id=user_id, slug=slug).first()
        if not page: raise

    # クリックログ
    try:
        db.session.add(TopicAnchorLog(
            user_id=user_id, site_id=site_id, page_id=page.id, source_url=source_url or "",
            position="unknown", anchor_text=clicked_anchor_text or "", event="click", latency_ms=elapsed_ms,
            topics_snapshot=(traits or None),
        ))
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logging.warning(f"[TopicAnchorLog click 保存失敗] {e}")

    # WP 反映フック
    if publish_to_wp:
        try:
            _publish_topic_to_wp(page)
        except Exception as e:
            logging.exception(f"[WP publish] failed: {e}")

    return page

# === ルート側ユーティリティ ===
def resolve_or_build_by_slug(
    *, user_id: int, slug: str, site_id: Optional[int] = None, source_url: Optional[str] = None,
    clicked_anchor_text: Optional[str] = None, topic_prompt_id: Optional[int] = None,
    fallback_topic_prompt: Optional[str] = None, current_title: Optional[str] = None,
    page_summary: Optional[str] = None, user_traits_json: Optional[Dict[str, Any]] = None,
    publish_to_wp: bool = True,
) -> TopicPage:
    tp = db.session.get(TopicPrompt, topic_prompt_id) if topic_prompt_id else None
    return get_or_generate_topic_page(
        user_id=user_id, slug=slug, site_id=site_id, source_url=source_url,
        clicked_anchor_text=clicked_anchor_text, topic_prompt=tp,
        fallback_topic_prompt=fallback_topic_prompt, current_title=current_title,
        page_summary=page_summary, user_traits_json=user_traits_json, publish_to_wp=publish_to_wp,
    )

# === WordPress 反映（フック） ===
def _publish_topic_to_wp(page: TopicPage) -> None:
    """
    TopicPage を WordPress に「記事として」公開し、URLとpost_idを保存する。
    - 既に published_url があればスキップ（多重投稿防止）
    - post_id はモデル変更なしで meta.wp_post_id に格納
    """
    if page.published_url:
        return
    if not page.site_id:
        logging.info("[WP publish] site_id is None, skip.")
        return
    try:
        from app.models import Site
        from app.wp_client import post_topic_to_wp
    except Exception as e:
        logging.info(f"[WP publish] import failed, skip: {e}")
        return

    site = db.session.get(Site, page.site_id)
    if not site:
        logging.info("[WP publish] Site not found, skip.")
        return

    title = page.title
    html_body = _topic_to_html(title, page.body or "")

    try:
        # カテゴリを付与したい場合は category_ids=[... ] を渡してください（数値ID）
        post_id, link = post_topic_to_wp(site=site, title=title, html=html_body, slug=page.slug)
        # 保存（post_id は meta に入れてモデル変更を回避）
        meta = dict(page.meta or {})
        meta["wp_post_id"] = post_id
        page.meta = meta
        page.published_url = link
        db.session.commit()
    except Exception as e:
        logging.exception(f"[WP publish] topic post failed: {e}")


def _topic_to_html(title: str, body_text: str) -> str:
    """
    TopicPage.body（テキスト）をWP投稿向けの簡易HTMLに変換。
    """
    # h2扱いのタイトル（指示通りh2相当）
    title_h2 = f"<h2 class='wp-heading'>{title}</h2>"
    # 段落化（改行ベース）
    paras = [f"<p>{p.strip()}</p>" for p in re.split(r"\n{2,}", body_text.strip()) if p.strip()]
    return title_h2 + "\n" + "\n".join(paras)

# app/services/internal_seo/indexer.py
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from flask import current_app

from app import db
from app.models import ContentIndex, Site, InternalSeoConfig
from app.wp_client import fetch_posts_paged, WPPost  # 既存wp_client拡張を利用

logger = logging.getLogger(__name__)


# ----------------------------
# HTML -> プレーンテキスト化
# ----------------------------
def html_to_text(html: str) -> str:
    """
    BeautifulSoup が入っていれば使い、なければ正規表現で簡易除去。
    """
    if not html:
        return ""
    try:
        from bs4 import BeautifulSoup  # type: ignore
        soup = BeautifulSoup(html, "html.parser")
        # 不要要素を除去
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        return re.sub(r"\n{2,}", "\n", text).strip()
    except Exception:
        # フォールバック：タグ除去 → 余分な空白整形
        no_tag = re.sub(r"<[^>]+>", " ", html)
        no_ent = re.sub(r"&[a-zA-Z#0-9]+;", " ", no_tag)
        text = re.sub(r"\s+", " ", no_ent)
        return text.strip()


# ----------------------------
# 軽量キーワード抽出（日本語OK版）
# ----------------------------
JP_TOKEN = re.compile(r"[一-龥ぁ-んァ-ンーA-Za-z0-9]{2,}")

def extract_keywords(text: str, top_k: int = 20) -> List[str]:
    """
    形態素なしでも概ね上位語を拾う簡易版。
    - 2文字以上の日本語/英数をトークン化
    - よくある機能語を軽く除外
    """
    if not text:
        return []
    tokens = JP_TOKEN.findall(text.lower())
    if not tokens:
        return []

    stop = {
        "こと", "これ", "それ", "ため", "よう", "です", "ます", "する", "いる",
        "ある", "ない", "なる", "そして", "また", "ので", "のでしょう", "でも",
        "について", "まとめ", "ポイント", "こちら", "今回", "場合", "可能", "原因",
        "方法", "対処", "基本", "入門", "注意", "解説", "詳細", "最新",
    }
    tokens = [t for t in tokens if t not in stop and not t.isdigit()]
    if not tokens:
        return []

    freq = Counter(tokens)
    # 単純頻度上位（将来TF-IDFに差し替え可能）
    common = [w for (w, c) in freq.most_common(top_k)]
    return common


# ----------------------------
# 日付ユーティリティ
# ----------------------------
def _parse_wp_dt_gmt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        # WPの modified_gmt は "YYYY-MM-DDTHH:MM:SS" を想定（末尾Zなし）
        # ここでは UTC naive として扱い、UTCにローカライズ
        dt = datetime.fromisoformat(s.replace("Z", ""))
        return dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


# ----------------------------
# UPSERT
# ----------------------------
def _upsert_content_index(site_id: int, wp_post: WPPost) -> None:
    """
    site_id + (wp_post.id or url) で既存を探し、更新 or 作成。
    """
    raw_text = html_to_text(wp_post.content_html)
    kws = extract_keywords(raw_text, top_k=20)

    published_at = None  # /posts APIの簡易レスポンスでは date は省いているので空でOK
    updated_at = _parse_wp_dt_gmt(wp_post.modified_gmt)

    row = (
        ContentIndex.query
        .filter_by(site_id=site_id, wp_post_id=wp_post.id)
        .one_or_none()
    )
    if row is None:
        # URLでの重複防止（念のため）
        row = (
            ContentIndex.query
            .filter_by(site_id=site_id, url=wp_post.link)
            .one_or_none()
        )

    if row is None:
        row = ContentIndex(
            site_id=site_id,
            wp_post_id=wp_post.id,
            title=html_to_text(wp_post.title_html)[:255] or wp_post.title_html[:255],
            url=wp_post.link,
            slug=wp_post.slug,
            status=wp_post.status,
            published_at=published_at,
            updated_at=updated_at,
            raw_text=raw_text,
            keywords=",".join(kws),
            last_indexed_at=datetime.utcnow(),
        )
        db.session.add(row)
    else:
        row.wp_post_id = wp_post.id
        row.title = html_to_text(wp_post.title_html)[:255] or row.title
        row.url = wp_post.link or row.url
        row.slug = wp_post.slug or row.slug
        row.status = wp_post.status or row.status
        # 更新日時は“より新しい方”を採用
        if updated_at:
            row.updated_at = updated_at
        if published_at and not row.published_at:
            row.published_at = published_at
        row.raw_text = raw_text or row.raw_text
        if kws:
            row.keywords = ",".join(kws)
        row.last_indexed_at = datetime.utcnow()


# ----------------------------
# 同期メイン
# ----------------------------
def sync_site_content_index(
    site_id: int,
    per_page: int = 100,
    max_pages: Optional[int] = None,
    incremental: bool = True,
    batch_commit_size: int = 200,
) -> Dict[str, int]:
    """
    指定サイトの公開記事を ContentIndex に同期。
    - topic含むURLは wp_client 層で除外済み
    - incremental=True の場合、site内の ContentIndex.updated_at の最大値をヒントに
      “最近更新分”から優先（WPの after が投稿日基準のケースもあるため、厳密差分は後続でページの早期停止を併用）
    - 戻り値: 処理統計
    """
    site = Site.query.get(site_id)
    if not site:
        raise ValueError(f"Site not found: {site_id}")

    cfg = InternalSeoConfig.query.filter_by(site_id=site_id).one_or_none()
    if not cfg:
        cfg = InternalSeoConfig(site_id=site_id)
        db.session.add(cfg)
        db.session.commit()

    # 既存の最新更新時刻（ヒント）
    last_known_updated = (
        db.session.query(ContentIndex.updated_at)
        .filter(ContentIndex.site_id == site_id, ContentIndex.updated_at.isnot(None))
        .order_by(ContentIndex.updated_at.desc())
        .limit(1)
        .scalar()
    )
    after_gmt = None
    if incremental and last_known_updated:
        # ISO8601（UTC）に整形（WordPressの 'after' は投稿日基準ことがある点は注意）
        after_gmt = last_known_updated.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

    page = 1
    total_pages = None
    processed = 0
    created_or_updated = 0

    while True:
        # ページ取得
        posts, total_pages = fetch_posts_paged(site, page=page, per_page=per_page, status="publish", after_gmt=after_gmt)
        if not posts:
            break

        for p in posts:
            _upsert_content_index(site_id, p)
            processed += 1
            created_or_updated += 1

            if processed % batch_commit_size == 0:
                db.session.commit()
                logger.info("[Indexer] site=%s committed batch (count=%s)", site_id, processed)

        db.session.commit()

        # 上限ページ制限
        if max_pages is not None and page >= max_pages:
            break

        page += 1
        if total_pages and page > total_pages:
            break

    stats = {
        "processed": processed,
        "created_or_updated": created_or_updated,
        "pages": page - 1,
    }
    logger.info("[Indexer] site=%s done: %s", site_id, stats)
    return stats

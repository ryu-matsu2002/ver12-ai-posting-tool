# app/services/internal_seo/applier.py
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from html import unescape
from typing import Dict, List, Optional, Tuple

from app import db
from app.models import (
    ContentIndex,
    InternalLinkAction,
    InternalLinkGraph,
    InternalSeoConfig,
    Site,
)
from app.wp_client import fetch_single_post, update_post_content

logger = logging.getLogger(__name__)

# ---- HTMLユーティリティ ----

_P_CLOSE = re.compile(r"</p\s*>", re.IGNORECASE)
_A_TAG = re.compile(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a\s*>', re.IGNORECASE | re.DOTALL)
_TAG_STRIP = re.compile(r"<[^>]+>")

def _split_paragraphs(html: str) -> List[str]:
    if not html:
        return []
    parts = _P_CLOSE.split(html)
    return [p for p in parts]  # </p> を落として配列化（貼る時に追加）

def _rejoin_paragraphs(paragraphs: List[str]) -> str:
    return "</p>".join(paragraphs)

def _html_to_text(s: str) -> str:
    if not s:
        return ""
    return _TAG_STRIP.sub(" ", unescape(s)).strip()

def _is_internal_url(site_url: str, href: str) -> bool:
    if not href:
        return False
    return href.startswith(site_url.rstrip("/"))

def _extract_links(html: str) -> List[Tuple[str, str]]:
    return [(m.group(1) or "", _html_to_text(m.group(2) or "")) for m in _A_TAG.finditer(html or "")]

def _wrap_link(html_fragment: str, anchor_text: str, href: str) -> str:
    # シンプルにアンカーを置く（applierでは前計画に従うのみ。精緻化は将来改良）
    anchor_escaped = anchor_text
    return f'{html_fragment}<a href="{href}">{anchor_escaped}</a>'

# ---- データ取得 ----

def _post_url(site_id: int, wp_post_id: int) -> Optional[str]:
    row = (
        ContentIndex.query
        .with_entities(ContentIndex.url)
        .filter_by(site_id=site_id, wp_post_id=wp_post_id)
        .one_or_none()
    )
    return row[0] if row else None

def _action_targets_with_urls(site_id: int, actions: List[InternalLinkAction]) -> Dict[int, str]:
    need_ids = list({a.target_post_id for a in actions})
    if not need_ids:
        return {}
    rows = (
        ContentIndex.query
        .with_entities(ContentIndex.wp_post_id, ContentIndex.url)
        .filter(ContentIndex.site_id == site_id)
        .filter(ContentIndex.wp_post_id.in_(need_ids))
        .all()
    )
    return {int(pid): (url or "") for (pid, url) in rows}

def _existing_internal_links_count(site: Site, html: str) -> int:
    site_url = site.url.rstrip("/")
    links = _extract_links(html)
    return sum(1 for (href, _) in links if _is_internal_url(site_url, href))

# ---- 差分作成 ----

@dataclass
class ApplyResult:
    applied: int = 0
    swapped: int = 0
    skipped: int = 0
    message: str = ""

def _apply_plan_to_html(
    site: Site,
    src_post_id: int,
    html: str,
    actions: List[InternalLinkAction],
    cfg: InternalSeoConfig,
    target_url_map: Dict[int, str],
) -> Tuple[str, ApplyResult]:
    """
    計画（plan/swap_candidate）のうち、本文内の挿入と置換を行う。
    - 本文内の内部リンク総数が min~max に収まるよう調整
    """
    res = ApplyResult()
    if not html:
        res.message = "empty-html"
        return html, res

    paragraphs = _split_paragraphs(html)
    if not paragraphs:
        res.message = "no-paragraphs"
        return html, res

    site_url = site.url.rstrip("/")
    # 既存内部リンクの個数
    existing_internal = _existing_internal_links_count(site, html)

    need_min = max(2, int(cfg.min_links_per_post or 2))
    need_max = min(5, int(cfg.max_links_per_post or 5))

    # 1) まずは reason='plan' を優先して挿入
    plan_actions = [a for a in actions if a.reason in ("plan", "review_approved")]
    swaps = [a for a in actions if a.reason == "swap_candidate"]

    inserted = 0
    for act in plan_actions:
        if existing_internal + inserted >= need_max:
            break
        # 位置指定 'p:{idx}'
        try:
            if not act.position.startswith("p:"):
                res.skipped += 1
                continue
            idx = int(act.position.split(":")[1])
        except Exception:
            res.skipped += 1
            continue
        if idx < 0 or idx >= len(paragraphs):
            res.skipped += 1
            continue
        href = target_url_map.get(act.target_post_id)
        if not href:
            res.skipped += 1
            continue
        # 段落末尾に自然に1本だけ追加（過密回避：既に内部リンク多い段落は避けても良いが簡易化）
        para_html = paragraphs[idx]
        # 同じURLが段落内に既にあるならスキップ
        if href in [h for (h, _) in _extract_links(para_html)]:
            res.skipped += 1
            continue
        new_para = _wrap_link(para_html, act.anchor_text, href)
        paragraphs[idx] = new_para
        act.status = "applied"
        act.applied_at = datetime.utcnow()
        res.applied += 1
        inserted += 1

    # 2) swap候補：既存内部リンクがある & まだ余裕がない場合に置換を試みる
    #   （簡易ルール：scoreの低そうな既存リンクをひとつだけ差し替え）
    if swaps and (existing_internal + inserted) >= need_min:
        # 既存リンク列挙
        existing = _extract_links(_rejoin_paragraphs(paragraphs))
        # 既存 internal のうちスコアが低いものを特定
        # URL -> post_id
        url_to_pid = {}
        rows = (
            ContentIndex.query
            .with_entities(ContentIndex.url, ContentIndex.wp_post_id)
            .filter(ContentIndex.site_id == site.id)
            .filter(ContentIndex.url.in_([u for (u, _) in existing]))
            .all()
        )
        for u, pid in rows:
            url_to_pid[u] = int(pid) if pid else None

        # そのpost_idのスコアを取り出し、最小スコアのリンクを交換対象に
        def score_of(dst_pid: Optional[int]) -> float:
            if not dst_pid:
                return 0.0
            row = (
                InternalLinkGraph.query
                .with_entities(InternalLinkGraph.score)
                .filter_by(site_id=site.id, source_post_id=src_post_id, target_post_id=dst_pid)
                .one_or_none()
            )
            return float(row[0]) if row and row[0] is not None else 0.0

        worst_url = None
        worst_score = 999.0
        for (href, _) in existing:
            if not _is_internal_url(site_url, href):
                continue
            sc = score_of(url_to_pid.get(href))
            if sc < worst_score:
                worst_score = sc
                worst_url = href

        if worst_url:
            # 最もスコアの高い swap 候補を1つ
            swaps_sorted = sorted(swaps, key=lambda a: a.id)  # 安定
            best_swap = None
            best_sc = -1.0
            for s in swaps_sorted:
                href = target_url_map.get(s.target_post_id)
                if not href:
                    continue
                row = (
                    InternalLinkGraph.query
                    .with_entities(InternalLinkGraph.score)
                    .filter_by(site_id=site.id, source_post_id=src_post_id, target_post_id=s.target_post_id)
                    .one_or_none()
                )
                sc = float(row[0]) if row and row[0] is not None else 0.0
                if sc > best_sc:
                    best_sc = sc
                    best_swap = (s, href)

            if best_swap and best_sc > worst_score + 0.10:  # マージン
                s_act, new_href = best_swap
                # HTML全体で1箇所 worst_url を new_href に差し替え（アンカー文は新しい案に）
                whole_html = _rejoin_paragraphs(paragraphs)
                # 最初の一致だけ置換（雑にやりすぎない）
                replaced = False
                def _replace_first(h, newh, text):
                    nonlocal replaced
                    if replaced:
                        return text
                    idx = text.find(f'href="{h}"')
                    if idx == -1:
                        return text
                    # アンカー文も差し替えたいが、正確にやるにはパースが必要。
                    # 簡易: href のみ差し替え、アンカー文は維持 or 追記。
                    replaced = True
                    return text.replace(f'href="{h}"', f'href="{newh}"', 1)

                whole_html = _replace_first(worst_url, new_href, whole_html)
                if replaced:
                    paragraphs = [whole_html]  # 再分割は不要。まとまりで返す
                    s_act.status = "applied"
                    s_act.reason = "swap"  # 採用された置換
                    s_act.applied_at = datetime.utcnow()
                    res.swapped += 1

    # 最終的な本文
    new_html = _rejoin_paragraphs(paragraphs)
    return new_html, res

# ---- パブリックAPI ----

def apply_actions_for_post(site_id: int, src_post_id: int, dry_run: bool = False) -> ApplyResult:
    """
    1記事分の pending を読み込んで差分適用（WP更新）。
    - dry_run=True の場合はWP更新せず、結果だけ返す
    """
    cfg = InternalSeoConfig.query.filter_by(site_id=site_id).one_or_none()
    if not cfg:
        cfg = InternalSeoConfig(site_id=site_id)
        db.session.add(cfg)
        db.session.commit()

    site = Site.query.get(site_id)
    wp_post = fetch_single_post(site, src_post_id)
    if not wp_post:
        return ApplyResult(message="fetch-failed-or-excluded")

    # 対象アクション（plan / swap_candidate）
    actions = (
        InternalLinkAction.query
        .filter_by(site_id=site_id, post_id=src_post_id, status="pending")
        .order_by(InternalLinkAction.created_at.asc())
        .all()
    )
    if not actions:
        return ApplyResult(message="no-pending")

    url_map = _action_targets_with_urls(site_id, actions)

    # 差分作成
    new_html, res = _apply_plan_to_html(site, src_post_id, wp_post.content_html, actions, cfg, url_map)

    if dry_run:
        # ドライランではDBを一切変更しない
        return res

    # 監査用の抜粋
    before_excerpt = _html_to_text(wp_post.content_html)[:280]
    after_excerpt = _html_to_text(new_html)[:280]

    # DB更新（監査ログの抜粋）
    for a in actions:
        if a.status == "applied":
            a.diff_before_excerpt = before_excerpt
            a.diff_after_excerpt = after_excerpt
        else:
            # 適用されなかった pending はスキップへ
            a.status = a.status if a.status == "applied" else "skipped"
            a.updated_at = datetime.utcnow()
    db.session.commit()

    # WPへ反映
    ok = update_post_content(site, src_post_id, new_html)
    if not ok:
        # 反映失敗時はロールバック扱いにしておく（applied→skipped）
        for a in actions:
            if a.status == "applied":
                a.status = "pending"  # 再試行できるよう pending に戻す
                a.applied_at = None
        db.session.commit()
        return ApplyResult(message="wp-update-failed")

    return res

def apply_actions_for_site(site_id: int, limit_posts: Optional[int] = 50, dry_run: bool = False) -> Dict[str, int]:
    """
    サイト全体で pending を本文に反映。limit_posts で刻んで安全に。
    """
    q = (
        InternalLinkAction.query
        .with_entities(InternalLinkAction.post_id)
        .filter_by(site_id=site_id, status="pending")
        .group_by(InternalLinkAction.post_id)
        .order_by(db.func.min(InternalLinkAction.created_at).asc())
    )
    if limit_posts:
        q = q.limit(limit_posts)
    src_ids = [int(pid) for (pid,) in q.all()]
    total = {"applied": 0, "swapped": 0, "skipped": 0, "processed_posts": 0}

    for src_post_id in src_ids:
        res = apply_actions_for_post(site_id, src_post_id, dry_run=dry_run)
        total["applied"] += res.applied
        total["swapped"] += res.swapped
        total["skipped"] += res.skipped
        total["processed_posts"] += 1

    logger.info("[Applier] site=%s result=%s", site_id, total)
    return total

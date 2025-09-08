# app/services/internal_seo/planner.py
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from html import unescape
from typing import Dict, Iterable, List, Optional, Tuple

from app import db
from app.models import (
    ContentIndex,
    InternalLinkAction,
    InternalLinkGraph,
    InternalSeoConfig,
    Site,
)
from app.wp_client import fetch_single_post  # 現在のHTMLを読む用（swap判定で使用）

logger = logging.getLogger(__name__)

# ---------- テキスト/段落分割系 ----------

_P_CLOSE = re.compile(r"</p\s*>", re.IGNORECASE)
_A_TAG = re.compile(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a\s*>', re.IGNORECASE | re.DOTALL)
_TAG_STRIP = re.compile(r"<[^>]+>")

JP_TOKEN = re.compile(r"[一-龥ぁ-んァ-ンーA-Za-z0-9]{2,}")

def _split_paragraphs_from_html(html: str) -> List[str]:
    if not html:
        return []
    parts = _P_CLOSE.split(html)
    cleaned = [p.strip() for p in parts if p.strip()]
    return cleaned

def _html_to_text(s: str) -> str:
    if not s:
        return ""
    return _TAG_STRIP.sub(" ", unescape(s)).strip()

def _candidate_anchor_from(title: str, para_text: str) -> Optional[str]:
    """
    ターゲット記事のタイトルを主軸に、段落テキストとのトークン重なりから自然なアンカーを作る。
    足りなければタイトル短縮版を返す。
    """
    title_txt = _html_to_text(title)[:80]
    if not para_text:
        return title_txt or None

    para_tokens = JP_TOKEN.findall(para_text.lower())
    title_tokens = JP_TOKEN.findall(title_txt.lower())

    # タイトルトークンのうち段落にも出る語を優先
    overlap = [t for t in title_tokens if t in para_tokens]
    # 2〜5語で作る
    if overlap:
        anchor = "".join(overlap[:5])
        if 2 <= len(anchor) <= 40:
            return anchor

    # それでも無理ならタイトル先頭 ~ 28文字
    if title_txt:
        return title_txt[:28]
    return None

def _pick_paragraph_slots(paragraphs: List[str], need: int, min_len: int) -> List[int]:
    """
    序盤・中盤・終盤に分散するように段落indexを選ぶ。
    短すぎる段落は除外。
    """
    eligible = [i for i, p in enumerate(paragraphs) if _html_to_text(p) and len(_html_to_text(p)) >= min_len]
    if not eligible:
        return []
    # 3ゾーンで均等抽出（必要数に応じて）
    zones = []
    n = len(eligible)
    if need <= 3:
        zones = [0, n // 2, n - 1]
        zones = [eligible[min(max(z, 0), n - 1)] for z in zones][:need]
        return sorted(set(zones))[:need]
    # need > 3 の場合は等間隔サンプリング
    step = max(1, n // need)
    slots = [eligible[min(i * step, n - 1)] for i in range(need)]
    return sorted(set(slots))[:need]

# ---------- 既存リンク抽出 & URL→post_id解決 ----------

def _extract_existing_links(html: str) -> List[Tuple[str, str]]:
    """
    (href, anchor_text)
    """
    out: List[Tuple[str, str]] = []
    for m in _A_TAG.finditer(html or ""):
        href = (m.group(1) or "").strip()
        anchor = _html_to_text(m.group(2) or "").strip()
        if href:
            out.append((href, anchor))
    return out

def _url_to_post_id_map(site_id: int, urls: Iterable[str]) -> Dict[str, Optional[int]]:
    """
    content_index からURL→wp_post_id を引く。
    """
    uniq = list({u for u in urls if u})
    if not uniq:
        return {}
    rows = (
        ContentIndex.query
        .with_entities(ContentIndex.url, ContentIndex.wp_post_id)
        .filter(ContentIndex.site_id == site_id)
        .filter(ContentIndex.url.in_(uniq))
        .all()
    )
    m = {u: pid for (u, pid) in rows}
    # 見つからなかったURLもキーだけ残す
    for u in uniq:
        m.setdefault(u, None)
    return m

# ---------- 計画作成メイン ----------

@dataclass
class PlanStats:
    planned_actions: int = 0
    swap_candidates: int = 0
    posts_processed: int = 0


def _top_targets_for_source(site_id: int, src_post_id: int, topk: int, min_score: float) -> List[Tuple[int, float]]:
    rows = (
        InternalLinkGraph.query
        .with_entities(InternalLinkGraph.target_post_id, InternalLinkGraph.score)
        .filter_by(site_id=site_id, source_post_id=src_post_id)
        .order_by(InternalLinkGraph.score.desc())
        .limit(topk * 3)  # 後でフィルタする余裕を持たせる
        .all()
    )
    return [(int(t), float(s)) for (t, s) in rows if s is not None and s >= min_score]


def plan_links_for_post(site_id: int, src_post_id: int, mode_swap_check: bool = True) -> PlanStats:
    """
    単一記事に対して本文内リンク 2-5本の計画を作成し、InternalLinkAction(pending) で保存。
    swapチェックONの場合は既存リンクと比較して置換候補も生成（pending, reason='swap_candidate'）。
    """
    stats = PlanStats()

    cfg = InternalSeoConfig.query.filter_by(site_id=site_id).one_or_none()
    if not cfg:
        cfg = InternalSeoConfig(site_id=site_id)
        db.session.add(cfg)
        db.session.commit()

    # 1) 対象本文の取得（最新HTML）
    site = Site.query.get(site_id)
    wp_post = fetch_single_post(site, src_post_id)
    if not wp_post:
        logger.info("[Planner] skip src=%s (fetch failed or excluded)", src_post_id)
        return stats

    paragraphs = _split_paragraphs_from_html(wp_post.content_html)
    if not paragraphs:
        logger.info("[Planner] skip src=%s (no paragraphs)", src_post_id)
        return stats

    # 2) 既存の内部リンク抽出（swap用 & 重複抑制）
    existing_links = _extract_existing_links(wp_post.content_html)
    existing_urls = [u for (u, a) in existing_links]
    url_to_pid = _url_to_post_id_map(site_id, existing_urls)
    existing_post_ids = {pid for pid in url_to_pid.values() if pid}  # 既にリンクしているpost_id

    # 3) 候補ターゲットの収集（スコア順）
    #    - 既存リンク先は優先度下げ（まずは新顔を入れたい）
    #    - 自身は除外
    raw_candidates = _top_targets_for_source(site_id, src_post_id, topk=20, min_score=0.10)
    candidates: List[int] = []
    for t, s in raw_candidates:
        if t == src_post_id:
            continue
        if t in existing_post_ids:
            continue
        candidates.append(t)
    if not candidates:
        logger.info("[Planner] src=%s no fresh candidates", src_post_id)

    # 4) 2〜5本の本数決定（Config固定仕様に従う）
    need_min = max(2, int(cfg.min_links_per_post or 2))
    need_max = min(5, int(cfg.max_links_per_post or 5))
    need = min(max(need_min, 2), need_max)

    # 5) 段落スロット選定
    slots = _pick_paragraph_slots(paragraphs, need=need, min_len=int(cfg.min_paragraph_len or 80))
    if not slots:
        logger.info("[Planner] src=%s no suitable paragraphs", src_post_id)
        return stats

    # 6) ターゲット記事のタイトルを取得（アンカー生成用）
    tgt_rows = (
        ContentIndex.query
        .with_entities(ContentIndex.wp_post_id, ContentIndex.title, ContentIndex.url)
        .filter(ContentIndex.site_id == site_id)
        .filter(ContentIndex.wp_post_id.in_(candidates[: len(slots)]))
        .all()
    )
    tgt_map: Dict[int, Tuple[str, str]] = {int(pid): (title or "", url or "") for (pid, title, url) in tgt_rows}

    # 7) スロット×ターゲットで pending アクションを作成
    actions_made = 0
    for slot_idx, tgt_pid in zip(slots, candidates):
        title, tgt_url = tgt_map.get(tgt_pid, ("", ""))
        para_text = _html_to_text(paragraphs[slot_idx])
        anchor = _candidate_anchor_from(title, para_text)
        if not anchor or not tgt_url:
            continue

        # 監査ログに pending で登録（position は 'p:{index}'）
        act = InternalLinkAction(
            site_id=site_id,
            post_id=src_post_id,
            target_post_id=tgt_pid,
            anchor_text=anchor[:255],
            position=f"p:{slot_idx}",
            status="pending",
            reason="plan",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.session.add(act)
        actions_made += 1

    if actions_made:
        db.session.commit()
    stats.planned_actions = actions_made
    stats.posts_processed = 1

    # 8) 定期検診：swap候補の作成（既存リンクより適合度の高い新顔があれば提案）
    if mode_swap_check and existing_post_ids:
        # 既存リンク先のスコアを取得
        exist_scores: Dict[int, float] = {}
        rows = (
            InternalLinkGraph.query
            .with_entities(InternalLinkGraph.target_post_id, InternalLinkGraph.score)
            .filter_by(site_id=site_id, source_post_id=src_post_id)
            .filter(InternalLinkGraph.target_post_id.in_(list(existing_post_ids)))
            .all()
        )
        for pid, sc in rows:
            exist_scores[int(pid)] = float(sc or 0.0)

        # 新顔候補のうち、既存平均より十分高いものを swap 候補に
        if exist_scores:
            baseline = sum(exist_scores.values()) / max(1, len(exist_scores))
            margin = 0.10  # 既存より +0.10 以上で置換候補
            better_rows = (
                InternalLinkGraph.query
                .with_entities(InternalLinkGraph.target_post_id, InternalLinkGraph.score)
                .filter_by(site_id=site_id, source_post_id=src_post_id)
                .order_by(InternalLinkGraph.score.desc())
                .limit(10)
                .all()
            )
            made_swaps = 0
            for pid, sc in better_rows:
                pid = int(pid)
                sc = float(sc or 0.0)
                if pid in existing_post_ids:
                    continue
                if sc >= baseline + margin:
                    # 置換候補：最初の長め段落に提案（厳密位置は applier で精緻化）
                    slot_for_swap = slots[0] if slots else 0
                    t_title, t_url = tgt_map.get(pid, ("", ""))
                    if not t_title or not t_url:
                        # ない場合はContentIndexから再取得
                        tr = (
                            ContentIndex.query
                            .with_entities(ContentIndex.title, ContentIndex.url)
                            .filter_by(site_id=site_id, wp_post_id=pid)
                            .one_or_none()
                        )
                        if tr:
                            t_title, t_url = tr[0] or "", tr[1] or ""
                    anc = _candidate_anchor_from(t_title, _html_to_text(paragraphs[slot_for_swap]))
                    if not anc or not t_url:
                        continue
                    db.session.add(InternalLinkAction(
                        site_id=site_id,
                        post_id=src_post_id,
                        target_post_id=pid,
                        anchor_text=anc[:255],
                        position=f"p:{slot_for_swap}",
                        status="pending",
                        reason="swap_candidate",
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow(),
                    ))
                    made_swaps += 1
            if made_swaps:
                db.session.commit()
                stats.swap_candidates = made_swaps

    return stats


def plan_links_for_site(
    site_id: int,
    limit_sources: Optional[int] = None,
    mode_swap_check: bool = True,
) -> Dict[str, int]:
    """
    サイト全体の計画を作る（pending行を蓄積）。大量サイト向けに limit_sources で刻み実行。
    """
    cfg = InternalSeoConfig.query.filter_by(site_id=site_id).one_or_none()
    if not cfg:
        cfg = InternalSeoConfig(site_id=site_id)
        db.session.add(cfg)
        db.session.commit()

    # 対象のソース記事（publish & wp_post_idあり）
    q = (
        ContentIndex.query
        .with_entities(ContentIndex.wp_post_id)
        .filter_by(site_id=site_id, status="publish")
        .filter(ContentIndex.wp_post_id.isnot(None))
        .order_by(ContentIndex.updated_at.desc().nullslast())
    )
    if limit_sources:
        q = q.limit(limit_sources)

    src_ids = [int(pid) for (pid,) in q.all()]
    planned = 0
    swaps = 0
    processed = 0

    for src_post_id in src_ids:
        st = plan_links_for_post(site_id, src_post_id, mode_swap_check=mode_swap_check)
        planned += st.planned_actions
        swaps += st.swap_candidates
        processed += st.posts_processed

    logger.info("[Planner] site=%s planned=%s swaps=%s processed=%s", site_id, planned, swaps, processed)
    return {"planned": planned, "swap_candidates": swaps, "processed": processed}

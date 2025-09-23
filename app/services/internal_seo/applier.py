# app/services/internal_seo/applier.py
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from html import unescape
from typing import Dict, List, Optional, Tuple
import os
import time
import random

from app import db
from app.models import (
    ContentIndex,
    InternalLinkAction,
    InternalLinkGraph,
    InternalSeoConfig,
    Site,
)
from app.wp_client import fetch_single_post, update_post_content
from app.services.internal_seo.legacy_cleaner import find_and_remove_legacy_links

logger = logging.getLogger(__name__)

# ---- HTMLユーティリティ ----

_P_CLOSE = re.compile(r"</p\s*>", re.IGNORECASE)
_A_TAG = re.compile(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a\s*>', re.IGNORECASE | re.DOTALL)
_TAG_STRIP = re.compile(r"<[^>]+>")
_SEO_CLASS = "ai-ilink"  # 互換用（生成時は使わない。既存の後方互換処理でのみ参照）

_H_TAG = re.compile(r"<h[1-6]\b[^>]*>", re.IGNORECASE)
_H_BLOCK = re.compile(r"(<h[1-6]\b[^>]*>)(.*?)(</h[1-6]\s*>)", re.IGNORECASE | re.DOTALL)
_TOC_HINT = re.compile(
    r'(id=["\']toc["\']|class=["\'][^"\']*(?:\btoctitle\b|\btoc\b|\bez\-toc\b)[^"\']*["\']|\[/?toc[^\]]*\])',
    re.IGNORECASE
)
_AI_STYLE_MARK = "<!-- ai-internal-link-style:v2 -->"

def _split_paragraphs(html: str) -> List[str]:
    if not html:
        return []
    # まず </p> で分割
    parts = [p for p in _P_CLOSE.split(html) if p is not None]
    # 段落が1つしか取れなかった場合は <br> でも分割
    if len(parts) <= 1:
        parts = re.split(r"<br\s*/?>", html, flags=re.IGNORECASE)
    # 最終的に空要素を除去
    parts = [p.strip() for p in parts if p and p.strip()]
    # 全く分割できなければ本文全体を1段落として返す
    return parts or [html]

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

_A_OPEN = re.compile(r"<a\b[^>]*>", re.IGNORECASE)
_A_CLOSE = re.compile(r"</a\s*>", re.IGNORECASE)

def _mask_existing_anchors(html: str) -> Tuple[str, Dict[str, str]]:
    """
    既存<a> ... </a> をプレースホルダに置き換えて保護する。
    置換後テキストで生の語句置換を行っても既存リンクを壊さないため。
    """
    placeholders: Dict[str, str] = {}
    out = []
    i = 0
    while i < len(html):
        m = _A_OPEN.search(html, i)
        if not m:
            out.append(html[i:])
            break
        # 直前までを追加
        out.append(html[i:m.start()])
        # 対応する </a> を探す
        mclose = _A_CLOSE.search(html, m.end())
        if not mclose:
            # 異常系。以後はそのまま
            out.append(html[m.start():])
            break
        seg = html[m.start(): mclose.end()]
        key = f"__A_PLACEHOLDER_{len(placeholders)}__"
        placeholders[key] = seg
        out.append(key)
        i = mclose.end()
    return "".join(out), placeholders

def _unmask_existing_anchors(html: str, placeholders: Dict[str, str]) -> str:
    for k, v in placeholders.items():
        html = html.replace(k, v)
    return html

def _linkify_first_occurrence(para_html: str, anchor_text: str, href: str) -> Optional[str]:
    """
    段落内の**未リンク領域**にある anchor_text の最初の出現を
    Wikipedia風の <a href="..." class="ai-ilink" title="...">anchor_text</a>
    に置換する。見つからなければ None。
    見出し/TOC を含むブロックでは実行しない。
    """
    if not (para_html and anchor_text and href):
        return None
    # 見出し・TOC っぽいブロックは一律除外
    if _H_TAG.search(para_html) or _TOC_HINT.search(para_html):
        return None
    masked, ph = _mask_existing_anchors(para_html)
    # 生テキストで最初の一致を探す（HTMLタグは残るが <a> はマスク済み）
    idx = masked.find(anchor_text)
    if idx == -1:
        return None
    # Wikipedia風：href + title のみ（class/style は付けない）
    linked = f'<a href="{href}" title="{anchor_text}">{anchor_text}</a>'
    masked = masked.replace(anchor_text, linked, 1)
    return _unmask_existing_anchors(masked, ph)

def _strip_links_in_headings(html: str) -> str:
    """
    見出し(H1〜H6)内の <a>…</a> を **テキスト/内側HTMLだけ残して除去** する。
    内部/外部リンクを問わず、見出しには一切リンクを残さない方針。
    例: <h2>foo <a href="/bar"><b>bar</b></a></h2> → <h2>foo <b>bar</b></h2>
    """
    if not html:
        return html
    def _drop_anchor_inner_keep(m: re.Match) -> str:
        open_h, inner, close_h = m.group(1), m.group(2), m.group(3)
        # 見出し内部の a を全て除去（中身は残す）
        inner = re.sub(
            r"<a\b[^>]*>(.*?)</a\s*>",
            r"\1",
            inner,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return open_h + inner + close_h
    # すべての見出しブロックに対して実行
    return _H_BLOCK.sub(_drop_anchor_inner_keep, html)

def _ensure_inline_underline_style(site: Site, html: str) -> str:
    """
    テーマCSSを触らず、リンク要素にも style を書かずに下線を効かせるため、
    記事先頭に 1度だけ最小の <style> を挿入する。
      対象: サイト内URLへ向く a 要素（Wikipedia と同じく「内部リンクは下線」）
    """
    if not html:
        return html
    # 旧/新マーカー付きの既存ブロックを一旦取り除いてから v2 を入れる
    html = re.sub(
        r'<!-- ai-internal-link-style:v[0-9]+ -->\s*<style>.*?</style>',
        '',
        html,
        flags=re.IGNORECASE | re.DOTALL
    )
    site_url = site.url.rstrip("/")
    # 本文(.ai-content)内の内部リンクのみ下線。見出し/目次は除外。
    css = (
        f'{_AI_STYLE_MARK}<style>'
        # 本文に限定
        f'.ai-content a[href^="{site_url}"]{{text-decoration:underline;}}'
        # 見出しは除外（上書き）
        f'.ai-content h1 a[href^="{site_url}"],'
        f'.ai-content h2 a[href^="{site_url}"],'
        f'.ai-content h3 a[href^="{site_url}"],'
        f'.ai-content h4 a[href^="{site_url}"],'
        f'.ai-content h5 a[href^="{site_url}"],'
        f'.ai-content h6 a[href^="{site_url}"]{{text-decoration:none;}}'
        # 代表的な TOC を除外（ez-toc / #toc / .toc / .toctitle）
        f'.ai-content .toctitle a,'
        f'.ai-content .toc a,'
        f'#toc a,'
        f'.ez-toc a{{text-decoration:none;}}'
        f'</style>'
    )
    return css + html

def _normalize_existing_internal_links(html: str) -> str:
    """
    既存の ai-ilink / inline-style を Wikipedia 風に正規化:
      <a href="..." class="ai-ilink" style="text-decoration:underline;">TEXT</a>
        → <a href="..." title="TEXT">TEXT</a>
    """
    if not html:
        return html
    # a タグの属性部（前後）と href、内側HTMLを分離
    pat = re.compile(
        r'<a\b([^>]*)\bhref=["\']([^"\']+)["\']([^>]*)>(.*?)</a\s*>',
        re.IGNORECASE | re.DOTALL
    )
    def _repl(m: re.Match) -> str:
        attrs_all = (m.group(1) or "") + (m.group(3) or "")
        href      = m.group(2) or ""
        inner     = m.group(4) or ""
        attrs_lc  = attrs_all.lower()
        # 既存の ai-ilink または inline style を含む a のみ正規化対象
        if ("ai-ilink" not in attrs_lc) and ("style=" not in attrs_lc):
            return m.group(0)
        text = _TAG_STRIP.sub(" ", unescape(inner)).strip()
        return f'<a href="{href}" title="{text}">{inner}</a>'
    return pat.sub(_repl, html)

def _add_attrs_to_first_anchor_with_href(html: str, href: str) -> str:
    """
    swap などで href を差し替えた最初の <a ... href="href"> を Wikipedia 風に正規化する。
    - class/style を除去
    - title が無ければ空で付与（後続の正規化で本文テキストに同期）
    """
    if not html or not href:
        return html
    pat = re.compile(
        rf'(<a\b[^>]*href=["\']{re.escape(href)}["\'][^>]*)(>)',
        re.IGNORECASE
    )
    def _repl(m):
        start, end = m.group(1), m.group(2)
        # class/style を丸ごと除去
        start = re.sub(r'\sclass=["\'][^"\']*["\']', '', start, flags=re.IGNORECASE)
        start = re.sub(r'\sstyle=["\'][^"\']*["\']', '', start, flags=re.IGNORECASE)
        # title が無ければ追加
        if not re.search(r'\btitle=["\']', start, re.IGNORECASE):
            start += ' title=""'
        return start + end
    # 最初の1件だけ注入
    return pat.sub(_repl, html, count=1)

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


def _all_url_to_title_map(site_id: int) -> Dict[str, str]:
    """
    サイト内すべての公開記事について URL→タイトル の辞書を返す。
    旧仕様削除判定に利用。
    """
    rows = (
        ContentIndex.query
        .with_entities(ContentIndex.url, ContentIndex.title)
        .filter_by(site_id=site_id, status="publish")
        .filter(ContentIndex.wp_post_id.isnot(None))
        .all()
    )
    return { (u or ""): (t or "") for (u, t) in rows if u }

def _all_url_to_pid_map(site_id: int) -> Dict[str, int]:
    """
    サイト内すべての公開記事について URL→wp_post_id の辞書を返す。
    旧仕様削除ログで target_post_id を紐づける用途。
    """
    rows = (
        ContentIndex.query
        .with_entities(ContentIndex.url, ContentIndex.wp_post_id)
        .filter_by(site_id=site_id, status="publish")
        .filter(ContentIndex.wp_post_id.isnot(None))
        .all()
    )
    out: Dict[str, int] = {}
    for (u, pid) in rows:
        if u and pid is not None:
            out[u] = int(pid)
    return out


# ---- 差分作成 ----

@dataclass
class ApplyResult:
    applied: int = 0
    swapped: int = 0
    skipped: int = 0
    legacy_deleted: int = 0
    message: str = ""

@dataclass
class PreviewItem:
    position: str
    anchor_text: str
    target_post_id: int
    target_url: str
    paragraph_index: int
    paragraph_excerpt_before: str
    paragraph_excerpt_after: str

def preview_apply_for_post(site_id: int, src_post_id: int) -> Tuple[str, ApplyResult, List[PreviewItem]]:
    """
    **副作用なし**のプレビュー。
    - pending actions を読み込み、_apply_plan_to_html を使って“仮適用”したHTMLを作る
    - DB更新もWP更新も行わない
    - どのアンカーが採用されたか（位置・テキスト・URL・前後抜粋）を返す
    """
    cfg = InternalSeoConfig.query.filter_by(site_id=site_id).one_or_none()
    if not cfg:
        cfg = InternalSeoConfig(site_id=site_id)
        db.session.add(cfg)
        db.session.commit()

    site = Site.query.get(site_id)
    wp_post = fetch_single_post(site, src_post_id)
    if not wp_post:
        return "", ApplyResult(message="fetch-failed-or-excluded"), []
    
    # 1) 旧仕様削除（プレビュー：DBは触らない）
    url_title_map = _all_url_to_title_map(site_id)
    cleaned_html, deletions = find_and_remove_legacy_links(wp_post.content_html or "", url_title_map)

    # 2) 新仕様の pending を取得
    actions = (
        InternalLinkAction.query
        .filter_by(site_id=site_id, post_id=src_post_id, status="pending")
        .order_by(InternalLinkAction.created_at.asc())
        .all()
    )

    url_map = _action_targets_with_urls(site_id, actions)

    original_paras = _split_paragraphs(wp_post.content_html or "")
    # 3) 旧仕様を除去した本文をベースに新仕様を仮適用
    base_html = cleaned_html if cleaned_html is not None else (wp_post.content_html or "")
    # 3.5) 見出し内リンクをサニタイズ（Hタグからはリンクを完全排除）
    base_html = _strip_links_in_headings(base_html)
    new_html, res = _apply_plan_to_html(site, src_post_id, base_html, actions, cfg, url_map)
    res.legacy_deleted = len(deletions or [])
    new_paras = _split_paragraphs(new_html)

    previews: List[PreviewItem] = []
    # _apply_plan_to_html 内で in-memory に a.status="applied" を立てるが、ここではcommitしない
    for a in actions:
        if a.status == "applied":
            try:
                pidx = int(a.position.split(":")[1]) if a.position and a.position.startswith("p:") else -1
            except Exception:
                pidx = -1
            before_snip = _html_to_text(original_paras[pidx])[:120] if (0 <= pidx < len(original_paras)) else ""
            after_snip  = _html_to_text(new_paras[pidx])[:120] if (0 <= pidx < len(new_paras)) else ""
            previews.append(PreviewItem(
                position=a.position or "",
                anchor_text=a.anchor_text or "",
                target_post_id=int(a.target_post_id),
                target_url=url_map.get(a.target_post_id, "") or "",
                paragraph_index=pidx,
                paragraph_excerpt_before=before_snip,
                paragraph_excerpt_after=after_snip,
            ))
    # pending が無くても、旧仕様削除プレビューは返す（message を調整）
    if not actions and res.legacy_deleted > 0:
        res.message = "legacy-clean-only"
    elif not actions:
        res.message = "no-pending"
    return new_html, res, previews   

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

    # 既定を 4〜8 に引き上げ（サイト設定があればそれを優先）
    need_min = max(4, int(cfg.min_links_per_post or 4))
    need_max = min(8, int(cfg.max_links_per_post or 8))

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

        # 段落本文を先に取り出してから各種チェックを行う（未定義エラー対策）
        para_html = paragraphs[idx]

        # 見出し/目次の段落はスキップ（Wikipedia方針）
        if _H_TAG.search(para_html) or _TOC_HINT.search(para_html):
            res.skipped += 1
            continue
        # 同じURLが段落内に既にあるならスキップ
        if href in [h for (h, _) in _extract_links(para_html)]:
            res.skipped += 1
            continue
        # 段落内の最初の未リンク出現をリンク化（Wikipedia風）
        new_para = _linkify_first_occurrence(para_html, act.anchor_text, href)
        if not new_para:
            # 見つからなければスキップ（このスロットの語句が無い）
            res.skipped += 1
            continue
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
                # 段落単位で置換し、見出し/TOC 段落は除外
                replaced = False
                for i, para in enumerate(paragraphs):
                    if _H_TAG.search(para) or _TOC_HINT.search(para):
                        continue
                    idx = para.find(f'href="{worst_url}"')
                    if idx == -1:
                        continue
                    new_para = para.replace(f'href="{worst_url}"', f'href="{new_href}"', 1)
                    # class/style を取り除き Wikipedia 風に（title は _normalize で整う）
                    new_para = _add_attrs_to_first_anchor_with_href(new_para, new_href)
                    paragraphs[i] = new_para
                    replaced = True
                    break
                if replaced:                    
                    s_act.status = "applied"
                    s_act.reason = "swap"  # 採用された置換
                    s_act.applied_at = datetime.utcnow()
                    res.swapped += 1

    # 本文を連結 → 既存の ai-ilink / inline-style を Wikipedia 風に正規化
    new_html = _rejoin_paragraphs(paragraphs)
    new_html = _normalize_existing_internal_links(new_html)
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

    # 1) 旧仕様削除（apply：後で削除ログを保存）
    url_title_map = _all_url_to_title_map(site_id)
    url_pid_map   = _all_url_to_pid_map(site_id)
    cleaned_html, deletions = find_and_remove_legacy_links(wp_post.content_html or "", url_title_map)

    # 2) 対象アクション（plan / swap_candidate）
    actions = (
        InternalLinkAction.query
        .filter_by(site_id=site_id, post_id=src_post_id, status="pending")
        .order_by(InternalLinkAction.created_at.asc())
        .all()
    )

    url_map = _action_targets_with_urls(site_id, actions)

    # 3) 差分作成（旧仕様削除済みの本文に新仕様を適用）
    base_html = cleaned_html if cleaned_html is not None else (wp_post.content_html or "")
    new_html, res = _apply_plan_to_html(site, src_post_id, base_html, actions, cfg, url_map)
    # 3.5) 見出し内リンクをサニタイズ（Hタグからはリンクを完全排除）
    base_html = _strip_links_in_headings(base_html)
    # 記事先頭に 1回だけ下線CSSを注入（テーマ非依存）
    new_html = _ensure_inline_underline_style(site, new_html)
    res.legacy_deleted = len(deletions or [])

    if dry_run:
        # ドライラン：DBを一切変更しない（旧仕様削除件数だけ反映）
        if not actions and res.legacy_deleted > 0:
            res.message = "legacy-clean-only"
        elif not actions:
            res.message = "no-pending"
        return res

    # 監査用の抜粋
    before_excerpt = _html_to_text(wp_post.content_html)[:280]
    after_excerpt = _html_to_text(new_html)[:280]

    # 4) DB更新（監査ログの抜粋）
    for a in actions:
        if a.status == "applied":
            a.diff_before_excerpt = before_excerpt
            a.diff_after_excerpt = after_excerpt
        else:
            # 適用されなかった pending はスキップへ
            a.status = a.status if a.status == "applied" else "skipped"
            a.updated_at = datetime.utcnow()

    # 5) 旧仕様削除ログを保存（1削除=1行、status='legacy_deleted'）
    if deletions:
        now = datetime.utcnow()
        for d in deletions:
            anchor_text = (d.get("anchor_text") or "")
            href        = (d.get("href") or "")
            position    = (d.get("position") or "")
            # target_post_id は cleaner が返すか、URL→PID で推定
            tpid = d.get("target_post_id")
            if not tpid and href:
                tpid = url_pid_map.get(href)
            try:
                tpid_int = int(tpid) if tpid is not None else None
            except Exception:
                tpid_int = None
            ila = InternalLinkAction(
                site_id=site_id,
                post_id=src_post_id,
                target_post_id=tpid_int,
                anchor_text=anchor_text,
                position=position,
                reason="legacy_cleanup",
                status="legacy_deleted",
                created_at=now,
                updated_at=now,
                diff_before_excerpt=before_excerpt,
                diff_after_excerpt=after_excerpt,
            )
            db.session.add(ila)        
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

        
        # --- レート制御：WP REST API 負荷保護 ---
        if not dry_run:
            try:
                # 1分あたりの最大件数（例: 120 → 0.5秒間隔）
                per_min = int(os.getenv("INTERNAL_SEO_RATE_LIMIT_PER_MIN", "0"))
                if per_min > 0:
                    base_sleep = 60.0 / max(1, per_min)
                else:
                    base_sleep = 0.5  # デフォルト最低500ms

                # 200〜500ms は最低保証
                base_sleep = max(base_sleep, 0.2)

                # ±30% のランダム揺らぎを加える
                sleep_time = base_sleep * random.uniform(0.7, 1.3)
                time.sleep(sleep_time)
            except Exception as e:
                logger.warning(f"[Applier] rate-limit sleep skipped due to error: {e}")

    logger.info("[Applier] site=%s result=%s", site_id, total)
    return total

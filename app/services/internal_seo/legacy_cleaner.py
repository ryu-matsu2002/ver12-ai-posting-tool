# app/services/internal_seo/legacy_cleaner.py
from __future__ import annotations
import re
import unicodedata
from html import unescape
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# aタグ抽出（オープンタグ全体も捕捉：版情報属性を読み取るため）
_A_TAG = re.compile(r'(<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>)(.*?)</a\s*>', re.I | re.S)
_TAG_STRIP = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")

# 版判定用：data-iseo="vX"（過去互換）、および直前コメント <!-- ai-internal-link:vX -->
_DATA_ISEO_RE = re.compile(r'\bdata-iseo=["\']([^"\']+)["\']', re.I)
_SPEC_COMMENT_TAIL_RE = re.compile(
    r'(?:<br\s*/?>\s*)*<!--\s*ai-internal-link:([a-zA-Z0-9._\-]+)\s*-->\s*$',
    re.I
)
# 最新仕様のアンカー末尾（この形以外は削除対象）
CTA_SUFFIX = "について詳しい解説はコチラ"
_TRAILING_PUNCT = re.compile(r"[、。．.;；.!！?？\s]+$")  # 末尾の余計な記号/空白
_CONTENT_CHARS = re.compile(r"[一-龥ぁ-んァ-ンーA-Za-z0-9]")

def _html_to_text(s: Optional[str]) -> str:
    return _TAG_STRIP.sub(" ", unescape(s or "")).strip()

def _norm(s: Optional[str]) -> str:
    """NFKC→空白除去→小文字→記号除去（和文記号も考慮）"""
    s = unicodedata.normalize("NFKC", s or "")
    s = _WS.sub("", s).lower()
    s = re.sub(r"[、。・,.!！?？:：;；\"'“”‘’()\[\]（）【】『』…\-_—–]", "", s)
    return s

@dataclass
class RemovedLink:
    href: str
    anchor_text: str
    position: str  # 例: "match:5"
    target_post_id: Optional[int] = None
    reason: str = "legacy_title_anchor"


def _is_legacy(anchor: str, title: str) -> bool:
    """
    旧仕様：本文に “リンク先の記事タイトル” をそのまま挿入し、
    その文字列にリンクしているケースのみ検出する。
    → 正規化後の厳密一致（完全一致）のみを採用。
    """
    na, nt = _norm(anchor), _norm(title)
    if not nt:
        return False
    return na == nt  # ← 部分一致は廃止し、厳密一致に限定

def _is_cta_compliant(anchor_text: str) -> bool:
    """
    「…について詳しい解説はコチラ」で厳密に終わることを必須にし、
    直前の本文部分が最低限の内容（“語”が複数）を持つかを簡易チェック。
    例外：
      - 末尾に句読点やセミコロンが付いている場合は不合格（旧仕様の名残りを排除）
    """
    if not anchor_text:
        return False
    text = anchor_text.strip()
    # 末尾の余計な記号を除去して判定（「コチラ；」「コチラ。」などを不合格に）
    if _TRAILING_PUNCT.search(text):
        # 余計な記号が付いている＝不合格
        return False
    if not text.endswith(CTA_SUFFIX):
        return False
    head = text[: -len(CTA_SUFFIX)].strip()
    # 前半が空や「について」で終わっている等は不完全とみなす
    if not head or head.endswith("について"):
        return False
    # “内容文字”がある程度含まれるかを緩めに確認（不完全な「向上させるため」単語のみ等を弾く）
    return len(_CONTENT_CHARS.findall(head)) >= 3


def find_and_remove_legacy_links(
    html: str,
    url_to_title: Dict[str, str],
    spec_version: Optional[str] = None,
) -> Tuple[str, List[Dict]]:
    """
    旧仕様リンク（V1〜V3、または“タイトル厳密一致リンク”）だけを削除する。
    版情報が無い「普通の内部リンク」は**残す**。
    内部SEO由来（data-iseo or 直前コメントの版マークがある）リンクに限り、
    CTA適合や重複hrefの整理も行う。
    引数:
      - html: 記事本文HTML
      - url_to_title: {URL: タイトル} の辞書（公開記事のみ）
    戻り値:
      - cleaned_html
      - deletions: 辞書の配列 [{href, anchor_text, position, target_post_id(任意)}]
    """
    if not html:
        return html, []

    removed: List[RemovedLink] = []
    out_parts: List[str] = []
    last = 0
    match_idx = 0

    # 直前コメントの判定では「マッチ直前のテキスト末尾」を見るため、各マッチごとに html の一部を参照
    # 重複hrefチェックは「内部SEO由来リンク」にのみ適用する
    seen_seo_hrefs: set[str] = set()

    # 既定の最新版（未指定なら v4 を最新版として扱う）
    latest = (spec_version or "v4").strip().lower()
    for m in _A_TAG.finditer(html):
        open_tag = m.group(1) or ""
        href = m.group(2) or ""
        inner_html = m.group(3) or ""
        anchor_text = _html_to_text(inner_html)

        # 内部記事（＝マップにあるURL）のみ対象
        title = url_to_title.get(href)
        if not title:
            continue
        
        
        # --- 版情報の検出（内部SEO由来の識別） ---
        # 1) data-iseo 属性（旧来 or 互換）
        ver_in_attr: Optional[str] = None
        m_attr = _DATA_ISEO_RE.search(open_tag)
        if m_attr:
            ver_in_attr = (m_attr.group(1) or "").strip().lower()
        # 2) 直前コメント <!-- ai-internal-link:vX -->
        ver_in_tail: Optional[str] = None
        prefix_tail = html[max(0, m.start() - 300): m.start()]  # 直前300文字程度を参照
        m_cmt = _SPEC_COMMENT_TAIL_RE.search(prefix_tail)
        if m_cmt:
            ver_in_tail = (m_cmt.group(1) or "").strip().lower()

        # 内部SEO由来か？（いずれかの版マークが付いている）
        is_seo_link = bool(ver_in_attr or ver_in_tail)
        # 当該リンクの「判定された版」
        detected_ver = ver_in_attr or ver_in_tail  # どちらかが入っていればその値

        # --- 削除・保持の判定方針 ---
        # 1) 内部SEO由来リンクのみ以下を適用
        #    - 旧版（detected_ver != latest）→ 削除
        #    - CTA不適合 → 削除
        #    - 同一hrefの重複（SEOリンク内で）→ 2本目以降を削除
        # 2) 版情報が無い普通の内部リンクは原則保持
        #    - 例外：旧仕様の“タイトル厳密一致リンク”は削除

        if is_seo_link:
            # 旧版（v1〜v3 など） → 削除
            if detected_ver and detected_ver != latest:
                out_parts.append(html[last:m.start()])
                last = m.end()
                removed.append(RemovedLink(
                    href=href,
                    anchor_text=anchor_text,
                    position=f"match:{match_idx}",
                    target_post_id=None,
                    reason="old_spec",
                ))
                match_idx += 1
                continue

            # 最新版でも CTA 不適合は削除（SEO由来のみ）
            if not _is_cta_compliant(anchor_text):
                out_parts.append(html[last:m.start()])
                last = m.end()
                removed.append(RemovedLink(
                    href=href,
                    anchor_text=anchor_text,
                    position=f"match:{match_idx}",
                    target_post_id=None,
                    reason="non_compliant_cta",
                ))
                match_idx += 1
                continue

            # 重複hrefは2本目以降を削除（SEO由来のみ）
            if href in seen_seo_hrefs:
                out_parts.append(html[last:m.start()])
                last = m.end()
                removed.append(RemovedLink(
                    href=href,
                    anchor_text=anchor_text,
                    position=f"match:{match_idx}",
                    target_post_id=None,
                    reason="duplicate_target_href",
                ))
                match_idx += 1
                continue
            else:
                seen_seo_hrefs.add(href)

        else:
            # 版情報なし：普通の内部リンクは保持。ただし“タイトル厳密一致”は旧仕様として削除。
            if _is_legacy(anchor_text, title):
                out_parts.append(html[last:m.start()])
                last = m.end()
                removed.append(RemovedLink(
                    href=href,
                    anchor_text=anchor_text,
                    position=f"match:{match_idx}",
                    target_post_id=None,
                    reason="legacy_title_anchor",
                ))
                match_idx += 1
                continue

        # ここまでで削除対象に当たらなければ保持
        match_idx += 1

    if removed:
        out_parts.append(html[last:])
        cleaned = "".join(out_parts)

        # aタグ削除により連続スペースや「 、。」の余りが出た場合の軽い整形（過度にいじらない）
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\s+([、。.,!！?？:：;；])", r"\1", cleaned)
        # dataclass → dict へ変換（applier 仕様）
        deletions = [
            {
                "href": r.href,
                "anchor_text": r.anchor_text,
                "position": r.position,
                # target_post_id は None（applier で URL→PID により補完）
                "target_post_id": r.target_post_id,
                "reason": r.reason,
            }
            for r in removed
        ]
        return cleaned, deletions
    else:
        return html, []

# ---- 互換ラッパー（旧API）。必要な場合のみDBからマップを構築して呼び出す ----
def clean_legacy_links(site_id: int, post_id: int, html: str) -> Tuple[str, List[RemovedLink]]:
    """
    互換用：旧API。サイトIDからURL→タイトルを組んで find_and_remove_legacy_links を呼ぶ。
    ※ 新コードからは `find_and_remove_legacy_links` を直接使ってください。
    """
    if not html:
        return html, []
    # 遅延インポートで循環/未使用を回避
    from app import db
    from app.models import ContentIndex
    rows = (
        db.session.query(ContentIndex.url, ContentIndex.title)
        .filter(ContentIndex.site_id == site_id, ContentIndex.status == "publish")
        .filter(ContentIndex.wp_post_id.isnot(None))
        .all()
    )
    url_to_title = { (u or ""): (t or "") for (u, t) in rows if u }
    # 既定で v4 を最新版として扱う
    cleaned, deletions = find_and_remove_legacy_links(html, url_to_title, spec_version="v4")
    # dict → RemovedLink へ戻す（呼び出し側が旧型を期待する場合のため）
    removed_objs = [
        RemovedLink(
            href=d.get("href") or "",
            anchor_text=d.get("anchor_text") or "",
            position=d.get("position") or "",
            target_post_id=d.get("target_post_id"),
            reason=d.get("reason") or "legacy_title_anchor",
        )
        for d in deletions
    ]
    return cleaned, removed_objs
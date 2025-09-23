# app/services/internal_seo/legacy_cleaner.py
from __future__ import annotations
import re
import unicodedata
from html import unescape
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# aタグ抽出
_A_TAG = re.compile(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a\s*>', re.I | re.S)
_TAG_STRIP = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")

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


def find_and_remove_legacy_links(
    html: str,
    url_to_title: Dict[str, str],
) -> Tuple[str, List[Dict]]:
    """
    旧仕様リンク（アンカー=リンク先記事タイトルと厳密一致）を検出し、
    <a>〜</a> を **丸ごと削除** する（＝追加されたタイトル文言ごと除去）。
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

    for m in _A_TAG.finditer(html):
        href = m.group(1) or ""
        inner_html = m.group(2) or ""
        anchor_text = _html_to_text(inner_html)

        # 内部記事（＝マップにあるURL）のみ対象
        title = url_to_title.get(href)
        if not title:
            continue

        if _is_legacy(anchor_text, title):
            # 旧仕様 → aタグ全体を削除（本文から丸ごと取り除く）
            out_parts.append(html[last:m.start()])
            last = m.end()

            removed.append(RemovedLink(
                href=href,
                anchor_text=anchor_text,
                position=f"match:{match_idx}",
                target_post_id=None,  # PIDは applier 側で URL→PID マップにより補完
                reason="legacy_title_anchor",
            ))
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
    cleaned, deletions = find_and_remove_legacy_links(html, url_to_title)
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
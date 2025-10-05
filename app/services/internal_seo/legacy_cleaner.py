# app/services/internal_seo/legacy_cleaner.py
from __future__ import annotations
import re
import unicodedata
from html import unescape
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
# topic判定を一元化（将来のルール変更にも追随）
from app.services.internal_seo.utils import is_topic_url

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
# 旧ボックス(v1)のコメント＆直後のボックス本体、黒ボックスの検出
_BOX_COMMENT_V1_RE = re.compile(r'<!--\s*ai-internal-link-box:v1\s*-->\s*', re.I)
_RELBOX_BLOCK_RE   = re.compile(r'<div[^>]*class=["\']ai-relbox[^>]*>.*?</div>', re.I | re.S)
_RELBOX_BLACK_RE   = re.compile(
    r'<div[^>]*class=["\']ai-relbox[^>]*style=["\'][^"\']*background\s*:\s*#111[^"\']*["\'][^>]*>.*?</div>',
    re.I | re.S
)
# --- 追加クリーンアップ: 空のプレースホルダ段落の削除 ---
def _drop_standalone_placeholders(text: str) -> str:
    """
    <p><!-- ai-internal-link... --> のみで構成される空段落を全て削除する。
    v0〜v13 まで（＝最新版 v14 以降は残す）。
    """
    return re.sub(
        r'<p\b[^>]*>\s*(?:<br\s*/?>\s*)*<!--\s*ai-internal-link(?:-box)?:v(?:[0-9]|1[0-3])\s*-->\s*</p\s*>',
        '',
        text,
        flags=re.I | re.S
    )
# vX コメント（リンク用/ボックス用）と “<p>…</p> で空行化されたマーカー” の検出
_LINK_MARK_RE      = re.compile(r'<!--\s*ai-internal-link:([a-z0-9._\-]+)\s*-->', re.I)
_BOX_MARK_RE       = re.compile(r'<!--\s*ai-internal-link-box:([a-z0-9._\-]+)\s*-->', re.I)
_P_LINK_MARK_RE    = re.compile(r'(?:<p[^>]*>\s*)<!--\s*ai-internal-link:([a-z0-9._\-]+)\s*-->\s*(?:</p\s*>)', re.I)
_P_BOX_MARK_RE     = re.compile(r'(?:<p[^>]*>\s*)<!--\s*ai-internal-link-box:([a-z0-9._\-]+)\s*-->\s*(?:</p\s*>)', re.I)
_ANY_EMPTY_P_RE    = re.compile(r'(?:<p[^>]*>\s*</p\s*>)', re.I)
# 最新仕様のアンカー末尾（この形以外は削除対象）
CTA_SUFFIX = "について詳しい解説はコチラ"
_TRAILING_PUNCT = re.compile(r"[、。．.;；.!！?？\s]+$")  # 末尾の余計な記号/空白
_CONTENT_CHARS = re.compile(r"[一-龥ぁ-んァ-ンーA-Za-z0-9]")
_H_OPEN_NEAR = re.compile(r"<h([1-6])\b[^>]*>", re.I)
_H_CLOSE_NEAR = re.compile(r"</h[1-6]\s*>", re.I)
_P_BLOCK = re.compile(r"(<p\b[^>]*>)(.*?)(</p\s*>)", re.I | re.S)

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

# 追加: “意味のないアンカー”を判定
_MEANINGLESS_WORDS = {
    "では", "まず", "こちら", "ここ", "そこ", "あそこ",
    "これ", "それ", "あれ", "この", "その", "あの",
}
def _is_meaningless_anchor(anchor_text: str) -> bool:
    t = _norm(anchor_text)
    if not t:
        return False
    if t in _MEANINGLESS_WORDS:
        return True
    # 1〜2文字や、指示語＋助詞系（例: こちらへ/こちらに）
    if len(t) <= 2:
        return True
    if t.startswith("こちら") or t in {"くわしくはこちら", "詳しくはこちら", "詳細はこちら"}:
        return True
    return False

def _inside_heading(html: str, a_start: int, a_end: int) -> bool:
    """
    aタグの前後をざっくり見て、見出し(H1〜H6)内にあるっぽければ True。
    （保守的に判定：直前200字内に<h..>があり、直後200字内に</h..>がある）
    """
    left = html[max(0, a_start-200):a_start]
    right = html[a_end:min(len(html), a_end+200)]
    return bool(_H_OPEN_NEAR.search(left) and _H_CLOSE_NEAR.search(right))

def _find_enclosing_p(html: str, a_start: int, a_end: int) -> Optional[Tuple[int,int,str,str,str]]:
    """
    aタグを内包している直近の<p>...</p>ブロックを返す。
    戻り値: (p_start, p_end, open_p, inner, close_p) / 無ければ None
    """
    # 直前の <p> を探す
    p_open_pos = html.rfind("<p", 0, a_start)
    if p_open_pos == -1:
        return None
    # その位置から最初の </p> を探す
    m = _P_BLOCK.search(html, p_open_pos)
    if not m:
        return None
    p_start, p_end = m.start(), m.end()
    if not (p_start <= a_start and a_end <= p_end):
        return None
    return (p_start, p_end, m.group(1) or "", m.group(2) or "", m.group(3) or "")

def _is_anchor_only_paragraph(inner_html: str) -> bool:
    """
    <p> の中身が「改行/空白 +（装飾を含む）a要素 だけ」で構成されているかを判定。
    aの中身テキスト以外に有意味テキストが無ければ True。
    """
    # a をプレースホルダ化して残滓を確認
    tmp = re.sub(r"<a\b[^>]*>.*?</a\s*>", "__A__", inner_html, flags=re.I|re.S)
    # a以外のタグは消す
    tmp = _TAG_STRIP.sub(" ", tmp)
    tmp = _WS.sub(" ", tmp).strip()
    # 「__A__」のみ（±前後に何もない）ならアンカー単独段落
    return tmp == "__A__"

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

    # --- 事前クリーンアップ：旧ボックス/孤立コメントの除去 -----------------
    # 1) v1コメント直後の ai-relbox を丸ごと削除
    #    （<!-- ai-internal-link-box:v1 --> の直後にある .ai-relbox を対象）
    def _drop_v1_box_blocks(text: str) -> str:
        pos = 0
        chunks = []
        while True:
            m = _BOX_COMMENT_V1_RE.search(text, pos)
            if not m:
                chunks.append(text[pos:])
                break
            # コメント以降で最初に現れる ai-relbox を探す
            after = m.end()
            mbox = _RELBOX_BLOCK_RE.search(text, after)
            if mbox and mbox.start() == after or mbox and text[after:mbox.start()].strip() == "":
                # コメント〜ボックス終端までを削除
                chunks.append(text[pos:m.start()])
                pos = mbox.end()
            else:
                # コメントだけ見つかった場合はコメントのみ削除
                chunks.append(text[pos:m.start()])
                pos = m.end()
        return "".join(chunks)

    # 2) 黒ボックス（background:#111 を含む .ai-relbox）を無条件で削除
    def _drop_black_boxes(text: str) -> str:
        return _RELBOX_BLACK_RE.sub("", text)

    html = _drop_black_boxes(_drop_v1_box_blocks(html))
    html = _drop_standalone_placeholders(html)

    # 既定の最新版（未指定なら v14 として扱う）
    latest = (spec_version or "v14").strip().lower()

    # 1) 旧版ボックス印（コメント）＋直後の .ai-relbox を丸ごと除去（v1〜v13 等を網羅）
    #    - <p><!-- ai-internal-link-box:vN --></p> 形式も対象
    def _drop_old_box_marks(text: str) -> str:
        pos = 0
        out = []
        while True:
            m = _BOX_MARK_RE.search(text, pos)
            if not m:
                out.append(text[pos:])
                break
            ver = (m.group(1) or "").strip().lower()
            if ver == latest:
                # 最新版のマークは保持
                out.append(text[pos:m.end()])
                pos = m.end()
                continue
            # 直前に <p>…> がある場合はそこから、無ければマーク開始から削る
            start = m.start()
            # 直前の <p> を探して「マークだけの段落」ならそこから落とす
            p_open = text.rfind("<p", 0, start)
            if p_open != -1:
                pm = _P_BOX_MARK_RE.search(text, p_open)
                if pm and pm.start() <= start <= pm.end():
                    start = p_open
            # 直後に ai-relbox が続いていればそれも含めて除去
            after = m.end()
            mbox = _RELBOX_BLOCK_RE.match(text, after)
            if mbox:
                end = mbox.end()
            else:
                end = after
            out.append(text[pos:start])
            pos = end
        return "".join(out)

    # 2) 旧版 “リンク用” マーカーのうち、**直後に <a> が無い孤立行** を除去
    #    （<p><!-- ai-internal-link:vN --></p> / 連打しているだけの残骸を掃除）
    def _drop_orphan_old_link_marks(text: str) -> str:
        # <p>…</p> 形式
        def repl_p(m: re.Match) -> str:
            ver = (m.group(1) or "").strip().lower()
            return "" if ver != latest else m.group(0)  # 最新は維持（後段で a が続かない場面は後処理で落とす）
        text = _P_LINK_MARK_RE.sub(repl_p, text)
        # コメント素の形式（ただし直後に <a> が来ないものだけ）
        def _repl_plain(m: re.Match) -> str:
            ver = (m.group(1) or "").strip().lower()
            tail = text[m.end(): m.end()+200]
            has_next_a = bool(re.match(r'\s*(?:<br\s*/?>\s*)*<a\b', tail, re.I))
            if (ver != latest) and (not has_next_a):
                return ""
            return m.group(0)
        return _LINK_MARK_RE.sub(_repl_plain, text)

    html = _drop_old_box_marks(html)
    html = _drop_orphan_old_link_marks(html)

    # 以降は aタグ単位の本処理
    for m in _A_TAG.finditer(html):
        open_tag = m.group(1) or ""
        href = m.group(2) or ""
        inner_html = m.group(3) or ""
        anchor_text = _html_to_text(inner_html)

        # 内部記事（＝マップにあるURL）のみ対象
        title = url_to_title.get(href)
        if not title:
            continue

        # --- topicページへのリンクは絶対に削除・変更しない ---
        # is_topic_url() による一元判定（小文字化/パターン拡張にも対応）
        if is_topic_url(href):
            match_idx += 1
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

        # 見出し内は一切触らない
        if _inside_heading(html, m.start(), m.end()):
            match_idx += 1
            continue

        if is_seo_link:
            # 旧版（v1〜v4 など） → 削除（直前に残っているボックスがあれば近傍も削除）
            if detected_ver and detected_ver != latest:
                # 直前にある ai-relbox をまとめて削除
                prefix = html[max(0, m.start()-500):m.start()]
                relbox_match = re.search(r'<div[^>]*class=["\']ai-relbox[^>]*>.*?</div>', prefix, re.I|re.S)
                if relbox_match:
                    # relbox 開始位置から削除
                    relbox_start = prefix.rfind("<div", 0, relbox_match.end())
                    if relbox_start != -1:
                        out_parts.append(html[last:m.start()-len(prefix)+relbox_start])
                        # 段落ごと落とせるなら優先して落とす
                        p = _find_enclosing_p(html, m.start(), m.end())
                        if p and _is_anchor_only_paragraph(p[3]):
                            last = p[1]
                        else:
                            last = m.end()
                else:
                    out_parts.append(html[last:m.start()])
                    p = _find_enclosing_p(html, m.start(), m.end())
                    if p and _is_anchor_only_paragraph(p[3]):
                        last = p[1]
                    else:
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
                p = _find_enclosing_p(html, m.start(), m.end())
                if p and _is_anchor_only_paragraph(p[3]):
                    last = p[1]
                else:
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
                p = _find_enclosing_p(html, m.start(), m.end())
                if p and _is_anchor_only_paragraph(p[3]):
                    last = p[1]
                else:
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
            # 版情報なし：原則保持。ただし
            #  - “タイトル厳密一致”リンク → a だけ剥がす（テキストは残す）
            #  - “意味のないアンカー”（では/まず/こちら.../1〜2文字）→ a だけ剥がす
            if _is_legacy(anchor_text, title) or _is_meaningless_anchor(anchor_text):
                # 直近の <p> … </p> が “アンカー単独段落”なら段落ごと削除
                p = _find_enclosing_p(html, m.start(), m.end())
                if p and _is_anchor_only_paragraph(p[3]):
                    out_parts.append(html[last:p[0]])  # <p>開始までを残す
                    last = p[1]                        # </p>の次から再開
                    reason = "drop_anchor_only_paragraph"
                else:
                    # aだけ剥がして中身テキストは残す
                    out_parts.append(html[last:m.start()])
                    out_parts.append(inner_html)
                    last = m.end()
                    reason = "unlink_legacy_or_meaningless"
                removed.append(RemovedLink(
                    href=href,
                    anchor_text=anchor_text,
                    position=f"match:{match_idx}",
                    target_post_id=None,
                    reason=reason,
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
                "reason": r.reason,  # old_spec / non_compliant_cta / duplicate_target_href / unlink_legacy_or_meaningless
            }
            for r in removed
        ]
        # 後処理：孤立マーカーの掃除（ただし最新版 latest は残す）
        def _drop_orphan_marks_latest_safe(text: str, latest: str) -> str:
            def _repl_link(m: re.Match) -> str:
                ver = (m.group(1) or "").strip().lower()
                return "" if ver != latest else m.group(0)
            def _repl_box(m: re.Match) -> str:
                ver = (m.group(1) or "").strip().lower()
                return "" if ver != latest else m.group(0)
            text = _P_LINK_MARK_RE.sub(_repl_link, text)
            text = _P_BOX_MARK_RE.sub(_repl_box,  text)
            return text
        cleaned = _drop_orphan_marks_latest_safe(cleaned, latest)
        # マーク削除で生じた空<p> の束を軽く整理
        cleaned = _ANY_EMPTY_P_RE.sub("", cleaned)
        return cleaned, deletions
    else:
        # 本処理で削除が無くても、旧版の孤立マーカーのみ落とし最新版は残す
        def _drop_only_old_marks(text: str, latest: str) -> str:
            def _repl_link(m: re.Match) -> str:
                ver = (m.group(1) or "").strip().lower()
                return "" if ver != latest else m.group(0)
            def _repl_box(m: re.Match) -> str:
                ver = (m.group(1) or "").strip().lower()
                return "" if ver != latest else m.group(0)
            text = _P_LINK_MARK_RE.sub(_repl_link, text)
            text = _P_BOX_MARK_RE.sub(_repl_box,  text)
            return text
        cleaned = _drop_only_old_marks(html, latest)
        cleaned = _ANY_EMPTY_P_RE.sub("", cleaned)
        return cleaned, []

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
    # 既定で v14 を最新版として扱う
    cleaned, deletions = find_and_remove_legacy_links(html, url_to_title, spec_version="v14")
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
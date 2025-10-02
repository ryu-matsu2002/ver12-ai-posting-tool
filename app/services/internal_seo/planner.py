# app/services/internal_seo/planner.py
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from html import unescape
from typing import Dict, Iterable, List, Optional, Tuple
from collections import Counter
import os
from app import db

from app.models import ContentIndex, InternalLinkAction, InternalLinkGraph, InternalSeoConfig, Site
from app.services.internal_seo.utils import (
    html_to_text,
    nfkc_norm,
    extract_terms_for_partial,
    extract_h2_sections,
    is_ng_anchor,
    STOPWORDS_N,
    title_tokens,
    keywords_set,
    jaccard,
    title_tfidf_cosine,
    is_natural_span,
)

from app.services.internal_seo.applier import (
    _H_TAG, _TOC_HINT, _mask_existing_anchors, _split_paragraphs, _link_version_int
)  # 再利用  # 既存互換のため残置（本改修では段落スロットは使用しない）

from datetime import datetime, UTC

from app.wp_client import fetch_single_post  # 現在のHTMLを読む用（swap判定で使用）

logger = logging.getLogger(__name__)

# ---------- テキスト/段落分割系 ----------

_P_CLOSE = re.compile(r"</p\s*>", re.IGNORECASE)
_BR_SPLIT = re.compile(r"<br\s*/?>", re.IGNORECASE)
_A_TAG = re.compile(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a\s*>', re.IGNORECASE | re.DOTALL)
_TAG_STRIP = re.compile(r"<[^>]+>")
_STYLE_BLOCK = re.compile(r"<style\b[^>]*>.*?</style\s*>", re.IGNORECASE | re.DOTALL)

JP_TOKEN = re.compile(r"[一-龥ぁ-んァ-ンーA-Za-z0-9]{2,}")
# アンカーテキストの推奨最大全角相当（長い文章リンクを避け、単語優先に）
# 必要に応じて環境変数 INTERNAL_SEO_MAX_ANCHOR_LEN で調整可能（デフォルト18）
MAX_ANCHOR_CHARS = int(os.getenv("INTERNAL_SEO_MAX_ANCHOR_LEN", "18"))
# 同一段落に許容する最大リンク本数（不足時の救済用：厳しめに既定=1）
MAX_LINKS_PER_PARA = int(os.getenv("INTERNAL_SEO_MAX_LINKS_PER_PARA", "1"))

# 追加：<style>タグが剥がれて“CSSテキストだけ”になっても本文扱いにしないための判定
_CSS_LIKE_TEXT = re.compile(
    r'(?:/\*.*?\*/)|'           # コメント記法 /* ... */
    r'(?:\{[^}]*\})|'           # {...} のプロパティ集合
    r'(?:\btext-decoration\b)|' # 代表的なCSSプロパティ
    r'(?:\bcolor\s*:)|'         # 代表的なCSSプロパティ
    r'(?:\.ai-content\b)|'      # 今回の注入CSSで使用
    r'(?:a\[href\^\=)'          # a[href^="..."] セレクタ
, re.IGNORECASE | re.DOTALL)

# 類似判定は少しだけ厳しめ（必要なら環境変数で調整）
GENRE_JACCARD_MIN = float(os.getenv("INTERNAL_SEO_GENRE_JACCARD_MIN", "0.30"))
TITLE_COSINE_MIN  = float(os.getenv("INTERNAL_SEO_TITLE_COSINE_MIN", "0.20"))
# LinkGraph が十分強いときのみ同ジャンル判定を免除するスコア下限（強め）
STRONG_LINKGRAPH_SCORE = float(os.getenv("INTERNAL_SEO_STRONG_LINKGRAPH_SCORE", "0.10"))

# 段落分割は applier と完全に同じ実装を使う（index不一致を避ける）
# _split_paragraphs は applier から import 済み

def _html_to_text(s: str) -> str:
    if not s:
        return ""
    return _TAG_STRIP.sub(" ", unescape(s)).strip()

def _candidate_anchor_from(title: str, para_text: str) -> Optional[str]:
    """
    段落テキスト内に**実在する**語句のみをアンカーとして採用する。
    - タイトルのトークン群を長い順に並べ、段落内に出現する最初のものを返す
    - 見つからなければ None（＝このスロットは作らない）
    """
    title_txt = _html_to_text(title)[:80]
    para_norm = (para_text or "").strip()
    if not title_txt or not para_norm:
        return None

    tokens = title_tokens(title_txt)

    for tk in tokens:
        if len(tk) < 2:
            continue
        if tk in para_norm and is_natural_span(para_norm, tk):
            # 過度に長いのは避ける
            cand = tk[: min(40, MAX_ANCHOR_CHARS)]
            if not is_ng_anchor(cand):
                return cand
    return None

def _candidate_anchor_from_target_content(
    site: Site,
    target_post_id: int,
    para_text: str,
    *,
    max_len: int = 40,
    min_token_len: int = 2,
) -> Optional[str]:
    """
    ターゲット記事“本文”から抽出した語を優先して、段落内の自然語句をアンカーにする。
    - ターゲット本文HTMLを取得→テキスト化→JP_TOKENでトークン化
    - 長い語句を優先しつつ、頻度もスコアに入れて順位付け
    - 段落テキスト内に最初に出現した語を採用（過剰長は切り詰め）
    フォールバックは従来のタイトル由来アンカーに任せる（呼び出し側）。
    """
    para_norm = (para_text or "").strip()
    if not para_norm:
        return None
    try:
        tgt = fetch_single_post(site, target_post_id)
    except Exception:
        tgt = None
    if not tgt or not (tgt.content_html or "").strip():
        return None
    tgt_txt = _html_to_text(tgt.content_html)
    if not tgt_txt:
        return None
    tokens = [t for t in JP_TOKEN.findall(tgt_txt) if len(t) >= min_token_len and (nfkc_norm(t) not in STOPWORDS_N)]
    if not tokens:
        return None
    # Wikipedia 的な振る舞い：段落内で最も早い位置に現れる語を選ぶ（タイは“より長い語”を優先）
    earliest: Optional[Tuple[int, int, str]] = None  # (start_idx, -len, token)
    seen = set()
    for tk in tokens:
        if tk in seen:
            continue
        seen.add(tk)
        idx = para_norm.find(tk)
        if idx >= 0:
            # STOPWORDS を除外
            if is_ng_anchor(tk):
                continue
            cand = (idx, -len(tk), tk)
            if (earliest is None) or (cand < earliest):
                earliest = cand
    if earliest:
        best = earliest[2][: min(max_len, MAX_ANCHOR_CHARS)]
        if not is_ng_anchor(best):
            return best        
    return None


def _candidate_anchor_by_partial(
    title: str,
    target_body_text: str,
    para_text: str,
    *,
    max_len: int = 40,
    min_token_len: int = 2,
) -> Optional[str]:
    """
    部分一致許容：タイトルおよびターゲット本文から抽出した代表トークン群（正規化/NFKC）を
    段落テキスト（正規化/NFKC）に対して substring マッチ。長い語優先。
    - “本文に出てこない語は挿入しない”方針は維持（必ず para に出現する語だけ返す）
    - 記号・全半角の揺れを吸収
    """
    p_norm = nfkc_norm((para_text or "").strip())
    if not p_norm:
        return None
    # 候補語：タイトル＋ターゲット本文から抽出
    terms: List[str] = []
    terms += extract_terms_for_partial(title or "")           # utils 由来（NFKC/フィルタ済み想定）
    terms += extract_terms_for_partial(target_body_text or "")
    # 重複除去し、長い順に
    uniq = sorted(
        {t for t in terms if len(t) >= min_token_len and (nfkc_norm(t) not in STOPWORDS_N)},
        key=lambda s: (-len(s), s)
    )
    for t in uniq:
        tn = nfkc_norm(t)
        if tn and tn in p_norm:
            cand = t[: min(max_len, MAX_ANCHOR_CHARS)]
            if not is_ng_anchor(cand):
                return cand
    return None


# === 追加: 段落×ターゲット “重なり語” からベストを選ぶ ===
def _extract_para_tokens(para_text: str) -> list[str]:
    toks = JP_TOKEN.findall(para_text or "")
    out = []
    for t in toks:
        if len(t) < 2:
            continue
        if nfkc_norm(t) in STOPWORDS_N:
            continue
        out.append(t)
    return out

def _extract_target_tokens_from_index(site_id: int, pid: int) -> tuple[list[str], set[str]]:
    """
    ContentIndex からタイトルとキーワードを軽量に取得（必ず (list, set) を返す）。
    - 戻り: (重要語リスト, キーワード集合)  ※キーワード集合はスコア加点に使用
    """
    try:
        tr = (
            ContentIndex.query
            .with_entities(ContentIndex.title, ContentIndex.keywords)
            .filter_by(site_id=site_id, wp_post_id=pid)
            .one_or_none()
        )
        if not tr:
            return [], set()
        title = tr[0] or ""
        kws_csv = tr[1] or ""
        toks = JP_TOKEN.findall(title.lower())
        # タイトルは重要なので重みを意識しつつ2回入れる（ここでは順序のみ利用）
        imp: list[str] = []
        for _ in range(2):
            for t in toks:
                if len(t) >= 2 and (nfkc_norm(t) not in STOPWORDS_N):
                    imp.append(t)
        kwset = {k.strip().lower() for k in (kws_csv or "").split(",") if k.strip()}
        return imp, kwset
    except Exception:
        # どんな例外でも安全にフォールバック
        return [], set()
   
   
def _get_keywords_set(site_id: int, pid: int) -> set[str]:
    """
    ContentIndex.keywords をセットで取得（小文字・トリム済み）
    """
    tr = (
        ContentIndex.query
        .with_entities(ContentIndex.keywords)
        .filter_by(site_id=site_id, wp_post_id=pid)
        .one_or_none()
    )
    if not tr:
        return set()
    kws_csv = tr[0] or ""
    return {k.strip().lower() for k in (kws_csv or "").split(",") if k.strip()}   

def _pick_anchor_from_overlap(site: Site, target_post_id: int, para_text: str) -> Optional[str]:
    """
    段落に“実在する語” ∩ “ターゲットが大事にする語（タイトル・キーワード）”の交差から選ぶ。
    スコア: 早く出てくるほど+ / 長いほど+ / キーワード命中で+ / NG語は即除外
    """
    para = (para_text or "").strip()
    if not para:
        return None
    para_tokens = _extract_para_tokens(para)
    if not para_tokens:
        return None
    imp_tokens, kwset = _extract_target_tokens_from_index(site.id, target_post_id)
    if not imp_tokens and not kwset:
        return None

    best = None
    best_score = -1.0
    for t in set(para_tokens):
        if is_ng_anchor(t):
            continue
        idx = para.find(t)
        if idx < 0:
            continue
        length_bonus = min(len(t), 40) * 0.02     # 2文字で+0.04, 10文字で+0.2 くらい
        pos_bonus    = max(0.0, 1.0 - (idx / max(1, len(para)))) * 0.3  # 早いほど+（最大0.3）
        kw_bonus     = (t.lower() in kwset) * 0.3  # キーワードに載っていれば+0.3
        imp_bonus    = (t in imp_tokens) * 0.2     # タイトル重要語なら+0.2
        score = length_bonus + pos_bonus + kw_bonus + imp_bonus
        if score > best_score:
            best_score = score
            best = t
    if best and not is_ng_anchor(best):
        return best[: min(40, MAX_ANCHOR_CHARS)]
    return None

# === 新規: 「タイトル語 ∩ 段落」の厳密一致（最優先） =====================
def _pick_anchor_from_title_overlap(title: str, para_text: str) -> Optional[str]:
    """
    要件：アンカーワードはリンク先記事タイトルの語と一致し、かつ段落に実在。
    ルール：長い語優先→段落中で早く出る語を優先。NG語/不自然位置は除外。
    """
    if not title or not para_text:
        return None
    toks = title_tokens(title)
    if not toks:
        return None
    ptxt = (para_text or "").strip()
    best = None
    best_key = None  # (idx, -len)
    for tk in toks:
        if is_ng_anchor(tk):
            continue
        idx = ptxt.find(tk)
        if idx < 0:
            continue
        if not is_natural_span(ptxt, tk):
            continue
        key = (idx, -len(tk))
        if (best is None) or (key < best_key):
            best = tk
            best_key = key
    if best:
        return best[: min(MAX_ANCHOR_CHARS, 40)]
    return None

def _pick_anchor_from_common_keywords(site_id: int, src_post_id: int, target_post_id: int, para_text: str) -> Optional[str]:
    """
    ソース記事とターゲット記事の keywords の共通語を最優先。
    - 共通keywordsを正規化して用意
    - 段落内に“実在する語”のうち、早く出る/長い語を優先（禁止語は除外）
    """
    para = (para_text or "").strip()
    if not para:
        return None
    para_tokens = _extract_para_tokens(para)  # 段落に実在する語（NG語はここでも除外）
    if not para_tokens:
        return None
    src_kw = _get_keywords_set(site_id, src_post_id)
    tgt_kw = _get_keywords_set(site_id, target_post_id)
    common = src_kw & tgt_kw
    if not common:
        return None
    # 正規化済みで照合するため、段落トークンを nfkc_norm したキーで判定
    best = None
    best_score = -1.0
    for t in set(para_tokens):
        if is_ng_anchor(t):
            continue
        tn = nfkc_norm(t).lower()
        if not tn or tn not in common:
            continue
        idx = para.find(t)
        if idx < 0:
            continue
        length_bonus = min(len(t), 40) * 0.02
        pos_bonus    = max(0.0, 1.0 - (idx / max(1, len(para)))) * 0.3
        score = length_bonus + pos_bonus
        if score > best_score:
            best_score = score
            best = t
    if best and not is_ng_anchor(best):
        return best[: min(40, MAX_ANCHOR_CHARS)]
    return None

def _pick_anchor_from_common_keywords_single(
    site_id: int,
    src_post_id: int,
    target_post_id: int,
    para_text: str
) -> Optional[str]:
    """
    ★新規★ ソース/ターゲットの共通keywordsに含まれる「単語」を本文中で最優先してアンカーにする。
      - ContentIndex.keywords の共通集合（小文字・トリム済み）を用意
      - 段落内トークンのうち、共通集合に入る単語を左から優先（同点は長い語優先）
      - 禁止語は除外、長すぎる語は採用しない（MAX_ANCHOR_CHARS）
    """
    para = (para_text or "").strip()
    if not para:
        return None
    para_tokens = JP_TOKEN.findall(para)
    if not para_tokens:
        return None
    common = _get_keywords_set(site_id, src_post_id) & _get_keywords_set(site_id, target_post_id)
    if not common:
        return None
    best: Optional[Tuple[int, int, str]] = None  # (start_idx, -len, token)
    seen = set()
    for t in para_tokens:
        if t in seen:
            continue
        seen.add(t)
        tn = nfkc_norm(t).lower()
        if (not tn) or (tn not in common):
            continue
        if is_ng_anchor(t):
            continue
        if len(t) > MAX_ANCHOR_CHARS:
            continue
        idx = para.find(t)
        if idx < 0:
            continue
        cand = (idx, -len(t), t)
        if (best is None) or (cand < best):
            best = cand
    if best:
        return best[2]
    return None


def _pick_paragraph_slots(
    paragraphs: List[str],
    need: int,
    min_len: int,
    *,
    allow_repeat: bool = False,
    max_per_para: int = 1,
) -> List[int]:
    """
    序盤・中盤・終盤に分散するように段落indexを選ぶ。
    短すぎる段落は除外。
    """
    # 見出し/TOC 段落は除外（applier と同方針）
    def _is_ok_para(p: str) -> bool:
        if _H_TAG.search(p) or _TOC_HINT.search(p):
            return False
        # スタイルブロックは本文対象外
        if _STYLE_BLOCK.search(p):
            return False
        txt = _html_to_text(p)
        # ★追加：タグが剥がれても“CSSっぽいテキスト”は本文から除外
        if _CSS_LIKE_TEXT.search(txt or ""):
            return False
        return bool(txt) and len(txt) >= min_len

    # 短い記事にも対応するため、閾値未満でも最終的にフォールバックで拾う
    eligible = [i for i, p in enumerate(paragraphs) if _is_ok_para(p)]
    # ★不足時の段階的フォールバック（base→70→60→50→40 まで。20 には下げない）
    if len(eligible) < need:
        for relaxed in (70, 60, 50, 40):
            def _is_ok_relax(p: str) -> bool:
                if _H_TAG.search(p) or _TOC_HINT.search(p) or _STYLE_BLOCK.search(p):
                    return False
                txt = _html_to_text(p)
                if _CSS_LIKE_TEXT.search(txt or ""):
                    return False
                return bool(txt) and len(txt) >= relaxed
            eligible = [i for i, p in enumerate(paragraphs) if _is_ok_relax(p)]
            if len(eligible) >= need or eligible:
                break
    if not eligible:
        # 1文でもある段落を対象にフォールバック（安全に末尾挿入）
        eligible = [
            i for i, p in enumerate(paragraphs)
            if (_html_to_text(p)
                and not (_H_TAG.search(p) or _TOC_HINT.search(p) or _STYLE_BLOCK.search(p)))
        ]
        if not eligible:
            return []
    # 3ゾーンで均等抽出（必要数に応じて）
    zones = []
    n = len(eligible)
    if need <= 3 and len(eligible) >= 1:
        zones = [0, n // 2, n - 1]
        zones = [eligible[min(max(z, 0), n - 1)] for z in zones][:need]
        return sorted(set(zones))[:need]
    # need > 3 の場合は等間隔サンプリング（ユニークに）
    step = max(1, n // max(1, need))
    uniq_slots = [eligible[min(i * step, n - 1)] for i in range(min(need, n))]
    uniq_slots = sorted(set(uniq_slots))[:min(need, n)]
    # --- 追加: 最小ギャップ=1（隣接 index を避ける） ---
    def _apply_min_gap(slots: List[int], gap: int = 1) -> List[int]:
        if not slots:
            return []
        kept: List[int] = []
        for s in sorted(slots):
            if not kept or (s - kept[-1] > gap):
                kept.append(s)
        return kept

    uniq_slots = _apply_min_gap(uniq_slots, gap=1)
    if (not allow_repeat) or (len(uniq_slots) >= need):
        return uniq_slots
    # ここから救済：同一段落に複数本（max_per_para）まで許容して need を満たす
    counts = {idx: 0 for idx in eligible}
    for idx in uniq_slots:
        counts[idx] += 1
    slots = list(uniq_slots)
    i = 0
    while len(slots) < need and eligible:
        idx = eligible[i % len(eligible)]
        # 追加: 直前に使った段落の「隣接」は避ける
        if counts[idx] < max_per_para:
            slots.append(idx)
            counts[idx] += 1
        i += 1
        # 念のためセーフガード
        if i > need * max(1, len(eligible)) * max(1, max_per_para):
            break
    # 最終的にも最小ギャップを適用して返す（不足はそのまま）
    slots = _apply_min_gap(sorted(slots), gap=1)
    return slots[:need]
    # ※ これ以上の緩和は行わない（関連性・可読性の担保）

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

# 既存の本文内「内部リンク」本数を概算（URL→post_id が解決できたものを内部リンクとみなす）
def _count_existing_internal_links(existing_links: List[Tuple[str, str]], url_to_pid: Dict[str, Optional[int]]) -> int:
    cnt = 0
    for href, _ in existing_links:
        if url_to_pid.get(href):
            cnt += 1
    return cnt

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
    return [(int(t), float(s)) for (t, s) in rows if s is not None and float(s) >= float(min_score)]


def _same_genre_ok(site_id: int, src_post_id: int, tgt_post_id: int) -> bool:
    """
    keywordsのJaccardで同ジャンル判定。足りなければタイトルコサイン近似でフォールバック。
    ContentIndexにカテゴリ/タグが無い前提で、keywordsとタイトルで代用。
    """
    src = (
        ContentIndex.query
        .with_entities(ContentIndex.keywords, ContentIndex.title)
        .filter_by(site_id=site_id, wp_post_id=src_post_id)
        .one_or_none()
    )
    tgt = (
        ContentIndex.query
        .with_entities(ContentIndex.keywords, ContentIndex.title)
        .filter_by(site_id=site_id, wp_post_id=tgt_post_id)
        .one_or_none()
    )
    if not src or not tgt:
        return False
    sj, tj = src[0] or "", tgt[0] or ""
    ja = jaccard(keywords_set(sj), keywords_set(tj))
    if ja >= GENRE_JACCARD_MIN:
        return True
    # フォールバック：タイトル近似
    ta, tb = src[1] or "", tgt[1] or ""
    cos = title_tfidf_cosine(ta, tb)
    return cos >= TITLE_COSINE_MIN


def plan_links_for_post(
    site_id: int,
    src_post_id: int,
    mode_swap_check: bool = True,
    min_score: float = 0.05,        # ← デフォルトを 0.05 に（link_graph と揃える）
    max_candidates: int = 60        # ← 余裕を持って拾う
) -> PlanStats:
    """
    単一記事に対して本文内リンク **1-8本** の計画を作成し、InternalLinkAction(pending) で保存
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

    html = wp_post.content_html or ""
    # --- ★同記事再ビルド用：今回の計画で使用する link_version を決定
    # ルール：
    #   - 過去の InternalLinkAction(link_version) の最大値 + 1 を基本とする
    #   - ただし仕様由来の整数版(_link_version_int)より小さくはしない
    try:
        prev_max_ver = (
            db.session.query(db.func.max(InternalLinkAction.link_version))
            .filter(InternalLinkAction.site_id == site_id,
                    InternalLinkAction.post_id == src_post_id)
            .scalar()
        ) or 0
    except Exception:
        prev_max_ver = 0
    target_link_version = max(int(prev_max_ver) + 1, int(_link_version_int()))
    # H2 セクション抽出（H2の“末尾＝次H2直前”に挿入するための座標を得る）
    sections = extract_h2_sections(html)
    # H2 が全く無い記事は本文末尾へのフォールバック（applier 側で解釈する position を使用）
    has_h2 = bool(sections)

    # 2) 既存の内部リンク抽出（swap用 & 重複抑制）
    existing_links = _extract_existing_links(wp_post.content_html)
    existing_urls = [u for (u, a) in existing_links]
    url_to_pid = _url_to_post_id_map(site_id, existing_urls)
    existing_post_ids = {pid for pid in url_to_pid.values() if pid}  # 既にリンクしているpost_id
    # 本文に既に存在する内部リンク本数（概算）
    existing_internal_links_count = _count_existing_internal_links(existing_links, url_to_pid)

    # 3) 候補ターゲットの収集（スコア順 → 同ジャンルでフィルタ）
    #    - 既存リンク先は優先度下げ（まずは新顔を入れたい）
    #    - 自身は除外
    raw_candidates = _top_targets_for_source(site_id, src_post_id, topk=max_candidates, min_score=min_score)
    candidates: List[int] = []
    strong_pass = 0
    genre_pass  = 0
    for t, s in raw_candidates:
        if t == src_post_id:
            continue
        if t in existing_post_ids:
            continue
        # ★LinkGraph が強ければ同ジャンル判定を免除（強い関連のみ）
        if s >= STRONG_LINKGRAPH_SCORE:
            candidates.append(t)
            strong_pass += 1
            continue
        # それ以外は同ジャンル判定（keywords Jaccard or タイトル類似）
        try:
            if _same_genre_ok(site_id, src_post_id, t):
                candidates.append(t)
                genre_pass += 1
        except Exception:
            continue
    if not candidates:
        logger.info("[Planner] src=%s no fresh candidates", src_post_id)

    # 4) 本数決定（最低2本・最大4本を保証。H2が少なければ H2 数まで）
    need_min = max(2, int(getattr(cfg, "min_links_per_post", 2) or 2))
    need_max = min(4, int(getattr(cfg, "max_links_per_post", 4) or 4))

    # 既存内部リンク数が少ない場合の底上げ
    target_min = need_min
    if existing_internal_links_count <= 1 and need_min < 3:
        target_min = 3

    # 最終必要本数
    need = max(target_min, need_min)
    need = min(need, need_max)
    if has_h2:
        need = min(need, len(sections))  # H2数を超えない

    # 5) H2インデックスの分散選定（同じ位置に偏らないよう等間隔抽出＋最小ギャップ1）
    chosen_h2_idx: List[int] = []
    if has_h2 and need > 0:
        n = len(sections)
        if need == 1:
            chosen_h2_idx = [min(n - 1, max(0, n // 2))]
        else:
            step = max(1, n // need)
            picked = [min(i * step, n - 1) for i in range(need)]
            # 最小ギャップ=1 を適用
            kept: List[int] = []
            for s in sorted(set(picked)):
                if not kept or (s - kept[-1] > 0):
                    kept.append(s)
            chosen_h2_idx = kept[:need]
    elif not has_h2 and need > 0:
        # H2が無い場合は本文末尾に 1〜2 本（仕様に合わせて最大2）
        need = min(need, 2)
        chosen_h2_idx = [-1] * need  # applier 側で "body_tail" として扱う

    # 6) ターゲット記事のタイトル/URL/キーワードを取得（アンカー生成の補助用）
    tgt_rows = (
        ContentIndex.query
        .with_entities(ContentIndex.wp_post_id, ContentIndex.title, ContentIndex.url, ContentIndex.keywords)
        .filter(ContentIndex.site_id == site_id)
        .filter(ContentIndex.wp_post_id.in_(candidates[: max(0, len(chosen_h2_idx))]))
        .all()
    )
    def _csv_to_list(csv: Optional[str]) -> List[str]:
        return [k.strip() for k in (csv or "").split(",") if k and k.strip()]
    # pid -> (title, url, dst_keywords[])
    tgt_map: Dict[int, Tuple[str, str, List[str]]] = {
        int(pid): (title or "", url or "", _csv_to_list(kws))
        for (pid, title, url, kws) in tgt_rows
    }

    # 7) H2スロット×ターゲットで pending アクションを作成
    #    ※ アンカー文は applier 側で生成／整形。ここでは「位置＝H2末尾」を示す。
    actions_made = 0
    pos_ptr = 0
    seen_target_pids = set()
    for tgt_pid in candidates:
        if pos_ptr >= len(chosen_h2_idx):
            break
        if tgt_pid in seen_target_pids:
            continue
        chosen = chosen_h2_idx[pos_ptr % max(1, len(chosen_h2_idx))]
        title, tgt_url, dst_kw_list = tgt_map.get(tgt_pid, ("", "", []))
        # ContentIndexに無い候補はフォールバックで1回だけ取得
        if (not title) or (not tgt_url):
            tr = (
                ContentIndex.query
                .with_entities(ContentIndex.title, ContentIndex.url, ContentIndex.keywords)
                .filter_by(site_id=site_id, wp_post_id=tgt_pid)
                .one_or_none()
            )
            if tr:
                title, tgt_url = tr[0] or "", tr[1] or ""
                dst_kw_list = _csv_to_list(tr[2] or "")
        if not tgt_url:
            continue
        # pending を登録（position は 'h2:{index}'。H2なしは 'h2:-1' を本文末尾扱い）
        act = InternalLinkAction(
            site_id=site_id,
            post_id=src_post_id,
            target_post_id=tgt_pid,
            anchor_text="",  # アンカー文は applier が LLM 生成
            position=f"h2:{int(chosen)}",
            status="pending",
            reason="plan:generated",
            link_version=int(target_link_version),
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )        
        db.session.add(act)
        actions_made += 1
        pos_ptr += 1
        seen_target_pids.add(tgt_pid)

    if actions_made:
        db.session.commit()
    stats.planned_actions = actions_made
    stats.posts_processed = 1

    # === 5.5) ★救済パス：1本も計画できなかった場合に「厳しめ条件のまま」最低1本を狙う（H2ベース）
    #  - LinkGraph が強い or 同ジャンルOK の候補に限定
    #  - セクション末尾の直前テキスト（=H2の本文末尾付近）に「共通keywordsの単語」または「タイトル語」が自然に実在することを必須に
    if stats.planned_actions == 0:
        rescue_candidates = []
        for t, s in raw_candidates:
            if t == src_post_id:
                continue
            if t in existing_post_ids:
                continue
            # 強いスコア or 同ジャンルOK のみ許容（緩めない）
            ok = (s >= STRONG_LINKGRAPH_SCORE)
            if not ok:
                try:
                    ok = _same_genre_ok(site_id, src_post_id, t)
                except Exception:
                    ok = False
            if ok:
                rescue_candidates.append((t, s))
        # スコア降順でチェック
        rescue_candidates.sort(key=lambda x: x[1], reverse=True)
        if rescue_candidates:
            # 最後のH2セクション（または本文末尾）を優先して使う
            if has_h2 and sections:
                sec_idx = max(0, len(sections) - 1)
                s = sections[sec_idx]
                # H2本文部分（h2_end〜tail_insert_pos）のテキスト
                sec_text = _html_to_text(html[s["h2_end"]:s["tail_insert_pos"]])
                pos_str = f"h2:{sec_idx}"
            else:
                sec_text = _html_to_text(html[-4000:])  # 末尾近辺
                pos_str = "h2:-1"
            if sec_text:
                for tgt_pid, sc in rescue_candidates[:5]:
                    # アンカーは「共通keywordsの単語」＞「タイトル語 ∩ 段落」
                    anc = None
                    try:
                        anc = _pick_anchor_from_common_keywords_single(site_id, src_post_id, tgt_pid, sec_text)
                    except Exception:
                        anc = None
                    if not anc:
                        try:
                            # タイトル語に限定（自然な位置のみ）
                            tr = (
                                ContentIndex.query
                                .with_entities(ContentIndex.title)
                                .filter_by(site_id=site_id, wp_post_id=tgt_pid)
                                .one_or_none()
                            )
                            t_title = tr[0] if tr else ""
                            anc = _pick_anchor_from_title_overlap(t_title, sec_text)
                        except Exception:
                            anc = None
                    if not anc:
                        continue
                    # URL 取得
                    tr2 = (
                        ContentIndex.query
                        .with_entities(ContentIndex.url)
                        .filter_by(site_id=site_id, wp_post_id=tgt_pid)
                        .one_or_none()
                    )
                    t_url = (tr2[0] or "") if tr2 else ""
                    if not t_url:
                        continue
                    db.session.add(InternalLinkAction(
                        site_id=site_id,
                        post_id=src_post_id,
                        target_post_id=tgt_pid,
                        anchor_text="",  # アンカー文は applier が LLM 生成（自然語抽出は上で担保）
                        position=pos_str,
                        status="pending",
                        reason="plan:rescue",
                        link_version=int(target_link_version),
                        created_at=datetime.now(UTC),
                        updated_at=datetime.now(UTC),
                    ))
                    db.session.commit()
                    stats.planned_actions += 1
                    break

    # 8) 定期検診：swap候補の作成（既存リンクより適合度の高い新顔があれば提案）
    if mode_swap_check and existing_post_ids:
        # swap 生成時に記事内重複アンカーを避けるための簡易セット
        seen_anchor_keys = set()
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
                    # 置換候補：先頭のH2セクション末尾に提案（厳密位置は applier で精緻化）
                    if has_h2 and sections:
                        sec_idx = 0
                        sec_text = _html_to_text(html[sections[sec_idx]["h2_end"]:sections[sec_idx]["tail_insert_pos"]])
                        pos_str = f"h2:{sec_idx}"
                    else:
                        sec_text = _html_to_text(html[:4000])
                        pos_str = "h2:-1"
                    t_title, t_url, t_kws = tgt_map.get(pid, ("", "", []))
                    if not t_title or not t_url:
                        # ない場合はContentIndexから再取得
                        tr = (
                            ContentIndex.query
                            .with_entities(ContentIndex.title, ContentIndex.url, ContentIndex.keywords)
                            .filter_by(site_id=site_id, wp_post_id=pid)
                            .one_or_none()
                        )
                        if tr:
                            t_title, t_url = tr[0] or "", tr[1] or ""
                            t_kws = _csv_to_list(tr[2] or "")
                     # swapでも「タイトル語 ∩ セクション本文末尾近傍」を厳守
                    anc = _pick_anchor_from_title_overlap(t_title, sec_text) or _candidate_anchor_from(t_title, sec_text)

                    if not anc or not t_url:
                        continue
                    # swap候補側でも同一記事内アンカーの重複を避ける
                    anc_key = nfkc_norm(anc or "").strip().lower()
                    if anc_key in seen_anchor_keys:
                        continue
                    seen_anchor_keys.add(anc_key)
                    db.session.add(InternalLinkAction(
                        site_id=site_id,
                        post_id=src_post_id,
                        target_post_id=pid,
                        anchor_text=anc[:255],
                        position=pos_str,
                        status="pending",
                        reason="swap_candidate:title_match",
                        link_version=int(target_link_version),
                        created_at=datetime.now(UTC),
                        updated_at=datetime.now(UTC),
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
    min_score: float = 0.05,
    max_candidates: int = 60,
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
        st = plan_links_for_post(
            site_id,
            src_post_id,
            mode_swap_check=mode_swap_check,
            min_score=min_score,
            max_candidates=max_candidates,
        )
        planned += st.planned_actions
        swaps += st.swap_candidates
        processed += st.posts_processed

    logger.info("[Planner] site=%s planned=%s swaps=%s processed=%s", site_id, planned, swaps, processed)
    return {"planned": planned, "swap_candidates": swaps, "processed": processed}
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
    is_ng_anchor,
    STOPWORDS_N,
    title_tokens,
    keywords_set,
    jaccard,
    title_tfidf_cosine,
    is_natural_span,
)

from app.services.internal_seo.applier import _H_TAG, _TOC_HINT, _mask_existing_anchors, _split_paragraphs  # 再利用  # 再利用


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
        if counts[idx] < max_per_para:
            slots.append(idx)
            counts[idx] += 1
        i += 1
        # 念のためセーフガード
        if i > need * max(1, len(eligible)) * max(1, max_per_para):
            break
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

    paragraphs = _split_paragraphs(wp_post.content_html or "")
    if not paragraphs:
        logger.info("[Planner] skip src=%s (no paragraphs)", src_post_id)
        return stats

    # 2) 既存の内部リンク抽出（swap用 & 重複抑制）
    existing_links = _extract_existing_links(wp_post.content_html)
    existing_urls = [u for (u, a) in existing_links]
    url_to_pid = _url_to_post_id_map(site_id, existing_urls)
    existing_post_ids = {pid for pid in url_to_pid.values() if pid}  # 既にリンクしているpost_id

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

    # 4) 本数決定（applier と揃える：最小2 / 最大4。サイト設定があれば尊重）
    need_min = max(2, int(getattr(cfg, "min_links_per_post", 2) or 2))
    need_max = min(4, int(getattr(cfg, "max_links_per_post", 4) or 4))
    need = min(max(need_min, 2), need_max)

    # 5) 段落スロット選定（段階的に緩和 → それでも不足なら同段落 MAX_LINKS_PER_PARA 本まで許容）
    base_min_len = int(getattr(cfg, "min_paragraph_len", 80) or 80)
    # 厳しめ→緩め の順で試行（重複排除・ユニーク優先）
    thresholds = sorted({base_min_len, 70, 60, 50, 40}, reverse=True)
    slots: List[int] = []
    for th in thresholds:
        slots = _pick_paragraph_slots(paragraphs, need=need, min_len=th)
        if len(slots) >= min(need, 2):  # まずは2本分以上確保できたらOK
            break
    if len(slots) < need:
        # 同一段落に複数本まで入れて必要数を満たす（MAX_LINKS_PER_PARA で上限管理）
        # 最後に試した閾値(thresholds[-1]＝最も緩い条件)で再度取得し直してから埋める
        th = thresholds[-1]
        slots = _pick_paragraph_slots(
            paragraphs,
            need=need,
            min_len=th,
            allow_repeat=True,
            max_per_para=max(1, MAX_LINKS_PER_PARA),
        )
    if not slots:
        logger.info("[Planner] src=%s no suitable paragraphs", src_post_id)
        return stats
    # 診断ログ（INTERNAL_SEO_PLANNER_DIAG=1 のときだけ）
    if os.getenv("INTERNAL_SEO_PLANNER_DIAG", "0") == "1":
        logger.info(
            "[PlannerDiag] src=%s paras=%s slots=%s need=%s strong_pass=%s genre_pass=%s raw=%s",
            src_post_id, len(paragraphs), slots, need, strong_pass, genre_pass, len(raw_candidates)
        )

   # 6) ターゲット記事のタイトル/URL/キーワードを取得（アンカー生成の補助用）
    tgt_rows = (
        ContentIndex.query
        .with_entities(ContentIndex.wp_post_id, ContentIndex.title, ContentIndex.url, ContentIndex.keywords)
        .filter(ContentIndex.site_id == site_id)
        .filter(ContentIndex.wp_post_id.in_(candidates[: len(slots)]))
        .all()
    )
    def _csv_to_list(csv: Optional[str]) -> List[str]:
        return [k.strip() for k in (csv or "").split(",") if k and k.strip()]
    # pid -> (title, url, dst_keywords[])
    tgt_map: Dict[int, Tuple[str, str, List[str]]] = {
        int(pid): (title or "", url or "", _csv_to_list(kws))
        for (pid, title, url, kws) in tgt_rows
    }

    # 7) スロット×ターゲットで pending アクションを作成
    #    ※ アンカー文は applier 側で LLM 生成するため、ここでは“メタ情報を丁寧に渡す”
    actions_made = 0
    slot_ptr = 0
    seen_target_pids = set()
    for tgt_pid in candidates:
        if slot_ptr >= len(slots):
            break
        if tgt_pid in seen_target_pids:
            continue
        # slots が不足している場合は循環利用（同一段落に最大2本まで）
        slot_idx = slots[slot_ptr % len(slots)]
        # 1段落2本の上限を守る（同じ index が3回以上使われないようにする）
        if slots.count(slot_idx) >= 2 and (slot_ptr // len(slots)) >= 2:
            slot_ptr += 1
            continue
        raw_para = paragraphs[slot_idx]
        if _H_TAG.search(raw_para) or _TOC_HINT.search(raw_para):
            continue  # 念のため
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
        # pending を登録（position は 'p:{index}'）
        act = InternalLinkAction(
            site_id=site_id,
            post_id=src_post_id,
            target_post_id=tgt_pid,
            anchor_text="",  # アンカー文は applier が LLM 生成
            position=f"p:{slot_idx}",
            status="pending",
            reason="plan:generated",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.session.add(act)
        actions_made += 1
        slot_ptr += 1
        seen_target_pids.add(tgt_pid)

    if actions_made:
        db.session.commit()
    stats.planned_actions = actions_made
    stats.posts_processed = 1

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
                    # 置換候補：最初の長め段落に提案（厳密位置は applier で精緻化）
                    slot_for_swap = slots[0] if slots else 0
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
                    slot_para_raw = paragraphs[slot_for_swap] if 0 <= slot_for_swap < len(paragraphs) else ""
                    slot_para_text = _html_to_text(slot_para_raw)
                    # swap 候補のアンカー決定でも部分一致を利用
                    tgt_body_for_swap = ""
                    try:
                        _t = fetch_single_post(Site.query.get(site_id), pid)
                        if _t and (_t.content_html or ""):
                            tgt_body_for_swap = html_to_text(_t.content_html)
                    except Exception:
                        tgt_body_for_swap = ""
                    # swapでも「タイトル語 ∩ 段落」を厳守
                    anc = _pick_anchor_from_title_overlap(t_title, slot_para_text) or _candidate_anchor_from(t_title, slot_para_text)

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
                        position=f"p:{slot_for_swap}",
                        status="pending",
                        reason="swap_candidate:title_match",
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
# app/services/internal_seo/applier.py
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, UTC
from html import unescape
from typing import Dict, List, Optional, Tuple
import os
import time
import random
import math

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
from app.services.internal_seo.utils import nfkc_norm, is_ng_anchor, title_tokens, extract_h2_sections

logger = logging.getLogger(__name__)

_GEN_SUFFIX = "について詳しい解説はコチラ"

def _is_ng_anchor_generated_line(text: str, tgt_title: str | None = None) -> bool:
    """
    V4の“文スタイル”アンカー用のNG判定ラッパ。
    固定終止句「〜について詳しい解説はコチラ」を外した“コア語”でNGを判定する。
    """
    if not text:
        return True
    core = re.sub(rf"{re.escape(_GEN_SUFFIX)}$", "", text).strip()
    # コアが消えてしまう（=固定句だけ）なら安全テンプレ扱いでNGにしない
    if not core:
        return False
    return is_ng_anchor(core, tgt_title)


# ---- HTMLユーティリティ ----

# ====== 新方式：ファイル内完結の設定・モデル・プロンプト ======
import os as _os
from typing import Any
try:
    from openai import OpenAI as _OpenAI, BadRequestError as _BadRequestError
    _OPENAI_CLIENT = _OpenAI(api_key=_os.getenv("OPENAI_API_KEY", ""))
except Exception:
    _OPENAI_CLIENT = None
    _BadRequestError = Exception

# ▼ 新旧の切替（ここを "legacy_phrase" にすれば従来動作）
ANCHOR_MODE: str = "generated_line"

# ▼ アンカー生成スタイル（llm / template）
#   - llm: ChatGPTで「KWを自然に含む定型文」を生成（推奨・既定）
#   - template: LLMを使わず {KW}について詳しい解説はコチラ の固定文
ANCHOR_STYLE: str = _os.getenv("INTERNAL_SEO_ANCHOR_STYLE", "llm").strip().lower() or "llm"

# ▼ LLM 呼び出し設定（記事生成と同等の流儀を最小限踏襲）
ISEO_ANCHOR_MODEL: str = _os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ISEO_ANCHOR_TEMPERATURE: float = 0.30
ISEO_ANCHOR_TOP_P: float = 0.9
ISEO_CTX_LIMIT: int = 4000
ISEO_SHRINK: float = 0.85
ISEO_MAX_TOKENS: int = 120           # 短文用にやや抑制
ISEO_ANCHOR_MAX_CHARS: int = 58      # 上限を少しタイトに

ANCHOR_SYSTEM_PROMPT = (
    "あなたはSEOに配慮する日本語編集者です。"
    "内部リンクのアンカー文（1行）を作成します。必ず以下を厳守：\n"
    "・日本語で1文のみ（改行/引用符/絵文字/装飾なし）\n"
    "・リンク先の主要キーワードを1〜2語だけ含める（タイトル丸写し禁止）\n"
    "・文末は必ず「について詳しい解説はコチラ」で終える（句点や読点を付けない）\n"
    "・煽り語や冗長表現は禁止（例：ぜひ／チェックしてみてください／参考にしてください など）\n"
    "・全体で40〜58字に収める\n"
)
ANCHOR_USER_PROMPT_TEMPLATE = (
    "【リンク先タイトル】{dst_title}\n"
    "【主要キーワード（重要度順）】{dst_keywords}\n"
    "【段落の要旨（文脈ヒント）】{src_hint}\n"
    "要件: 上記の仕様どおり、1文・40〜60字で、"
    "【リンク先タイトル】の名詞系キーワードを必ず1語以上含め、「〜について詳しい解説はコチラ」で結ぶアンカー文のみ出力。"
    "動詞や連体修飾だけで主語が無い文（例: 向上させるため…）は不可。"
)

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
_STYLE_BLOCK = re.compile(r"<style\b[^>]*>.*?</style\s*>", re.IGNORECASE | re.DOTALL)
_AI_STYLE_MARK = "<!-- ai-internal-link-style:v2 -->"

# ==== 内部SEO 仕様バージョン（新規） ====
# <a> には一切属性を付けない方針。代替として直前コメントで版管理を行う。
INTERNAL_SEO_SPEC_VERSION = "v4"
INTERNAL_SEO_SPEC_MARK = f"<!-- ai-internal-link:{INTERNAL_SEO_SPEC_VERSION} -->"
ILINK_BOX_MARK = "<!-- ai-internal-link-box:v1 -->"

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

# ---- LLMユーティリティ（applier内だけで完結）----
def _clean_gpt_output(text: str) -> str:
    text = re.sub(r"```(?:html)?", "", text or "")
    text = re.sub(r"```", "", text)
    text = text.replace("\u3000", " ")
    text = text.strip()
    # 改行は1行に潰す
    text = re.sub(r"\s*\n+\s*", " ", text)
    # 先頭末尾の引用符・鉤括弧系は剥がす
    text = re.sub(r'^[\'"「『（\(\[]\s*', "", text)
    text = re.sub(r'\s*[\'"」』）\)\]]$', "", text)
    # 禁止・冗長フレーズの簡易除去（順序大事：長いもの→短いもの）
    STOP_PHRASES = [
        "ぜひチェックしてみてください",
        "チェックしてみてください",
        "参考にしてください",
        "ぜひご覧ください",
        "ぜひ参考に",
        "ぜひチェック",
        "ぜひとも",
        "ぜひ",
        # 主語呼びかけ系（後続の「は」ごと除去）
        "気になる方は",
        "知りたい方は",
        "詳しく知りたい方は",
    ]
    for s in STOP_PHRASES:
        text = text.replace(s, "")
    # よく出る文法崩れの補正
    text = re.sub(r"\s+", " ", text)              # 連続スペース
    text = re.sub(r"(は|が)\s*について", "について", text)  # 「〜は/が について」→「について」
    text = re.sub(r"に\s*ついて", "について", text)        # 全半角ゆらぎ
    text = re.sub(r"はどうなっているのか", "の概要", text)   # 冗長疑問形の簡約
    # 多重スペース整理
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def _iseo_tok(s: str) -> int:
    return int(len(s or "") / 1.8)

def _iseo_chat(msgs: List[Dict[str, str]], max_t: int, temp: float, user_id: Optional[int] = None) -> str:
    """
    内部SEO用の軽ラッパ。記事生成の挙動を簡略化して踏襲。
    TokenUsageLog 記録は user_id が無ければスキップ。
    """
    if _OPENAI_CLIENT is None:
        raise RuntimeError("OpenAI client is not available")
    used = sum(_iseo_tok(m.get("content", "")) for m in msgs)
    available = ISEO_CTX_LIMIT - used - 16
    max_t = max(1, min(max_t, available))

    def _call(m: int) -> str:
        res = _OPENAI_CLIENT.chat.completions.create(
            model=ISEO_ANCHOR_MODEL,
            messages=msgs,
            max_tokens=m,
            temperature=temp,
            top_p=ISEO_ANCHOR_TOP_P,
            timeout=60,
        )
        # TokenUsageLog（任意）
        try:
            if hasattr(res, "usage") and user_id:
                from app.models import TokenUsageLog
                usage = res.usage
                log = TokenUsageLog(
                    user_id=user_id,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0),
                    completion_tokens=getattr(usage, "completion_tokens", 0),
                    total_tokens=getattr(usage, "total_tokens", 0),
                )
                db.session.add(log)
                db.session.commit()
        except Exception as _e:
            logger.warning(f"[ISEO TokenLog warn] { _e }")
        content = (res.choices[0].message.content or "").strip()
        return _clean_gpt_output(content)

    try:
        return _call(max_t)
    except _BadRequestError as e:
        if "max_tokens" in str(e):
            retry_t = max(1, int(max_t * ISEO_SHRINK))
            return _call(retry_t)
        raise

def _generate_anchor_text_via_llm(
    dst_title: str,
    dst_keywords: List[str] | Tuple[str, ...] | None,
    src_hint: str = "",
    user_id: Optional[int] = None,
) -> str:
    """
    LLMで「KWを1〜2語含む・40〜60字・…について詳しい解説はコチラ」文を生成。
    最低限のバリデーションを行い、要件を満たさなければテンプレで補正。
    """
    kw_csv = ", ".join([k for k in (dst_keywords or []) if k]) if isinstance(dst_keywords, (list, tuple)) else (dst_keywords or "")
    sys = ANCHOR_SYSTEM_PROMPT
    usr = ANCHOR_USER_PROMPT_TEMPLATE.format(
        dst_title=(dst_title or "")[:200],
        dst_keywords=(kw_csv or "")[:200],
        src_hint=(src_hint or "")[:200],
    )
    out = _iseo_chat(
        [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
        ISEO_MAX_TOKENS,
        ISEO_ANCHOR_TEMPERATURE,
        user_id=user_id,
    )
    # 最終正規化（1行・装飾なし）
    out = _clean_gpt_output(out)
    # --- 追加バリデーション（厳格版） ---
    text = out
    # 語尾を強制整形：末尾句点・読点・空白を除去
    text = re.sub(r"[、。．.\s]+$", "", text)
    # 規定の終止句で終わらせる（重複防止のため一旦削る）
    text = re.sub(r"(について詳しい解説はコチラ)$", r"\1", text)
    if not text.endswith("について詳しい解説はコチラ"):
        # 末尾が別表現なら置換
        text = re.sub(r"(について.*)$", "について詳しい解説はコチラ", text)
        if not text.endswith("について詳しい解説はコチラ"):
            # どうしても整わない場合はテンプレで作り直し
            first_kw = ""
            if isinstance(dst_keywords, (list, tuple)) and dst_keywords:
                first_kw = str(dst_keywords[0]).strip()
            if not first_kw:
                try:
                    toks = title_tokens(dst_title or "") or []
                    first_kw = toks[0] if toks else ""
                except Exception:
                    first_kw = ""
            base = (first_kw or (dst_title or "")[:20]).strip()
            text = f"{base}について詳しい解説はコチラ"
    # 文字数を最終調整（超過は丸め、末尾の読点類は除去）
    if len(text) > ISEO_ANCHOR_MAX_CHARS:
        text = text[:ISEO_ANCHOR_MAX_CHARS]
        text = re.sub(r"[、。．.\s]+$", "", text)
        # 超過丸めで終止句が欠けた場合はテンプレで復元
        if not text.endswith("について詳しい解説はコチラ"):
            first_kw = ""
            if isinstance(dst_keywords, (list, tuple)) and dst_keywords:
                first_kw = str(dst_keywords[0]).strip()
            if not first_kw:
                try:
                    toks = title_tokens(dst_title or "") or []
                    first_kw = toks[0] if toks else ""
                except Exception:
                    first_kw = ""
            text = f"{first_kw or (dst_title or '')[:20]}について詳しい解説はコチラ"
    # --- ここから 追加の厳格バリデーション ---
    text = _clean_gpt_output(text)
    text = re.sub(r"[、。．.\s]+$", "", text)

    # 主要キーワード候補（dst_keywords優先→無ければタイトル主要トークン）
    kw_candidates: List[str] = []
    if isinstance(dst_keywords, (list, tuple)):
        kw_candidates = [str(k).strip() for k in dst_keywords if str(k).strip()]
    if not kw_candidates:
        try:
            kw_candidates = [w for w in (title_tokens(dst_title or "") or []) if w][:6]
        except Exception:
            kw_candidates = []

    def _norm(s: str) -> str:
        return nfkc_norm((s or "").strip()).lower()

    ntext = _norm(text)
    nkeys = [_norm(k) for k in kw_candidates if _norm(k)]

    # 1) 出力に主要キーワードが1語も含まれない → テンプレにフォールバック
    has_kw = any((nk in ntext) for nk in nkeys) if nkeys else False

    # 2) 冒頭の品質: 助詞・連用形だけ等の弱い始まりを弾く（よく出るNGの簡易検知）
    BAD_START_PAT = re.compile(r"^(?:について|により|に関して|における|によって|に対して|向上させるため|成功させるため|選ぶため|知っておくべきこと|活用法|方法|ポイント)")
    bad_start = bool(BAD_START_PAT.search(nfkc_norm(text)))

    if (not has_kw) or bad_start:
        first_kw = ""
        if nkeys:
            # なるべく長い語を優先
            nkeys_sorted = sorted(nkeys, key=len, reverse=True)
            first_kw = nkeys_sorted[0]
        if not first_kw:
            try:
                tks = [w for w in (title_tokens(dst_title or "") or []) if w]
                first_kw = _norm(tks[0]) if tks else ""
            except Exception:
                first_kw = ""
        base = next((k for k in (kw_candidates or []) if _norm(k) == first_kw), "") or (dst_title or "")[:20]
        text = f"{base}について詳しい解説はコチラ"
        text = re.sub(r"[、。．.\s]+$", "", text)

    return text

def _postprocess_anchor_text(text: str) -> str:
    """
    生成後の日本語を軽く整形:
      - 「得られますについて」→「について」
      - 「留学in〇〇」→「〇〇留学」
      - 末尾の句読点・空白除去
    """
    s = _clean_gpt_output(text)
    s = re.sub(r"得られますについて", "について", s)
    s = re.sub(r"得られる?について", "について", s)
    # 留学inスウェーデン → スウェーデン留学（一般化: 留学inXXXX → XXXX留学）
    s = re.sub(r"留学in([一-龥ぁ-んァ-ンA-Za-z0-9ー]+)", r"\1留学", s)
    s = re.sub(r"[、。．.\s]+$", "", s)
    # 終止句の統一（壊れていたら付け直す）
    if not s.endswith("について詳しい解説はコチラ"):
        s = re.sub(r"(について.*)$", "について詳しい解説はコチラ", s)
        if not s.endswith("について詳しい解説はコチラ"):
            s = s + "について詳しい解説はコチラ"
    return s

def _safe_anchor_from_keywords(dst_kw_list: List[str], dst_title: str) -> str:
    """
    NG時のセーフフォールバック用に、主要KWから安全テンプレを生成。
    ・KWが無い場合はタイトル主要語
    ・「留学in◯◯」→「◯◯留学」に正規化
    """
    base = ""
    pool = [k for k in (dst_kw_list or []) if k]
    if not pool:
        try:
            pool = [w for w in (title_tokens(dst_title or "") or []) if w]
        except Exception:
            pool = []
    if pool:
        # より一般的で長めの語を優先
        pool = sorted(set(pool), key=len, reverse=True)
        base = pool[0]
    base = re.sub(r"留学in([一-龥ぁ-んァ-ンA-Za-z0-9ー]+)", r"\\1留学", base or "")
    base = (base or (dst_title or "")[:20]).strip()
    return f"{base}について詳しい解説はコチラ"

def _is_anchor_quality_ok(text: str, dst_keywords: List[str], dst_title: str) -> bool:
    """最低品質チェック：名詞系KWを1語以上含む／文頭が述語だけにならない／長さ"""
    if not text:
        return False
    # 末尾は既に「について詳しい解説はコチラ」で正規化済み前提
    body = re.sub(r"について詳しい解説はコチラ$", "", text).strip()
    if not (24 <= len(text) <= ISEO_ANCHOR_MAX_CHARS):
        return False
    # 先頭が助詞・補助動詞・動詞語幹っぽい始まりはNG（簡易）
    if re.match(r"^(について|により|に向け|のため|ために|向上させるため|改善するため|選ぶため|知るため)", body):
        return False
    # キーワードを必ず1語以上含む（title_tokensからも補完）
    kw_pool = set(k for k in (dst_keywords or []) if k)
    try:
        kw_pool.update(title_tokens(dst_title or "") or [])
    except Exception:
        pass
    # 2文字以上のキーワードに限定して含有判定
    kw_pool = {k for k in kw_pool if len(k) >= 2}
    if not kw_pool:
        return True  # どうしても無い場合は通す（上位でテンプレに落とすため緩め）
    normalized = body
    hit = any(k in normalized for k in kw_pool)
    return bool(hit)

def _emit_anchor_html(href: str, text: str) -> str:
    text_safe = _TAG_STRIP.sub(" ", unescape(text or "")).strip()
    # 構造は維持（<a href ... title ...>）。版情報は直前コメントで表現。
    return f'{INTERNAL_SEO_SPEC_MARK}<a href="{href}" title="{text_safe}">{text_safe}</a>'

def _emit_recommend_box() -> str:
    """
    「この記事を読んでる方がよく読んでる記事」を黒地・白文字の囲みで出力。
    ※リンクは入れない（直下に別途 <a> を置く）。
    """
    return (
        f'{ILINK_BOX_MARK}'
        '<div class="ai-relbox" '
        'style="margin:1.2em 0 0.4em; padding:10px 12px; border-radius:8px; '
        'background:#111; color:#fff; font-weight:600;">'
        'この記事を読んでる方がよく読んでる記事'
        '</div>'
    )


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

def _extract_anchor_text_set(html: str) -> set[str]:
    """記事中に既に存在する <a>…</a> のテキスト（正規化済み）集合"""
    out = set()
    for _, atext in _extract_links(html or ""):
        key = nfkc_norm((atext or "").strip()).lower()
        if key:
            out.add(key)
    return out

def _collect_internal_hrefs(site: Site, html: str) -> set[str]:
    """記事全体の内部リンク href セット（記事単位の重複抑止に使用）"""
    site_prefix = (site.url or "").rstrip("/")
    hrefs = set()
    for h, _ in _extract_links(html or ""):
        if h and site_prefix and h.startswith(site_prefix):
            hrefs.add(h)
    return hrefs

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

def _linkify_first_occurrence(
    para_html: str,
    anchor_text: str,
    href: str,
    tgt_title: Optional[str] = None,
) -> Optional[str]:
    """
    段落内の**未リンク領域**にある anchor_text の最初の出現を
    Wikipedia風の <a href="..." class="ai-ilink" title="...">anchor_text</a>
    に置換する。見つからなければ None。
    見出し/TOC を含むブロックでは実行しない。
    """
    if not (para_html and anchor_text and href):
        return None
    # 見出し・TOC・STYLE ブロックは一律除外（本文以外は触らない）
    if _H_TAG.search(para_html) or _TOC_HINT.search(para_html) or _STYLE_BLOCK.search(para_html):
        return None
    # NGアンカーは即中止（タイトル関連性も考慮）
    if is_ng_anchor(anchor_text, tgt_title):
        return None
    masked, ph = _mask_existing_anchors(para_html)
    # 生テキストで最初の一致を探す（HTMLタグは残るが <a> はマスク済み）
    idx = masked.find(anchor_text)
    if idx == -1:
        return None
    # 簡易“語の境界”チェック：
    # - 既存<a>はマスク済みなので、素のテキスト連結のみ判定
    # - 前後が「語文字」でも、日本語の助詞なら“柔らかい境界”として許容する
    def _is_word_char(ch: str) -> bool:
        return bool(re.match(r"[A-Za-z0-9一-龥ぁ-んァ-ンー]", ch))
    SOFT_BOUNDARIES = set(list("でをにがはともへやの"))  # 日本語の主要助詞
    before = masked[idx - 1] if idx > 0 else ""
    after  = masked[idx + len(anchor_text)] if (idx + len(anchor_text)) < len(masked) else ""
    # “両側が語文字かつ助詞でもない”ときだけ不自然として拒否
    if (before and _is_word_char(before) and before not in SOFT_BOUNDARIES) and \
       (after  and _is_word_char(after)  and after  not in SOFT_BOUNDARIES):
        return None
    # Wikipedia風：href + title（class/style なし）。版情報は直前コメントで表現。
    linked = f'{INTERNAL_SEO_SPEC_MARK}<a href="{href}" title="{anchor_text}">{anchor_text}</a>'
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
    テーマCSSを触らず、リンク要素にも style を書かずに下線と色を効かせるため、
    記事先頭に 1度だけ最小の <style> を挿入する。
      対象: サイト内URLへ向く a 要素（Wikipedia と同じく「内部リンクは下線＋青」）
    """
    if not html:
        return html
    
    # ★追加：環境変数で注入を無効化（WPが<style>を剥がす場合に有効）
    import os
    if os.getenv("INTERNAL_SEO_EMBED_STYLE", "1") == "0":
        return html
    # 旧/新マーカー付きの既存ブロックを一旦取り除いてから v2 を入れる
    html = re.sub(
        r'<!-- ai-internal-link-style:v[0-9]+ -->\s*<style>.*?</style>',
        '',
        html,
        flags=re.IGNORECASE | re.DOTALL
    )
    site_url = site.url.rstrip("/")

    css = f'''{_AI_STYLE_MARK}<style>
/* 本文に限定：内部リンクは下線＋青(#0645ad)（テーマに勝てるよう !important） */
:where(.ai-content,.entry-content,.post-content,article,.content) a[href^="{site_url}"] {{
  text-decoration: underline !important;
  color: #0645ad !important;
}}
/* 見出しは除外（下線/色とも継承で上書き） */
:where(.ai-content,.entry-content,.post-content,article,.content) h1 a[href^="{site_url}"],
:where(.ai-content,.entry-content,.post-content,article,.content) h2 a[href^="{site_url}"],
:where(.ai-content,.entry-content,.post-content,article,.content) h3 a[href^="{site_url}"],
:where(.ai-content,.entry-content,.post-content,article,.content) h4 a[href^="{site_url}"],
:where(.ai-content,.entry-content,.post-content,article,.content) h5 a[href^="{site_url}"],
:where(.ai-content,.entry-content,.post-content,article,.content) h6 a[href^="{site_url}"] {{
  text-decoration: none !important;
  color: inherit !important;
}}
/* 代表的な TOC を除外（ez-toc / #toc / .toctitle） */
.toctitle a, .toc a, #toc a, .ez-toc a {{
  text-decoration: none !important;
  color: inherit !important;
}}
</style>'''
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
            # data-iseo は使用しない（属性は追加しない）。コメントマークは別で扱う。
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

def _action_targets_meta(site_id: int, actions: List[InternalLinkAction]) -> Dict[int, Tuple[str, str]]:
    need_ids = list({a.target_post_id for a in actions})
    if not need_ids:
        return {}
    rows = (
        ContentIndex.query
        .with_entities(ContentIndex.wp_post_id, ContentIndex.url, ContentIndex.title)
        .filter(ContentIndex.site_id == site_id)
        .filter(ContentIndex.wp_post_id.in_(need_ids))
        .all()
    )
    # return: {pid: (url, title)}
    return {int(pid): ((url or ""), (title or "")) for (pid, url, title) in rows}

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

    site = db.session.get(Site, site_id)
    wp_post = fetch_single_post(site, src_post_id)
    if not wp_post:
        return "", ApplyResult(message="fetch-failed-or-excluded"), []
    
    # 1) 旧仕様削除（プレビュー：DBは触らない）— 新シグネチャ優先、未対応環境はフォールバック
    url_title_map = _all_url_to_title_map(site_id)
    try:
        cleaned_html, deletions = find_and_remove_legacy_links(
            wp_post.content_html or "", url_title_map, spec_version=INTERNAL_SEO_SPEC_VERSION
        )
    except TypeError:
        # cleaner 未更新環境でも動くよう後方互換
        cleaned_html, deletions = find_and_remove_legacy_links(wp_post.content_html or "", url_title_map)

    # 2) 新仕様の pending を取得
    actions = (
        InternalLinkAction.query
        .filter_by(site_id=site_id, post_id=src_post_id, status="pending")
        .order_by(InternalLinkAction.created_at.asc())
        .all()
    )

    meta_map = _action_targets_meta(site_id, actions)

    original_paras = _split_paragraphs(wp_post.content_html or "")
    # 3) 旧仕様を除去した本文をベースに新仕様を仮適用
    base_html = cleaned_html if cleaned_html is not None else (wp_post.content_html or "")
    # 3.5) 見出し内リンクをサニタイズ（Hタグからはリンクを完全排除）
    base_html = _strip_links_in_headings(base_html)
    new_html, res = _apply_plan_to_html(site, src_post_id, base_html, actions, cfg, meta_map)
    new_html = _strip_links_in_headings(new_html)  # ← 適用後も再サニタイズ
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
            # meta は一度だけ取得
            meta_obj = getattr(a, "meta", None)
            meta_dict = meta_obj if isinstance(meta_obj, dict) else {}
            _dst_url_from_meta = (meta_dict.get("dst_url") or "").strip()
            previews.append(PreviewItem(
                position=a.position or "",
                anchor_text=a.anchor_text or "",
                target_post_id=int(a.target_post_id),
                target_url=_dst_url_from_meta or (meta_map.get(a.target_post_id, ("",""))[0]) or "",
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
    target_meta_map: Dict[int, Tuple[str, str]],  # pid -> (url, title)
) -> Tuple[str, ApplyResult]:
    """
    計画（plan/swap_candidate）のうち、本文内の挿入と置換を行う。
    - 本文内の内部リンク総数が min~max に収まるよう調整
    """
    res = ApplyResult()
    if not html:
        res.message = "empty-html"
        return html, res

    # 以降、H2末尾に直接挿入するため、段落分割は後段（swap等）で都度再計算する
    paragraphs = _split_paragraphs(html) or [html]

    site_url = site.url.rstrip("/")
    # 作業用の本文
    html_work = html
    # 既存内部リンクの個数
    existing_internal = _existing_internal_links_count(site, html_work)
    # 記事全体で既に使われている内部href（重複抑止用）
    article_href_set = _collect_internal_hrefs(site, html_work)
    # 記事全体で既に使われているアンカーテキスト（重複抑止用）
    existing_anchor_text_set = _extract_anchor_text_set(html_work)
    # 同一ターゲットPIDは記事内で1回まで
    used_target_pids: set[int] = set()

    # 既定を 2〜4 に変更（サイト設定があればそれを優先）
    need_min = max(2, int(cfg.min_links_per_post or 2))
    need_max = min(4, int(cfg.max_links_per_post or 4))

    # 1) まずは reason='plan' を優先して挿入
    def _reason_prefix(a):
        return (a.reason or "").split(":", 1)[0]
    plan_actions = [a for a in actions if _reason_prefix(a) in ("plan", "review_approved")]
    swaps = [a for a in actions if _reason_prefix(a) == "swap_candidate"]

    inserted = 0
    # 記事内で同一キーワード（アンカーテキスト）は1回まで
    seen_anchor_keys = set()
    # --- 新仕様: H2末尾に挿入するアクションを分離 ---
    h2_actions = [a for a in plan_actions if (a.position or "").startswith("h2:")]
    p_actions  = [a for a in plan_actions if (a.position or "").startswith("p:")]

    # 1-A) H2末尾への挿入（同じ記事内での位置ずれを避けるため “末尾座標の降順” で処理）
    if h2_actions:
        # 現在の本文から H2 セクション座標を取得
        sections = extract_h2_sections(html_work)
        # (tail_pos, act, h2_idx) を作る（-1 は本文末尾）
        h2_plan: List[Tuple[int, InternalLinkAction, int]] = []
        for act in h2_actions:
            try:
                h2_idx = int((act.position or "h2:-1").split(":")[1])
            except Exception:
                h2_idx = -1
            if h2_idx >= 0 and sections and 0 <= h2_idx < len(sections):
                tail_pos = int(sections[h2_idx]["tail_insert_pos"])
            else:
                tail_pos = len(html_work)
                h2_idx = -1
            h2_plan.append((tail_pos, act, h2_idx))
        # 末尾から処理（挿入による以後位置のシフトを回避）
        h2_plan.sort(key=lambda x: x[0], reverse=True)

        for tail_pos, act, h2_idx in h2_plan:
            if existing_internal + inserted >= need_max:
                break
            # --- meta優先でリンク先情報を取得 ---
            meta_obj = getattr(act, "meta", None)
            _meta: dict = meta_obj if isinstance(meta_obj, dict) else {}
            href0, tgt_title0 = target_meta_map.get(act.target_post_id, ("", ""))
            href_meta  = (_meta.get("dst_url") or "").strip()
            title_meta = (_meta.get("dst_title") or "").strip()
            kw_meta = _meta.get("dst_keywords")
            if isinstance(kw_meta, str):
                kw_meta_list = [k.strip() for k in kw_meta.split(",") if k.strip()]
            elif isinstance(kw_meta, (list, tuple)):
                kw_meta_list = [str(k).strip() for k in kw_meta if str(k).strip()]
            else:
                kw_meta_list = []
            href = href_meta or href0
            tgt_title = title_meta or tgt_title0
            if not href:
                act.status = "skipped"
                act.updated_at = datetime.now(UTC)
                res.skipped += 1
                continue
            # 自己リンク禁止
            try:
                if int(act.target_post_id) == int(src_post_id):
                    act.status = "skipped"
                    act.reason = "self-link"
                    act.updated_at = datetime.now(UTC)
                    res.skipped += 1
                    continue
            except Exception:
                pass
            # 同一URL重複・同一PID重複
            if href in article_href_set:
                act.status = "skipped"; act.reason = "duplicate-href-in-article"
                act.updated_at = datetime.now(UTC); res.skipped += 1; continue
            try:
                if int(act.target_post_id) in used_target_pids:
                    act.status = "skipped"; act.reason = "duplicate-target-in-article"
                    act.updated_at = datetime.now(UTC); res.skipped += 1; continue
            except Exception:
                pass

            # セクション本文末尾近傍のテキストをヒントに LLM 生成
            if h2_idx >= 0 and sections and 0 <= h2_idx < len(sections):
                s = sections[h2_idx]
                ctx = _html_to_text(html_work[s["h2_end"]:s["tail_insert_pos"]])[:120]
            else:
                ctx = _html_to_text(html_work[-4000:])[:120]
            try:
                dst_kw_list = kw_meta_list or [w for w in (title_tokens(tgt_title or "") or []) if w][:6]
            except Exception:
                dst_kw_list = kw_meta_list or []
            try:
                if ANCHOR_STYLE == "template":
                    base = (dst_kw_list[0] if dst_kw_list else (tgt_title or "")[:20]).strip()
                    anchor_text = f"{base}について詳しい解説はコチラ"
                else:
                    anchor_text = _generate_anchor_text_via_llm(
                        dst_title=tgt_title or "", dst_keywords=dst_kw_list, src_hint=ctx, user_id=None
                    )
                anchor_text = _postprocess_anchor_text(anchor_text)
            except Exception as e:
                logger.warning(f"[GEN-ANCHOR:H2] LLM failed: {e}")
                key = (tgt_title or "").strip()
                anchor_text = (f"{key}について詳しい解説はコチラ")[:58] if key else "内部リンクについて詳しい解説はコチラ"

            # 最低品質/NGチェックとフォールバック
            if _is_ng_anchor_generated_line(anchor_text, tgt_title):
                fb = _postprocess_anchor_text(_safe_anchor_from_keywords(dst_kw_list, tgt_title or ""))
                if len(fb) > ISEO_ANCHOR_MAX_CHARS:
                    fb = re.sub(r"[、。．.\s]+$", "", fb[:ISEO_ANCHOR_MAX_CHARS])
                    if not fb.endswith("について詳しい解説はコチラ"):
                        fb = _safe_anchor_from_keywords(dst_kw_list, tgt_title or "")
                        fb = _postprocess_anchor_text(fb)
                if _is_ng_anchor_generated_line(fb, tgt_title):
                    act.status = "skipped"; act.reason = "ng-anchor"
                    act.updated_at = datetime.now(UTC); res.skipped += 1; continue
                anchor_text = fb

            # アンカーテキスト重複抑止
            anchor_key = nfkc_norm(anchor_text).lower()
            if anchor_key and (anchor_key in seen_anchor_keys or anchor_key in existing_anchor_text_set):
                act.status = "skipped"; act.reason = "duplicate-anchor-in-article"
                act.updated_at = datetime.now(UTC); res.skipped += 1; continue

            # 生成HTML（囲みボックス → 改行 → <a>）
            box_html = _emit_recommend_box()
            a_html   = _emit_anchor_html(href, anchor_text)
            insert_html = box_html + "<br>" + a_html

            # 実挿入
            html_work = html_work[:tail_pos] + insert_html + html_work[tail_pos:]
            # 追跡セット更新
            article_href_set.add(href)
            if anchor_key:
                seen_anchor_keys.add(anchor_key)
                existing_anchor_text_set.add(anchor_key)
            try:
                used_target_pids.add(int(act.target_post_id))
            except Exception:
                pass
            # 状態更新
            act.anchor_text = anchor_text
            act.status = "applied"
            act.applied_at = datetime.now(UTC)
            res.applied += 1
            inserted += 1

            # 次のH2位置計算がずれないよう、必要なら再抽出
            sections = extract_h2_sections(html_work)

    # 1-B) 旧互換：p:{idx} 指定が残っている場合は従来どおり段落末に <br><a> で追加
    paragraphs = _split_paragraphs(html_work) or [html_work]
    for act in p_actions:
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
        # --- meta優先でリンク先情報を取得（fallbackはContentIndex由来のtarget_meta_map） ---
        # meta は一度だけ取得
        meta_obj = getattr(act, "meta", None)
        _meta: dict = meta_obj if isinstance(meta_obj, dict) else {}
        href0, tgt_title0 = target_meta_map.get(act.target_post_id, ("", ""))
        href_meta = (_meta.get("dst_url") or "").strip()
        title_meta = (_meta.get("dst_title") or "").strip()
        # dst_keywords は list/tuple/str いずれにも対応（最終的にlist化）
        kw_meta = _meta.get("dst_keywords")
        if isinstance(kw_meta, str):
            kw_meta_list = [k.strip() for k in kw_meta.split(",") if k.strip()]
        elif isinstance(kw_meta, (list, tuple)):
            kw_meta_list = [str(k).strip() for k in kw_meta if str(k).strip()]
        else:
            kw_meta_list = []
        href = href_meta or href0
        tgt_title = title_meta or tgt_title0
        if not href:
            res.skipped += 1
            continue

        # --- 自己リンク禁止（src_post_id == target_post_id の場合） ---
        try:
            if int(act.target_post_id) == int(src_post_id):
                act.status = "skipped"
                act.reason = "self-link"
                act.updated_at = datetime.now(UTC)
                res.skipped += 1
                continue
        except Exception:
            pass

        # （緩和）planner 由来の anchor_text は使わず、generated_line では LLM で毎回生成する

        # --- 同一URLへの重複アンカー禁止（既に同じhrefを別アンカーで使用している場合） ---
        existing_links = [h for (h, atext) in _extract_links(_rejoin_paragraphs(paragraphs))]
        if href in existing_links:
            # ただし同一アンカーテキストなら既存を置換対象に回すのでOK
            if act.anchor_text and not any(atext == act.anchor_text and h == href for (h, atext) in _extract_links(_rejoin_paragraphs(paragraphs))):
                act.status = "skipped"
                act.reason = "duplicate-href-anchor"
                act.updated_at = datetime.now(UTC)
                res.skipped += 1
                continue
        # ---- 新方式 / 旧方式の分岐 ----
        if ANCHOR_MODE == "generated_line":
            # 段落本文（安全ガード/禁止領域チェック）
            para_html = paragraphs[idx]
            if _H_TAG.search(para_html) or _TOC_HINT.search(para_html):
                res.skipped += 1
                continue
            # 記事全体で同一hrefが既に存在したらスキップ（段落をまたいだ重複防止）
            if href in article_href_set:
                act.status = "skipped"
                act.reason = "duplicate-href-in-article"
                act.updated_at = datetime.now(UTC)
                res.skipped += 1
                continue
            # 同一target_post_idは記事内で1回まで
            try:
                if int(act.target_post_id) in used_target_pids:
                    act.status = "skipped"
                    act.reason = "duplicate-target-in-article"
                    act.updated_at = datetime.now(UTC)
                    res.skipped += 1
                    continue
            except Exception:
                pass
            # 同一段落に同じhrefが既にあるなら多重リンク回避
            if href in [h for (h, _) in _extract_links(para_html)]:
                res.skipped += 1
                continue

            # --- 惹句テキスト：LLM 生成（緩和版：プロンプト準拠・最低限の整形のみ） ---
            try:
                src_hint = _html_to_text(para_html)[:120]
                dst_kw_list = kw_meta_list or []
                if not dst_kw_list:
                    try:
                        dst_kw_list = [w for w in (title_tokens(tgt_title or "") or []) if w][:6]
                    except Exception:
                        dst_kw_list = []
                if ANCHOR_STYLE == "template":
                    base = (dst_kw_list[0] if dst_kw_list else (tgt_title or "")[:20]).strip()
                    anchor_text = f"{base}について詳しい解説はコチラ"
                else:
                    anchor_text = _generate_anchor_text_via_llm(
                        dst_title=tgt_title or "",
                        dst_keywords=dst_kw_list,
                        src_hint=src_hint,
                        user_id=None,
                    )
                # 事後整形（日本語の不自然さを軽減）
                anchor_text = _postprocess_anchor_text(anchor_text)
                if len(anchor_text) > ISEO_ANCHOR_MAX_CHARS:
                    anchor_text = anchor_text[:ISEO_ANCHOR_MAX_CHARS]
                    anchor_text = re.sub(r"[、。．.\s]+$", "", anchor_text)
                    if not anchor_text.endswith("について詳しい解説はコチラ"):
                        base = (dst_kw_list[0] if dst_kw_list else (tgt_title or "")[:20]).strip()
                        anchor_text = f"{base}について詳しい解説はコチラ"
            except Exception as e:
                logger.warning(f"[GEN-ANCHOR] LLM failed: {e}")
                key = (tgt_title or "").strip()
                anchor_text = (f"{key}について詳しい解説はコチラ")[:ISEO_ANCHOR_MAX_CHARS] if key else "内部リンクについて詳しい解説はコチラ"

            # NGアンカー最終チェック（最低限）
            if _is_ng_anchor_generated_line(anchor_text, tgt_title):
                # --- ★リカバリ1：安全テンプレ再構成 → 日本語補正 → 再判定
                fallback = _safe_anchor_from_keywords(dst_kw_list, tgt_title or "")
                fallback = _postprocess_anchor_text(fallback)
                # 長すぎる場合の丸め（語尾を保つ）
                if len(fallback) > ISEO_ANCHOR_MAX_CHARS:
                    fallback = fallback[:ISEO_ANCHOR_MAX_CHARS]
                    fallback = re.sub(r"[、。．.\\s]+$", "", fallback)
                    if not fallback.endswith("について詳しい解説はコチラ"):
                        fallback = _safe_anchor_from_keywords(dst_kw_list, tgt_title or "")
                if not _is_ng_anchor_generated_line(fallback, tgt_title):
                    anchor_text = fallback
                else:
                    act.status = "skipped"
                    act.reason = "ng-anchor"
                    act.updated_at = datetime.now(UTC)
                    res.skipped += 1
                    continue

            # 直前段落がすでに内部リンクで終わっていれば（版マークあり）連続行を回避
            if idx - 1 >= 0:
                prev_tail = paragraphs[idx - 1][-200:]
                if INTERNAL_SEO_SPEC_MARK in prev_tail:
                    act.status = "skipped"
                    act.reason = "avoid-consecutive-link-paragraphs"
                    act.updated_at = datetime.now(UTC)
                    res.skipped += 1
                    continue

            # 記事内でのアンカーテキスト重複抑止（既存＋今回実行内）
            anchor_key = nfkc_norm(anchor_text).lower()
            if anchor_key and anchor_key in seen_anchor_keys:
                act.status = "skipped"
                act.reason = "duplicate-anchor-in-article"
                act.updated_at = datetime.now(UTC)
                res.skipped += 1
                continue

            if anchor_key and anchor_key in existing_anchor_text_set:
                act.status = "skipped"
                act.reason = "duplicate-anchor-existing"
                act.updated_at = datetime.now(UTC)
                res.skipped += 1
                continue

            anchor_html = _emit_anchor_html(href, anchor_text)
            # p: では囲みボックスは使わない（新仕様は h2: のみ）
            paragraphs[idx] = para_html + "<br>" + anchor_html
            article_href_set.add(href)
            try:
                used_target_pids.add(int(act.target_post_id))
            except Exception:
                pass
            act.anchor_text = anchor_text
            act.status = "applied"
            act.applied_at = datetime.now(UTC)
            res.applied += 1
            inserted += 1
            if anchor_key:
                seen_anchor_keys.add(anchor_key)
                existing_anchor_text_set.add(anchor_key)
            logger.info(f"[GEN-ANCHOR] p={idx} text='{anchor_text}' -> {href}")
            continue
        # 同一target_post_idは記事内で1回まで
        try:
            if int(act.target_post_id) in used_target_pids:
                act.status = "skipped"
                act.reason = "duplicate-target-in-article"
                act.updated_at = datetime.now(UTC)
                res.skipped += 1
                continue
        except Exception:
            pass
            # 同じURLが段落内に既にあるならスキップ（多重リンク回避）
            if href in [h for (h, _) in _extract_links(para_html)]:
                res.skipped += 1
                continue

            # 惹句テキスト：generated_line モードでは **毎回 LLM 生成**（plannerの anchor_text は使わない）
            try:
                # 文脈ヒント：当該段落のプレーンテキストを抜粋
                src_hint = _html_to_text(para_html)[:120]
                # 主要キーワード：plannerのmeta(dst_keywords)を最優先、無ければタイトルから抽出
                dst_kw_list = kw_meta_list
                if not dst_kw_list:
                    try:
                        dst_kw_list = [w for w in (title_tokens(tgt_title or "") or []) if w][:6]
                    except Exception:
                        dst_kw_list = []
                # 生成 → 品質チェック → 最大2回まで再生成、最後はテンプレで確定
                def _gen_once() -> str:
                    if ANCHOR_STYLE == "template":
                        base = (dst_kw_list[0] if dst_kw_list else (tgt_title or "")[:20]).strip()
                        return f"{base}について詳しい解説はコチラ"
                    return _generate_anchor_text_via_llm(
                        dst_title=tgt_title or "",
                        dst_keywords=dst_kw_list,
                        src_hint=src_hint,
                        user_id=None,
                    )
                anchor_text = _gen_once()
                tries = 0
                while tries < 2 and not _is_anchor_quality_ok(anchor_text, dst_kw_list, tgt_title or ""):
                    anchor_text = _gen_once()
                    tries += 1
                if not _is_anchor_quality_ok(anchor_text, dst_kw_list, tgt_title or ""):
                    base = (dst_kw_list[0] if dst_kw_list else (tgt_title or "")[:20]).strip()
                    anchor_text = f"{base}について詳しい解説はコチラ"
            except Exception as e:
                logger.warning(f"[GEN-ANCHOR] LLM failed: {e}")
                # フォールバックの定型（軽いCTA + タイトルの主要語）
                key = (tgt_title or "").strip()
                anchor_text = (f"{key}の詳しい解説はこちら。")[:80] if key else "詳しい解説はこちら。"
            # NGアンカー最終チェック
            if is_ng_anchor(anchor_text, tgt_title):
                act.status = "skipped"
                act.reason = "ng-anchor"
                act.updated_at = datetime.now(UTC)
                res.skipped += 1
                continue

            # 記事内での重複抑止
            anchor_key = nfkc_norm(anchor_text).lower()
            if anchor_key and anchor_key in seen_anchor_keys:
                act.status = "skipped"
                act.reason = "duplicate-anchor-in-article"
                act.updated_at = datetime.now(UTC)
                res.skipped += 1
                continue

            anchor_html = _emit_anchor_html(href, anchor_text)
            # 要求構造維持：<br><a ...> の形だが、版マークは直前コメントとして <a> の直前に置く
            paragraphs[idx] = para_html + "<br>" + anchor_html
            # 記事レベルの重複抑止セットを更新
            article_href_set.add(href)
            try:
                used_target_pids.add(int(act.target_post_id))
            except Exception:
                pass

            # 状態更新
            act.anchor_text = anchor_text  # 生成結果を保存（再試行時のキャッシュにもなる）
            act.status = "applied"
            act.updated_at = datetime.now(UTC)
            res.applied += 1
            inserted += 1
            if anchor_key:
                seen_anchor_keys.add(anchor_key)
            logger.info(f"[GEN-ANCHOR] p={idx} text='{anchor_text}' -> {href}")

        else:
            # ---- 旧方式：語句の最初の出現をリンク化（従来ロジックを維持） ----
            # ★安全ガード（緩和版）：ターゲットタイトルへの部分一致
            if nfkc_norm(act.anchor_text) not in nfkc_norm(tgt_title or ""):
                act.status = "skipped"
                act.reason = "skipped:anchor-not-in-target-title"
                act.updated_at = datetime.now(UTC)
                res.skipped += 1
                continue

            # 重複抑止
            anchor_key = nfkc_norm((act.anchor_text or "").strip()).lower()
            if anchor_key and anchor_key in seen_anchor_keys:
                act.status = "skipped"
                act.reason = "duplicate-anchor-in-article"
                act.updated_at = datetime.now(UTC)
                res.skipped += 1
                continue

            para_html = paragraphs[idx]
            if _H_TAG.search(para_html) or _TOC_HINT.search(para_html):
                res.skipped += 1
                continue
            if href in [h for (h, _) in _extract_links(para_html)]:
                res.skipped += 1
                continue
            if is_ng_anchor(act.anchor_text, tgt_title):
                act.status = "skipped"
                act.reason = "ng-anchor"
                act.updated_at = datetime.now(UTC)
                res.skipped += 1
                continue
            new_para = _linkify_first_occurrence(para_html, act.anchor_text, href, tgt_title)
            if not new_para:
                res.skipped += 1
                continue
            paragraphs[idx] = new_para
            article_href_set.add(href)
            try:
                used_target_pids.add(int(act.target_post_id))
            except Exception:
                pass
            act.status = "applied"
            act.applied_at = datetime.now(UTC)
            res.applied += 1
            inserted += 1
            if anchor_key:
                seen_anchor_keys.add(anchor_key)

    # h2/p 適用後の本文を連結
    new_html_mid = _rejoin_paragraphs(paragraphs)
    #   （簡易ルール：scoreの低そうな既存リンクをひとつだけ差し替え）
    # 2) swap候補：既存内部リンクがある & まだ余裕がない場合に置換を試みる
    if swaps and (existing_internal + inserted) >= need_min:
        # 既存リンク列挙
        existing = _extract_links(new_html_mid)
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
                href, _t = target_meta_map.get(s.target_post_id, ("",""))
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
                    s_act.applied_at = datetime.now(UTC)
                    res.swapped += 1

    # 本文を連結 → 既存の ai-ilink / inline-style を Wikipedia 風に正規化
    new_html = _rejoin_paragraphs(paragraphs)
    new_html = _normalize_existing_internal_links(new_html)
    # 正規化で a をいじっても、直前コメント（INTERNAL_SEO_SPEC_MARK）は HTML と独立で残る想定
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

    site = db.session.get(Site, site_id)
    # ▼ topic スキップ（記事URLに 'topic' を含む場合は一切触らない）
    # （重複ブロックを削除：上の分岐だけ残す
    # ▼ topic スキップ（記事URLに 'topic' を含む場合は一切触らない）
    try:
        if os.getenv("INTERNAL_SEO_SKIP_TOPIC", "1") != "0":
            src_url = _post_url(site_id, src_post_id) or ""
            if "topic" in (src_url or "").lower():
                return ApplyResult(message="skip-topic-page")
    except Exception:
        pass
    wp_post = fetch_single_post(site, src_post_id)
    if not wp_post:
        return ApplyResult(message="fetch-failed-or-excluded")

    # 1) 旧仕様削除（apply：後で削除ログを保存）— 新シグネチャ優先、未対応環境はフォールバック
    url_title_map = _all_url_to_title_map(site_id)
    url_pid_map   = _all_url_to_pid_map(site_id)
    try:
        cleaned_html, deletions = find_and_remove_legacy_links(
            wp_post.content_html or "", url_title_map, spec_version=INTERNAL_SEO_SPEC_VERSION
        )
    except TypeError:
        cleaned_html, deletions = find_and_remove_legacy_links(wp_post.content_html or "", url_title_map)

    # 2) 対象アクション（plan / swap_candidate）
    actions = (
        InternalLinkAction.query
        .filter_by(site_id=site_id, post_id=src_post_id, status="pending")
        .order_by(InternalLinkAction.created_at.asc())
        .all()
    )

    meta_map = _action_targets_meta(site_id, actions)

    # 3) 差分作成（旧仕様削除済みの本文に新仕様を適用）
    base_html = cleaned_html if cleaned_html is not None else (wp_post.content_html or "")
    # 3.5) まず入力HTMLをサニタイズ：見出し(H1〜H6)内の <a> を除去（中身は残す）
    base_html = _strip_links_in_headings(base_html)
    # サニタイズ済みの本文を元に新仕様を適用
    new_html, res = _apply_plan_to_html(site, src_post_id, base_html, actions, cfg, meta_map)
    # 念のため、適用後HTMLにも再サニタイズ（他要因でH内に混入しても除去）
    new_html = _strip_links_in_headings(new_html)
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
            a.updated_at = datetime.now(UTC)

    # 5) 旧仕様削除ログを保存（1削除=1行、status='legacy_deleted'）
    if deletions:
        now = datetime.now(UTC)
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

# ==== 追加：ユーザー単位でサイト横断適用 ====
def apply_actions_for_user(user_id: int, limit_posts: int = 50, dry_run: bool = False) -> Dict[str, object]:
    """
    指定ユーザーに紐づく全サイトを対象に、pending の内部リンク適用を実行する。
    - `limit_posts`: この呼び出し（1 tick）で処理する「記事数」の総予算
    - 予算はサイトごとの pending 件数を見て「水割り（均等＋余り前寄せ）」で配分

    戻り値:
      {
        "applied": int,
        "swapped": int,
        "skipped": int,
        "processed_posts": int,
        "pending_total": int,         # 開始時点の「未処理記事」総数（distinct post_id）
        "site_breakdown": [           # サイトごとの実行結果サマリ
          {
            "site_id": int,
            "allocated_posts": int,   # 今回割り当てた記事数
            "pending_posts": int,     # 開始時点のサイト内 pending 記事数
            "result": {"applied":..,"swapped":..,"skipped":..,"processed_posts":..}
          },
          ...
        ]
      }
    """
    # 1) ユーザーのサイト抽出
    sites = (
        Site.query
        .with_entities(Site.id)
        .filter(Site.user_id == user_id)
        .all()
    )
    site_ids = [int(sid) for (sid,) in sites]
    if not site_ids:
        return {
            "applied": 0, "swapped": 0, "skipped": 0,
            "processed_posts": 0, "pending_total": 0,
            "site_breakdown": []
        }

    # 2) サイトごとの「pending のある投稿数（distinct post_id）」を集計
    pending_rows = (
        db.session.query(
            InternalLinkAction.site_id,
            db.func.count(db.func.distinct(InternalLinkAction.post_id)).label("pending_posts")
        )
        .filter(InternalLinkAction.site_id.in_(site_ids),
                InternalLinkAction.status == "pending")
        .group_by(InternalLinkAction.site_id)        
        .all()
    )
    pending_map = {int(sid): int(cnt) for (sid, cnt) in pending_rows}

    # pending がゼロなら何もしない
    pending_total = sum(pending_map.values())
    if pending_total == 0 or (limit_posts or 0) <= 0:
        return {
            "applied": 0, "swapped": 0, "skipped": 0,
            "processed_posts": 0, "pending_total": pending_total,
            "site_breakdown": [
                {"site_id": sid, "allocated_posts": 0, "pending_posts": pending_map.get(sid, 0), "result": {"applied":0,"swapped":0,"skipped":0,"processed_posts":0}}
                for sid in site_ids
            ]
        }

    # 3) 予算配分（均等割り＋余りを pending が多いサイトから加算）
    targets = [sid for sid in site_ids if pending_map.get(sid, 0) > 0]
    if not targets:
        targets = []  # 念のため
    n = len(targets)
    budget = max(0, int(limit_posts or 0))
    base = budget // n if n else 0
    rem  = budget % n if n else 0

    # 余りは pending 多い順に +1
    targets_sorted = sorted(targets, key=lambda sid: pending_map.get(sid, 0), reverse=True)
    allocation = {sid: 0 for sid in site_ids}
    for sid in targets_sorted:
        allocation[sid] = base
    for i in range(rem):
        allocation[targets_sorted[i]] += 1

    # 4) サイトごとに実行（上限は pending_posts を超えない）
    total = {"applied": 0, "swapped": 0, "skipped": 0, "processed_posts": 0}
    breakdown: List[Dict[str, object]] = []
    for sid in site_ids:
        pending_posts = pending_map.get(sid, 0)
        alloc = min(allocation.get(sid, 0), pending_posts)
        if alloc <= 0:
            breakdown.append({
                "site_id": sid,
                "allocated_posts": 0,
                "pending_posts": pending_posts,
                "result": {"applied":0,"swapped":0,"skipped":0,"processed_posts":0}
            })
            continue
        res = apply_actions_for_site(sid, limit_posts=alloc, dry_run=dry_run)
        # 集計
        total["applied"] += int(res.get("applied", 0))
        total["swapped"] += int(res.get("swapped", 0))
        total["skipped"] += int(res.get("skipped", 0))
        total["processed_posts"] += int(res.get("processed_posts", 0))
        breakdown.append({
            "site_id": sid,
            "allocated_posts": alloc,
            "pending_posts": pending_posts,
            "result": res
        })

    total.update({"pending_total": pending_total, "site_breakdown": breakdown})
    return total
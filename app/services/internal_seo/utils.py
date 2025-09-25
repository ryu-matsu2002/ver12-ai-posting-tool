# app/services/internal_seo/utils.py
import re
import unicodedata
from html import unescape
from collections import Counter
from math import sqrt

_TAG = re.compile(r"<[^>]+>")
_WS  = re.compile(r"\s+")
# 句読点・全角記号も含めた記号類
_PUNCT = re.compile(r"[、。・,.!！?？:：;；\"'“”‘’()\[\]（）【】『』…\-_—–/]+")
_JP_TOKEN = re.compile(r"[一-龥ぁ-んァ-ンーA-Za-z0-9]{2,}")
_PAREN_AROUND = re.compile(r"[（(][^）)]{0,12}$")  # 直前が開き括弧付近
_PAREN_INSIDE = re.compile(r"^[^（)]{0,12}[）)]")  # 直後が閉じ括弧付近

def html_to_text(s: str | None) -> str:
    return _TAG.sub(" ", unescape(s or "")).strip()

def nfkc_norm(s: str | None) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = _WS.sub("", s).lower()
    s = _PUNCT.sub("", s)
    return s

def extract_terms_for_partial(title: str, min_len: int = 2) -> list[str]:
    """
    タイトルから “本文に載りやすい語” を抽出（漢字/かな/英数の2文字以上）。
    - まず長い塊を抽出
    - さらに助詞（の/に/を/へ/と/が/で/や/から/まで）で分割
    - カタカナ→漢字の境界でも分割（例: ワーホリ｜国）
    最後に重複を除去し、上限8語に抑制。
    """
    raw = unicodedata.normalize("NFKC", title or "")
    # 基本の長めトークンを取得
    base_toks = re.findall(r"[ぁ-んァ-ヴー一-龥A-Za-z0-9]{%d,}" % min_len, raw)

    # 追加の短語候補を生成（助詞・境界で分割）
    extra: list[str] = []
    for t in base_toks:
        # 助詞で分割
        parts = re.split(r"(?:から|まで|の|に|を|へ|と|が|で|や)", t)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            # カタカナ→漢字の境界でも分割（例: ワーホリ｜国）
            subparts = re.split(r"(?<=[ァ-ヴー])(?=[一-龥])", p)
            for q in subparts:
                q = q.strip()
                if len(q) >= min_len:
                    extra.append(q)

    toks = base_toks + extra
    # 重複除去（順序維持）
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:8]  # 上限（過剰に増やさない）


# === 追加: アンカー品質チェック ===
# NG語（機能語・指示語など）。正規化（nfkc_norm）後で比較。
STOPWORDS: set[str] = {
    "また","こちら","ここ","それ","これ","あれ","そちら","こちらも","そして","さらに","しかし",
    "まず","次に","一方","例えば","つまり","なお","ただし","です","ます","する","いる","ある",
    "ない","なる","まとめ","ポイント","今回","場合","可能","方法","基本","注意","詳細",
    # よくある誘導文
    "詳しくはこちら","詳細はこちら","こちらをご覧ください","こちらから","次はこちら",
    # 英語の汎用
    "here","this","that","click","more","readmore","readmore",
}

# 追加：正規化済みのNG語セット
STOPWORDS_N: set[str] = {  # <- これを公開して他モジュールでも使う
    (lambda x: (unicodedata.normalize("NFKC", x)).lower())(w)
    .replace(" ", "")
    for w in STOPWORDS
}

_KANA_ONLY = re.compile(r"^[ぁ-んァ-ンー]+$")
# 有意文字の判定：英数・漢字に加えて「かな（ひら/カタカナ/長音）」も許可
_CJK_OR_WORD = re.compile(r"[A-Za-z0-9一-龥ぁ-んァ-ンー]")

def is_ng_anchor(s: str | None) -> bool:
    """
    “ダメなアンカー”判定（短すぎ/指示語/機能語/不自然な語）。
    - 正規化して STOPWORDS に入っていればNG
    - 長さが短すぎ（正規化3未満）はNG
    - かなだけで短い（4未満）はNG
    - 記号しかない/有意な文字が無い場合もNG
    """
    if not s:
        return True
    n = nfkc_norm(s)
    if not n:
        return True
    if n in STOPWORDS_N:
        return True
    if len(n) < 3:
        return True
    if _KANA_ONLY.match(n) and len(n) < 4:
        return True
    # 有意な文字（英数/漢字/かな）が1つも含まれない場合はNG
    if not _CJK_OR_WORD.search(s):
        return True
    return False


# === 追加: タイトル語抽出 / 類似度ユーティリティ =========================
def title_tokens(title: str) -> list[str]:
    """
    タイトルから“アンカー候補になり得る語”を抽出。
    - NFKC 正規化
    - まず長めの塊を取り、その後
      * 助詞（の/に/を/へ/と/が/で/や/から/まで）で分割
      * カタカナ→漢字の境界でも分割（例: ワーホリ｜国）
    - 2文字以上を採用し、重複を除去して長い順に並べる
    """
    raw = unicodedata.normalize("NFKC", title or "")
    # ベース：漢字/かな/英数の2文字以上の塊
    base = re.findall(r"[ぁ-んァ-ヴー一-龥A-Za-z0-9]{2,}", raw)
    parts: list[str] = []
    for t in base:
        # 助詞で分割
        for p in re.split(r"(?:から|まで|の|に|を|へ|と|が|で|や)", t):
            p = p.strip()
            if not p:
                continue
            # カタカナ→漢字の境界でさらに分割
            for q in re.split(r"(?<=[ァ-ヴー])(?=[一-龥])", p):
                q = q.strip()
                if len(q) >= 2:
                    parts.append(q)
    uniq = sorted(set(parts), key=lambda s: (-len(s), s))
    return uniq

def keywords_set(csv_like: str | None) -> set[str]:
    if not csv_like:
        return set()
    return { (unicodedata.normalize("NFKC", k).strip().lower()) for k in (csv_like or "").split(",") if k.strip() }

def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    union = len(a | b)
    return inter / max(1, union)

def _tf(counter: Counter) -> dict[str, float]:
    total = sum(counter.values()) or 1
    return {k: v/total for k, v in counter.items()}

def title_tfidf_cosine(a_title: str, b_title: str) -> float:
    """
    簡易：タイトル語のTFコサイン（IDFなし）。0〜1近似。
    """
    at = Counter(title_tokens(a_title))
    bt = Counter(title_tokens(b_title))
    if not at or not bt:
        return 0.0
    fa, fb = _tf(at), _tf(bt)
    keys = set(fa) | set(fb)
    num = sum(fa.get(k,0.0) * fb.get(k,0.0) for k in keys)
    da = sqrt(sum((fa.get(k,0.0))**2 for k in keys))
    db = sqrt(sum((fb.get(k,0.0))**2 for k in keys))
    if da == 0 or db == 0:
        return 0.0
    return num/(da*db)

def is_natural_span(context: str, anchor: str) -> bool:
    """
    段落中で anchor をリンク化しても読みに違和感が出にくい位置かの軽量チェック。
    - 直前直後が助詞/記号のみでない
    - 括弧の直後/直前に寄りすぎていない
    """
    if not context or not anchor:
        return False
    idx = context.find(anchor)
    if idx < 0:
        return False
    before = context[max(0, idx-1):idx]
    after  = context[idx+len(anchor):idx+len(anchor)+1]
    # 直前直後が完全に連結文字の場合は避ける
    if re.match(r"[A-Za-z0-9一-龥ぁ-んァ-ンー]", before or "") and re.match(r"[A-Za-z0-9一-龥ぁ-んァ-ンー]", after or ""):
        return False
    # 極端に括弧に寄っていないか
    left = context[:idx][-12:]
    right = context[idx+len(anchor):][:12]
    if _PAREN_AROUND.search(left) or _PAREN_INSIDE.search(right):
        return False
    return True
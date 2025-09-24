# app/services/internal_seo/utils.py
import re
import unicodedata
from html import unescape

_TAG = re.compile(r"<[^>]+>")
_WS  = re.compile(r"\s+")
# 句読点・全角記号も含めた記号類
_PUNCT = re.compile(r"[、。・,.!！?？:：;；\"'“”‘’()\[\]（）【】『』…\-_—–/]+")

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
    長すぎる複合は避け、Wikipedia方針の“本文に出てくる語にだけリンク”に寄せる。
    """
    raw = unicodedata.normalize("NFKC", title or "")
    # 記号で分割 → 2文字以上の漢字・かな・英数の塊
    toks = re.findall(r"[ぁ-んァ-ヴー一-龥A-Za-z0-9]{%d,}" % min_len, raw)
    # 重複除去（順序維持）
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t); out.append(t)
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
_CJK_OR_WORD = re.compile(r"[A-Za-z0-9一-龥]")

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
    # A-Z/0-9/漢字が1つも含まれず、かなだけで極端に短いものは避ける
    if not _CJK_OR_WORD.search(s):
        return True
    return False
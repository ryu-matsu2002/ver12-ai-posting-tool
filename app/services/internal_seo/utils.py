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

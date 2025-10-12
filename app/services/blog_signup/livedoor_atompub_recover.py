#app/services/blog_signup/livedoor_atompub_recover.py

import asyncio
import os
import random
import time
from datetime import datetime
from pathlib import Path
import logging
import re as _re
from urllib.parse import urlparse, urlsplit, urlunsplit
from typing import List

logger = logging.getLogger(__name__)

# OpenAI（タイトル生成用）
try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None
OPENAI_MODEL_FOR_TITLES = os.getenv("OPENAI_TITLE_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_SEC = float(os.getenv("OPENAI_TIMEOUT_SEC", "12"))

# 可能ならサインアップ時の CAPTCHA 手動入力ツールを流用（存在しなければフォールバックへ）
try:
    from app.services.blog_signup.livedoor_signup import (
        prepare_captcha as ld_prepare_captcha,
        submit_captcha as ld_submit_captcha,
    )
except Exception:
    ld_prepare_captcha = None
    ld_submit_captcha = None

# 可能なら pwctl（セッションIDや一時保存ディレクトリの流儀を合わせるため）
try:
    from app.services.pw_controller import pwctl  # noqa
except Exception:
    pwctl = None

from app.utils.locks import pg_advisory_lock

# ビルド識別（デプロイ反映チェック用）
BUILD_TAG = "2025-09-12 livedoor-create-guarded + handoff-tab"
HANDOFF_MODE = True  # ✅ 手動ハンドオフ中は自動作成ロジックを無効化
logger.info(f"[LD-Recover] loaded build {BUILD_TAG}")

# 直列化・バックオフ・成功検知タイムアウト
LD_CREATE_LOCK_KEY = "livedoor:create_blog:global"
LD_CREATE_MAX_RETRIES = int(os.getenv("LD_CREATE_MAX_RETRIES", "3"))
LD_CREATE_BACKOFF_MIN = int(os.getenv("LD_CREATE_BACKOFF_MIN", "300"))    # 5分
LD_CREATE_BACKOFF_MAX = int(os.getenv("LD_CREATE_BACKOFF_MAX", "600"))    # 10分
LD_CREATE_SUCCESS_TIMEOUT_MS = int(os.getenv("LD_CREATE_SUCCESS_TIMEOUT_MS", "30000"))  # 30秒

def _rand_backoff() -> int:
    return random.randint(LD_CREATE_BACKOFF_MIN, LD_CREATE_BACKOFF_MAX)

def _human_sleep(a: float = 1.8, b: float = 3.2) -> None:
    time.sleep(random.uniform(a, b))


async def _save_shot(page, prefix: str) -> tuple[str, str]:
    """
    現在ページを /app/static/ld_dumps/{prefix}_{ts}.{png,html} で保存してパスを返す。
    失敗時は full_page=False にフォールバック。
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.getenv("LD_DUMP_DIR", "/var/www/ver12-ai-posting-tool/app/static/ld_dumps")
    try:
        Path(base_dir).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    png = f"{base_dir}/{prefix}_{ts}.png"
    html = f"{base_dir}/{prefix}_{ts}.html"
    try:
        await page.screenshot(path=png, full_page=True)
    except Exception:
        try:
            await page.screenshot(path=png)
        except Exception:
            pass
    try:
        Path(html).write_text(await page.content(), encoding="utf-8")
    except Exception:
        pass
    logger.info("[LD-Recover] dump saved: %s , %s", png, html)
    return png, html


# ─────────────────────────────────────────────
# 安定インデックス・文字種判定・正規化などのユーティリティ
# ─────────────────────────────────────────────
def _deterministic_index(salt: str, n: int) -> int:
    """
    salt（文字列）から 0..n-1 の安定インデックスを決める。
    - ランタイム/プロセスを跨いでも同じ salt, n なら同じ値
    - n <= 0 の場合は 0
    """
    if n <= 0:
        return 0
    # 32bit rolling hash（Python の hash は起動ごとに変わるため使わない）
    acc = 0
    for ch in str(salt):
        acc = (acc * 131 + ord(ch)) & 0xFFFFFFFF
    return acc % n


def _has_cjk(s: str) -> bool:
    return bool(_re.search(r"[\u3040-\u30FF\u3400-\u9FFF]", s or ""))


def _norm(s: str) -> str:
    """比較用：空白/記号を落として小文字化"""
    s = (s or "").lower()
    s = _re.sub(r"[\s\-_／|｜/・]+", "", s)
    return s


def _domain_tokens(url: str) -> list[str]:
    """ドメインを単語に分割（tld等は除外）"""
    try:
        netloc = urlparse(url or "").netloc.lower()
    except Exception:
        netloc = ""
    if ":" in netloc:
        netloc = netloc.split(":", 1)[0]
    parts = [p for p in netloc.split(".") if p and p not in ("www", "com", "jp", "net", "org", "co")]
    words = []
    for p in parts:
        words.extend([w for w in p.replace("_", "-").split("-") if w])
    return words


# ─────────────────────────────────────────────
# サイト名トークン化・ジャンル推定・日本語タイトル生成
# ─────────────────────────────────────────────
STOPWORDS_JP = {
    "株式会社", "有限会社", "合同会社", "公式", "オフィシャル", "ブログ", "サイト", "ホームページ",
    "ショップ", "ストア", "サービス", "工房", "教室", "情報", "案内", "チャンネル", "通信", "マガジン"
}
STOPWORDS_EN = {
    "inc", "ltd", "llc", "official", "blog", "site", "homepage", "shop", "store",
    "service", "studio", "channel", "magazine", "info", "news"
}


def _name_tokens(name: str) -> list[str]:
    """サイト名を雑にトークン化（日本語/英語混在対応・記号で分割）"""
    if not name:
        return []
    parts = _re.split(r"[\s\u3000\-/＿_・|｜／]+", str(name))
    toks: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # 記号を除去しすぎない程度に掃除
        p = _re.sub(r"[^\w\u3040-\u30FF\u3400-\u9FFFー]+", "", p)
        if p:
            toks.append(p)
    return toks


def _keyword_seed_from_site(site) -> tuple[str | None, bool]:
    """
    サイト名から「1語」を安定に選ぶ。
    戻り値: (seed, is_jp)  /  抽出できなければ (None, False)
    """
    name = (getattr(site, "name", "") or "").strip()
    # salt は id+name のみ（URLは含めない）
    salt = f"{getattr(site, 'id', '')}-{name}"

    name_toks = _name_tokens(name)

    # 日本語と英語で候補を分ける
    jp_cands = [t for t in name_toks if _has_cjk(t) and t not in STOPWORDS_JP]
    en_cands = [t for t in name_toks if not _has_cjk(t)]
    en_cands = [t for t in en_cands if t.lower() not in STOPWORDS_EN]

    # 長さフィルタ（1文字や長すぎは除外）
    jp_cands = [t for t in jp_cands if 2 <= len(t) <= 12]
    en_cands = [t for t in en_cands if 2 <= len(t) <= 15]

    # 同一サイトでは安定して同じ語を選ぶ（塩＝site.id+name）
    def _pick(stable_list: list[str]) -> str | None:
        if not stable_list:
            return None
        idx = _deterministic_index(salt, len(stable_list))
        return stable_list[idx]

    seed = _pick(jp_cands) or _pick(en_cands)
    if seed:
        return seed, _has_cjk(seed)
    return None, False


def _guess_genre(site) -> tuple[str, bool]:
    """
    サイトからジャンル語(日本語/英語)と日本語フラグを推定。
    1) 明示属性（primary_genre_name / genre_name / genre.name など）
    2) site.name の語からヒューリスティック（URLは参照しない）
    """
    # 1) 明示属性
    for attr in ("primary_genre_name", "genre_name", "genre", "main_genre", "category", "category_name"):
        v = getattr(site, attr, None)
        if isinstance(v, str) and v.strip():
            txt = v.strip()
            return txt, _has_cjk(txt)
        name = getattr(v, "name", None)
        if isinstance(name, str) and name.strip():
            txt = name.strip()
            return txt, _has_cjk(txt)

    # 2) ヒューリスティック（サイト名のみ）
    name = (getattr(site, "name", "") or "")
    txt = name.lower()

    JP = [
        ("ピラティス", ("pilates", "ピラティス", "yoga", "体幹", "姿勢", "fitness", "stretch")),
        ("留学", ("studyabroad", "abroad", "留学", "ielts", "toefl", "海外", "study")),
        ("旅行", ("travel", "trip", "観光", "hotel", "onsen", "温泉", "tour")),
        ("美容", ("beauty", "esthetic", "skin", "hair", "美容", "コスメ", "メイク")),
        ("ビジネス", ("business", "marketing", "sales", "seo", "経営", "起業", "副業")),
    ]
    for label, keys in JP:
        if any(k in txt for k in keys):
            return label, True

    EN = [
        ("Pilates", ("pilates", "yoga", "fitness", "posture", "stretch")),
        ("Study Abroad", ("studyabroad", "abroad", "study", "ielts", "toefl")),
        ("Travel", ("travel", "trip", "hotel", "onsen", "tour")),
        ("Beauty", ("beauty", "esthetic", "skin", "hair", "cosme", "makeup")),
        ("Business", ("business", "marketing", "sales", "seo", "startup")),
    ]
    for label, keys in EN:
        if any(k in txt for k in keys):
            return label, False

    # どれにも該当しなければ汎用
    return ("日々", _has_cjk(name))


def _too_similar_to_site(title: str, site) -> bool:
    """
    タイトルがサイト名/ドメイン由来語と似すぎなら True。
    - 正規化同士の完全一致
    - 片方がもう片方を包含
    - ドメイン語幹（tokens）が含まれる/含まれる
    """
    t = _norm(title)
    site_name = (getattr(site, "name", "") or "")
    site_url = (getattr(site, "url", "") or "")
    n = _norm(site_name)

    if not t:
        return True

    # 完全一致 / 包含
    if t == n or (t and n and (t in n or n in t)):
        return True

    # ドメイン語幹との照合
    toks = set(_domain_tokens(site_url))
    toks |= {w for w in _name_tokens(site_name) if not _has_cjk(w)}  # 英字トークンも禁止寄り
    toks = {_norm(w) for w in toks if w}

    for w in toks:
        if not w:
            continue
        if w in t or t in w:
            return True

    return False

# ─────────────────────────────
# タイトル生成ユーティリティ
# ─────────────────────────────
def _strip_numbering(s: str) -> str:
    """行頭の番号・記号を除去"""
    s = (s or "").strip()
    s = _re.sub(r"^[\s\-\*\u2022\u25CF\u25A0\u30FB\d]+[.)、．]?\s*", "", s)
    return s.strip("　").strip()

def _split_lines_as_titles(text: str) -> List[str]:
    lines = [_strip_numbering(x) for x in (text or "").splitlines()]
    seen, out = set(), []
    for t in lines:
        t = t.strip()
        if not t:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out

# ─────────────────────────────────────────────
# LLMでジャンル推定（短い日本語一般名詞）＋ 指定プロンプトの生成
# ─────────────────────────────────────────────
async def _infer_genre_with_openai(site) -> str:
    """サイト名から '医療脱毛' '不動産投資' などの一般名詞ジャンルを1〜4語で抽出"""
    if AsyncOpenAI is None:
        return ""
    name = (getattr(site, "name", "") or "").strip()
    url  = (getattr(site, "url", "") or "").strip()
    bad  = ", ".join(_domain_tokens(url))
    prompt = f"""以下の情報から、そのサイトのジャンル名を日本語で1〜4語に要約して返してください。
ブランド名・固有名詞・ドメイン語は使わず、一般名詞で。
出力はジャンル名だけ（句読点・説明なし）。

サイト名: {name}
参考URLドメインの語（避ける語）: {bad or "なし"}
例: 医療脱毛 / 不動産投資 / 子育て / 料理レシピ / 中古車査定 / プログラミング学習"""
    try:
        client = AsyncOpenAI(timeout=OPENAI_TIMEOUT_SEC)
        res = await client.chat.completions.create(
            model=OPENAI_MODEL_FOR_TITLES,
            temperature=0.2,
            top_p=0.95,
            max_tokens=50,
            messages=[
                {"role": "system", "content": "You are a concise classifier that answers only with a short Japanese noun phrase."},
                {"role": "user", "content": prompt},
            ],
        )
        txt = (res.choices[0].message.content or "").strip()
        genre = _strip_numbering((txt.splitlines()[0] if txt else "")).replace("ジャンル", "").strip('「」"\'：: 　')
        # ドメイン語が混入したら落とす
        for w in _domain_tokens(url):
            if w and w.lower() in genre.lower():
                genre = genre.replace(w, "").strip()
        return genre[:24]
    except Exception as e:
        logger.warning("[GenreInfer] OpenAI error: %s", e)
        return ""

def _title_prompt_from_genre(genre: str) -> str:
    """あなた指定のプロンプト本文をそのまま使用。末尾でジャンルを渡す。"""
    base = """あなたはSEOとコンテンツマーケティングの専門家です。
キャッチコピーを考える天才です。

入力されたサイトジャンルから連想して
WEBサイトの「サイトタイトル」を32文字以内で10個考えてください。 
日本一のマーケッター神田昌典さんが考えたような感じでお願いします！

タイトルごとに改行してください


###条件###
タイトルの中にキャッチコピー的なテキストを入れてください
役に立つ情報を発信していくサイトです

###具体例###
「転職」というキーワードに対する出力文：
転職アドバイザーが全力で教える転職ノウハウ"""
    g = (genre or "総合情報").strip()
    return f"{base}\n\n【サイトジャンル】{g}\n"

def _score_title(t: str, site, prefer_kw: str | None = None) -> float:
    """
    “必ず10案から選ぶ”ためのスコアリング。
    0〜100想定。高いほど良い。
    """
    if not t:
        return -1
    score = 100.0
    # 長さ（32超は強い減点、32以内は微加点）
    L = len(t)
    if L > 32:
        score -= 2.5 * (L - 32) + 20
    else:
        score += max(0, 6 - abs(28 - L) // 3)  # ざっくり32付近を好む
    # 日本語らしさ
    if not _has_cjk(t):
        score -= 35
    # 類似（強い減点）
    if _too_similar_to_site(t, site):
        score -= 60
    # 記号だらけなど軽微な減点
    if _re.search(r"[!！?？]{3,}", t):
        score -= 5

    # ★ プロンプト反映：推定ジャンル語が含まれていれば微加点
    if prefer_kw:
        try:
            if _norm(prefer_kw) and _norm(prefer_kw) in _norm(t):
                score += 8
        except Exception:
            pass    
    return score

def _pick_best_from_candidates(cands: List[str], site, prefer_keyword: str | None = None) -> str | None:
    """まず厳格フィルタ→空ならスコアで10案から必ず選ぶ"""
    strict = [
        t for t in cands
        if t and len(t) <= 32 and _has_cjk(t) and not _too_similar_to_site(t, site)
        and t not in {"日々のブログ", "ひびのブログ"}
    ]
    pool = strict if strict else cands  # 空なら全候補から選ぶ
    if not pool:
        return None
    # スコア降順、同点はsaltで安定選択
    scored = sorted(pool, key=lambda x: (_score_title(x, site, prefer_keyword)), reverse=True)
    top_score = _score_title(scored[0], site, prefer_keyword)
    # 同点群を抽出
    ties = [t for t in scored if abs(_score_title(t, site, prefer_keyword) - top_score) < 1e-6]
    
    # 同点が1つだけならそれを採用。ただし32字トリム後の最終類似を確認
    if len(ties) == 1:
        cand = ties[0][:32]
        return cand if not _too_similar_to_site(cand, site) else None
    # 同点が複数なら salt で安定選択 → それでも似すぎなら次点を順に試す
    salt = f"{getattr(site,'id','')}-{getattr(site,'name','')}-{getattr(site,'url','')}"
    order = list(ties)
    start = _deterministic_index(salt, len(order))
    order = order[start:] + order[:start]
    for t in order:
        cand = t[:32]
        if _has_cjk(cand) and not _too_similar_to_site(cand, site):
            return cand
    # ここまで来たら、同点以外（scored全体）からも順に拾う
    for t in scored:
        cand = t[:32]
        if _has_cjk(cand) and not _too_similar_to_site(cand, site):
            return cand
    # ✅ それでも全滅なら「最初の10案から必ず選ぶ」ポリシーで最上位を返す
    return (scored[0][:32] if scored else (cands[0][:32] if cands else None))

async def _gen_titles_with_openai(site, genre: str) -> List[str]:
    """
    1回だけ呼び出して10案取得（再生成しない）。
    失敗時は空配列。
    """
    if AsyncOpenAI is None:
        return []
    try:
        client = AsyncOpenAI(timeout=OPENAI_TIMEOUT_SEC)
        res = await client.chat.completions.create(
            model=OPENAI_MODEL_FOR_TITLES,
            temperature=0.7,
            top_p=0.95,
            max_tokens=400,
            messages=[
                {"role": "system", "content": "You are a skilled Japanese copywriter and SEO strategist."},
                {"role": "user", "content": _title_prompt_from_genre(genre)},
            ],
        )
        text = ""
        try:
            if res and getattr(res, "choices", None):
                text = (res.choices[0].message.content or "").strip()
        except Exception:
            # choices が空/不正でも全体はフォールバックに流れる
            text = ""
        cands = _split_lines_as_titles(text)
        # 取り過ぎた場合も10件に丸める
        return cands[:10]
    except Exception as e:
        logger.warning("[TitleGen] OpenAI error (single shot): %s", e)
        return []

async def generate_blog_title(site) -> str:
    """
    もとのサイト名→ジャンル推定→指定プロンプトで10案→ベスト選択。
    API失敗/0件のみフォールバック。
    """
    # 1) ジャンル推定（LLM）。だめなら既存ヒューリスティック。
    genre = await _infer_genre_with_openai(site)
    if not genre:
        g, _ = _guess_genre(site)
        genre = g or "総合情報"
    # 2) 指定プロンプトで10案生成
    cands = await _gen_titles_with_openai(site, genre)
    if cands:
        # 3) プロンプト反映（ジャンル語）＋“似すぎ回避”で最良案
        best = _pick_best_from_candidates(cands, site, prefer_keyword=genre)
        if best:
            logger.info("[TitleGen] picked: %s (from %d candidates)", best, len(cands))
            return best[:32]
    # 失敗時のみフォールバック
    try:
        return _craft_blog_title(site)[:32]
    except Exception:
        return "こつこつブログ"


def _templates_jp(topic: str) -> list[str]:
    base = (topic or "").strip() or "日々"
    return [
        f"{base}ブログ",
        f"{base}ブログ日記",
        f"{base}のブログ",
        f"{base}の記録ブログ",
        f"{base}の暮らしブログ",
        f"{base}のメモ帳",
        f"{base}の覚え書き",
        f"{base}のジャーナル",
        f"{base}手帖",
        f"{base}ノート",
        f"{base}の小部屋",
        f"{base}ログ",
    ]


def _templates_en(topic: str) -> list[str]:
    base = topic.strip() or "Notes"
    return [f"{base} Blog"]  # ダミー（呼ばれない想定）


def _japanese_base_word(site) -> str:
    """
    1) まずジャンル推定で日本語ラベルを取得（ピラティス/旅行/美容/ビジネス…）
    2) 取れなければ「日々」
    ※ “サイト名そのもの”は使わない（似すぎ回避）
    """
    topic, is_jp = _guess_genre(site)
    if _has_cjk(topic):
        return topic.strip()
    return "日々"


def _craft_blog_title(site) -> str:
    """
    仕様（ご指定反映）：
      - 生成結果は日本語ベース
      - サイト名/URLから抽出したキーワードやジャンル語を使って“ブログ風”に
      - 「日々のブログ」にはしない（明示的に禁止）
      - 元サイト名/ドメインに似すぎない
      - 同一サイトでは決定論的に安定
    """
    site_name = (getattr(site, "name", "") or "").strip()
    site_url = (getattr(site, "url", "") or "").strip()
    salt = f"{getattr(site, 'id', '')}-{site_name}-{site_url}"

    # まずはサイトから1語シードを取る（日本語があれば優先）
    seed, seed_is_jp = _keyword_seed_from_site(site)
    if not seed:
        # ジャンル推定語（JPなら優先）
        topic, is_jp = _guess_genre(site)
        seed = topic if _has_cjk(topic) else "暮らし"  # デフォルトは「暮らし」
        seed_is_jp = True

    # ブログ風テンプレ（“ブログ”を含むパターン中心＋バリエーション）
    base = seed.strip()
    # 「日々のブログ」は禁止語として明示除外
    banned_exact = {"日々のブログ", "ひびのブログ"}
    candidates = [
        f"{base}ブログ",
        f"{base}のブログ",
        f"{base}ブログ記録",
        f"{base}の記録ブログ",
        f"{base}のメモブログ",
        f"{base}のノート",
        f"{base}ログ",
        f"{base}手帖",
    ]

    # 許容判定
    def acceptable(title: str) -> bool:
        if not title or not title.strip():
            return False
        if title in banned_exact:
            return False
        if _too_similar_to_site(title, site):
            return False
        # 日本語らしさ：少なくとも1文字はCJK
        if not _has_cjk(title):
            return False
        return True

    # saltで開始位置を決め、順回しで最初に通ったものを採用
    start = _deterministic_index(salt, len(candidates))
    for i in range(len(candidates)):
        t = candidates[(start + i) % len(candidates)]
        if acceptable(t):
            return t[:48]

    # 最終フォールバック（禁止の「日々のブログ」は含めない）
    fallbacks = [f"{base}ブログ", f"{base}ログ", f"{base}手帖", "こつこつブログ"]
    return fallbacks[_deterministic_index(salt, len(fallbacks))][:48]

# ─────────────────────────────────────────────
# 追加: AtomPub エンドポイント軽量プローブ
# ─────────────────────────────────────────────
try:
    import requests  # 最小フォールバック用（ない環境では False を返す）
except Exception:
    requests = None  # type: ignore

def _normalize_atompub_endpoint(raw: str) -> str:
    """
    'example.com' → 'https://example.com/atompub' に正規化。
    末尾スラッシュは付けない（'/atompub' で統一）。
    """
    v = (raw or "").strip()
    if not v:
        return v
    if not _re.match(r"^https?://", v, _re.I):
        v = "https://" + v
    parts = urlsplit(v)
    path = _re.sub(r"/{2,}", "/", parts.path or "/")
    if not _re.search(r"/atompub/?$", path, _re.I):
        path = path.rstrip("/") + "/atompub"
    path = path.rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))

def probe_auth(*, endpoint: str, api_key: str | None = None, blog_id: str | None = None, timeout: int = 6) -> bool:
    """
    Livedoor AtomPub の **超軽量疎通チェック**。
    - 認証ヘッダは扱わず、2xx を疎通OKとみなす。
    - 401/403 は“認証エラー”として False。
    - それ以外／例外は False。
    ルート側の「接続テスト」用途（存在チェック）に特化。
    """
    if not endpoint or requests is None:
        return False
    url = _normalize_atompub_endpoint(endpoint)
    # まずは素のエンドポイント
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code // 100 == 2:
            return True
        if r.status_code in (401, 403):
            return False
    except Exception as e:
        logger.debug("[LD-AtomPub] probe GET base error: %s", e)
    # blog_id 付きでも何パターンか当ててみる（環境差対策）
    if blog_id:
        for suffix in (f"/blog/{blog_id}", f"/blogs/{blog_id}", f"/{blog_id}"):
            try:
                r2 = requests.get(url + suffix, timeout=timeout)
                if r2.status_code // 100 == 2:
                    return True
                if r2.status_code in (401, 403):
                    return False
            except Exception as e:
                logger.debug("[LD-AtomPub] probe GET suffix error (%s): %s", suffix, e)
    return False



# ─────────────────────────────────────────────
# 追加：フレーム横断・同意チェック・エラーテキスト採取
# ─────────────────────────────────────────────
async def _maybe_close_overlays(page):
    selectors = [
        'button#iubenda-cs-accept-btn',
        'button#iubenda-cs-accept',
        'button:has-text("同意")',
        'button:has-text("許可")',
        'button:has-text("OK")',
        '.cookie-accept', '.cookie-consent-accept',
        '.modal-footer button:has-text("閉じる")',
        'div[role="dialog"] button:has-text("OK")',
    ]
    for sel in selectors:
        try:
            if await page.locator(sel).first.is_visible():
                await page.locator(sel).first.click(timeout=1000)
        except Exception:
            pass
    # 透明オーバーレイの一般除去
    try:
        await page.evaluate("""
            (() => {
              const blocks = Array.from(document.querySelectorAll('div,section'))
                .filter(n => {
                  const s = getComputedStyle(n);
                  if (!s) return false;
                  const r = n.getBoundingClientRect();
                  return r.width>300 && r.height>200 &&
                         s.position !== 'static' &&
                         parseFloat(s.zIndex||'0') >= 1000 &&
                         s.pointerEvents !== 'none' &&
                         (s.backgroundColor && s.backgroundColor !== 'rgba(0, 0, 0, 0)');
                });
              blocks.slice(0,3).forEach(n => n.style.pointerEvents='none');
            })();
        """)
    except Exception:
        pass


async def _maybe_accept_terms(page) -> bool:
    """利用規約の同意チェックがあればON。チェック状態が変わったら True を返す。"""
    sels = [
        'input[type="checkbox"][name*="agree"]',
        'input#agree', 'input#agreement', 'input#termsAgree',
        'input#accept-terms', 'input[name="agreement"]', 'input[name="accept"]'
    ]
    changed = False
    for sel in sels:
        try:
            loc = page.locator(sel).first
            if await loc.count() > 0:
                try:
                    await loc.check()
                    changed = True
                except Exception:
                    # check() で失敗した際の直接操作フォールバック
                    try:
                        handle = await loc.element_handle()
                        if handle:
                            await page.evaluate(
                                "(el)=>{if(!el.checked){el.checked=true;} el.dispatchEvent(new Event('change',{bubbles:true}))}", handle
                            )
                            changed = True
                    except Exception:
                        pass
                if changed:
                    logger.info("[LD-Recover] ✅ 規約同意チェック: %s", sel)
                    break
        except Exception:
            pass
    return changed


async def _has_blog_id_input(page) -> bool:
    for sel in [
        '#blogId', 'input[name="blog_id"]', 'input[name="livedoor_blog_id"]',
        'input[name="blogId"]', 'input#livedoor_blog_id', 'input[placeholder*="ブログURL"]',
        '#sub', 'input[name="sub"]'   # ← これが生きる
    ]:
        try:
            if await page.locator(sel).count() > 0:
                return True
        except Exception:
            pass
    return False



async def _log_inline_errors(page):
    """画面上の代表的なエラーメッセージを収集してログ出力"""
    sels = [
        '.error', '.error-message', '.errors li', 'p.error', 'span.error',
        '.alert-danger', '.alert.alert-danger', 'div.errorMessage', 'li.error',
        'div.formError', 'div#notice .error', 'div.notice.error'
    ]
    texts = []
    for sel in sels:
        try:
            loc = page.locator(sel)
            cnt = await loc.count()
            for i in range(min(cnt, 10)):
                t = (await loc.nth(i).inner_text()).strip()
                if t:
                    texts.append(t.replace("\n", " "))
        except Exception:
            pass
    if texts:
        logger.warning("[LD-Recover] inline errors: %s", " | ".join(texts[:5]))


async def _find_in_any_frame(page, selectors, timeout_ms=15000):
    """全フレーム走査。最初に見つかったframeとselectorを返す。"""
    logger.info("[LD-Recover] frame-scan start selectors=%s timeout=%sms", selectors[:2], timeout_ms)
    deadline = asyncio.get_event_loop().time() + (timeout_ms / 1000)
    while asyncio.get_event_loop().time() < deadline:
        try:
            for fr in page.frames:
                for sel in selectors:
                    try:
                        if await fr.locator(sel).count() > 0:
                            logger.info("[LD-Recover] frame-scan hit: frame=%s sel=%s", getattr(fr, 'url', None), sel)
                            return fr, sel
                    except Exception:
                        continue
        except Exception:
            pass
        await asyncio.sleep(0.25)
    logger.warning("[LD-Recover] frame-scan timeout selectors=%s", selectors[:3])
    return None, None


async def _wait_enabled_and_click(page, locator, *, timeout=8000, label_for_log=""):
    try:
        await locator.wait_for(state="visible", timeout=timeout)
    except Exception:
        try:
            await locator.wait_for(state="attached", timeout=int(timeout/2))
        except Exception:
            pass

    # ★ ElementHandle を取得
    try:
        handle = await locator.element_handle()
    except Exception:
        handle = None

    # enabled/表示状態
    if handle:
        try:
            await page.wait_for_function(
                "(el) => el && !el.disabled && el.offsetParent !== null",
                arg=handle, timeout=timeout
            )
        except Exception:
            pass

    try:
        await locator.scroll_into_view_if_needed(timeout=1500)
    except Exception:
        pass
    try:
        await locator.focus()
    except Exception:
        pass

    # クリック多段
    try:
        await locator.click(timeout=timeout)
        logger.info("[LD-Recover] clicked %s (normal)", label_for_log or "")
        return True
    except Exception:
        try:
            await locator.click(timeout=timeout, force=True)
            logger.info("[LD-Recover] clicked %s (force)", label_for_log or "")
            return True
        except Exception:
            if handle:
                try:
                    await page.evaluate("(el)=>el.click()", handle)
                    logger.info("[LD-Recover] clicked %s (evaluate)", label_for_log or "")
                    return True
                except Exception:
                    pass
            logger.warning("[LD-Recover] click failed %s", label_for_log, exc_info=True)
            return False


# ─────────────────────────────────────────────
# ブログ作成ページ：タイトル入力＆送信
# ─────────────────────────────────────────────
async def _set_title_and_submit(page, desired_title: str) -> bool:
    """
    ブログタイトル入力 → 『ブログを作成する』クリック だけ。
    それ以外のUI操作は行わない。
    """
    title_selectors = ['#blogTitle', 'input[name="title"]']
    button_selectors = [
        'input[type="submit"][value="ブログを作成する"]',
        'button[type="submit"]',
        'input[type="submit"]',
    ]

    # タイトル入力
    title_loc = None
    for sel in title_selectors:
        try:
            await page.wait_for_selector(sel, state="visible", timeout=20000)
            cand = page.locator(sel).first
            if await cand.count() > 0:
                title_loc = cand
                break
        except Exception:
            continue
    if not title_loc:
        logger.warning("[LD-Recover] タイトル入力欄が見つかりません")
        return False

    try:
        try:
            await title_loc.fill("")
        except Exception:
            try:
                await title_loc.click()
                await title_loc.press("Control+A")
                await title_loc.press("Delete")
            except Exception:
                pass
        await title_loc.fill(desired_title)
        logger.info("[LD-Recover] ブログタイトルを設定: %s", desired_title)
    except Exception:
        logger.warning("[LD-Recover] タイトル入力に失敗", exc_info=True)
        return False

    # 作成ボタン
    btn = None
    for sel in button_selectors:
        try:
            cand = page.locator(sel).first
            if await cand.count() > 0:
                btn = cand
                break
        except Exception:
            continue
    if not btn:
        logger.warning("[LD-Recover] 『ブログを作成する』ボタンが見つかりません")
        return False

    # クリック（遷移が発生しないUIでも1回だけ押す）
    try:
        async with page.expect_navigation(wait_until="load", timeout=15000):
            await btn.click()
        logger.info("[LD-Recover] 『ブログを作成する』をクリック")
    except Exception:
        # 遷移イベントが取れなくても、1回だけフォールバッククリック
        try:
            await btn.click(timeout=5000)
            logger.info("[LD-Recover] 『ブログを作成する』をクリック（fallback）")
        except Exception:
            logger.warning("[LD-Recover] 作成ボタンクリックに失敗", exc_info=True)
            return False

    # 追加の安定待ち（軽く）
    try:
        await page.wait_for_load_state("networkidle", timeout=8000)
    except Exception:
        pass
    return True



# ─────────────────────────────────────────────
# 作成ページ CAPTCHA 検出・処理（人力ツール優先／FSフォールバック）
# ─────────────────────────────────────────────
async def _detect_create_captcha(page) -> tuple[bool, str | None]:
    try:
        img_sel = '#captcha_image, #captcha-img, img.captcha'  # 実体＋互換
        box_sel = 'input[name="captcha_code"], input[name="captcha"], #captcha'
        has_img = await page.locator(img_sel).first.count() > 0
        has_box = await page.locator(box_sel).first.count() > 0
        if not (has_img and has_box):
            return False, None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.getenv("LD_DUMP_DIR", "/var/www/ver12-ai-posting-tool/app/static/ld_dumps")
        try:
            Path(base_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        path = f"{base_dir}/ld_create_captcha_{ts}.png"
        try:
            await page.locator(img_sel).first.screenshot(path=path)
        except Exception:
            await page.screenshot(path=path, full_page=True)
        logger.info("[LD-Recover] create CAPTCHA captured: %s", path)
        return True, path
    except Exception:
        return False, None



async def _fill_captcha_and_submit(page, text: str) -> bool:
    try:
        box_sel = 'input[name="captcha_code"], input[name="captcha"], #captcha'
        loc = page.locator(box_sel).first
        await loc.fill("")
        await loc.fill(text)

        # 送信ボタン（代表選抜）
        btn = page.locator('input[type="submit"][value="ブログを作成する"]').first
        if await btn.count() == 0:
            btn = page.locator(
                'button[type="submit"], input[type="submit"][value*="作成"], input[type="submit"][value*="登録"]'
            ).first

        await _save_shot(page, "ld_create_before_submit_with_captcha")
        try:
            async with page.expect_navigation(wait_until="load", timeout=15000):
                await btn.click()
        except Exception:
            await _wait_enabled_and_click(page, btn, timeout=8000, label_for_log="create-after-captcha")
            try:
                await page.wait_for_load_state("networkidle", timeout=8000)
            except Exception:
                pass
        return True
    except Exception:
        logger.warning("[LD-Recover] CAPTCHA 入力→送信に失敗", exc_info=True)
        return False



async def _wait_success_after_submit(page) -> tuple[bool, str | None]:
    """/welcome 遷移 or 成功導線検出を待って成功可否と URL 由来 blog_id を返す。"""
    try:
        await page.wait_for_url(_re.compile(r"/welcome($|[/?#])"), timeout=LD_CREATE_SUCCESS_TIMEOUT_MS)
        return True, _extract_blog_id_from_url(page.url)
    except Exception:
        pass

    # 文言 or 導線
    try:
        await page.wait_for_selector('text=ブログの作成が完了しました', timeout=LD_CREATE_SUCCESS_TIMEOUT_MS)
        return True, _extract_blog_id_from_url(page.url)
    except Exception:
        pass
    try:
        await page.wait_for_selector('text=ブログの作成が完了しました！', timeout=LD_CREATE_SUCCESS_TIMEOUT_MS)
        return True, _extract_blog_id_from_url(page.url)
    except Exception:
        pass

    fr, _ = await _find_in_any_frame(
        page,
        ['a:has-text("最初のブログを書く")', 'a.button:has-text("はじめての投稿")', ':has-text("ブログが作成されました")', 'a:has-text("投稿する")', 'a:has-text("ブログ設定")'],
        timeout_ms=LD_CREATE_SUCCESS_TIMEOUT_MS
    )
    if fr:
        return True, _extract_blog_id_from_url(page.url)

    return False, None


import inspect

async def _human_tool_captcha_flow(
    page,
    image_path: str,
    *,
    livedoor_id: str | None = None,
    password: str | None = None,
) -> bool:
    """
    サインアップ時の手動入力ツールを優先利用。
    - ld_prepare_captcha / ld_submit_captcha が使える場合はそれを使う
    - 使えない場合は FS 監視フォールバック
    """
    # 1) まずは既存ツール（関数が見つかれば使う）
    if ld_prepare_captcha and ld_submit_captcha:
        try:
            logger.info("[LD-Recover] using signup captcha tool on create-page")
            # 既存実装の引数差異を吸収
            def _callable_with(args_map, fn):
                sig = inspect.signature(fn)
                kwargs = {}
                for name in sig.parameters.keys():
                    if name in args_map and args_map[name] is not None:
                        kwargs[name] = args_map[name]
                return kwargs
            args_map = {
                "page": page,
                "livedoor_id": livedoor_id,
                "password": password,
            }
            await ld_prepare_captcha(**_callable_with(args_map, ld_prepare_captcha))
            await ld_submit_captcha(**_callable_with(args_map, ld_submit_captcha))
            return True
        except Exception:
            logger.warning("[LD-Recover] signup captcha tool failed; fallback to FS watcher", exc_info=True)

    # 2) フォールバック：/tmp を監視して人間が置く回答ファイルを待つ
    try:
        ans_dir = Path("/tmp/captcha_answers")
        ans_dir.mkdir(parents=True, exist_ok=True)
        base = Path(image_path).stem  # ld_create_captcha_yyyymmdd_hhmmss
        ans_file = ans_dir / f"{base}.txt"

        # ヒントファイル（UI/オペレータ向け）
        hint_file = ans_dir / f"{base}.readme"
        hint = (
            f"[LD-Recover] 手動CAPTCHA回答の受け付け\n"
            f"- 画像: {image_path}\n"
            f"- 回答ファイルにテキストで解答を保存してください: {ans_file}\n"
        )
        try:
            hint_file.write_text(hint, encoding="utf-8")
        except Exception:
            pass

        logger.info("[LD-Recover] waiting human answer file: %s (up to 180s)", ans_file)
        for _ in range(180):  # 最大180秒
            if ans_file.exists():
                try:
                    text = ans_file.read_text(encoding="utf-8").strip()
                except Exception:
                    text = ""
                if text:
                    logger.info("[LD-Recover] got manual captcha answer: %s (len=%d)", text, len(text))
                    await _fill_captcha_and_submit(page, text)
                    return True
                else:
                    logger.warning("[LD-Recover] answer file empty, keep waiting: %s", ans_file)
            await asyncio.sleep(1.0)
        logger.warning("[LD-Recover] timeout waiting for human captcha answer")
        return False
    except Exception:
        logger.warning("[LD-Recover] FS watcher fallback failed", exc_info=True)
        return False


# ─────────────────────────────────────────────
# メイン：ブログ作成→AtomPub APIキー取得
# ─────────────────────────────────────────────
def _extract_blog_id_from_url(url: str) -> str | None:
    try:
        m = _re.search(r"/blog/([^/]+)/", url)
        return m.group(1) if m else None
    except Exception:
        return None

async def _extract_public_url(page) -> str | None:
    # 設定/ウェルカムにある「ブログを見る」リンクを探す
    sels = [
        'a:has-text("ブログを見る")',
        'a[target="_blank"][href*="livedoor.blog"]',
        'a[href^="https://blog.livedoor.com/"]',
        'a[href^="https://blog.livedoor.jp/"]',
    ]
    for sel in sels:
        try:
            loc = page.locator(sel).first
            if await loc.count() > 0 and await loc.is_visible():
                href = await loc.get_attribute("href")
                if href:
                    # 相対URLで返るケースに備えて絶対化
                    abs_href = await page.evaluate("href => new URL(href, location.href).href", href)
                    return abs_href
        except Exception:
            pass
    return None


async def recover_atompub_key(page, livedoor_id: str | None, nickname: str, email: str, password: str, site,
                              desired_blog_id: str | None = None) -> dict:
    # ✅ ハンドオフ運用中はここを使わず、routes側の open_create_tab_for_handoff で別タブに渡す
    if HANDOFF_MODE:
        return {"success": False, "error": "recover_disabled_in_handoff_mode"}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("[LD-Recover] args: livedoor_id=%s desired_blog_id=%s email=%s", livedoor_id, desired_blog_id, email)

    async def _dump_error(prefix: str):
        html = await page.content()
        base_dir = os.getenv("LD_DUMP_DIR", "/var/www/ver12-ai-posting-tool/app/static/ld_dumps")
        try:
            Path(base_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        error_html = f"{base_dir}/{prefix}_{timestamp}.html"
        error_png = f"{base_dir}/{prefix}_{timestamp}.png"
        try:
            Path(error_html).write_text(html, encoding="utf-8")
        except Exception:
            pass
        try:
            await page.screenshot(path=error_png, full_page=True)
        except Exception:
            try:
                await page.screenshot(path=error_png)
            except Exception:
                pass
        return error_html, error_png

    try:
        # 1) ブログ作成ページへ
        logger.info("[LD-Recover] ブログ作成ページに遷移")
        await page.goto("https://livedoor.blogcms.jp/member/blog/create", wait_until="load")
        try:
            await page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass
        await _save_shot(page, "ld_create_landing")
        logger.info("[LD-Recover] create到達: url=%s title=%s", page.url, (await page.title()))

        # 2) タイトル生成（10案から必ず1つ選ぶ）
        try:
            desired_title = await generate_blog_title(site)
        except Exception:
            desired_title = "こつこつブログ"  # 最終フォールバック

        with pg_advisory_lock(LD_CREATE_LOCK_KEY):
            for attempt in range(1, LD_CREATE_MAX_RETRIES + 1):

                _human_sleep()
                ok_submit = await _set_title_and_submit(page, desired_title)
                if not ok_submit:
                    err_html, err_png = await _dump_error("ld_create_ui_notfound")
                    return {"success": False, "error": "タイトル/送信UIが見つからない", "html_path": err_html, "png_path": err_png}

                try:
                    await page.wait_for_load_state("networkidle", timeout=LD_CREATE_SUCCESS_TIMEOUT_MS)
                except Exception:
                    pass

                success, blog_id_from_url = await _wait_success_after_submit(page)
                if success:
                    break  # 成功

                # 失敗：ダンプ＆エラーメッセージ検査（s497 判定）
                await _save_shot(page, "ld_create_after_submit_failed_minimal")
                await _log_inline_errors(page)
                html = await page.content()
                if "ブログの作成に失敗しました" in html and "s497" in html:
                    logger.warning("[LD-Recover] s497 detected (attempt %d/%d)", attempt, LD_CREATE_MAX_RETRIES)
                    if attempt < LD_CREATE_MAX_RETRIES:
                        await asyncio.sleep(_rand_backoff())
                        await page.goto("https://livedoor.blogcms.jp/member/blog/create", wait_until="load")
                        try:
                            await page.wait_for_load_state("networkidle", timeout=15000)
                        except Exception:
                            pass
                        continue
                    err_html, err_png = await _dump_error("ld_atompub_create_fail_s497")
                    logger.error("[LD-Recover] ブログ作成に失敗（s497）")
                    return {"success": False, "error": "blog create rejected (s497)", "html_path": err_html, "png_path": err_png}
                else:
                    if attempt < LD_CREATE_MAX_RETRIES:
                        await asyncio.sleep(_rand_backoff())
                        await page.goto("https://livedoor.blogcms.jp/member/blog/create", wait_until="load")
                        try:
                            await page.wait_for_load_state("networkidle", timeout=15000)
                        except Exception:
                            pass
                        continue
                    err_html, err_png = await _dump_error("ld_atompub_create_fail_minimal")
                    logger.error("[LD-Recover] ブログ作成に失敗（createに留まる）")
                    return {"success": False, "error": "blog create failed", "html_path": err_html, "png_path": err_png}
 
        # ここまでで作成成功している想定（blog_id_from_url は None の可能性あり）
        # （以下は従来どおり：blog_id抽出→設定→APIキー取得）
        blog_id = blog_id_from_url
        if not blog_id:
            await page.goto("https://livedoor.blogcms.jp/member/", wait_until="load")
            try:
                await page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                pass
            # blog_idをメニューから推定
            blog_settings_selectors = [
                'a[title="ブログ設定"]',
                'a:has-text("ブログ設定")',
                'a[href^="/blog/"][href$="/config/"]',
                'a[href*="/config/"]'
            ]
            href = None
            for sel in blog_settings_selectors:
                try:
                    loc = page.locator(sel).first
                    if await loc.count() > 0:
                        try:
                            await loc.wait_for(state="visible", timeout=8000)
                        except Exception:
                            pass
                        href = await loc.get_attribute("href"); 
                        if href: break
                except Exception:
                    continue
            if href:
                try:
                    parts = href.split("/")
                    blog_id = parts[2] if len(parts) > 2 else None
                except Exception:
                    blog_id = None
            if not blog_id and "/blog/" in page.url:
                blog_id = page.url.split("/blog/")[1].split("/")[0]
            if not blog_id:
                err_html, err_png = await _dump_error("ld_atompub_member_fail")
                return {"success": False, "error": "member page missing blog link", "html_path": err_html, "png_path": err_png}

        # 設定→APIキー
        config_url = f"https://livedoor.blogcms.jp/blog/{blog_id}/config/"
        await page.goto(config_url, wait_until="load")
        try:
            await page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            pass

        api_nav_selectors = [
            'a.configIdxApi[title="API Keyの発行・確認"]',
            'a[title*="API Key"]',
            'a:has-text("API Key")',
            'a:has-text("API Keyの発行")',
            'a[href*="/api"]',
            'a:has-text("AtomPub")',
        ]
        api_link = None
        for sel in api_nav_selectors:
            try:
                loc = page.locator(sel).first
                if await loc.count() > 0:
                    api_link = loc; break
            except Exception:
                continue
        if not api_link:
            fr, sel = await _find_in_any_frame(page, api_nav_selectors, timeout_ms=8000)
            if fr: api_link = fr.locator(sel).first
        if not api_link:
            err_html, err_png = await _dump_error("ld_atompub_nav_fail")
            return {"success": False, "error": "api nav link not found", "html_path": err_html, "png_path": err_png}

        await _wait_enabled_and_click(page, api_link, timeout=8000, label_for_log="api-nav")
        try:
            await page.wait_for_load_state("load", timeout=10000)
        except Exception:
            pass
        if "member" in page.url:
            err_html, err_png = await _dump_error("ld_atompub_redirect_fail")
            return {"success": False, "error": "redirected to member", "html_path": err_html, "png_path": err_png}

        base_dir = os.getenv("LD_DUMP_DIR", "/var/www/ver12-ai-posting-tool/app/static/ld_dumps")
        try:
            Path(base_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        success_png = f"{base_dir}/ld_atompub_page_{timestamp}.png"
        try:
            await page.screenshot(path=success_png, full_page=True)
        except Exception:
            try: await page.screenshot(path=success_png)
            except Exception: pass

        # 発行
        await page.wait_for_selector('input#apiKeyIssue', timeout=12000)
        await _wait_enabled_and_click(page, page.locator('input#apiKeyIssue').first, timeout=6000, label_for_log="api-issue")
        await page.wait_for_selector('button:has-text("実行")', timeout=12000)
        await _wait_enabled_and_click(page, page.locator('button:has-text("実行")').first, timeout=6000, label_for_log="api-issue-confirm")

        async def _read_endpoint_and_key():
            endpoint_val = ""
            for sel in ['input.input-xxlarge[readonly]','input[readonly][name*="endpoint"]','input[readonly][id*="endpoint"]']:
                try:
                    await page.wait_for_selector(sel, timeout=8000)
                    endpoint_val = await page.locator(sel).first.input_value()
                    if endpoint_val: break
                except Exception:
                    continue
            await page.wait_for_selector('input#apiKey', timeout=15000)
            for _ in range(30):
                key_val = (await page.locator('input#apiKey').input_value()).strip()
                if key_val: return endpoint_val, key_val
                await asyncio.sleep(0.5)
            return endpoint_val, ""

        endpoint, api_key = await _read_endpoint_and_key()
        if not api_key:
            await page.reload(wait_until="load")
            try: await page.wait_for_load_state("networkidle", timeout=8000)
            except Exception: pass
            await page.wait_for_selector('input#apiKeyIssue', timeout=15000)
            await _wait_enabled_and_click(page, page.locator('input#apiKeyIssue').first, timeout=6000, label_for_log="api-issue-retry")
            await page.wait_for_selector('button:has-text("実行")', timeout=15000)
            await _wait_enabled_and_click(page, page.locator('button:has-text("実行")').first, timeout=6000, label_for_log="api-issue-confirm-retry")
            endpoint, api_key = await _read_endpoint_and_key()
        if not api_key:
            err_html, err_png = await _dump_error("ld_atompub_no_key")
            return {"success": False, "error": "api key empty", "html_path": err_html, "png_path": err_png}

        public_url = await _extract_public_url(page)
        if not public_url:
            try:
                await page.goto(f"https://livedoor.blogcms.jp/blog/{blog_id}/config/", wait_until="load")
                try: await page.wait_for_load_state("networkidle", timeout=6000)
                except Exception: pass
                public_url = await _extract_public_url(page)
            except Exception:
                pass

        return {
            "success": True,
            "blog_id": blog_id,
            "api_key": api_key,
            "endpoint": endpoint,
            "blog_title": desired_title,
            "public_url": public_url
        }

    except Exception as e:
        err_html, err_png = await _dump_error("ld_atompub_fail")
        logger.error("[LD-Recover] AtomPub処理エラー", exc_info=True)
        return {"success": False, "error": str(e), "html_path": err_html, "png_path": err_png}


# ─────────────────────────────────────────────
# NEW（タスクB）: 手動ハンドオフ用に“同一セッションの新タブ”を開く
# ─────────────────────────────────────────────
async def _open_create_tab_for_handoff(page, site, *, prefill_title: bool = True) -> dict:
    """
    既存のログイン済み Page から **同一ブラウザコンテキスト**で新しいタブを開き、
    /member/blog/create に遷移して人手作業を受け渡しやすい状態に整える。

    - 規約同意チェックがあればON
    - タイトル入力欄があれば、可能なら generate_blog_title(site) を事前入力（送信はしない）
    - blog_id 入力欄の有無を返す（人がIDを決めて入力する指標）
    """
    try:
        ctx = page.context
        newp = await ctx.new_page()
    except Exception as e:
        logger.error("[LD-Recover] failed to open new tab for handoff: %s", e)
        return {"ok": False, "error": "cannot_open_new_tab"}

    try:
        await newp.goto("https://livedoor.blogcms.jp/member/blog/create", wait_until="load")
        try:
            await newp.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass
        await _maybe_close_overlays(newp)
        await _maybe_accept_terms(newp)

        prefilled = None
        if prefill_title:
            try:
                desired_title = await generate_blog_title(site)
                # タイトル欄が見つかった場合のみ入力（送信はしない）
                title_sels = ['#blogTitle', 'input[name="title"]']
                for sel in title_sels:
                    try:
                        await newp.wait_for_selector(sel, state="attached", timeout=4000)
                        inp = newp.locator(sel).first
                        if await inp.count() > 0:
                            try:
                                await inp.fill("")
                            except Exception:
                                try:
                                    await inp.click(); await inp.press("Control+A"); await inp.press("Delete")
                                except Exception:
                                    pass
                            await inp.fill(desired_title[:48])
                            prefilled = desired_title[:48]
                            logger.info("[LD-Recover] prefilled blog title for handoff: %s", prefilled)
                            break
                    except Exception:
                        continue
            except Exception:
                pass

        has_id_box = await _has_blog_id_input(newp)
        # 使いやすいよう blog_id 欄にフォーカスしておく（あれば）
        if has_id_box:
            for sel in ['#blogId', 'input[name="blog_id"]', 'input[name="livedoor_blog_id"]', 'input[name="blogId"]', 'input#livedoor_blog_id', 'input[name="sub"]']:
                try:
                    loc = newp.locator(sel).first
                    if await loc.count() > 0:
                        try:
                            await loc.focus()
                        except Exception:
                            pass
                        break
                except Exception:
                    continue

        # 画面の安定化
        try:
            await newp.wait_for_load_state("networkidle", timeout=3000)
        except Exception:
            pass

        return {
            "ok": True,
            "url": newp.url,
            "prefilled_title": prefilled,
            "has_blog_id_box": has_id_box,
        }
    except Exception as e:
        logger.error("[LD-Recover] open handoff tab error: %s", e, exc_info=True)
        return {"ok": False, "error": "handoff_navigation_failed"}


def open_create_tab_for_handoff(session_id: str, site, *, prefill_title: bool = True) -> dict:
    """
    同期ラッパー：pwctl でセッションから Page を取得し、
    同一セッションの新タブを準備して **人手作業にバトン**を渡す。

    戻り値例:
      {"ok": True, "url": ".../member/blog/create", "prefilled_title": "…", "has_blog_id_box": True}

    注意：ここでは **送信を行わない**。以降の操作（ID入力/送信/CAPTCHA対応）は人手前提。
    """
    if pwctl is None:
        return {"ok": False, "error": "pwctl_unavailable"}
    # 既存セッションの page を取得（無ければ revive）
    page = pwctl.run(pwctl.get_page(session_id))
    if page is None:
        page = pwctl.run(pwctl.revive(session_id))
        if page is None:
            return {"ok": False, "error": f"session_not_found:{session_id}"}
    # 念のため最新の storage_state を保存してから実行（別ワーカーでも状態共有）
    try:
        pwctl.run(pwctl.save_storage_state(session_id))
    except Exception:
        pass
    # 実処理
    result = pwctl.run(_open_create_tab_for_handoff(page, site, prefill_title=prefill_title))
    # 戻りに合わせて state も再保存（新タブでクッキーが増えた場合に備える）
    try:
        pwctl.run(pwctl.save_storage_state(session_id))
    except Exception:
        pass
    return result
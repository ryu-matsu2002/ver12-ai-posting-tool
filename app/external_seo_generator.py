# app/external_seo_generator.py

import random
import logging
from typing import List, Tuple, Set, Iterable, Optional
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin, urlparse

import requests
from flask import current_app
from xml.etree import ElementTree as ET

from concurrent.futures import ThreadPoolExecutor, as_completed  # ★ 並列化
import html as _html  # ★ タイトルのエスケープ用
import re as _re      # ★ タイトル抽出にも利用（既存と衝突しないよう別名に合わせる）

from app import db
from app.models import Site, Keyword, Article, ExternalArticleSchedule
from app.google_client import (
    fetch_top_queries_for_site,   # impressions降順のquery上位取得（40件）
    fetch_top_pages_for_site,     # impressions降順のpage上位取得（任意件）
)
from .article_generator import _chat, _compose_body, TOKENS, TEMP, _generate  # ★ _generate を流用

# 追加: タイトルのフォールバック生成用
def _fallback_title_from_keyword(kw: str) -> str:
    """タイトルが空のときに必ず返すフォールバック"""
    kw = (kw or "").strip()
    if not kw:
        return "自動生成記事"
    base = kw[:60]
    tails = ["の完全ガイド", "の基礎知識", "の始め方", "のポイント", "で失敗しないコツ"]
    return f"{base}{tails[sum(map(ord, base)) % len(tails)]}"

def _safe_title(proposed: Optional[str], kw: str) -> str:
    """候補が空/空白ならキーワードからフォールバックを生成し、120文字に収める"""
    t = (proposed or "").strip()
    if not t:
        t = _fallback_title_from_keyword(kw)
    return t[:120]

# タイムゾーン設定
JST = timezone(timedelta(hours=9))

# ===============================
# 固定プロンプト（タイトル / 本文）
# ===============================
TITLE_PROMPT = """あなたはSEOとコンテンツマーケティングの専門家です。

入力されたキーワードを使って
WEBサイトのQ＆A記事コンテンツに使用する「Q＆A記事タイトル」を「1個」考えてください。

Q＆A記事タイトルには必ず入力されたキーワードを全て使ってください
キーワードの順番は入れ替えないでください
最後は「？」で締めてください


###具体例###

「由布院 観光 おすすめ スポット」というキーワードに対する出力文
↓↓↓
由布院観光でカップルに人気のおすすめスポットは？

「由布院 観光 モデルコース」というキーワードに対する出力文
↓↓↓
由布院観光のモデルコースでおすすめの一日プランは？
"""

BODY_PROMPT = """あなたはSEOとコンテンツマーケティングの専門ライターです。

これから、**Q&A形式**の記事本文を作成してもらいます。
必ず、以下の【執筆ルール】と【出力条件】に**厳密に従って**ください。

---

### ✅【執筆ルール】

#### 1. 文章構成（記事全体の流れ）
- 記事は「問題提起」→「読者への共感」→「解決策の提示」の順番で構成してください。
- もしくは「結論ファースト」→「読者への共感」→「体験談」や「レビュー風」→「権威性（資格・実績）や専門性」の順番で構成してください。
- 記事タイトルの素になった検索キーワードの出現率を7%前後にする

#### 2. 読者視点
- 読者は、Q&Aタイトルに悩んで検索してきた「1人のユーザー」です。
- 必ず、**読者が本当に知りたいこと**をわかりやすく伝えてください。
- 呼びかけは「あなた」に統一してください（「皆さん」などの複数形はNGです）。
- 語り口は「親友に話すような親しみを込めた敬語」で書いてください。
- 検索意図に100%応えるように書いてください

#### 3. 文章スタイル
- **段落内で改行しないでください**（1文のまとまり＝1つの文章ブロックにしてください）。
- 1つの文章ブロック（いわゆる「文章の島」）は**1～3行以内**に収めてください。
- 各文章ブロックの間は、**2行分空けて**ください（ダブル改行）。

#### 4. 文字数
- 記事の本文は、必ず**2,500～3,500文字**の範囲で書いてください。

#### 5. 小見出し
- hタグ（h2, h3）を使って、内容を適切に整理してください。

#### 6. ほかのサイトへのリンクについて
- アンカーテキストはジャンルのキーワードに合わせてリンクしてください
- 何度でもほかのサイトへリンクしてOKです
- 商品やサービスを紹介する場合は「不自然な押し売り」にならないように、**ごく自然に紹介**してください。

---

### ✅【出力条件】

- 記事冒頭には、**Q&Aタイトルを表示しないでください**（タイトルなしで本文からスタートしてください）。
- 余計な前置きや挨拶も入れないでください。例：「承知しました」「この記事では～」など不要です。
- そのままコピペしたいので、必ず、すぐに本文を書き始めてください。

---

### 🔥【特に重要なポイントまとめ】

- 「あなた」呼びで統一
- 「親しみを込めた敬語」で
- Q＆A記事タイトルの素になったキーワードの出現頻度は7%前後で。
- 段落中に改行禁止、文章の島は1～3行
- 各島は**2行空け**
- タイトル表示なし、本文から即スタート
"""

# ===============================
# 投稿スロット（JST 10:00〜21:59）
# ===============================
RANDOM_MINUTE_CHOICES = [3, 7, 11, 13, 17, 19, 23, 27, 31, 37, 41, 43, 47, 53]
RANDOM_SECOND_CHOICES = [5, 12, 17, 23, 35, 42, 49]


def _random_minutes(n: int) -> List[int]:
    """重複なく n 個の分を選ぶ。候補が足りない場合はプールを拡張"""
    if n <= len(RANDOM_MINUTE_CHOICES):
        return random.sample(RANDOM_MINUTE_CHOICES, n)
    pool = RANDOM_MINUTE_CHOICES[:]
    while len(pool) < n:
        for m in RANDOM_MINUTE_CHOICES:
            if len(pool) >= n:
                break
            if m not in pool:
                pool.append(m)
    return random.sample(pool, n)


def _daily_slots_jst(per_day: int) -> List[Tuple[int, int]]:
    """
    1日の投稿スロット（JST）を返す。
    10:00〜21:59 の各“時”をベースに、分は“切りの良くない分”からランダム。
    ※ 同一の“時”は1本のみ → 最低1時間以上間隔を担保
    """
    base_hours = list(range(10, 22))  # 10..21 の12時間
    hours = sorted(random.sample(base_hours, per_day))  # 例: 10本/日 → 10時間を抽選
    minutes = _random_minutes(per_day)
    return list(zip(hours, minutes))


def _ensure_http_url(u: str) -> str:
    return u.strip()

#
# ====== 追加：リンク先タイトル取得ユーティリティ ======
#
_ANCHOR_UA = "ai-posting-tool/1.0 (+title-fetch)"

def _extract_html_title(text: str) -> Optional[str]:
    """HTML文字列から <meta property='og:title'> もしくは <title> を抽出"""
    if not text:
        return None
    # og:title 優先
    m = _re.search(r'<meta[^>]+property=["\']og:title["\'][^>]*content=["\']([^"\']+)["\']', text, flags=_re.I)
    if m and m.group(1).strip():
        return m.group(1).strip()
    # 一般的な <title>
    m = _re.search(r'<title[^>]*>(.*?)</title\s*>', text, flags=_re.I | _re.S)
    if m:
        # 改行・余白の整理
        t = _re.sub(r'\s+', ' ', (m.group(1) or '').strip())
        return t or None
    return None

def _fallback_anchor_from_url(u: str) -> str:
    """
    タイトルが取れない場合のフォールバック：
      1) パス末尾のスラッグっぽい部分をスペース区切りタイトル化
      2) それでもダメならドメイン
      3) 最後にURL全体
    """
    try:
        pu = urlparse(u)
        # スラッグ候補
        path = (pu.path or "").rstrip("/")
        slug = path.split("/")[-1] if path else ""
        slug = _re.sub(r'[-_]+', ' ', slug).strip()
        slug = slug.title() if slug else ""
        if slug:
            return slug[:120]
        if pu.netloc:
            return pu.netloc
    except Exception:
        pass
    return u

def _clean_anchor_text(url: str, title: str) -> str:
    """
    取得したページタイトルから「 - サイト名」「｜サイト名」「| サイト名」「：サイト名」などの
    末尾ブランド表記を除去して、アンカーに使う“記事タイトルだけ”を返す。
    - 区切りは「空白付きハイフン」「パイプ」「全角縦棒」「コロン」を対象
    - 左右の空白を除去し、最も“長い”セグメントを採用（多くのサイトで記事タイトルが最長）
    """
    t = (title or "").strip()
    if not t:
        return t
    # 区切りで末尾のサイト名を外す（例: "記事タイトル - サイト名" / "サイト名 | 記事タイトル" など）
    parts = _re.split(r'(?:\s-\s|\s–\s|\s—\s|\s\|\s|｜|：|:|»)', t)
    parts = [p.strip() for p in parts if p and p.strip()]
    if len(parts) >= 2:
        # 記事タイトルが最も長くなるケースが多いので長さ優先で選ぶ
        t = max(parts, key=lambda s: len(s))
    return t[:120]

def _fetch_page_title(u: str, timeout: int = 8) -> Optional[str]:
    """URLへHTTP GETしてタイトルを取得（短時間タイムアウト/軽量UA）"""
    try:
        # http/https のみ対象（mailto:, javascript: 等は除外）
        pu = urlparse(u)
        if pu.scheme not in ("http", "https"):
            return None
        r = requests.get(u, timeout=timeout, headers={"User-Agent": _ANCHOR_UA})
        if r.status_code != 200:
            return None
        # HTML以外は除外
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "text/html" not in ctype:
            return None
        # エンコーディングはrequestsが推定する、失敗時はtextが空になる可能性あり
        return _extract_html_title(r.text or "")
    except Exception:
        return None

def _prefetch_anchor_texts(urls: List[str], max_workers: int = 8) -> dict:
    """
    渡されたURLのページタイトルを並列で事前取得して dict で返す。
    取得失敗時は dict に入れない（呼び出し側でフォールバック）。
    """
    anchors: dict[str, str] = {}
    # ユニーク化して負荷を抑制
    uniq = list({u for u in urls if isinstance(u, str)})
    def _job(u: str):
        t = _fetch_page_title(u)
        if t and t.strip():
            anchors[u] = _clean_anchor_text(u, t.strip())
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_job, u) for u in uniq]
            for _ in as_completed(futs):
                pass
    except Exception:
        logging.exception("[external_seo] タイトル事前取得中に例外")
    return anchors


def _base_and_topic(site: Site) -> Tuple[str, str]:
    base = site.url.rstrip("/")
    # topic の末尾に必ずスラッシュを付ける
    return base, f"{base}/topic/"


# ===============================
# URL収集（固定5リンク & ランダム候補）
# ===============================
def _build_fixed_links(site: Site) -> List[str]:
    """
    固定5リンク：
      - base（サイトTOP）
      - base/topic
      - GSC page impressions 上位3件
    """
    base, sales = _base_and_topic(site)
    fixed = [base, sales]

    try:
        top_pages = fetch_top_pages_for_site(site, days=28, limit=3) or []
        # 返り値が dict の配列でも str 配列でも吸収
        def _page_to_url(p):
            if isinstance(p, str):
                return p
            if isinstance(p, dict):
                return p.get("page") or p.get("url")
            return None

        pages = []
        for p in top_pages:
            url = _page_to_url(p)
            if url and isinstance(url, str):
                pages.append(url.strip())
        # base, sales と被ったら除外（補充はしない）
        for u in pages:
            if u not in fixed:
                fixed.append(u)
    except Exception as e:
        logging.warning(f"[external_seo] 固定リンク: GSC上位page取得失敗: {e}")

    # 固定は最大5本に丸める（不足はそのまま）
    fixed = fixed[:5]
    if len(fixed) < 5:
        logging.warning(f"[external_seo] 固定リンクが {len(fixed)} 件しか用意できません（想定:5件）: {fixed}")
    return fixed


def _fetch_xml(url: str, timeout: int = 10) -> Optional[ET.Element]:
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "ai-posting-tool/1.0"})
        if resp.status_code != 200 or not resp.content:
            return None
        return ET.fromstring(resp.content)  # XML宣言やcharset差異にも強めに
    except Exception:
        return None


def _extract_loc_values(root: ET.Element) -> Iterable[str]:
    # sitemapindex -> sitemap -> loc
    for sm in root.findall(".//{*}sitemap"):
        loc = sm.find("{*}loc")
        if loc is not None and loc.text:
            yield loc.text.strip()
    # urlset -> url -> loc
    for u in root.findall(".//{*}url"):
        loc = u.find("{*}loc")
        if loc is not None and loc.text:
            yield loc.text.strip()


def _collect_all_site_urls(site: Site, max_nested: int = 50, max_total: int = 5000) -> Set[str]:
    """
    なるべく多くの内部URLを収集する。
    1) sitemap_index.xml（あれば）→各sitemap
    2) sitemap.xml
    3) wp-sitemap.xml（WP標準）
    4) それでも不足なら GSC page上位（最大1000）
    5) 最後の保険で WP REST を叩く
    """
    base = site.url.rstrip("/")
    candidates = set()

    sitemap_entries = [
        f"{base}/sitemap_index.xml",
        f"{base}/sitemap.xml",
        f"{base}/wp-sitemap.xml",
    ]

    for sm_url in sitemap_entries:
        root = _fetch_xml(sm_url)
        if not root:
            continue

        locs = list(_extract_loc_values(root))
        if any(tag in root.tag for tag in ("sitemapindex", "index")) and locs:
            # nested sitemaps
            for child_url in locs[:max_nested]:
                cr = _fetch_xml(child_url)
                if not cr:
                    continue
                for u in _extract_loc_values(cr):
                    if u.startswith(base):
                        candidates.add(_ensure_http_url(u))
                        if len(candidates) >= max_total:
                            break
                if len(candidates) >= max_total:
                    break
        else:
            for u in locs:
                if u.startswith(base):
                    candidates.add(_ensure_http_url(u))
                    if len(candidates) >= max_total:
                        break

        if len(candidates) >= max_total:
            break

    # GSC page で補完
    if len(candidates) < 50:
        try:
            gsc_pages = fetch_top_pages_for_site(site, days=180, limit=1000) or []
            def _page_to_url(p):
                if isinstance(p, str):
                    return p
                if isinstance(p, dict):
                    return p.get("page") or p.get("url")
                return None
            for p in gsc_pages:
                url = _page_to_url(p)
                if url and isinstance(url, str) and url.startswith(base):
                    candidates.add(url.strip())
        except Exception as e:
            logging.warning(f"[external_seo] GSC page補完に失敗: {e}")

    # WordPress REST（最後の保険）
    if len(candidates) < 50:
        try:
            page = 1
            while len(candidates) < 200:
                api = f"{base}/wp-json/wp/v2/posts?per_page=100&page={page}"
                r = requests.get(api, timeout=8, headers={"User-Agent": "ai-posting-tool/1.0"})
                if r.status_code != 200:
                    break
                arr = r.json()
                if not isinstance(arr, list) or not arr:
                    break
                for it in arr:
                    link = it.get("link")
                    if isinstance(link, str) and link.startswith(base):
                        candidates.add(link.strip())
                page += 1
        except Exception:
            pass

    return candidates


def _pick_random_unique(urls: Iterable[str], n: int, excluded: Iterable[str] = ()) -> List[str]:
    pool = list({u for u in urls if isinstance(u, str)})
    ex = set(excluded or [])
    pool = [u for u in pool if u not in ex]
    if len(pool) < n:
        logging.error(f"[external_seo] ランダム用URLが不足しています（必要:{n}, 取得:{len(pool)}）。要件を満たせません。")
        return random.sample(pool, len(pool)) if pool else []
    return random.sample(pool, n)


# ===============================
# 外部SEO：並列生成 + スケジューリング
# ===============================
def generate_and_schedule_external_articles(
    user_id: int,
    site_id: int,
    blog_account_id: int,
    count: int = 100,
    per_day: int = 10,
    start_day_jst: Optional[datetime] = None,
) -> int:
    """
    外部SEO記事を一括生成し、1日 per_day 本、JST 10:00-21:59 のランダム分でスケジュール登録。
    実装（高速化版）:
      1) まず 100 本分の Article(status="pending") と ExternalArticleSchedule を一気に作成
      2) その後、本文生成を ThreadPoolExecutor(max_workers=4) で並列実行し、
         本文末尾にその記事専用リンクを追記して Article.status="done" にする
      3) 投稿は既存の _run_external_post_job が拾って行う
    """
    app = current_app._get_current_object()
    site = Site.query.get(site_id)
    assert site, "Site not found"

    # === キーワード上位40件（impressions降順） ===
    try:
        top40 = fetch_top_queries_for_site(site, days=28, limit=40) or []
        def _to_query(d):
            if isinstance(d, str):
                return d
            if isinstance(d, dict):
                return d.get("query")
            return None

        kw40 = []
        for d in top40:
            q = _to_query(d)
            if q and isinstance(q, str):
                kw40.append(q.strip())
        kw40 = kw40[:40]
    except Exception as e:
        logging.exception(f"[external_seo] GSCクエリ取得に失敗: {e}")
        kw40 = []

    # 補完（不足時）：既存Keywordから新しい順
    if len(kw40) < 40:
        need = 40 - len(kw40)
        extra = (
            Keyword.query
            .filter(Keyword.site_id == site_id)
            .order_by(Keyword.id.desc())
            .limit(need)
            .all()
        )
        kw40 += [k.keyword for k in extra if isinstance(k.keyword, str)]
        kw40 = kw40[:40]

    if len(kw40) < 40:
        raise RuntimeError(f"[external_seo] 上位キーワードが40件に満たないため中断しました（取得:{len(kw40)}件）。")

    # === 100本の配分 ===
    dist: List[Tuple[str, int]] = [(kw, 3) for kw in kw40[:20]] + [(kw, 2) for kw in kw40[20:40]]
    gen_queue: List[str] = [kw for (kw, n) in dist for _ in range(n)]
    assert len(gen_queue) == 100, f"配分エラー: {len(gen_queue)} != 100"

    # === リンク計画 ===
    fixed5 = _build_fixed_links(site)  # [base, base/sales, top_page1, top_page2, top_page3]
    if len(fixed5) < 2:
        raise RuntimeError("[external_seo] 固定リンク（base, base/sales）が確保できませんでした。サイトURLをご確認ください。")

    fixed50 = []
    for u in fixed5:
        fixed50 += [u] * 10
    fixed50 = fixed50[:50]  # 念のため丸め

    # ランダム50（固定5は除外、同一URL再利用不可）
    all_urls = _collect_all_site_urls(site)
    random50 = _pick_random_unique(all_urls, 50, excluded=set(fixed5))
    if len(random50) < 50:
        raise RuntimeError(f"[external_seo] ランダムリンクが50件に満たないため中断しました（{len(random50)}件）。サイトのURL収集設定をご確認ください。")

    link_plan: List[str] = fixed50 + random50
    assert len(link_plan) == 100

    # ★ 新規：リンク先のページタイトルを事前取得（アンカーテキスト用）
    anchor_map = _prefetch_anchor_texts(link_plan)

    # 記事とリンクの対応をランダム化（均等性担保のため記事側をシャッフル）
    random.shuffle(gen_queue)

    # === スケジュール開始日（翌日） ===
    base_jst = datetime.now(JST).replace(hour=0, minute=0, second=0, microsecond=0)
    start_day_jst = (start_day_jst or base_jst) + timedelta(days=1)

    created_cnt = 0
    day_offset = 0
    idx = 0

    # 1) まず枠を作成（Article: pending / ExternalArticleSchedule: pending）
    article_ids: List[int] = []
    per_article_link: List[Tuple[str, str]] = []  # (url, anchor_text)

    with app.app_context():
        try:
            while idx < len(gen_queue):
                # その日のスロット（1時間1本／分は“切りの良くない分”）
                slots = _daily_slots_jst(per_day)
                slots.sort()

                for h, m in slots:
                    if idx >= len(gen_queue):
                        break

                    kw_str = gen_queue[idx]
                    link = link_plan[idx]

                    # Keyword 取得 or 生成（external ソース）
                    kobj = (
                        Keyword.query
                        .filter_by(user_id=user_id, site_id=site_id, keyword=kw_str)
                        .first()
                    )
                    if not kobj:
                        kobj = Keyword(
                            user_id=user_id,
                            site_id=site_id,
                            keyword=kw_str,
                            created_at=datetime.now(JST),
                            source="external",
                            status="pending",
                            used=False,
                        )
                        db.session.add(kobj)
                        db.session.flush()

                    # JST → UTC naive に変換（秒はばらす）
                    when_jst = (start_day_jst + timedelta(days=day_offset)).replace(
                        hour=h,
                        minute=m,
                        second=random.choice(RANDOM_SECOND_CHOICES),
                        microsecond=0,
                    )
                    when_utc = when_jst.astimezone(timezone.utc)
                    when_naive = when_utc.replace(tzinfo=None)

                    # ★ 修正: title に必ず非空の仮タイトルを入れる
                    placeholder_title = _safe_title(None, kw_str)

                    art = Article(
                        keyword=kw_str,
                        title=placeholder_title,  # ← プレースホルダを実際に設定
                        body="",
                        user_id=user_id,
                        site_id=site_id,
                        status="pending",
                        progress=0,
                        source="external",
                        scheduled_at=when_naive,
                    )
                    db.session.add(art)
                    db.session.flush()

                    sched = ExternalArticleSchedule(
                        blog_account_id=blog_account_id,
                        article_id=art.id,
                        keyword_id=kobj.id,
                        scheduled_date=when_naive,
                        status="pending",
                    )
                    db.session.add(sched)

                    article_ids.append(art.id)
                    # ★ URLに対応するアンカーテキスト（タイトル）を用意
                    anchor_txt = anchor_map.get(link) or _fallback_anchor_from_url(link)
                    # 念のためここでもクリーンアップ（フォールバック時は影響なし）
                    anchor_txt = _clean_anchor_text(link, anchor_txt)
                    # HTML表示用にエスケープしておく（挿入箇所で二重エスケープしないようここで）
                    safe_anchor_txt = _html.escape(anchor_txt, quote=True)[:120]

                    per_article_link.append((link, safe_anchor_txt))

                    created_cnt += 1
                    idx += 1

                day_offset += 1

            db.session.commit()
            logging.info(f"[external_seo] 枠作成完了: {created_cnt} 件（site_id={site_id}）")

        except Exception as e:
            db.session.rollback()
            logging.exception(f"[external_seo] 枠作成中にエラー: {e}")
            raise


        # 本文の「中間」にリンクブロックを差し込む
    def _insert_link_mid(html: str, link_url: str, anchor_text: str) -> str:
        """
        「関連情報はこちら：{アンカーテキスト}」の形で挿入。
        別タブ遷移（target=_blank）、安全のため rel を付与。
        """
        safe_url = _html.escape(link_url, quote=True)
        # anchor_text は事前にエスケープ済み
        snippet = (
            f"<p>関連情報はこちら："
            f"<a href='{safe_url}' target='_blank' rel='nofollow noopener noreferrer'>{anchor_text}</a>"
            f"</p>"
        )

        if not html:
            return snippet

        # 1) </p> の直後に入れる：段落を保てるので最優先
        closings = [m.end() for m in _re.finditer(r'</p\s*>', html, flags=_re.I)]
        if closings:
            mid = max(0, len(closings) // 2 - 1)  # 真ん中の段落終端の直後
            pos = closings[mid]
            return html[:pos] + snippet + html[pos:]

        # 2) 段落が無ければ、ダブル改行でブロック分割して中間に入れる
        parts = _re.split(r'\n{2,}', html)
        if len(parts) >= 2:
            mid = len(parts) // 2
            return '\n\n'.join(parts[:mid]) + '\n\n' + snippet + '\n\n' + '\n\n'.join(parts[mid:])

        # 3) それも厳しければ末尾に念のため追加（フォールバック）
        return html + '\n\n' + snippet

    # 2) 並列で本文生成 → 本文末尾にリンク追記 → done
    # external_seo_generator.py 内 _gen_and_append より

    from app.article_generator import _unique_title

    def _gen_and_append(aid: int, link_url: str, anchor_text: str):
        _generate(app, aid, TITLE_PROMPT, BODY_PROMPT,
                  format="html", self_review=False, user_id=user_id)

        with app.app_context():
            art = Article.query.get(aid)
            if not art:
                return

            # 🔧 タイトルが空ならフォールバック
            if not art.title or not art.title.strip():
                art.title = _fallback_title_from_keyword(art.keyword or "")

            # 🔧 類似タイトルがある場合はユニーク化
            art.title = _unique_title(art.keyword, TITLE_PROMPT)

            # 本文に「タイトル付きリンク」を差し込み（別タブ）
            art.body = _insert_link_mid(art.body or "", link_url, anchor_text)

            if art.status not in ("done", "gen"):
                art.status = "done"
            art.progress = 100
            art.updated_at = datetime.utcnow()
            db.session.commit()

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for aid, pair in zip(article_ids, per_article_link):
                url, anchor = pair
                futures.append(executor.submit(_gen_and_append, aid, url, anchor))
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    logging.exception(f"[external_seo] 並列生成で例外: {e}")
    except Exception:
        # 並列実行自体が落ちても、作成済み枠は残る（再実行や再生成の余地を残す）
        logging.exception("[external_seo] 並列生成フェーズでエラー")

    logging.info(f"[external_seo] 生成フェーズ完了（site_id={site_id}, total={created_cnt}）")
    return created_cnt

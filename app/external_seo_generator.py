import random
import logging
import threading
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import current_app
from app import db
from app.models import Article, Site
from app.google_client import fetch_search_queries
from .article_generator import _chat, clean_gpt_output, _compose_body, TOKENS, TEMP

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
# スケジュール生成（1日10記事固定・時間ランダム）
# ===============================
def _generate_slots_external(app, site_id: int, n: int) -> list[datetime]:
    POST_HOURS = list(range(10, 22))  # 10時〜21時
    slots = []
    day = datetime.now(JST).date() + timedelta(days=1)  # 翌日から
    while len(slots) < n:
        # その日の10本分の時間をランダム選択
        hours_for_day = random.sample(POST_HOURS, 10)
        for hour in sorted(hours_for_day):
            minute = random.randint(0, 59)  # 分もランダム化
            local = datetime.combine(day, datetime.min.time(), tzinfo=JST).replace(hour=hour, minute=minute)
            slots.append(local.astimezone(timezone.utc))
            if len(slots) >= n:
                break
        day += timedelta(days=1)
    return slots


# ===============================
# ランダムリンク選択
# ===============================
def choose_random_link(site_id: int) -> str:
    """
    .comリンク / 固定セールスページ / 上位10記事URL からランダム選択
    """
    site = Site.query.get(site_id)
    base_url = site.url.rstrip("/")
    sales_url = f"{base_url}/sales"

    # GSC上位10記事（URL）取得
    top_articles = []
    try:
        queries = fetch_search_queries(site_id, days=28, limit=10, by_page=True)
        top_articles = [q["page"] for q in queries if q.get("page")]
    except Exception as e:
        logging.warning(f"GSC上位記事取得失敗: {e}")

    link_pool = [base_url, sales_url] + top_articles
    return random.choice(link_pool)

# ===============================
# 外部SEO記事生成メイン関数
# ===============================
from app.models import ExternalArticleSchedule, ExternalBlogAccount

def generate_external_seo_articles(user_id: int, site_id: int, blog_id: int, account: ExternalBlogAccount):
    app = current_app._get_current_object()

    # 1. GSC上位キーワード取得（1件だけ）
    queries = fetch_search_queries(site_id, days=28, limit=1)
    keywords = [q["query"] for q in queries] or ["テストキーワード"]

    # 2. テスト用: 1記事だけ
    total_articles = 1
    from datetime import datetime, timedelta, timezone
    JST = timezone(timedelta(hours=9))
    scheduled_time = (datetime.now(JST) + timedelta(minutes=2)).astimezone(timezone.utc)

    def _bg():
        with app.app_context():
            schedules = []
            try:
                kw = keywords[0]
                title_prompt = f"{TITLE_PROMPT}\n\nキーワード: {kw}"
                title = _chat(
                    [{"role": "system", "content": "あなたはSEOに強い日本語ライターです。"},
                     {"role": "user", "content": title_prompt}],
                    TOKENS["title"], TEMP["title"], user_id=user_id
                )

                body = _compose_body(
                    kw=kw,
                    pt=BODY_PROMPT,
                    format="html",
                    self_review=False,
                    user_id=user_id
                )

                link = choose_random_link(site_id)
                body += f"\n\n<a href='{link}' target='_blank'>{link}</a>"

                # 記事保存
                art = Article(
                    keyword=kw,
                    title=title,
                    body=body,
                    user_id=user_id,
                    site_id=site_id,
                    status="done",
                    progress=100,
                    scheduled_at=scheduled_time,
                    source="external_seo"
                )
                db.session.add(art)
                db.session.flush()

                # スケジュール保存
                sched = ExternalArticleSchedule(
                    blog_account_id=account.id,
                    keyword_id=None,
                    scheduled_date=scheduled_time,
                    status="pending"
                )
                schedules.append(sched)

                if schedules:
                    db.session.bulk_save_objects(schedules)
                db.session.commit()

            except Exception as e:
                db.session.rollback()
                logging.exception(f"[外部SEO記事生成テスト中エラー] site_id={site_id}, error={e}")

    threading.Thread(target=_bg, daemon=True).start()

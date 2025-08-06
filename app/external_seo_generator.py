import random
import logging
import threading
from app.models import Site
from app.models import Keyword
from datetime import datetime, timedelta, timezone
from flask import current_app
from app import db
from app.models import Article
from app.google_client import fetch_search_queries_for_site
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
# ランダムリンク選択（修正版）
# ===============================
def choose_random_link(site_id: int) -> str:
    site = Site.query.get(site_id)
    base_url = site.url.rstrip("/")
    sales_url = f"{base_url}/sales"

    top_articles = []
    try:
        # 不要な by_page を削除、limit → row_limit に変更
        queries = fetch_search_queries_for_site(site, days=28, row_limit=10)
        if queries and isinstance(queries[0], dict):
            top_articles = [q.get("page") for q in queries if q.get("page")]
    except Exception as e:
        logging.warning(f"GSC上位記事取得失敗: {e}")

    link_pool = [base_url, sales_url] + top_articles
    return random.choice(link_pool)

# ===============================
# 外部SEO記事生成（テスト用）
# ===============================
from app.models import ExternalArticleSchedule, ExternalBlogAccount

def generate_external_seo_articles(user_id: int, site_id: int, blog_id: int, account: ExternalBlogAccount):
    app = current_app._get_current_object()

    # DetachedInstanceError対策で事前にIDだけ退避
    blog_account_id = account.id

    # Siteオブジェクト取得
    site_obj = Site.query.get(site_id)

    # 1. GSC上位キーワード取得（1件だけ）
    try:
        queries = fetch_search_queries_for_site(site_obj, days=28, row_limit=1)
        if queries and isinstance(queries[0], dict):
            keywords = [q.get("query") for q in queries if q.get("query")]
        else:
            keywords = list(queries) if queries else []
    except Exception as e:
        logging.warning(f"[外部SEOテスト] GSCキーワード取得失敗: {e}")
        keywords = []

    keywords = keywords or ["テストキーワード"]

    # 2. テスト用: 1記事だけ、2分後に投稿
    scheduled_time = (datetime.now(JST) + timedelta(minutes=2)).astimezone(timezone.utc)

    def _bg():
        with app.app_context():
            schedules = []
            try:
                kw = keywords[0]

                # タイトル生成
                title_prompt = f"{TITLE_PROMPT}\n\nキーワード: {kw}"
                title = _chat(
                    [{"role": "system", "content": "あなたはSEOに強い日本語ライターです。"},
                     {"role": "user", "content": title_prompt}],
                    TOKENS["title"], TEMP["title"], user_id=user_id
                )

                # 本文生成
                body = _compose_body(
                    kw=kw,
                    pt=BODY_PROMPT,
                    format="html",
                    self_review=False,
                    user_id=user_id
                )

                # リンク挿入
                link = choose_random_link(site_id)
                body += f"\n\n<a href='{link}' target='_blank'>{link}</a>"

                # 記事保存前にキーワードをKeywordテーブルに保存 or 取得
                keyword_obj = Keyword.query.filter_by(
                    user_id=user_id,
                    site_id=site_id,
                    keyword=kw
                ).first()
                if not keyword_obj:
                    keyword_obj = Keyword(
                        user_id=user_id,
                        site_id=site_id,
                        keyword=kw,
                        created_at=datetime.now(JST)
                    )
                    db.session.add(keyword_obj)
                    db.session.flush()


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

                # keyword_id にはKeywordテーブルのIDを入れる
                # keyword_id にはKeywordテーブルのIDを入れる
                sched = ExternalArticleSchedule(
                    blog_account_id=blog_account_id,
                    keyword_id=keyword_obj.id,
                    scheduled_date=scheduled_time,
                    status="pending"
                )
                db.session.add(sched)  # bulk_save_objectsではなくadd

            
                db.session.commit()

            except Exception as e:
                db.session.rollback()
                logging.exception(f"[外部SEO記事生成テスト中エラー] site_id={site_id}, error={e}")

    threading.Thread(target=_bg, daemon=True).start()

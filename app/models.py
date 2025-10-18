# app/models.py

from datetime import datetime
from flask_login import UserMixin
from sqlalchemy import DateTime
from typing import Tuple
from . import db
from app import db
from sqlalchemy import Text

try:
    # Postgres なら JSONB に自動マッピングされます（SQLAlchemy の JSON 型）
    from sqlalchemy import JSON as SA_JSON
except Exception:
    SA_JSON = db.JSON

# ──── ユーザ ────
class User(db.Model, UserMixin):
    id       = db.Column(db.Integer, primary_key=True)

    # 基本情報
    email    = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(300), nullable=False)
    username = db.Column(db.String(50), unique=True, nullable=False)

    # 区分（個人 or 法人）
    user_type = db.Column(db.String(20), nullable=False, default="personal")  # "personal" or "corporate"

    # 法人用
    company_name = db.Column(db.String(100), nullable=True)
    company_kana = db.Column(db.String(100), nullable=True)

    # 氏名
    last_name  = db.Column(db.String(50), nullable=False)
    first_name = db.Column(db.String(50), nullable=False)

    # フリガナ
    last_kana  = db.Column(db.String(50), nullable=False)
    first_kana = db.Column(db.String(50), nullable=False)

    # 住所
    postal_code = db.Column(db.String(10), nullable=False)
    address = db.Column(db.String(200), nullable=True)  # ← 一旦nullable=True

    # 電話番号
    phone = db.Column(db.String(20), nullable=False)

    # 管理用
    is_admin   = db.Column(db.Boolean, default=False)
    is_special_access = db.Column(db.Boolean, default=False)
    has_purchased = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # ── プレゼンス表示用（UTCで保存、表示は相対表記などで）
    last_seen_at = db.Column(DateTime(timezone=True), nullable=True)
    share_presence = db.Column(db.Boolean, nullable=False, default=True)

    # リレーション（削除時に関連データも自動削除）
    prompts = db.relationship("PromptTemplate", backref="user", lazy=True, cascade="all, delete-orphan")
    articles = db.relationship("Article", backref="user", lazy='selectin', cascade="all, delete-orphan")
    sites = db.relationship("Site", backref="user", lazy=True, cascade="all, delete-orphan")
    keywords = db.relationship("Keyword", back_populates="user", cascade="all, delete-orphan")
    # ✅ 追加：GSCキーワードのみ抽出用（source='gsc'）
    gsc_keywords = db.relationship(
        "Keyword",
        primaryjoin="and_(User.id==Keyword.user_id, Keyword.source=='gsc')",
        viewonly=True,
        lazy=True
    )
    site_quota = db.relationship("UserSiteQuota", backref="user", lazy=True, uselist=False, cascade="all, delete-orphan")
    payment_logs = db.relationship("PaymentLog", backref="user", lazy=True, cascade="all, delete-orphan")
    token_logs = db.relationship("TokenUsageLog", backref="user", lazy=True, cascade="all, delete-orphan")
    gsc_tokens = db.relationship("GSCAuthToken", backref="user", lazy=True, cascade="all, delete-orphan")
    site_quota_logs = db.relationship("SiteQuotaLog", backref="user", lazy=True, cascade="all, delete-orphan")
    # app/models.py （class User 内のリレーション定義の末尾あたりに追記）
    # 内部SEO スケジュール（ユーザー単位）
    internal_seo_schedule = db.relationship(
        "InternalSeoUserSchedule",
        backref="user",
        uselist=False,
        cascade="all, delete-orphan",
        lazy=True,
    )
    internal_seo_user_runs = db.relationship(
        "InternalSeoUserRun",
        backref="user",
        lazy="dynamic",
        cascade="all, delete-orphan",
    )
    # ── Topic 自動生成（新規）
    topic_prompts = db.relationship("TopicPrompt", backref="user", lazy=True, cascade="all, delete-orphan")
    topic_pages = db.relationship("TopicPage", backref="user", lazy=True, cascade="all, delete-orphan")
    topic_anchor_logs = db.relationship("TopicAnchorLog", backref="user", lazy=True, cascade="all, delete-orphan")



# ──── サイトジャンル（ユーザーごとに管理可能） ────
class Genre(db.Model):
    __tablename__ = 'genres'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)

    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)  # ✅ 追加（ユーザー専用）

    # 🔄 関連
    user = db.relationship("User", backref="genres")  # ✅ 追加
    sites = db.relationship("Site", backref="genre", lazy=True)

# ──── エラー記録 ────

class Error(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    article_id = db.Column(db.Integer, nullable=False)
    user_id = db.Column(db.Integer, nullable=False)
    site_id = db.Column(db.Integer, nullable=False)
    error_message = db.Column(db.String(512), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Error {self.article_id} - {self.error_message}>'



# ──── WP サイト ────
class Site(db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    name     = db.Column(db.String(100), nullable=False)
    url      = db.Column(db.String(255), nullable=False)
    username = db.Column(db.String(100), nullable=False)
    app_pass = db.Column(db.String(200), nullable=False)
    user_id  = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    gsc_connected = db.Column(db.Boolean, default=False)  # ✅ 追加
    gsc_generation_started = db.Column(db.Boolean, default=False)  # ✅ GSC記事生成ボタンの実行フラグ
    # ✅ GSCオートジェンの基準日（これ以降に初観測のクエリのみ対象）
    gsc_autogen_since = db.Column(db.Date, nullable=True, index=True)
    # （任意）最後にオートジェンを回した時刻を残したい場合
    # last_gsc_autogen_at = db.Column(DateTime(timezone=True), nullable=True, index=True)
    # Site モデル内
    gsc_autogen_daily = db.Column(db.Boolean, default=False, nullable=False)
    clicks = db.Column(db.Integer, default=0)         # 総クリック数（GSC）
    impressions = db.Column(db.Integer, default=0)    # 表示回数（GSC）
    genre_id = db.Column(db.Integer, db.ForeignKey('genres.id'), nullable=True)  # ← 追加
    total_sites = db.Column(db.Integer, nullable=False, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    # リレーション
    articles = db.relationship("Article", backref="site", lazy='selectin')
    plan_type = db.Column(db.String(50), nullable=True)  # 'affiliate' または 'business'

    external_jobs = db.relationship(
        "ExternalSEOJob",
        back_populates="site",
        cascade="all, delete-orphan",
        lazy="selectin"
    )



# ──── プロンプト ────
class PromptTemplate(db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    genre    = db.Column(db.String(100), nullable=False)
    title_pt = db.Column(db.Text,       nullable=False)
    body_pt  = db.Column(db.Text,       nullable=False)
    user_id  = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)


# ──── 記事 ────
class Article(db.Model):
    __tablename__ = 'articles'
    id = db.Column(db.Integer, primary_key=True)
    keyword     = db.Column(db.String(255), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    body = db.Column(db.Text, nullable=True)
    featured_image = db.Column(db.String(255), nullable=True)  # アイキャッチ画像のURLを保存するカラムを追加
    image_url   = db.Column(db.String(500))
    status      = db.Column(db.String(20),  default="pending")   # pending/gen/done/error
    progress    = db.Column(db.Integer,     default=0)           # 0-100
    posted_url = db.Column(db.String(512), nullable=True)  # ✅ ←追加部分
    # ✅ NEW: WordPress上のpost ID（公開後に確定）。無ければURLから解決も可能
    wp_post_id  = db.Column(db.Integer, nullable=True, index=True)

    source = db.Column(db.String(50), default="manual")  # "manual", "gsc", "other"

    title_prompt = db.Column(db.Text, nullable=True)   # 生成時に使ったタイトルプロンプト
    body_prompt  = db.Column(db.Text, nullable=True)   # 生成時に使った本文プロンプト

    # === Title & Meta: メタ説明（最大180文字で運用）・品質ラベル・手動フラグ ===
    meta_description = db.Column(db.Text, nullable=True)
    meta_desc_quality = db.Column(
        db.Enum(
            "empty",
            "too_short",
            "too_long",
            "duplicate",
            "suspected_ai",
            "ok",
            name="meta_desc_quality_enum",
        ),
        nullable=True,
        index=True,
    )
    meta_desc_last_updated_at = db.Column(DateTime(timezone=True), nullable=True, index=True)
    # 手動編集が入った記事は自動上書きしない
    is_manual_meta = db.Column(db.Boolean, nullable=False, default=False, index=True)

    # タイムゾーン対応カラム (UTC保持、表示時にJSTに変換)
    created_at   = db.Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow
    )
    scheduled_at = db.Column(
        DateTime(timezone=True),
        nullable=True
    )
    posted_at    = db.Column(
        DateTime(timezone=True),
        nullable=True
    )

    site_id     = db.Column(db.Integer, db.ForeignKey("site.id"))
    user_id     = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

class ArticleAudit(db.Model):
    """
    Title/Meta の変更監査ログ（ロールバックと追跡用の最小構成）
    """
    __tablename__ = "article_audits"

    id = db.Column(db.Integer, primary_key=True)
    article_id = db.Column(db.Integer, db.ForeignKey("articles.id"), nullable=False, index=True)
    # どのフィールドを変更したか（例: 'meta_description'）
    field = db.Column(db.String(50), nullable=False)
    before = db.Column(db.Text, nullable=True)
    after  = db.Column(db.Text, nullable=True)
    job_id = db.Column(db.String(64), nullable=True, index=True)  # バッチ/ジョブのひも付け
    created_at = db.Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)

    article = db.relationship("Article", backref=db.backref("audit_logs", lazy="dynamic"))
    

class Keyword(db.Model):
    __tablename__ = "keywords"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    # ↓ まず site_id を "nullable=True" にしておく（仮対応）
    site_id = db.Column(db.Integer, db.ForeignKey("site.id"), nullable=True)


    keyword = db.Column(db.String(255), nullable=False)
    used = db.Column(db.Boolean, default=False)
    used_at = db.Column(db.DateTime, nullable=True)
    times_used = db.Column(db.Integer, default=0)
    status = db.Column(db.String(20), default="unprocessed", nullable=False)

    genre = db.Column(db.String(100), nullable=True)  # 任意（将来的な分類にも使える）
    status = db.Column(db.String(20), default="unprocessed")  # "unprocessed", "generating", "done", "error"
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  # 任意だが有用
    source = db.Column(db.String(20), default='manual')  # 'manual' または 'gsc'
    # リレーション
    user = db.relationship("User", back_populates="keywords")
    site = db.relationship("Site", backref="keywords")

class UserSiteQuota(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    total_quota = db.Column(db.Integer, default=0)  # 支払いで獲得したサイト数
    used_quota = db.Column(db.Integer, default=0)   # 登録済みサイト数
    plan_type = db.Column(db.String(20))            # 例: 'affiliate' or 'business'


class PaymentLog(db.Model):
    __tablename__ = 'payment_log'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)  # 紐付けが不明な場合はnull
    email = db.Column(db.String(255), nullable=False)
    amount = db.Column(db.Integer, nullable=False) 
    fee = db.Column(db.Integer, nullable=True) 
    net_income = db.Column(db.Integer, nullable=True) 
    plan_type = db.Column(db.String(50), nullable=True)  # "affiliate" or "business"
    stripe_payment_id = db.Column(db.String(100), nullable=False, unique=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default="succeeded")  # ← これを追加
    manual_fee = db.Column(db.Integer, nullable=True)  

    # ✅ 以下を追加（既存データはそのまま）
    product_name = db.Column(db.String(100), nullable=True)
    is_subscription = db.Column(db.Boolean, default=False)
    quantity = db.Column(db.Integer, default=1)
    currency = db.Column(db.String(10), default="JPY")

    def __repr__(self):
        return f"<PaymentLog {self.email} {self.amount}円 {self.created_at}>"

# ──── API使用ログ ────
class TokenUsageLog(db.Model):
    __tablename__ = 'token_usage_logs'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    prompt_tokens = db.Column(db.Integer, default=0)
    completion_tokens = db.Column(db.Integer, default=0)
    total_tokens = db.Column(db.Integer, default=0)

    created_at = db.Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)



# ✅ GSC 認証トークン保存用モデル
class GSCAuthToken(db.Model):
    __tablename__ = 'gsc_auth_tokens'

    id = db.Column(db.Integer, primary_key=True)
    site_id = db.Column(db.Integer, db.ForeignKey('site.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    access_token = db.Column(db.String(500), nullable=False)
    refresh_token = db.Column(db.String(500), nullable=True)
    token_expiry = db.Column(db.DateTime, nullable=True)

    # 関連リレーション
    site = db.relationship("Site", backref="gsc_tokens")

    def is_expired(self):
        from datetime import datetime, timedelta
        if not self.token_expiry:
            return True
        return datetime.utcnow() >= self.token_expiry - timedelta(minutes=5)
    
# ──── NEW: URL Inspection 結果キャッシュ ────
class GSCUrlStatus(db.Model):
    """
    URLごとのURL Inspection結果の軽量キャッシュ。
    - 直近の coverage_state / verdict / last_crawl_time などを保持
    - 生JSONは raw_json に保存（解析はアプリ側）
    """
    __tablename__ = "gsc_url_status"

    id = db.Column(db.Integer, primary_key=True)
    site_id = db.Column(db.Integer, db.ForeignKey("site.id"), nullable=False, index=True)
    article_id = db.Column(db.Integer, db.ForeignKey("articles.id"), nullable=True, index=True)
    url = db.Column(db.String(1024), nullable=False, index=True)

    indexed = db.Column(db.Boolean, nullable=False, default=False)           # verdict近似
    coverage_state = db.Column(db.String(128), nullable=True, index=True)    # 例: "Submitted and indexed", "Discovered - currently not indexed"
    verdict = db.Column(db.String(64), nullable=True)                        # PASS/FAIL/NEUTRAL など
    last_crawl_time = db.Column(db.DateTime, nullable=True)                  # GSCのクロール時刻
    robots_txt_state = db.Column(db.String(64), nullable=True)               # ALLOWED/BLOCKED など
    page_fetch_state = db.Column(db.String(64), nullable=True)               # SUCCESS/FAILED など
    last_inspected_at = db.Column(DateTime(timezone=True), nullable=True, index=True)

    raw_json = db.Column(SA_JSON, nullable=True)                             # Inspection API の生データ
    error = db.Column(db.Text, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint("site_id", "url", name="uq_gsc_url_status_site_url"),
        db.Index("ix_gsc_url_status_site_article", "site_id", "article_id"),
    )

    site = db.relationship("Site", backref=db.backref("gsc_url_statuses", lazy="dynamic"))
    article = db.relationship("Article", backref=db.backref("gsc_url_status", uselist=False))

# ──── NEW: URL Inspection 取得キュー ────
class GSCInspectionQueue(db.Model):
    """
    URL Inspectionを段階的に処理するためのキュー。
    サーバ停止時も再開できるよう、statusとattemptsを持つ。
    """
    __tablename__ = "gsc_inspection_queue"

    id = db.Column(db.Integer, primary_key=True)
    site_id = db.Column(db.Integer, db.ForeignKey("site.id"), nullable=False, index=True)
    article_id = db.Column(db.Integer, db.ForeignKey("articles.id"), nullable=True, index=True)
    url = db.Column(db.String(1024), nullable=False)

    priority = db.Column(db.Integer, nullable=False, default=100)            # 小さいほど優先
    status = db.Column(db.String(16), nullable=False, default="queued", index=True)  # queued/running/done/error
    attempts = db.Column(db.Integer, nullable=False, default=0)
    last_error = db.Column(db.Text, nullable=True)

    scheduled_at = db.Column(DateTime(timezone=True), nullable=True, index=True)
    updated_at = db.Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = db.Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)

    __table_args__ = (
        db.UniqueConstraint("site_id", "article_id", "url", name="uq_gsc_inspect_site_article_url"),
        db.Index("ix_gsc_inspect_status_priority", "status", "priority", "created_at"),
    )

    site = db.relationship("Site", backref=db.backref("gsc_inspection_queue", lazy="dynamic"))
    article = db.relationship("Article")    

# ──── GSC キーワード単位のパフォーマンス記録 ────
class GSCMetric(db.Model):
    __tablename__ = "gsc_metrics"

    id = db.Column(db.Integer, primary_key=True)
    site_id = db.Column(db.Integer, db.ForeignKey("site.id"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

    date = db.Column(db.Date, nullable=False)  # 日付
    query = db.Column(db.String(255), nullable=False)  # 検索クエリ

    impressions = db.Column(db.Integer, default=0)
    clicks = db.Column(db.Integer, default=0)
    ctr = db.Column(db.Float, default=0.0)
    position = db.Column(db.Float, default=0.0)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship("User", backref="gsc_metrics")
    site = db.relationship("Site", backref="gsc_metrics")


class SiteQuotaLog(db.Model):
    __tablename__ = 'site_quota_logs'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    stripe_payment_id = db.Column(db.String(255), nullable=True)
    site_count = db.Column(db.Integer, nullable=False)
    reason = db.Column(db.String(100), nullable=False, default="Stripe支払い")
    plan_type = db.Column(db.String(50), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class GSCConfig(db.Model):
    __tablename__ = 'gsc_configs'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    site_id = db.Column(db.Integer, db.ForeignKey("site.id"), nullable=False)
    property_uri = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship("User", backref="gsc_configs")
    site = db.relationship("Site", backref="gsc_configs")

# app/models.py の末尾付近（既存モデル群のあと）に新規クラスを追加
class GSCAutogenDaily(db.Model):
    """
    GSC自動記事化の“日次サマリー”
    1サイト×1日につき1行（DRYRUNでも記録）
    """
    __tablename__ = "gsc_autogen_daily"

    id = db.Column(db.Integer, primary_key=True)
    site_id = db.Column(db.Integer, db.ForeignKey("site.id"), nullable=False, index=True)

    # 実行対象日（UTC基準でOK。UIでJSTに変換表示）
    run_date = db.Column(db.Date, nullable=False, index=True)

    started_at  = db.Column(DateTime(timezone=True), nullable=True)
    finished_at = db.Column(DateTime(timezone=True), nullable=True)

    picked         = db.Column(db.Integer, nullable=False, default=0)   # 抽出総数
    queued         = db.Column(db.Integer, nullable=False, default=0)   # キュー投入数
    dup            = db.Column(db.Integer, nullable=False, default=0)   # 既存重複で除外
    limit_skipped  = db.Column(db.Integer, nullable=False, default=0)   # 上限で除外
    dryrun         = db.Column(db.Boolean, nullable=False, default=False)

    # 画面でサンプル表示するための10件など
    sample_keywords_json = db.Column(SA_JSON, nullable=True)
    # 任意：例外要約
    error = db.Column(db.Text, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint("site_id", "run_date", name="uq_gsc_autogen_daily_site_date"),
        db.Index("ix_gsc_autogen_daily_site_run", "site_id", "run_date"),
    )

    site = db.relationship("Site", backref=db.backref("gsc_autogen_daily_logs", lazy="dynamic"))




# app/models.py
class ChatLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    role = db.Column(db.String(10))  # "user" or "assistant"
    content = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class RyunosukeDeposit(db.Model):
    __tablename__ = "ryunosuke_deposits"

    id = db.Column(db.Integer, primary_key=True)
    deposit_date = db.Column(db.Date, nullable=False)
    amount = db.Column(db.Integer, nullable=False)
    memo = db.Column(db.String(255), nullable=True)  # 任意：備考

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# ──── 🔸 NEW: 外部ブログ自動投稿機能 ────
from app.enums import BlogType

class ExternalBlogAccount(db.Model):
    __tablename__ = "external_blog_account"

    id          = db.Column(db.Integer, primary_key=True)
    site_id     = db.Column(db.Integer, db.ForeignKey("site.id"), nullable=False, index=True)
    blog_type   = db.Column(db.Enum(BlogType), nullable=False)
    email       = db.Column(db.String(120), nullable=False, unique=True)
    username    = db.Column(db.String(100), nullable=False)
    password    = db.Column(db.String(255), nullable=False)          # 🔐 salted-hash 予定
    nickname    = db.Column(db.String(100), nullable=False, default="")   # ← ADD
    cookie_path = db.Column(db.Text,         nullable=True)               # ← ADD  Playwright storage_state 保存先
    status      = db.Column(db.String(30), default="active")         # active / done / error
    message = db.Column(db.Text, nullable=True)  # signupエラー時の説明メッセージ
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)
    posted_cnt          = db.Column(db.Integer,  default=0,  nullable=False)
    next_batch_started  = db.Column(db.Boolean,  default=False, nullable=False)

    livedoor_blog_id  = db.Column(db.String(50),  nullable=True, index=True)
    atompub_key_enc   = db.Column(db.String(255), nullable=True)
    atompub_endpoint  = db.Column(db.String(255), nullable=True)  # ← この行を追加
    api_post_enabled  = db.Column(db.Boolean,     default=False, nullable=False)
    blog_name   = db.Column(db.String(200), nullable=True, index=True)
        # 🔸 CAPTCHA分離ステップ用のフラグとセッション識別子
    is_captcha_completed = db.Column(db.Boolean, default=False, nullable=False)  # CAPTCHAが完了したか
    captcha_session_id = db.Column(db.String(64), nullable=True, index=True)     # CAPTCHA対応中セッションの識別子（UUIDなど）
    captcha_image_path = db.Column(db.String(255), nullable=True)                # 表示中のCAPTCHA画像のローカルパス
    generation_locked = db.Column(db.Boolean, nullable=False, default=False)  # 1回実行後に True
    generation_locked_at = db.Column(db.DateTime, nullable=True)              # いつロックしたか


    site        = db.relationship("Site", backref="external_accounts")
    schedules   = db.relationship("ExternalArticleSchedule", backref="blog_account", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<ExtBlogAccount {self.blog_type}:{self.username}>"

class ExternalArticleSchedule(db.Model):
    __tablename__ = "external_article_schedule"

    id               = db.Column(db.Integer, primary_key=True)
    blog_account_id  = db.Column(db.Integer, db.ForeignKey("external_blog_account.id"), nullable=False, index=True)
    keyword_id       = db.Column(db.Integer, db.ForeignKey("keywords.id"), nullable=False, index=True)
    # 追加：同一キーワードから複数記事を許容するため記事IDを持つ
    article_id       = db.Column(db.Integer, db.ForeignKey("articles.id"), nullable=True, index=True)

    # UTC naive を前提に走査するので index 付与
    scheduled_date   = db.Column(db.DateTime, nullable=False, index=True)
    status           = db.Column(db.String(30), default="pending")   # pending / posting / posted / error
    created_at       = db.Column(db.DateTime, default=datetime.utcnow)
    posted_url       = db.Column(db.String(512), nullable=True)
    message          = db.Column(db.Text, nullable=True)
    posted_at        = db.Column(db.DateTime, nullable=True)

    keyword          = db.relationship("Keyword")
    article          = db.relationship("Article")

    __table_args__ = (
        # 旧: db.UniqueConstraint("blog_account_id", "keyword_id", name="uq_blog_kw")
        db.UniqueConstraint("blog_account_id", "article_id", name="uq_blog_article"),
    )

    def __repr__(self):
        return f"<ExtArticleSched blog={self.blog_account_id} kw={self.keyword_id} art={self.article_id}>"

# ──── NEW: 外部SEOジョブステータス ────
class ExternalSEOJob(db.Model):
    __tablename__ = "external_seo_jobs"

    id         = db.Column(db.Integer, primary_key=True)
    site_id    = db.Column(db.Integer, db.ForeignKey("site.id"), nullable=False, index=True)
    blog_type  = db.Column(db.Enum(BlogType), nullable=False,
                           default=BlogType.LIVEDOOR)  # ← 変更
    status     = db.Column(db.String(20), default="queued")   # queued / running / success / error
    step       = db.Column(db.String(50),  default="waiting") # signup / generating / posting / finished
    message    = db.Column(db.Text)                           # エラーや備考を残す
    article_cnt= db.Column(db.Integer, default=0)             # 生成した記事数
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    site = db.relationship(
    "Site",
    back_populates="external_jobs"   # ← ここを back_populates に
    )
    # ExternalSEOJob にリレーションを追加
    logs = db.relationship("ExternalSEOJobLog", back_populates="job", lazy="dynamic")


    def __repr__(self) -> str:
        return f"<SEOJob site={self.site_id} status={self.status} step={self.step}>"

class ExternalSEOJobLog(db.Model):
    __tablename__ = "external_seo_job_logs"

    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey("external_seo_jobs.id"), nullable=False)
    step = db.Column(db.String(64))
    message = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    job = db.relationship("ExternalSEOJob", back_populates="logs")

# ──── GSC 日次合計（ランキング・ダッシュボード用の“正”データ） ────
class GSCDailyTotal(db.Model):
    __tablename__ = "gsc_daily_totals"

    id = db.Column(db.Integer, primary_key=True)

    site_id = db.Column(db.Integer, db.ForeignKey("site.id"), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)

    # GSCのプロパティURI（URLプレフィックスは末尾 / 必須、ドメインプロパティは sc-domain:example.com）
    property_uri = db.Column(db.String(255), nullable=False)

    # GSC側の “日付” 単位（JSTで集計した1日分を1レコードとして保存）
    date = db.Column(db.Date, nullable=False, index=True)

    # GSC UI と完全一致させるための合計値（加工なし）
    clicks = db.Column(db.Integer, nullable=False, default=0)
    impressions = db.Column(db.Integer, nullable=False, default=0)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # 速度 & 一意性
    __table_args__ = (
        db.UniqueConstraint(
            "site_id", "property_uri", "date",
            name="uq_gsc_daily_totals_site_prop_date"
        ),
        db.Index("ix_gsc_daily_totals_site_date", "site_id", "date"),
    )

    # 参照（必要に応じて利用）
    site = db.relationship("Site", backref=db.backref("gsc_daily_totals", lazy="dynamic"))
    user = db.relationship("User", backref=db.backref("gsc_daily_totals", lazy="dynamic"))

# >>> INTERNAL SEO: モデル追加（設定 / インデックス / グラフ / 監査ログ）

class InternalSeoConfig(db.Model):
    __tablename__ = "internal_seo_configs"

    id = db.Column(db.Integer, primary_key=True)
    site_id = db.Column(db.Integer, db.ForeignKey("site.id"), nullable=False, unique=True, index=True)

    # 実装仕様に合わせたデフォルト（本文リンクは最小2/最大5を厳守）
    min_links_per_post = db.Column(db.Integer, nullable=False, default=2)
    max_links_per_post = db.Column(db.Integer, nullable=False, default=5)
    insert_related_block = db.Column(db.Boolean, nullable=False, default=True)
    related_block_size = db.Column(db.Integer, nullable=False, default=4)
    min_paragraph_len = db.Column(db.Integer, nullable=False, default=80)

    # モード: auto（自動適用）/ assist（レビュー経由）/ off（無効）
    mode = db.Column(db.String(10), nullable=False, default="assist")

    # アンカー多様化のクールダウン（日数）
    avoid_exact_anchor_repeat_days = db.Column(db.Integer, nullable=False, default=30)

    # 対象記事フィルタ
    exclude_topic_in_url = db.Column(db.Boolean, nullable=False, default=True)
    link_to_status = db.Column(db.String(20), nullable=False, default="publish")  # 例: publish のみ

    # レート制限
    rate_limit_per_minute = db.Column(db.Integer, nullable=False, default=10)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    site = db.relationship("Site", backref=db.backref("internal_seo_config", uselist=False))


class ContentIndex(db.Model):
    """
    WP公開記事のインデックスキャッシュ（検索・類似度計算のため）
    """
    __tablename__ = "content_index"
    id = db.Column(db.Integer, primary_key=True)
    site_id = db.Column(db.Integer, db.ForeignKey("site.id"), nullable=False, index=True)
    wp_post_id = db.Column(db.Integer, nullable=True, index=True)

    title = db.Column(db.String(255), nullable=False)
    url = db.Column(db.String(512), nullable=False, index=True)
    slug = db.Column(db.String(255), nullable=True)
    status = db.Column(db.String(20), nullable=False, default="publish")
    published_at = db.Column(db.DateTime, nullable=True, index=True)
    updated_at = db.Column(db.DateTime, nullable=True, index=True)

    # 解析用（HTML除去済の本文）
    raw_text = db.Column(db.Text, nullable=True)
    keywords = db.Column(db.Text, nullable=True)  # CSVの軽い保存（将来は正規化/別テーブルも可）

    # ベクトル埋め込みは後で拡張（別テーブル or JSON等）。ここではプレースホルダ
    embedding = db.Column(db.LargeBinary, nullable=True)

    last_indexed_at = db.Column(db.DateTime, default=datetime.utcnow)

    site = db.relationship("Site", backref=db.backref("content_index", lazy="dynamic"))

    __table_args__ = (
        db.Index("ix_content_index_site_url", "site_id", "url"),
    )


class InternalLinkGraph(db.Model):
    """
    記事間の類似度と候補リンク（source -> target）
    """
    __tablename__ = "internal_link_graph"
    id = db.Column(db.Integer, primary_key=True)
    site_id = db.Column(db.Integer, db.ForeignKey("site.id"), nullable=False, index=True)
    source_post_id = db.Column(db.Integer, nullable=False, index=True)  # 対象: ContentIndex.wp_post_id or Article.wp_post_id
    target_post_id = db.Column(db.Integer, nullable=False, index=True)

    score = db.Column(db.Float, nullable=False, default=0.0)  # 0-1
    reason = db.Column(db.String(50), nullable=True)  # "topic_cluster", "keyword_match", "embedding" など
    last_evaluated_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        db.UniqueConstraint("site_id", "source_post_id", "target_post_id", name="uq_ilg_site_src_tgt"),
        db.Index("ix_ilg_site_src_score", "site_id", "source_post_id", "score"),
    )

# === 内部リンクのステータス定義（アプリ内参照用。DB制約は設けない） ===
# pending       : 未適用
# applied       : 本文へ適用済み（現行の有効リンク）
# reverted      : 手動またはロールバックで取り消し
# skipped       : 今回の適用対象から除外
# legacy_deleted: 旧仕様リンクの自動削除ログ（監査用）
# superseded    : 再ビルド等により“上書き置換”され、旧バージョンとしてクローズ
INTERNAL_LINK_STATUS_CHOICES: Tuple[str, ...] = (
    "pending",
    "applied",
    "reverted",
    "skipped",
    "legacy_deleted",
    "superseded",
)

class InternalLinkAction(db.Model):
    """
    適用/差し戻しなどの監査ログ（本文内/関連記事ブロック/置換も含む）
    置換によるクローズは 'superseded' を用いて 'reverted' と区別する。
    """
    __tablename__ = "internal_link_actions"
    id = db.Column(db.Integer, primary_key=True)
    site_id = db.Column(db.Integer, db.ForeignKey("site.id"), nullable=False, index=True)

    post_id = db.Column(db.Integer, nullable=False, index=True)         # ソースのWP post id
    target_post_id = db.Column(db.Integer, nullable=False, index=True)  # ターゲットのWP post id

    # 内部リンクの“世代”を明示する。置き換え（全差し替え）時に +1 する運用。
    link_version = db.Column(db.Integer, nullable=False, default=1, index=True)

    anchor_text = db.Column(db.String(255), nullable=False)
    position = db.Column(db.String(50), nullable=False)  # 'p:3'（段落3）/ 'related_block' など
    status = db.Column(db.String(20), nullable=False, default="pending")  # pending/applied/reverted/skipped
    # status は INTERNAL_LINK_STATUS_CHOICES を参照
    status = db.Column(db.String(20), nullable=False, default="pending")  # pending/applied/reverted/skipped/legacy_deleted/superseded

    applied_at = db.Column(db.DateTime, nullable=True, index=True)
    reverted_at = db.Column(db.DateTime, nullable=True, index=True)

    # 監査用の短い抜粋
    diff_before_excerpt = db.Column(db.Text, nullable=True)
    diff_after_excerpt = db.Column(db.Text, nullable=True)

    job_id = db.Column(db.String(64), nullable=True, index=True)
    reason = db.Column(db.String(50), nullable=True)  # 'auto_apply', 'review_approved', 'swap' など

    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        db.Index("ix_ila_site_post_status", "site_id", "post_id", "status"),
        # 記事×バージョンでの集計・検索を高速化
        db.Index("ix_ila_site_post_version", "site_id", "post_id", "link_version"),
    )

class InternalSeoRun(db.Model):
    """
    内部SEOの 1 回の実行ログ（インデックス→グラフ→計画→適用までのサマリー）
    ダッシュボードでの可視化・失敗時の原因追跡・再実行トリガに利用。
    """
    __tablename__ = "internal_seo_runs"

    id = db.Column(db.Integer, primary_key=True)
    site_id = db.Column(db.Integer, db.ForeignKey("site.id"), nullable=False, index=True)

    # 実行メタ
    job_kind = db.Column(db.String(20), nullable=False, default="manual")  # manual / regular / daily_sweep など
    status = db.Column(db.String(20), nullable=False, default="running")   # running / success / error
    message = db.Column(Text, nullable=True)                               # 失敗時の要約や注意

    started_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)
    ended_at   = db.Column(db.DateTime, nullable=True, index=True)
    duration_ms = db.Column(db.Integer, nullable=True)                     # 便利な所要時間(ms)

    # 各フェーズの統計を JSON で保存（例：
    # {"indexer":{"processed":930,"created_or_updated":930,"pages":9},
    #  "graph":{"sources":930,"edges_upserted":5736},
    #  "planner":{"planned":304,"swap_candidates":0,"processed":200},
    #  "applier":{"applied":88,"swapped":0,"skipped":0,"processed_posts":50}}
    stats = db.Column(SA_JSON, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    site = db.relationship("Site", backref=db.backref("internal_seo_runs", lazy="dynamic"))

    __table_args__ = (
        db.Index("ix_internal_seo_runs_site_status_started", "site_id", "status", "started_at"),
    )    

# === Internal SEO: ジョブキュー（ナイトリー投入＆ワーカー消化用） ===
class InternalSeoJobQueue(db.Model):
    __tablename__ = "internal_seo_job_queue"

    id = db.Column(db.Integer, primary_key=True)
    site_id = db.Column(db.Integer, db.ForeignKey("site.id"), nullable=False, index=True)

    # 実行パラメータ（ENVのデフォルトと同等）
    pages = db.Column(db.Integer, nullable=True)
    per_page = db.Column(db.Integer, nullable=True)
    min_score = db.Column(db.Float, nullable=True)
    max_k = db.Column(db.Integer, nullable=True)
    limit_sources = db.Column(db.Integer, nullable=True)
    limit_posts = db.Column(db.Integer, nullable=True)
    incremental = db.Column(db.Boolean, nullable=False, default=True)

    job_kind = db.Column(db.String(40), nullable=False, default="nightly-enqueue")  # nightly-enqueue / worker / manual etc.
    status = db.Column(db.String(20), nullable=False, default="queued")            # queued / running / done / error
    message = db.Column(db.Text, nullable=True)

    created_at = db.Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
    started_at = db.Column(DateTime(timezone=True), nullable=True)
    ended_at   = db.Column(DateTime(timezone=True), nullable=True)

    site = db.relationship("Site", backref=db.backref("internal_seo_job_queue_items", lazy="dynamic"))

    __table_args__ = (
        # タスク5の要件： (status, created_at) にインデックス
        db.Index("idx_internal_seo_job_queue_status_created", "status", "created_at"),
    )



# ──── NEW: 外部サインアップ一時タスク（方式A用・追加のみ） ────
class ExternalSignupTask(db.Model):
    """
    方式A（ローカル・ヘルパー）での“アカウント作成タスク”を一時的に保持するテーブル。
    既存機能に影響を与えないよう **追加のみ**。既存モデルは変更しない。
    """
    __tablename__ = "external_signup_tasks"

    id = db.Column(db.Integer, primary_key=True)

    # ワンタイムトークン（短TTL・ユニーク）
    token = db.Column(db.String(64), unique=True, nullable=False, index=True)

    # 実行主体のひも付け（監査と多ユーザー環境のため）
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    site_id = db.Column(db.Integer, db.ForeignKey("site.id"), nullable=False, index=True)

    # プロバイダ（例: 'livedoor'）—将来拡張に備える
    provider = db.Column(db.String(32), nullable=False, default="livedoor", index=True)

    # 入力パラメータ（希望 blog_id、ニックネーム等をJSONで保持）
    payload = db.Column(db.JSON, nullable=True)

    # サーバー側で取得した検証URLを短時間だけ保持（ヘルパーへ渡す）
    verification_url = db.Column(db.Text, nullable=True)

    # 結果（blog_id, api_key, endpoint, public_url など）をJSONで受け取り保存
    result = db.Column(db.JSON, nullable=True)

    # ステータス：pending / running / done / failed / expired
    status = db.Column(db.String(16), nullable=False, default="pending", index=True)

    # 失敗内容などの短いメッセージ
    message = db.Column(db.Text, nullable=True)

    # 期限（TTL切れで expired 扱いにする）
    expires_at = db.Column(DateTime(timezone=True), nullable=False)

    # 監査用タイムスタンプ
    created_at = db.Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
    updated_at = db.Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 参照
    user = db.relationship("User", backref=db.backref("external_signup_tasks", lazy="dynamic"))
    site = db.relationship("Site", backref=db.backref("external_signup_tasks", lazy="dynamic"))

    def is_expired(self) -> bool:
        return datetime.utcnow() >= (self.expires_at if self.expires_at else datetime.utcnow())

    def __repr__(self) -> str:
        return f"<ExternalSignupTask token={self.token} status={self.status} provider={self.provider}>"

# app/models.py （末尾に追加）

class InternalSeoUserSchedule(db.Model):
    """
    ユーザー単位で内部SEOスケジュールを管理するスイッチ＆状態。
    管理画面の「開始/停止/一時停止/再開」や、tick 間隔・1tickの処理件数を保持。
    """
    __tablename__ = "internal_seo_user_schedules"

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, unique=True, index=True)

    # 管理フラグ＆状態
    is_enabled = db.Column(db.Boolean, nullable=False, default=False)                  # ON/OFF
    status     = db.Column(db.String(20), nullable=False, default="idle")             # idle|queued|running|paused|error
    last_error = db.Column(db.Text, nullable=True)

    # スケジュール
    last_run_at = db.Column(DateTime(timezone=True), nullable=True, index=True)
    next_run_at = db.Column(DateTime(timezone=True), nullable=True, index=True)

    # 実行パラメータ
    tick_interval_sec = db.Column(db.Integer, nullable=False, default=90)             # 次tickまでの秒
    budget_per_tick   = db.Column(db.Integer, nullable=False, default=50)             # 1tickで処理する最大ポスト数
    rate_limit_per_min = db.Column(db.Integer, nullable=True)                         # ユーザー固有のレート（未指定ならapplier側のENV）

    created_at = db.Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = db.Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        db.Index("ix_iseo_user_sched_status_next", "status", "next_run_at"),
    )


class InternalSeoUserRun(db.Model):
    """
    ユーザー単位の実行履歴（1tick もしくは1バッチ）。
    管理画面の進行状況サマリーに使用。
    """
    __tablename__ = "internal_seo_user_runs"

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)

    started_at  = db.Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
    finished_at = db.Column(DateTime(timezone=True), nullable=True, index=True)
    status      = db.Column(db.String(20), nullable=False, default="running")         # running|success|failed|partial

    # applier の戻り値を集計
    applied         = db.Column(db.Integer, nullable=False, default=0)
    swapped         = db.Column(db.Integer, nullable=False, default=0)
    skipped         = db.Column(db.Integer, nullable=False, default=0)
    processed_posts = db.Column(db.Integer, nullable=False, default=0)

    notes = db.Column(SA_JSON, nullable=True)  # 補足メモ（サイト数/残件など）

    created_at = db.Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    __table_args__ = (
        db.Index("ix_iseo_user_runs_user_started", "user_id", "started_at"),
        db.Index("ix_iseo_user_runs_status", "status"),
    )

# =======================================================================
#                           TOPIC: 自動生成モデル
#   ステップ1（ユーザー情報なし版）の「アンカー生成 → クリック時に瞬間生成」
#   を実現するための最小セット。以後の拡張（Topics API統合等）も想定。
# =======================================================================

class TopicPrompt(db.Model):
    """
    Topicページ生成の“設計図”。1ユーザー内で複数保持可。
    - prompt: 生成に使うプロンプト雛形（JSONでも文字列でも可）
    - site_id: ひも付けるサイト（任意）
    """
    __tablename__ = "topic_prompts"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    site_id = db.Column(db.Integer, db.ForeignKey("site.id"), nullable=True, index=True)

    name = db.Column(db.String(120), nullable=False)                 # 画面で識別する名称
    prompt = db.Column(SA_JSON, nullable=True)                       # 本文/見出し/一覧などの生成指示（構造化）
    tags = db.Column(db.Text, nullable=True)                         # 補助用のCSVタグ
    is_active = db.Column(db.Boolean, nullable=False, default=True)

    created_at = db.Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = db.Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    site = db.relationship("Site", backref=db.backref("topic_prompts", lazy="dynamic"))

    __table_args__ = (
        db.Index("ix_topic_prompts_user_site", "user_id", "site_id"),
    )


class TopicPage(db.Model):
    """
    生成済みのTopicページ本体。
    - クリック瞬間生成後に保存（以降はキャッシュとして即返却）
    - slug: ルーティング用（/t/<slug>）
    """
    __tablename__ = "topic_pages"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    site_id = db.Column(db.Integer, db.ForeignKey("site.id"), nullable=True, index=True)
    source_prompt_id = db.Column(db.Integer, db.ForeignKey("topic_prompts.id"), nullable=True, index=True)

    slug = db.Column(db.String(191), nullable=False)                 # URLスラッグ（ユーザー内一意）
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=True)
    body = db.Column(db.Text, nullable=True)                         # 生成HTML（サニタイズ後）

    topics_json = db.Column(SA_JSON, nullable=True)                  # 生成時点のTopics APIスナップショット（後で使う）
    meta = db.Column(SA_JSON, nullable=True)                         # 生成メタ（見出し配列/関連記事IDs等）
    generated_ms = db.Column(db.Integer, nullable=True)              # 生成所要時間(ms)の記録

    created_at = db.Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
    updated_at = db.Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 参考：将来WP等へ公開した場合のURL
    published_url = db.Column(db.String(512), nullable=True)

    prompt = db.relationship("TopicPrompt", backref=db.backref("topic_pages", lazy="dynamic"))
    site = db.relationship("Site", backref=db.backref("topic_pages", lazy="dynamic"))

    __table_args__ = (
        db.UniqueConstraint("user_id", "slug", name="uq_topic_pages_user_slug"),
        db.Index("ix_topic_pages_user_created", "user_id", "created_at"),
    )


class TopicAnchorLog(db.Model):
    """
    アンカー表示/クリックの監査ログ。
    - event: impression / click
    - click 時は latency_ms に「クリック→生成完了」までの時間を記録
    """
    __tablename__ = "topic_anchor_logs"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    site_id = db.Column(db.Integer, db.ForeignKey("site.id"), nullable=True, index=True)
    page_id = db.Column(db.Integer, db.ForeignKey("topic_pages.id"), nullable=True, index=True)

    source_url = db.Column(db.String(512), nullable=False)           # アンカーを表示/クリックした元記事URL
    position = db.Column(db.String(50), nullable=False)              # 'slot_top' / 'slot_bottom' / 'p:3' 等
    anchor_text = db.Column(db.String(255), nullable=False)

    event = db.Column(db.String(12), nullable=False, default="impression")  # impression|click
    latency_ms = db.Column(db.Integer, nullable=True)                # click時のみ

    topics_snapshot = db.Column(SA_JSON, nullable=True)              # 当時のユーザーTopics（将来の分析用）
    created_at = db.Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)

    page = db.relationship("TopicPage", backref=db.backref("anchor_logs", lazy="dynamic"))
    site = db.relationship("Site", backref=db.backref("topic_anchor_logs", lazy="dynamic"))

    __table_args__ = (
        db.Index("ix_topic_anchor_logs_user_event_created", "user_id", "event", "created_at"),
    )
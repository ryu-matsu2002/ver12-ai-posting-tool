# app/models.py

from datetime import datetime
from flask_login import UserMixin
from sqlalchemy import DateTime
from . import db
from app import db

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

    source = db.Column(db.String(50), default="manual")  # "manual", "gsc", "other"

    title_prompt = db.Column(db.Text, nullable=True)   # 生成時に使ったタイトルプロンプト
    body_prompt  = db.Column(db.Text, nullable=True)   # 生成時に使った本文プロンプト

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

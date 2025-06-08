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
    site_quota = db.relationship("UserSiteQuota", backref="user", lazy=True, uselist=False, cascade="all, delete-orphan")
    payment_logs = db.relationship("PaymentLog", backref="user", lazy=True, cascade="all, delete-orphan")
    token_logs = db.relationship("TokenUsageLog", backref="user", lazy=True, cascade="all, delete-orphan")
    gsc_tokens = db.relationship("GSCAuthToken", backref="user", lazy=True, cascade="all, delete-orphan")
    site_quota_logs = db.relationship("SiteQuotaLog", backref="user", lazy=True, cascade="all, delete-orphan")


# ──── WP サイト ────
class Site(db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    name     = db.Column(db.String(100), nullable=False)
    url      = db.Column(db.String(255), nullable=False)
    username = db.Column(db.String(100), nullable=False)
    app_pass = db.Column(db.String(200), nullable=False)
    user_id  = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    gsc_connected = db.Column(db.Boolean, default=False)  # ✅ 追加
    # リレーション
    articles = db.relationship("Article", backref="site", lazy='selectin')
    plan_type = db.Column(db.String(50), nullable=True)  # 'affiliate' または 'business'



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

    genre = db.Column(db.String(100), nullable=True)  # 任意（将来的な分類にも使える）
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
    amount = db.Column(db.Integer, nullable=False)  # 単位：円
    fee = db.Column(db.Integer, nullable=True)      # Stripe手数料（運営計算）
    net_income = db.Column(db.Integer, nullable=True)  # 運営の取り分
    plan_type = db.Column(db.String(50), nullable=True)  # "affiliate" or "business"
    stripe_payment_id = db.Column(db.String(100), nullable=False, unique=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default="succeeded")  # ← これを追加

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
    user = db.relationship("User", backref="gsc_tokens")
    site = db.relationship("Site", backref="gsc_tokens")

    def is_expired(self):
        from datetime import datetime, timedelta
        if not self.token_expiry:
            return True
        return datetime.utcnow() >= self.token_expiry - timedelta(minutes=5)

# ──── サイト登録枠の取得ログ ────
class SiteQuotaLog(db.Model):
    __tablename__ = 'site_quota_logs'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    plan_type = db.Column(db.String(50), nullable=False)  # 'affiliate', 'business' など
    count = db.Column(db.Integer, nullable=False)          # 加算（または減算）されたサイト数
    reason = db.Column(db.String(255), nullable=False)     # 例："Stripe支払い", "管理者手動追加"
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship("User", backref="site_quota_logs")

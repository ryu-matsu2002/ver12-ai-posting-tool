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
    prefecture  = db.Column(db.String(20), nullable=False)
    city        = db.Column(db.String(50), nullable=False)
    address1    = db.Column(db.String(100), nullable=False)
    address2    = db.Column(db.String(100), nullable=True)

    # 電話番号
    phone = db.Column(db.String(20), nullable=False)

    # 管理用
    is_admin   = db.Column(db.Boolean, default=False)
    is_special_access = db.Column(db.Boolean, default=False)
    has_purchased = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # リレーション
    prompts  = db.relationship("PromptTemplate", backref="user", lazy=True)
    articles = db.relationship("Article", backref="user", lazy='selectin')
    sites    = db.relationship("Site",           backref="user", lazy=True)



# ──── WP サイト ────
class Site(db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    name     = db.Column(db.String(100), nullable=False)
    url      = db.Column(db.String(255), nullable=False)
    username = db.Column(db.String(100), nullable=False)
    app_pass = db.Column(db.String(200), nullable=False)
    user_id  = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

    # リレーション
    articles = db.relationship("Article", backref="site", lazy='selectin')


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

    # リレーション
    user = db.relationship("User", backref="keywords")
    site = db.relationship("Site", backref="keywords")

class UserSiteQuota(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    total_quota = db.Column(db.Integer, default=0)  # 支払いで獲得したサイト数
    used_quota = db.Column(db.Integer, default=0)   # 登録済みサイト数
    plan_type = db.Column(db.String(20))            # 例: 'affiliate' or 'business'

    user = db.relationship('User', backref=db.backref('site_quota', uselist=False))
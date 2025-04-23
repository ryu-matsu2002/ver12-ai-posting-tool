# app/models.py
from datetime import datetime
from flask_login import UserMixin
from . import db

# ──── ユーザ ────
class User(db.Model, UserMixin):
    id       = db.Column(db.Integer, primary_key=True)
    email    = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    prompts  = db.relationship("PromptTemplate", backref="user", lazy=True)
    articles = db.relationship("Article",        backref="user", lazy=True)
    sites    = db.relationship("Site",           backref="user", lazy=True)

# ──── WP サイト ────
class Site(db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    name     = db.Column(db.String(100), nullable=False)
    url      = db.Column(db.String(255), nullable=False)
    username = db.Column(db.String(100), nullable=False)
    app_pass = db.Column(db.String(200), nullable=False)
    user_id  = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    articles = db.relationship("Article", backref="site", lazy=True)

# ──── プロンプト ────
class PromptTemplate(db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    genre    = db.Column(db.String(100), nullable=False)
    title_pt = db.Column(db.Text,       nullable=False)
    body_pt  = db.Column(db.Text,       nullable=False)
    user_id  = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

# ──── 記事 ────
class Article(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    keyword     = db.Column(db.String(255), nullable=False)
    title       = db.Column(db.Text)
    body        = db.Column(db.Text)
    image_url   = db.Column(db.String(500))
    status      = db.Column(db.String(20),  default="pending")   # pending/gen/done/error
    progress    = db.Column(db.Integer,     default=0)           # 0-100
    created_at  = db.Column(db.DateTime,    default=datetime.utcnow)
    scheduled_at= db.Column(db.DateTime)                         # ★ 予約投稿時刻 (UTC)
    posted_at   = db.Column(db.DateTime)                         # WP 投稿完了時刻
    site_id     = db.Column(db.Integer, db.ForeignKey("site.id"))
    user_id     = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

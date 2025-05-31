# app/__init__.py
#─────────────────────────────────────────────
#  Flask アプリ factory / 拡張初期化 / Celery factory
#─────────────────────────────────────────────
from __future__ import annotations

import os
from dotenv import load_dotenv
load_dotenv()

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate
from celery import Celery




# ── Flask-拡張の“空”インスタンスを先に作成 ───────────
db            = SQLAlchemy()
login_manager = LoginManager()
migrate       = Migrate()

# --------------------------------------------------
# 1) Flask App Factory
# --------------------------------------------------
def create_app() -> Flask:
    """Flask application factory."""
    app = Flask(
        __name__,
        static_folder="static",
        template_folder="templates",
    )

    # ─── 基本設定 ─────────────────────────────
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-key")
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
        "DATABASE_URL", "sqlite:///instance/local.db"
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["STRIPE_WEBHOOK_SECRET"] = os.getenv("STRIPE_WEBHOOK_SECRET")

    # ─── SQLAlchemy 接続プール設定 ──────────────
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_size": int(os.getenv("POOL_SIZE", 10)),
        "max_overflow": int(os.getenv("MAX_OVERFLOW", 20)),
        "pool_timeout": int(os.getenv("POOL_TIMEOUT", 30)),
    }

    # ─── 拡張をバインド ───────────────────────
    db.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)
    app.logger.setLevel("DEBUG")

    # ─── Blueprints 登録とスケジューラ起動 ───────
    with app.app_context():
        # Blueprint の登録（main用 + admin用）
        from .routes import bp as main_bp, admin_bp, stripe_webhook_bp
        app.register_blueprint(main_bp)
        app.register_blueprint(admin_bp)
        app.register_blueprint(stripe_webhook_bp)

        from . import models

        # 自動投稿ジョブを APScheduler に登録して起動
        from .tasks import init_scheduler
        init_scheduler(app)

        # Flask-Login: user_loader
        from .models import User  # 循環 import 回避
        @login_manager.user_loader
        def load_user(user_id: str) -> User | None:  # type: ignore[name-defined]
            return User.query.get(int(user_id))

    login_manager.login_view = "main.login"
    return app


# --------------------------------------------------
# 2) Celery Factory
# --------------------------------------------------
def make_celery(app: Flask) -> Celery:
    """Flask アプリの context 付き Celery を返す."""
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    celery = Celery(app.import_name, broker=redis_url, backend=redis_url)

    # Flask 設定を Celery にコピー
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        """Flask アプリコンテキストでタスクを実行させる."""
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask  # type: ignore[attr-defined]
    return celery

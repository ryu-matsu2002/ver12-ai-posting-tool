# app/__init__.py
#─────────────────────────────────────────────
#  Flask アプリ factory / 拡張初期化 / Celery factory
#─────────────────────────────────────────────
from __future__ import annotations

import os
import logging  # ✅ 追加
from logging.handlers import RotatingFileHandler  # ✅ 追加
from dotenv import load_dotenv
load_dotenv()

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate
from celery import Celery
from .controllers.external_seo import external_bp    # ← 追加

# ── Flask-拡張の“空”インスタンスを先に作成 ───────────
db            = SQLAlchemy()
login_manager = LoginManager()
migrate       = Migrate()

# --------------------------------------------------
# 1) Flask App Factory
# --------------------------------------------------
def comma_filter(value):
    return "{:,}".format(value)


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
        "pool_size": int(os.getenv("POOL_SIZE", 50)),
        "max_overflow": int(os.getenv("MAX_OVERFLOW", 100)),
        "pool_timeout": int(os.getenv("POOL_TIMEOUT", 60)),
        "pool_recycle": 1800,  # ✅ 追加（切断予防）
    }

    # ─── 拡張をバインド ───────────────────────
    db.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)

    login_manager.login_message = "このページを開くにはログインが必要です。"
    login_manager.login_message_category = "info"  # Bootstrapの黄色表示

    # ✅ ログ出力設定（logs/system.log に出力）
    if not os.path.exists("logs"):
        os.makedirs("logs")  # logsフォルダがなければ作成

    file_handler = RotatingFileHandler("logs/system.log", maxBytes=1024 * 1024, backupCount=3)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s in %(module)s: %(message)s')
    file_handler.setFormatter(formatter)

    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info("✅ Flaskアプリが初期化されました")  # 明示ログ

    # ─── Blueprints 登録とスケジューラ起動 ───────
    with app.app_context():
        # Blueprint の登録（main用 + admin用 + webhook）
        from .routes import bp as main_bp, admin_bp, stripe_webhook_bp
        app.register_blueprint(main_bp)
        app.register_blueprint(admin_bp)
        app.register_blueprint(stripe_webhook_bp)
        app.jinja_env.filters["comma"] = comma_filter
        from . import models

        # Flask-Login: user_loader
        from .models import User  # 循環 import 回避
        @login_manager.user_loader
        def load_user(user_id: str) -> User | None:  # type: ignore[name-defined]
            return User.query.get(int(user_id))

    # ✅ スケジューラー起動（--preload により1回のみ呼ばれる）
    if os.getenv("SCHEDULER_ENABLED") == "1":
        app.logger.info("✅ init_scheduler() を起動します（is_main_process チェックなし）")
        from .tasks import init_scheduler
        init_scheduler(app)

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

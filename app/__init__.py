# app/__init__.py
#─────────────────────────────────────────────
#  Flask アプリ factory / 拡張初期化 / Celery factory
#─────────────────────────────────────────────
from __future__ import annotations

import os
import logging  # ✅ 追加
import redis
import fcntl  # 単一起動のためのファイルロック（Linux）
from logging.handlers import RotatingFileHandler  # ✅ 追加
from dotenv import load_dotenv
# .env は systemd から渡された環境変数を上書きしない（明示）
load_dotenv(override=False)

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate
from celery import Celery
from multiprocessing import current_process
from app.utils.datetime import to_jst  # ← 追加
# app/__init__.py の先頭 import 群のどこか（Flask拡張の init より前でOK）


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
        "pool_pre_ping": True,  # ✅ この行を追加！
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
    # --- Presence: Jinja フィルタ登録（相対時間の日本語表示） ---
    from app.utils.presence import timeago_jp
    @app.template_filter("timeago_jp")
    def _timeago_filter(dt):
        return timeago_jp(dt)

    # --- 追加: Jinja から to_jst() を直接呼べるようにする ---
    @app.context_processor
    def _inject_utils():
        # テンプレートで {{ to_jst(...) }} として呼べます
        return dict(to_jst=to_jst)

    # ─── Blueprints 登録とスケジューラ起動 ───────
    with app.app_context():
        # Blueprint の登録（main用 + admin用 + webhook）
        from .routes import bp as main_bp, admin_bp, stripe_webhook_bp
        app.register_blueprint(main_bp)
        app.register_blueprint(admin_bp)
        app.register_blueprint(stripe_webhook_bp)
        # --- Presence API BluePrint 登録（UI変更なし / APIのみ追加） ---
        from .blueprints.presence import bp as presence_bp
        app.register_blueprint(presence_bp)

        app.jinja_env.filters["comma"] = comma_filter
        from . import models

        # Flask-Login: user_loader
        from .models import User  # 循環 import 回避
        @login_manager.user_loader
        def load_user(user_id: str) -> User | None:  # type: ignore[name-defined]
            # SQLAlchemy 2.x 推奨のセッションAPI。例外連鎖を避けやすい
            return db.session.get(User, int(user_id))
        
        # ✅ 修正①: external_bp の import & 登録は app context 内で最後に行う
        #from .controllers.external_seo import external_bp
        #app.register_blueprint(external_bp)
    # --- Presence: すべてのリクエストで Redis TTL を更新（DBは触らない） ---
    from flask_login import current_user
    from app.utils.presence import mark_online as _mark_online
    @app.before_request
    def _touch_presence():
        try:
            if current_user.is_authenticated:
                _mark_online(current_user.id)
        except Exception:
            # Redis が落ちていてもアプリ全体は止めない
            pass

    # --- DBセッションのクリーンアップ: リクエスト終了時にロールバック＆解放 ---
    @app.teardown_request
    def _cleanup_session(exception):
        try:
            if exception is not None:
                # 直前にDB例外が起きていたら必ずROLLBACK
                db.session.rollback()
        finally:
            # 正常/異常に関わらず remove して接続とスコープを解放
            db.session.remove()    

    # ✅ スケジューラー起動（jobsロールのプロセスだけ）
    #    systemd から JOBS_ROLE=jobs を与えたときのみ起動する
    if os.getenv("SCHEDULER_ENABLED") == "1" and os.getenv("JOBS_ROLE", "web") == "jobs":
        lock_path = "/tmp/ai_posting_scheduler.lock"
        try:  
            app._scheduler_lockfile = open(lock_path, "w")
            try:
               # 取得できたプロセスだけが起動
               fcntl.flock(app._scheduler_lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
               app.logger.info("✅ SCHEDULER: lock acquired -> init_scheduler() を起動します")
               from .tasks import init_scheduler
               init_scheduler(app)
            except BlockingIOError:
               app.logger.info("ℹ️ SCHEDULER: lock already held -> 他プロセスで稼働中のためスキップ")
        except Exception as e:
            app.logger.exception("⚠️ SCHEDULER: lock 初期化に失敗したためスキップ: %s", e)
    else:
        app.logger.info("ℹ️ SCHEDULER: skipped (SCHEDULER_ENABLED=%s, JOBS_ROLE=%s)",
                        os.getenv("SCHEDULER_ENABLED"), os.getenv("JOBS_ROLE"))        

    # === PWController 起動（“jobs”ロールのみ）========================
    role = os.getenv("JOBS_ROLE", "").strip().lower()

    def _start_pw_controller_once():
        try:
            from app.services.pw_controller import pwctl  # type: ignore
            headless = os.getenv("PWCTL_HEADLESS", "1") == "1"
            pwctl.start(headless=headless)
            app.logger.info("✅ PWController started (headless=%s, role=%s)", headless, role)
        except Exception as e:
            app.logger.exception("⚠️ PWController start failed: %s", e)

    if role == "jobs":
        _hook = getattr(app, "before_serving", None) or getattr(app, "before_first_request", None)
        if callable(_hook):
            _hook(_start_pw_controller_once)
        else:
            try:
                _start_pw_controller_once()
            except Exception:
                app.logger.exception("⚠️ PWController immediate start failed")
    else:
        app.logger.info("ℹ️ PWController: disabled on role=%s", role)


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
# --------------------------------------------------
# Redis client（Flask全体で使い回す）
# --------------------------------------------------
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.StrictRedis.from_url(
    redis_url,
    decode_responses=True,
    socket_connect_timeout=0.5,  # 接続タイムアウト
    socket_timeout=0.8,          # コマンドタイムアウト
)    
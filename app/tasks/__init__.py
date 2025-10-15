# app/tasks/__init__.py
from .title_meta_backfill import run_title_meta_backfill  # re-export

def init_scheduler(app):
    """
    ダミー実装：
    - 既存コードが from app.tasks import init_scheduler を行うため、
      未実装期間は no-op でImportErrorを防ぐ。
    - きちんとしたジョブ定義を入れるときはここを差し替える。
    """
    app.logger.info("ℹ️ init_scheduler(): no-op (dummy). Replace with real scheduler when ready.")
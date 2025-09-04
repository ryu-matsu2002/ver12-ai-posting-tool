# app/jobs_runner.py
import os
import signal
from app import create_app  # 既存のファクトリ
# create_app() 内の既存ロジックで、SCHEDULER_ENABLED=1 なら
#   - /tmp/ai_posting_scheduler.lock のファイルロックを取得し
#   - from .tasks import init_scheduler を呼び出して scheduler.start()
# まで実行されます（あなたの app/__init__.py の実装をそのまま利用）

def main():
    # 念のため、環境変数が無い場合のデフォルトを指定（systemdでも設定します）
    os.environ.setdefault("SCHEDULER_ENABLED", "1")

    app = create_app()
    app.logger.info("✅ Jobs runner booted (scheduler should be started by create_app)")

    # 常駐（APSchedulerはバックグラウンドスレッドなので、プロセスが即終了しないよう待機）
    signal.pause()

if __name__ == "__main__":
    main()

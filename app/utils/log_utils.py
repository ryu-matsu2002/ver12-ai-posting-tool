import re

def parse_logs(raw: str) -> list[str]:
    lines = raw.strip().split("\n")
    parsed = []

    for line in lines:
        msg = line.strip()

        # 翻訳マッピング
        if "Scheduler started" in msg:
            msg = msg.replace("Scheduler started", "🕒 スケジューラが起動しました")
        if "TokenUsageLog保存失敗" in msg:
            msg = "❗ トークン使用量の記録に失敗しました"
        if "WARNING" in msg:
            msg = "⚠️ 警告：" + msg
        if "ERROR" in msg or "Traceback" in msg:
            msg = "❌ エラー：" + msg

        parsed.append(msg)

    return parsed[-30:]  # 最新30行に限定

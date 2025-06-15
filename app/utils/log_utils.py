import re

def parse_logs(raw_lines) -> list[dict]:
    """
    ログ行を整形して、ログレベルごとの絵文字・翻訳・色クラス付きで返す。
    入力は文字列または行リストどちらにも対応。
    """
    if isinstance(raw_lines, str):
        lines = raw_lines.strip().split("\n")
    else:
        lines = raw_lines

    parsed = []

    for line in lines:
        msg = line.strip()

        # 初期値
        level = "info"
        emoji = "ℹ️"
        color_class = "text-blue-600"

        # 翻訳マッピング（キーワード検出）
        if "Scheduler started" in msg:
            msg = msg.replace("Scheduler started", "🕒 スケジューラが起動しました")

        if "TokenUsageLog保存失敗" in msg:
            msg = "❗ トークン使用量の記録に失敗しました"

        # ログレベル判定
        if "ERROR" in msg or "Traceback" in msg:
            level = "error"
            emoji = "❌"
            color_class = "text-red-600"
            msg = f"{emoji} エラー: {msg}"

        elif "WARNING" in msg:
            level = "warning"
            emoji = "⚠️"
            color_class = "text-yellow-600"
            msg = f"{emoji} 警告: {msg}"

        elif "DEBUG" in msg:
            level = "debug"
            emoji = "🐞"
            color_class = "text-gray-500"
            msg = f"{emoji} デバッグ: {msg}"

        else:
            level = "info"
            emoji = "ℹ️"
            color_class = "text-blue-600"
            msg = f"{emoji} {msg}"

        parsed.append({
            "level": level,
            "color": color_class,
            "message": msg
        })

    return parsed[-30:]  # 最新30件に限定

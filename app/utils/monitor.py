import os
import subprocess
from datetime import datetime

def get_memory_usage():
    """メモリとスワップの使用量を取得"""
    result = {}
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
            mem_total = int([x for x in lines if "MemTotal" in x][0].split()[1]) // 1024
            mem_free = int([x for x in lines if "MemAvailable" in x][0].split()[1]) // 1024
            swap_total = int([x for x in lines if "SwapTotal" in x][0].split()[1]) // 1024
            swap_free = int([x for x in lines if "SwapFree" in x][0].split()[1]) // 1024
            result = {
                "mem_total": mem_total,
                "mem_used": mem_total - mem_free,
                "swap_total": swap_total,
                "swap_used": swap_total - swap_free
            }
    except Exception as e:
        result = {"error": str(e)}
    return result

def get_cpu_load():
    """Load averageを取得"""
    try:
        load1, load5, load15 = os.getloadavg()
        return {
            "load_1min": round(load1, 2),
            "load_5min": round(load5, 2),
            "load_15min": round(load15, 2)
        }
    except Exception as e:
        return {"error": str(e)}

def get_latest_restart_log(n=10):
    """再起動ログファイルの最新n行を返す"""
    log_path = "/var/log/ai_restart.log"
    if not os.path.exists(log_path):
        return ["ログファイルが存在しません。"]
    try:
        output = subprocess.check_output(["tail", f"-n{n}", log_path], text=True)
        return output.strip().split("\n")
    except Exception as e:
        return [f"エラー: {e}"]

def get_last_restart_time():
    """systemdの最終起動時間"""
    try:
        output = subprocess.check_output(
            ["systemctl", "show", "ai-posting.service", "--property=ActiveEnterTimestamp"],
            text=True
        )
        return output.strip().split("=")[-1]
    except Exception as e:
        return f"取得エラー: {e}"

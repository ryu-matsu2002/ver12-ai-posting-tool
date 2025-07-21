# /var/www/ver12-ai-posting-tool/scripts/log_memory.py

import psutil
from datetime import datetime

log_file = "/var/log/mem_usage.log"

def get_memory_info():
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "used": round(mem.used / (1024 * 1024), 2),
        "available": round(mem.available / (1024 * 1024), 2),
        "swap_used": round(swap.used / (1024 * 1024), 2),
        "swap_total": round(swap.total / (1024 * 1024), 2)
    }

def log_memory():
    info = get_memory_info()
    log_line = (
        f"{info['timestamp']} | RAM Used: {info['used']}MB | "
        f"Available: {info['available']}MB | "
        f"Swap Used: {info['swap_used']}MB / {info['swap_total']}MB\n"
    )
    with open(log_file, "a") as f:
        f.write(log_line)

if __name__ == "__main__":
    log_memory()

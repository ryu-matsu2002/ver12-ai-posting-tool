import asyncio
import threading
import time
from typing import Any, Dict, Optional
from concurrent.futures import TimeoutError as FutureTimeout

# --- 内部状態 ---
_captcha_sessions: Dict[str, Any] = {}
_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_thread: Optional[threading.Thread] = None
_loop_ready = threading.Event()   # ← 追加：ループ準備完了シグナル
_lock = threading.Lock()          # dict への同時アクセス保護

# --- バックグラウンドイベントループを常駐起動 ---
def _start_loop_in_thread() -> None:
    global _loop, _loop_thread
    if _loop_thread and _loop_thread.is_alive():
        return

    def runner():
        global _loop
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        _loop_ready.set()         # ← ループ作成完了を通知
        _loop.run_forever()

    _loop_ready.clear()
    _loop_thread = threading.Thread(target=runner, name="pw-controller-loop", daemon=True)
    _loop_thread.start()

_start_loop_in_thread()

# ========= ここから async API（Playwright 側から使う） =========

async def store_session(session_id: str, page: Any) -> None:
    with _lock:
        _captcha_sessions[session_id] = page

async def get_session(session_id: str) -> Optional[Any]:
    with _lock:
        return _captcha_sessions.get(session_id)

async def delete_session(session_id: str) -> None:
    with _lock:
        page = _captcha_sessions.pop(session_id, None)
    if page:
        try:
            await page.close()
        except Exception:
            pass

# ========= ここから同期ラッパ（Flask など同期コードから使う） =========

def _ensure_loop_ready(timeout: float = 5.0) -> asyncio.AbstractEventLoop:
    """
    バックグラウンドイベントループの起動を待機して返す。
    """
    _start_loop_in_thread()
    # ループができるまで待つ（最大 timeout 秒）
    if not _loop_ready.wait(timeout=timeout):
        raise RuntimeError("Background asyncio loop failed to start in time")
    assert _loop is not None
    return _loop

def run_coro_sync(coro, timeout: Optional[float] = None):
    """
    同期コードから async を安全に実行。
    gthread の別スレッドからでも OK。
    """
    loop = _ensure_loop_ready()
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return fut.result(timeout=timeout)
    except FutureTimeout:
        fut.cancel()
        raise

def store_session_sync(session_id: str, page: Any) -> None:
    run_coro_sync(store_session(session_id, page))

def get_session_sync(session_id: str) -> Optional[Any]:
    return run_coro_sync(get_session(session_id))

def delete_session_sync(session_id: str) -> None:
    run_coro_sync(delete_session(session_id))

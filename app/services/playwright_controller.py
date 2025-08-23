# app/services/playwright_controller.py
import asyncio
import threading
from typing import Any, Dict, Optional

# --- 内部状態 ---
_captcha_sessions: Dict[str, Any] = {}
_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_thread: Optional[threading.Thread] = None
_lock = threading.Lock()  # dict への同時アクセス保護（保険）

# --- バックグラウンドイベントループを常駐起動 ---
def _start_loop_in_thread() -> None:
    global _loop, _loop_thread
    if _loop and _loop.is_running():
        return

    def runner():
        global _loop
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        _loop.run_forever()

    _loop_thread = threading.Thread(target=runner, name="pw-controller-loop", daemon=True)
    _loop_thread.start()

_start_loop_in_thread()


# ========= ここから async API（Playwright 側から使う） =========

async def store_session(session_id: str, page: Any) -> None:
    # Playwright 実行スレッドから await される想定
    with _lock:
        _captcha_sessions[session_id] = page

async def get_session(session_id: str) -> Optional[Any]:
    with _lock:
        return _captcha_sessions.get(session_id)

async def delete_session(session_id: str) -> None:
    with _lock:
        page = _captcha_sessions.pop(session_id, None)
    # page.close() は await が必要（Playwright の Page は async close）
    if page:
        try:
            await page.close()
        except Exception:
            # close 失敗は無視してよい（ブラウザ側で既に閉じている等）
            pass


# ========= ここから同期ラッパ（Flask など同期コードから使う） =========

def _ensure_loop_ready() -> asyncio.AbstractEventLoop:
    # バックグラウンドループ起動保証
    _start_loop_in_thread()
    assert _loop is not None
    return _loop

def run_coro_sync(coro):
    """
    同期コードから async 関数を実行して結果を返す。
    （gthread ワーカーの別スレッドでも安全）
    """
    loop = _ensure_loop_ready()
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    return fut.result()

def store_session_sync(session_id: str, page: Any) -> None:
    run_coro_sync(store_session(session_id, page))

def get_session_sync(session_id: str) -> Optional[Any]:
    return run_coro_sync(get_session(session_id))

def delete_session_sync(session_id: str) -> None:
    run_coro_sync(delete_session(session_id))

# app/services/playwright_controller.py
import asyncio
import threading
from typing import Any, Dict, Optional, Tuple
from concurrent.futures import TimeoutError as FutureTimeout

# セッションは (page, owner_loop) を保存
_Session = Tuple[Any, asyncio.AbstractEventLoop]

_captcha_sessions: Dict[str, _Session] = {}
_lock = threading.Lock()

# ---- 汎用: 任意の loop で coro を同期実行 ----
def _run_in_loop_sync(loop: asyncio.AbstractEventLoop, coro, timeout: Optional[float] = None):
    if loop.is_closed():
        raise RuntimeError("Owner event loop is closed")
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return fut.result(timeout=timeout)
    except FutureTimeout:
        fut.cancel()
        raise

# ========= async API（Playwright 側から使う） =========
async def store_session(session_id: str, page: Any) -> None:
    owner_loop = asyncio.get_running_loop()   # ← page を作った “そのループ”
    with _lock:
        _captcha_sessions[session_id] = (page, owner_loop)

async def get_session(session_id: str) -> Optional[_Session]:
    with _lock:
        return _captcha_sessions.get(session_id)

async def delete_session(session_id: str) -> None:
    # ページと所有ループを取り出して、所有ループ上で close する
    with _lock:
        sess = _captcha_sessions.pop(session_id, None)

    if not sess:
        return

    page, owner_loop = sess
    try:
        # page.close() は “その page を作ったループ” で
        await asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(page.close(), owner_loop)
        )
    except Exception:
        pass

# ========= 同期ラッパ（Flask などから使う） =========
def get_session_sync(session_id: str) -> Optional[_Session]:
    # get はどのループでもよいので “今のスレッド” の一時ループで実行
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(get_session(session_id))
    finally:
        loop.close()

def run_on_owner_loop_sync(owner_loop: asyncio.AbstractEventLoop, coro, timeout: Optional[float] = None):
    return _run_in_loop_sync(owner_loop, coro, timeout=timeout)

def delete_session_sync(session_id: str) -> None:
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(delete_session(session_id))
    finally:
        loop.close()

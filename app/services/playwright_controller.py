# app/services/playwright_controller.py
import asyncio
import threading
from typing import Any, Dict, Optional

# セッション保管（PlaywrightのPageを格納）
_captcha_sessions: Dict[str, Any] = {}

# オーナーイベントループ（常駐）
_owner_loop: Optional[asyncio.AbstractEventLoop] = None
_owner_thread: Optional[threading.Thread] = None

# スレッド安全のためのロック
_map_lock = threading.Lock()
_loop_lock = threading.Lock()


def _owner_loop_alive() -> bool:
    """オーナーループが有効かを判定"""
    global _owner_loop, _owner_thread
    if _owner_loop is None or _owner_thread is None:
        return False
    if _owner_loop.is_closed():
        return False
    if not _owner_thread.is_alive():
        return False
    return True


def _start_owner_loop() -> None:
    """オーナーイベントループをバックグラウンドスレッドで起動"""
    global _owner_loop, _owner_thread

    with _loop_lock:
        # すでに健全に動いていれば何もしない
        if _owner_loop_alive():
            return

        # もし存在していれば捨てる（閉じ済み想定）
        _owner_loop = None
        _owner_thread = None

        def _runner():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # set back to globals inside the thread
            global _owner_loop
            _owner_loop = loop
            loop.run_forever()

        t = threading.Thread(target=_runner, name="pw-owner-loop", daemon=True)
        t.start()
        _owner_thread = t


def _ensure_owner_loop() -> asyncio.AbstractEventLoop:
    """常に使えるオーナーループを返す（必要なら起動し直す）"""
    _start_owner_loop()
    assert _owner_loop is not None
    return _owner_loop


def _run_in_loop_sync(loop: asyncio.AbstractEventLoop, coro, timeout: Optional[float] = None):
    """与えられたループでコルーチンを同期実行"""
    if loop.is_closed():
        # ここで閉じていたら再起動してやり直す
        _start_owner_loop()
        loop = _ensure_owner_loop()

    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    return fut.result(timeout=timeout)


# ======== 外部公開：同期ラッパ ========

def run_on_owner_loop_sync(coro, timeout: Optional[float] = None):
    """
    同期コード（Flask/gthread）から Playwright の async 関数を安全に叩く。
    ループが閉じていても自動で再起動。
    """
    loop = _ensure_owner_loop()
    return _run_in_loop_sync(loop, coro, timeout=timeout)


def store_session_sync(session_id: str, page: Any) -> None:
    with _map_lock:
        _captcha_sessions[session_id] = page


def get_session_sync(session_id: str) -> Optional[Any]:
    with _map_lock:
        return _captcha_sessions.get(session_id)


def delete_session_sync(session_id: str) -> None:
    """
    セッションからPageを取り出し、必ず async close() を実行（同期的に待つ）。
    """
    page = None
    with _map_lock:
        page = _captcha_sessions.pop(session_id, None)

    if page is not None:
        async def _close():
            try:
                await page.close()
            except Exception:
                # 既に閉じられている等は無視
                pass
        # ここで確実にawaitさせる（「was never awaited」を防止）
        run_on_owner_loop_sync(_close())


# ======== 互換：async API（必要であれば利用可） ========

async def store_session(session_id: str, page: Any) -> None:
    with _map_lock:
        _captcha_sessions[session_id] = page

async def get_session(session_id: str) -> Optional[Any]:
    with _map_lock:
        return _captcha_sessions.get(session_id)

async def delete_session(session_id: str) -> None:
    page = None
    with _map_lock:
        page = _captcha_sessions.pop(session_id, None)
    if page is not None:
        try:
            await page.close()
        except Exception:
            pass

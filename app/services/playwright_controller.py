# app/services/playwright_controller.py
import asyncio
import threading
from typing import Any, Dict, Optional, Tuple

# セッション保管：session_id -> (page, owner_loop)
_captcha_sessions: Dict[str, Tuple[Any, Optional[asyncio.AbstractEventLoop]]] = {}

# オーナーイベントループ（常駐：フォールバック用）
_owner_loop: Optional[asyncio.AbstractEventLoop] = None
_owner_thread: Optional[threading.Thread] = None

# スレッド安全のためのロック
_map_lock = threading.Lock()
_loop_lock = threading.Lock()


def _owner_loop_alive() -> bool:
    """常駐オーナーループが有効かを判定"""
    global _owner_loop, _owner_thread
    if _owner_loop is None or _owner_thread is None:
        return False
    if _owner_loop.is_closed():
        return False
    if not _owner_thread.is_alive():
        return False
    return True


def _start_owner_loop() -> None:
    """常駐オーナーイベントループをバックグラウンドスレッドで起動"""
    global _owner_loop, _owner_thread

    with _loop_lock:
        if _owner_loop_alive():
            return

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
    """常駐オーナーループを返す（必要なら起動）"""
    _start_owner_loop()
    assert _owner_loop is not None
    return _owner_loop


def _run_in_loop_sync(loop: asyncio.AbstractEventLoop, coro, timeout: Optional[float] = None):
    """指定されたイベントループでコルーチンを同期実行"""
    if loop.is_closed():
        raise RuntimeError("Owner event loop is closed")
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    return fut.result(timeout=timeout)


# ======== 外部公開：同期ラッパ ========

def run_on_owner_loop_sync(coro, timeout: Optional[float] = None, loop: Optional[asyncio.AbstractEventLoop] = None):
    """
    同期コード（Flask/gthread）から async 関数を安全に叩く。
    loop を明示指定した場合はそのループで、未指定なら常駐ループで実行。
    """
    target_loop = loop or _ensure_owner_loop()
    return _run_in_loop_sync(target_loop, coro, timeout=timeout)


def store_session_sync(session_id: str, page: Any, owner_loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
    """
    同期側からセッションを保存。owner_loop を渡せる場合は一緒に保存。
    渡せない場合は (page, None) として保存し、close 時は安全フォールバック。
    """
    with _map_lock:
        _captcha_sessions[session_id] = (page, owner_loop)


def get_session_sync(session_id: str) -> Optional[Tuple[Any, Optional[asyncio.AbstractEventLoop]]]:
    """
    同期側からセッション取得。常に (page, owner_loop) を返す。
    """
    with _map_lock:
        sess = _captcha_sessions.get(session_id)
    if sess is None:
        return None
    # 旧データ互換（page 単体で保存されていた場合）
    if not isinstance(sess, tuple):
        return (sess, None)
    return sess


def delete_session_sync(session_id: str) -> None:
    """
    同期側：セッションから (page, owner_loop) を取り出し、
    可能なら owner_loop 上で page.close() を await（同期待機）する。
    """
    with _map_lock:
        sess = _captcha_sessions.pop(session_id, None)

    if not sess:
        return

    # 旧互換：単体 page の可能性
    if not isinstance(sess, tuple):
        page, owner_loop = sess, None
    else:
        page, owner_loop = sess

    if page is None:
        return

    async def _close():
        try:
            await page.close()
        except Exception:
            # 既に閉じられている等は無視
            pass

    # まずは owner_loop が健在ならそこで実行
    if owner_loop is not None and not owner_loop.is_closed():
        try:
            _run_in_loop_sync(owner_loop, _close(), timeout=10.0)
            return
        except Exception:
            # ダメならフォールバックへ
            pass

    # フォールバック：常駐ループ上で close を試みる（多くの場合は
    # 作成ループ以外での操作は不可だが、RuntimeWarning を避ける目的で実施）
    try:
        run_on_owner_loop_sync(_close(), timeout=10.0)
    except Exception:
        # それでも失敗する場合は諦めて破棄（リーク防止のため既に pop 済み）
        pass


# ======== 互換：async API（必要であれば利用可） ========

async def store_session(session_id: str, page: Any) -> None:
    """
    async 側（Playwright 実行スレッドなど）から保存。
    ページを生成したイベントループ（= 現在のループ）を一緒に保存する。
    """
    owner_loop = asyncio.get_running_loop()
    with _map_lock:
        _captcha_sessions[session_id] = (page, owner_loop)


async def get_session(session_id: str) -> Optional[Tuple[Any, Optional[asyncio.AbstractEventLoop]]]:
    """
    async 側：常に (page, owner_loop) を返す。
    """
    with _map_lock:
        sess = _captcha_sessions.get(session_id)
    if sess is None:
        return None
    if not isinstance(sess, tuple):
        return (sess, None)
    return sess


async def delete_session(session_id: str) -> None:
    """
    async 側：保存された owner_loop 上で close する必要があるが、
    ここは「今いるループで」await できるので通常の close を試す。
    もし作成ループと異なることでエラーになる場合は、呼び出し元で
    delete_session_sync() を使うのが安全。
    """
    with _map_lock:
        sess = _captcha_sessions.pop(session_id, None)

    if not sess:
        return

    page = sess[0] if isinstance(sess, tuple) else sess
    if page is None:
        return

    try:
        await page.close()
    except Exception:
        pass

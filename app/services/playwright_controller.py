# app/services/playwright_controller.py
import asyncio
import threading
import logging
from typing import Any, Dict, Optional, Tuple, Union

"""
Playwright セッション管理（page と、それを生成した owner_loop を”対”で保持）
- 同期側（Flask/gthread）から安全に async 関数を叩くためのラッパを提供
- セッションは session_id -> (page, owner_loop) のタプルで保存（後方互換あり）
"""

logger = logging.getLogger(__name__)

# セッション保管：session_id -> (page, owner_loop)
#   後方互換：page 単体で保存されている場合にも対応する
_captcha_sessions: Dict[str, Tuple[Any, Optional[asyncio.AbstractEventLoop]]] = {}

# オーナーイベントループ（常駐：フォールバック用）
_owner_loop: Optional[asyncio.AbstractEventLoop] = None
_owner_thread: Optional[threading.Thread] = None

# スレッド安全のためのロック
_map_lock = threading.Lock()
_loop_lock = threading.Lock()


# ─────────────────────────────────────────────
# 内部：常駐オーナーループの起動・判定
# ─────────────────────────────────────────────
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

        # 既存を破棄（閉じ済み想定）
        _owner_loop = None
        _owner_thread = None

        def _runner():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # set back to globals inside the thread
            global _owner_loop
            _owner_loop = loop
            logger.info("[pwctl] owner loop started: %r", loop)
            try:
                loop.run_forever()
            finally:
                logger.info("[pwctl] owner loop finished: %r", loop)

        t = threading.Thread(target=_runner, name="pw-owner-loop", daemon=True)
        t.start()
        _owner_thread = t


def _ensure_owner_loop() -> asyncio.AbstractEventLoop:
    """常駐オーナーループを返す（必要なら起動）"""
    _start_owner_loop()
    assert _owner_loop is not None
    return _owner_loop


# ─────────────────────────────────────────────
# 内部：指定ループで同期実行
# ─────────────────────────────────────────────
def _run_in_loop_sync(loop: asyncio.AbstractEventLoop, coro, timeout: Optional[float] = None):
    """
    指定されたイベントループでコルーチンを同期実行。
    - 非常駐の owner_loop が閉じている場合は RuntimeError を投げる
    - 常駐オーナーループが閉じている場合は再起動して再試行
    """
    if loop.is_closed():
        # loop が常駐オーナーループなら再起動して再試行
        if loop is _owner_loop:
            logger.warning("[pwctl] owner loop was closed; restarting...")
            _start_owner_loop()
            fresh = _ensure_owner_loop()
            fut = asyncio.run_coroutine_threadsafe(coro, fresh)
            return fut.result(timeout=timeout)
        # 非常駐（＝page 生成元のオリジン）なら失敗として扱う
        raise RuntimeError("Owner event loop is closed")

    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    return fut.result(timeout=timeout)


# ─────────────────────────────────────────────
# 公開：同期ラッパ
# ─────────────────────────────────────────────
def run_on_owner_loop_sync(
    coro,
    timeout: Optional[float] = None,
    loop: Optional[asyncio.AbstractEventLoop] = None,
):
    """
    同期コード（Flask/gthread）から async 関数を安全に実行。
    - loop を明示指定した場合：そのループで実行（閉じていれば RuntimeError）
    - 指定なしの場合：常駐オーナーループで実行（閉じていれば自動再起動）
    """
    target_loop = loop or _ensure_owner_loop()
    logger.debug("[pwctl] run_on_owner_loop_sync: loop=%r, coro=%r", target_loop, getattr(coro, "__name__", type(coro)))
    return _run_in_loop_sync(target_loop, coro, timeout=timeout)


# ─────────────────────────────────────────────
# 公開：同期 API（セッション管理）
# ─────────────────────────────────────────────
def store_session_sync(session_id: str, page: Any, owner_loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
    """
    同期側からセッションを保存。owner_loop を渡せる場合は一緒に保存。
    渡せない場合は (page, None) として保存し、close 時は安全フォールバック。
    """
    with _map_lock:
        _captcha_sessions[session_id] = (page, owner_loop)
    logger.info("[pwctl] store_session_sync: session_id=%s, owner_loop=%r, page=%r", session_id, owner_loop, page)


def get_session_sync(session_id: str) -> Optional[Tuple[Any, Optional[asyncio.AbstractEventLoop]]]:
    """
    同期側からセッション取得。常に (page, owner_loop) を返す。
    """
    with _map_lock:
        sess = _captcha_sessions.get(session_id)
    if sess is None:
        logger.info("[pwctl] get_session_sync: session_id=%s -> None", session_id)
        return None
    # 旧データ互換（page 単体で保存されていた場合）
    if not isinstance(sess, tuple):
        logger.warning("[pwctl] get_session_sync: legacy entry detected (page only). session_id=%s", session_id)
        return (sess, None)
    logger.info("[pwctl] get_session_sync: session_id=%s -> (page=%r, owner_loop=%r)", session_id, sess[0], sess[1])
    return sess


def delete_session_sync(session_id: str) -> None:
    """
    同期側：セッションから (page, owner_loop) を取り出し、
    可能なら owner_loop 上で page.close() を await（同期待機）する。
    ※ ここでは「コルーチンを作る前に」ループの健全性を確認する。
    """
    with _map_lock:
        sess: Optional[Union[Tuple[Any, Optional[asyncio.AbstractEventLoop]], Any]] = _captcha_sessions.pop(session_id, None)

    if not sess:
        logger.info("[pwctl] delete_session_sync: session_id=%s -> no entry", session_id)
        return

    # 旧互換：単体 page の可能性
    if not isinstance(sess, tuple):
        page, owner_loop = sess, None
    else:
        page, owner_loop = sess

    if page is None:
        logger.info("[pwctl] delete_session_sync: session_id=%s -> page is None", session_id)
        return

    # まずはループ健全性をチェック（ここではまだコルーチンを作らない）
    if owner_loop is None or owner_loop.is_closed():
        logger.warning("[pwctl] delete_session_sync: owner_loop is None/closed (session_id=%s). Skip close()", session_id)
        # ループ不明/終了時は close を諦める（リークは dict から既に除去済み）
        return

    # ここで初めてコルーチンを生成して実行
    try:
        logger.info("[pwctl] delete_session_sync: closing page on owner_loop (session_id=%s)", session_id)
        run_on_owner_loop_sync(page.close(), timeout=10.0, loop=owner_loop)
    except Exception:
        # close 失敗は無視（既に閉じている等）
        logger.exception("[pwctl] delete_session_sync: page.close() failed (ignored). session_id=%s", session_id)


# ─────────────────────────────────────────────
# 公開：async API（必要に応じて利用可）
# ─────────────────────────────────────────────
async def store_session(session_id: str, page: Any) -> None:
    """
    async 側（Playwright 実行スレッドなど）から保存。
    ページを生成したイベントループ（= 現在のループ）を一緒に保存する。
    """
    owner_loop = asyncio.get_running_loop()
    with _map_lock:
        _captcha_sessions[session_id] = (page, owner_loop)
    logger.info("[pwctl] store_session(async): session_id=%s, owner_loop=%r, page=%r", session_id, owner_loop, page)


async def get_session(session_id: str) -> Optional[Tuple[Any, Optional[asyncio.AbstractEventLoop]]]:
    """
    async 側：常に (page, owner_loop) を返す。
    """
    with _map_lock:
        sess = _captcha_sessions.get(session_id)
    if sess is None:
        logger.info("[pwctl] get_session(async): session_id=%s -> None", session_id)
        return None
    if not isinstance(sess, tuple):
        logger.warning("[pwctl] get_session(async): legacy entry detected (page only). session_id=%s", session_id)
        return (sess, None)
    logger.info("[pwctl] get_session(async): session_id=%s -> (page=%r, owner_loop=%r)", session_id, sess[0], sess[1])
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
        logger.info("[pwctl] delete_session(async): session_id=%s -> no entry", session_id)
        return

    page = sess[0] if isinstance(sess, tuple) else sess
    if page is None:
        logger.info("[pwctl] delete_session(async): session_id=%s -> page is None", session_id)
        return

    try:
        logger.info("[pwctl] delete_session(async): closing page (session_id=%s)", session_id)
        await page.close()
    except Exception:
        logger.exception("[pwctl] delete_session(async): page.close() failed (ignored). session_id=%s", session_id)
        pass

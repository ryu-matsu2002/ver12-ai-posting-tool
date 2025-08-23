# app/services/playwright_controller.py
import asyncio
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)
logger.info("[PW-Controller] bootstrap start")

# セッションをメモリに保持（単一ワーカー前提）
_SESSIONS: Dict[str, dict] = {}
_LOCK = asyncio.Lock()

async def store_session(session_id: str, page) -> None:
    """
    Playwrightのpage(と関連context/browser)をsession_idで保持。
    """
    context = getattr(page, "context", None)
    browser = getattr(context, "browser", None) if context else None
    async with _LOCK:
        _SESSIONS[session_id] = {"page": page, "context": context, "browser": browser}
    logger.debug("[PW-Controller] stored session %s", session_id)

def load_session(session_id: str):
    """
    保存したpageを取得。存在しなければNone。
    """
    data = _SESSIONS.get(session_id)
    return data["page"] if data else None

def has_session(session_id: str) -> bool:
    return session_id in _SESSIONS

async def close_session(session_id: str) -> None:
    """
    page/context/browser を可能な範囲でクローズして登録を削除。
    """
    async with _LOCK:
        data = _SESSIONS.pop(session_id, None)

    if not data:
        return

    page = data.get("page")
    context = data.get("context")
    browser = data.get("browser")

    try:
        if page:
            await page.close()
    except Exception:
        pass
    try:
        if context:
            await context.close()
    except Exception:
        pass
    try:
        if browser:
            await browser.close()
    except Exception:
        pass

    logger.debug("[PW-Controller] closed session %s", session_id)
# --- backward compatibility wrappers for routes.py ---

def get_session(session_id: str):
    """routes.py 互換: load_session() の別名"""
    try:
        return load_session(session_id)  # 既存関数
    except NameError:
        # 念のため: 古い環境で load_session 未定義なら None 返し
        return None

def delete_session(session_id: str):
    """routes.py 互換: close_session() を同期的に呼ぶラッパ"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # もし既にイベントループが動いている場合は非同期タスクで破棄
            return asyncio.create_task(close_session(session_id))
        else:
            return loop.run_until_complete(close_session(session_id))
    except RuntimeError:
        # ループが無ければ新規に回して実行
        return asyncio.run(close_session(session_id))

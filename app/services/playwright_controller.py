# app/services/playwright_controller.py
import asyncio
import logging
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)

# 単一ワーカー前提のインメモリ保管
captcha_sessions: Dict[str, Any] = {}
_lock = asyncio.Lock()

async def store_session(session_id: str, page) -> None:
    async with _lock:
        captcha_sessions[session_id] = page
    logger.debug("[PW-Controller] store pid=%s id=%s", os.getpid(), session_id)

async def get_session(session_id: str):
    # awaitable として呼べることが重要（run_until_complete対応）
    page = captcha_sessions.get(session_id)
    logger.debug("[PW-Controller] get   pid=%s id=%s hit=%s",
                 os.getpid(), session_id, bool(page))
    return page

async def delete_session(session_id: str) -> None:
    async with _lock:
        page = captcha_sessions.pop(session_id, None)

    if not page:
        return

    # できる限り丁寧にクローズ
    try:
        ctx = getattr(page, "context", None)
        brw = getattr(ctx, "browser", None) if ctx else None

        try:
            await page.close()
        except Exception:
            pass
        try:
            if ctx:
                await ctx.close()
        except Exception:
            pass
        try:
            if brw:
                await brw.close()
        except Exception:
            pass
    except Exception:
        logger.exception("[PW-Controller] delete_session close failed id=%s", session_id)

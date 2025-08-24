# app/services/pw_controller.py
"""
PWController: Playwright をアプリ稼働中ずっと保持する長寿命コントローラ。
- 単一スレッド上の asyncio イベントループで Playwright/Browser を起動・維持
- セッション(session_id)ごとに {context, page, storage_state_path, provider, step, last_seen} を管理
- storage_state の保存/復旧で "ステートレス再開" を可能に
- run(coro) で同期コード(Flask)から安全に同一ループ上のコルーチンを実行
"""

from __future__ import annotations

import asyncio
import threading
import logging
import time
import uuid
import os
from pathlib import Path
from typing import Dict, Optional, Any, Tuple

from playwright.async_api import async_playwright, Page, BrowserContext

logger = logging.getLogger(__name__)

# セッション保存先（storage_state の JSON を置く場所）
SESS_DIR = Path("/tmp/captcha_sessions")
SESS_DIR.mkdir(parents=True, exist_ok=True)


class PWController:
    """Playwright を単一ループで管理するコントローラ（シングルトン運用想定）"""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._playwright = None
        self._browser = None
        self._started = False

        # session_id -> 情報
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._sessions_lock = threading.Lock()

        # セッションの TTL（秒）。アクセスがなければ GC で自動クローズ
        self.ttl_sec: int = int(os.environ.get("PWCTL_TTL_SEC", "1800"))  # 30min

    # ────────────────────────────────────────────
    # ライフサイクル
    # ────────────────────────────────────────────
    def start(self, headless: bool = True) -> None:
        """コントローラ起動（アプリ起動時に一度だけ呼ぶ）"""
        if self._started:
            return
        self._started = True

        def _runner() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            logger.info("[pwctl] event loop started: %r", loop)
            loop.run_forever()
            logger.info("[pwctl] event loop finished: %r", loop)

        t = threading.Thread(target=_runner, name="pwctl-loop", daemon=True)
        t.start()
        self._thread = t

        # Playwright/Browser をループ上で起動
        self.run(self._boot(headless=headless))

        # GC タスク起動
        self.run(self._gc_task())

    def is_alive(self) -> bool:
        return bool(self._loop and self._thread and self._thread.is_alive())

    def run(self, coro, timeout: Optional[float] = None):
        """同期側から、内部ループ上でコルーチンを実行して結果を返す"""
        if not self.is_alive():
            raise RuntimeError("PWController loop is not alive. Did you call pwctl.start()?")

        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)  # type: ignore[arg-type]
        return fut.result(timeout=timeout)

    async def _boot(self, headless: bool = True) -> None:
        """内部ループ上で Playwright/Browser を起動"""
        if self._playwright is None:
            self._playwright = await async_playwright().start()
        if self._browser is None:
            # 必要なら env で channel="chrome" など切替可
            self._browser = await self._playwright.chromium.launch(headless=headless)
        logger.info("[pwctl] browser launched (headless=%s)", headless)

    async def shutdown(self) -> None:
        """明示的終了（通常は使わない）"""
        with self._sessions_lock:
            sids = list(self.sessions.keys())
        for sid in sids:
            await self._close_session(sid)
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                logger.exception("[pwctl] browser.close failed (ignored)")
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                logger.exception("[pwctl] playwright.stop failed (ignored)")

    # ────────────────────────────────────────────
    # セッション管理
    # ────────────────────────────────────────────
    async def create_session(
        self,
        *,
        storage_state_path: Optional[str] = None,
        user_agent: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Tuple[str, Page]:
        """
        storage_state を引き継いで context/page を生成。新しい session_id を返す。
        """
        ctx_kwargs = {}
        if storage_state_path and os.path.exists(storage_state_path):
            ctx_kwargs["storage_state"] = storage_state_path
        if user_agent:
            ctx_kwargs["user_agent"] = user_agent

        context: BrowserContext = await self._browser.new_context(**ctx_kwargs)  # type: ignore[arg-type]
        page: Page = await context.new_page()

        sid = str(uuid.uuid4())
        with self._sessions_lock:
            self.sessions[sid] = {
                "context": context,
                "page": page,
                "storage_state_path": storage_state_path,
                "provider": provider,
                "step": "init",
                "created_at": time.time(),
                "last_seen": time.time(),
            }
        logger.info("[pwctl] create_session: sid=%s provider=%s", sid, provider)
        return sid, page

    async def get_page(self, session_id: str) -> Optional[Page]:
        with self._sessions_lock:
            s = self.sessions.get(session_id)
        if not s:
            return None
        s["last_seen"] = time.time()
        return s.get("page")

    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._sessions_lock:
            s = self.sessions.get(session_id)
        if not s:
            return None
        s["last_seen"] = time.time()
        return s

    async def save_storage_state(self, session_id: str) -> Optional[str]:
        """現在の context の storage_state を JSON に保存し、パスを返す"""
        with self._sessions_lock:
            s = self.sessions.get(session_id)
        if not s:
            return None
        context: BrowserContext = s["context"]
        path = str(SESS_DIR / f"{session_id}.json")
        await context.storage_state(path=path)
        s["storage_state_path"] = path
        logger.info("[pwctl] save_storage_state: sid=%s -> %s", session_id, path)
        return path

    async def revive(self, session_id: str) -> Optional[Page]:
        """
        既存の page/context が壊れた時に、保存済み storage_state から新規に復旧。
        """
        with self._sessions_lock:
            s = self.sessions.get(session_id)
        if not s:
            logger.warning("[pwctl] revive: no session sid=%s", session_id)
            return None

        path = s.get("storage_state_path")
        try:
            # 古いものを掃除
            if s.get("page"):
                try:
                    await s["page"].close()
                except Exception:
                    pass
            if s.get("context"):
                try:
                    await s["context"].close()
                except Exception:
                    pass
        finally:
            pass

        ctx_kwargs = {}
        if path and os.path.exists(path):
            ctx_kwargs["storage_state"] = path

        context: BrowserContext = await self._browser.new_context(**ctx_kwargs)  # type: ignore[arg-type]
        page: Page = await context.new_page()

        with self._sessions_lock:
            s["context"] = context
            s["page"] = page
            s["last_seen"] = time.time()

        logger.info("[pwctl] revive: sid=%s restored (storage_state=%s)", session_id, bool(path))
        return page

    async def _close_session(self, session_id: str) -> None:
        with self._sessions_lock:
            s = self.sessions.pop(session_id, None)

        if not s:
            return
        # page/context を安全にクローズ
        try:
            if s.get("page"):
                try:
                    await s["page"].close()
                except Exception:
                    logger.exception("[pwctl] page.close failed (ignored) sid=%s", session_id)
            if s.get("context"):
                try:
                    await s["context"].close()
                except Exception:
                    logger.exception("[pwctl] context.close failed (ignored) sid=%s", session_id)
        finally:
            pass
        logger.info("[pwctl] close_session: sid=%s", session_id)

    def close_session(self, session_id: str) -> None:
        """同期側からセッションを即時クローズ"""
        try:
            self.run(self._close_session(session_id))
        except Exception:
            logger.exception("[pwctl] close_session failed (ignored) sid=%s", session_id)

    # ────────────────────────────────────────────
    # ユーティリティ
    # ────────────────────────────────────────────
    async def _gc_task(self) -> None:
        """TTL 監視の常駐GC。last_seen から ttl_sec 経過したセッションを順次クローズ"""
        while True:
            try:
                now = time.time()
                with self._sessions_lock:
                    victims = [sid for sid, s in self.sessions.items()
                               if now - s.get("last_seen", now) > self.ttl_sec]
                for sid in victims:
                    logger.info("[pwctl] GC: closing expired session sid=%s", sid)
                    await self._close_session(sid)
            except Exception:
                logger.exception("[pwctl] GC loop error (ignored)")
            await asyncio.sleep(60)

    # 便利ヘルパ（任意）：状態更新
    async def set_step(self, session_id: str, step: str) -> None:
        with self._sessions_lock:
            s = self.sessions.get(session_id)
            if s:
                s["step"] = step
                s["last_seen"] = time.time()

    async def set_provider(self, session_id: str, provider: Optional[str]) -> None:
        with self._sessions_lock:
            s = self.sessions.get(session_id)
            if s:
                s["provider"] = provider
                s["last_seen"] = time.time()


# シングルトン・インスタンス
pwctl = PWController()

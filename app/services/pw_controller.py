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
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from playwright.async_api import BrowserContext, Page, async_playwright

logger = logging.getLogger(__name__)

# セッション保存先（storage_state の JSON を置く場所）
SESS_DIR = Path("/tmp/captcha_sessions")
SESS_DIR.mkdir(parents=True, exist_ok=True)


def _human_age_sec(ts: float) -> str:
    try:
        return f"{int(time.time() - ts)}s"
    except Exception:
        return "n/a"


def _find_latest_state_json(max_age_sec: int = 3600, prefer_member_cookie: bool = True) -> Optional[str]:
    """
    直近で更新された storage_state JSON を返す（フォールバック用）。
    - max_age_sec 以内のものを採用
    - prefer_member_cookie=True の場合、member.livedoor.com の Cookie を含むものを優先
    """
    try:
        candidates = sorted(SESS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            return None

        def has_member_cookie(path: Path) -> bool:
            if not prefer_member_cookie:
                return True
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                for c in (data.get("cookies") or []):
                    d = c.get("domain") or ""
                    if "member.livedoor.com" in d:
                        return True
            except Exception:
                pass
            return False

        now = time.time()

        # まず条件に合うもの（member cookie あり）を探す
        for p in candidates:
            age = now - p.stat().st_mtime
            if age <= max_age_sec and has_member_cookie(p):
                return str(p)

        # 見つからなければ年齢条件だけで妥協
        for p in candidates:
            age = now - p.stat().st_mtime
            if age <= max_age_sec:
                return str(p)
    except Exception:
        logger.exception("[pwctl] _find_latest_state_json failed (ignored)")
    return None


class PWController:
    """Playwright を単一バックグラウンドループで管理するコントローラ（シングルトン運用想定）"""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._loop_ready = threading.Event()
        self._lock = threading.RLock()

        self._pw = None
        self._browser = None
        self._booted = False  # _boot 完了フラグ

        # session_id -> 情報
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._sessions_lock = threading.Lock()

        # セッションの TTL（秒）。アクセスがなければ GC で自動クローズ
        self.ttl_sec: int = int(os.environ.get("PWCTL_TTL_SEC", "1800"))  # 30min

    # ────────────────────────────────────────────
    # ループ管理
    # ────────────────────────────────────────────
    def _loop_alive(self) -> bool:
        return (
            self._loop is not None
            and not self._loop.is_closed()
            and self._thread is not None
            and self._thread.is_alive()
        )

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """
        バックグラウンドのイベントループを必ず起動して返す。
        既に起動済みならそのループを返す。
        """
        with self._lock:
            if self._loop_alive():
                return self._loop  # type: ignore[return-value]

            # 新規起動
            loop = asyncio.new_event_loop()

            def runner():
                asyncio.set_event_loop(loop)
                logger.info("[pwctl] event loop started: %r", loop)
                self._loop_ready.set()
                try:
                    loop.run_forever()
                finally:
                    logger.info("[pwctl] event loop finished: %r", loop)

            t = threading.Thread(target=runner, name="pwctl-loop", daemon=True)
            t.start()

            # 起動同期（race防止）
            self._thread = t
            self._loop = loop
            self._loop_ready.wait(timeout=5.0)
            return loop

    # ────────────────────────────────────────────
    # コルーチン実行（同期側 API）
    # ────────────────────────────────────────────
    def run(self, coro, timeout: Optional[float] = None):
        """
        同期側から内部ループ上でコルーチンを実行して結果を返す。
        ループ未起動でもここで必ず起動する。
        """
        loop = self._ensure_loop()
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        return fut.result(timeout=timeout)

    def spawn(self, coro):
        """
        Fire-and-forget: 結果を待たずにループ上でコルーチンを実行。
        戻り値は concurrent.futures.Future。
        """
        loop = self._ensure_loop()
        return asyncio.run_coroutine_threadsafe(coro, loop)

    # ────────────────────────────────────────────
    # ブラウザ起動・停止
    # ────────────────────────────────────────────
    async def _boot(self, headless: bool = True) -> None:
        """内部ループ上で Playwright/Browser を起動（多重起動安全）"""
        if self._booted:
            return
        if self._pw is None:
            self._pw = await async_playwright().start()
        if self._browser is None:
            self._browser = await self._pw.chromium.launch(
                headless=headless,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
        self._booted = True
        logger.info("[pwctl] booted (headless=%s)", headless)

    def start(self, headless: bool = True) -> None:
        """
        コントローラ起動（アプリ起動時に呼ぶ）。複数回呼ばれても安全。
        - ループを起動
        - ループ上で _boot 完了まで待機
        - GC タスクをバックグラウンドで起動（待たない）
        """
        self._ensure_loop()
        # ブラウザの確実な起動
        self.run(self._boot(headless=headless))
        # 無限 GC を“待たない”で起動
        self.spawn(self._gc_task())

    def is_alive(self) -> bool:
        return self._loop_alive()

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
            finally:
                self._browser = None

        if self._pw:
            try:
                await self._pw.stop()
            except Exception:
                logger.exception("[pwctl] playwright.stop failed (ignored)")
            finally:
                self._pw = None

        self._booted = False

    # ────────────────────────────────────────────
    # セッション管理
    # ────────────────────────────────────────────
    async def _ensure_browser(self) -> None:
        """create_session 等でブラウザ未起動の場合の保険"""
        if self._booted:
            return
        headless = os.getenv("PWCTL_HEADLESS", "1") == "1"
        await self._boot(headless=headless)

    async def create_session(
        self,
        *,
        storage_state_path: Optional[str] = None,
        user_agent: Optional[str] = None,
        provider: Optional[str] = None,
        # 追加：パス未指定時に直近の state を自動採用するフォールバック
        auto_load_latest: bool = True,
        # 追加：古すぎる state は採用しない（デフォルト 1 時間）
        latest_state_max_age_sec: int = 3600,
    ) -> Tuple[str, Page]:
        """
        storage_state を引き継いで context/page を生成。新しい session_id を返す。
        - storage_state_path が与えられればそれを採用し、ログに [pwctl] load_storage_state を出す
        - それが無い場合でも auto_load_latest=True なら /tmp/captcha_sessions の直近 JSON を採用
        """
        await self._ensure_browser()

        # ここで sid を先に払い出しておく（セッション管理のキー）
        sid = str(uuid.uuid4())

        # storage_state の決定
        chosen_state: Optional[str] = None
        chosen_note: str = ""
        if storage_state_path and os.path.exists(storage_state_path):
            chosen_state = storage_state_path
            try:
                mtime = Path(chosen_state).stat().st_mtime
                chosen_note = f"(explicit; age={_human_age_sec(mtime)})"
            except Exception:
                chosen_note = "(explicit)"
        elif auto_load_latest:
            latest = _find_latest_state_json(max_age_sec=latest_state_max_age_sec, prefer_member_cookie=True)
            if latest and os.path.exists(latest):
                chosen_state = latest
                try:
                    mtime = Path(chosen_state).stat().st_mtime
                    chosen_note = f"(auto; age={_human_age_sec(mtime)})"
                except Exception:
                    chosen_note = "(auto)"

        ctx_kwargs: Dict[str, Any] = {}
        if chosen_state:
            ctx_kwargs["storage_state"] = chosen_state
            logger.info("[pwctl] load_storage_state: sid=%s -> %s %s", sid, chosen_state, chosen_note)
        else:
            logger.info("[pwctl] load_storage_state: sid=%s -> (none)", sid)

        if user_agent:
            ctx_kwargs["user_agent"] = user_agent

        context: BrowserContext = await self._browser.new_context(**ctx_kwargs)  # type: ignore[arg-type]
        page: Page = await context.new_page()

        with self._sessions_lock:
            self.sessions[sid] = {
                "context": context,
                "page": page,
                "storage_state_path": chosen_state,
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
        await self._ensure_browser()

        with self._sessions_lock:
            s = self.sessions.get(session_id)
        if not s:
            logger.warning("[pwctl] revive: no session sid=%s", session_id)
            return None

        path = s.get("storage_state_path")
        # 古いものを掃除
        if s.get("page"):
            try:
                await s["page"].close()
            except Exception:
                logger.exception("[pwctl] revive: old page.close failed (ignored)")
        if s.get("context"):
            try:
                await s["context"].close()
            except Exception:
                logger.exception("[pwctl] revive: old context.close failed (ignored)")

        ctx_kwargs: Dict[str, Any] = {}
        if path and os.path.exists(path):
            ctx_kwargs["storage_state"] = path
            try:
                mtime = Path(path).stat().st_mtime
                note = f"(age={_human_age_sec(mtime)})"
            except Exception:
                note = ""
            logger.info("[pwctl] load_storage_state: sid=%s -> %s %s", session_id, path, note)
        else:
            logger.info("[pwctl] load_storage_state: sid=%s -> (none)", session_id)

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
    # 常駐 GC
    # ────────────────────────────────────────────
    async def _gc_task(self) -> None:
        """TTL 監視の常駐GC。last_seen から ttl_sec 経過したセッションを順次クローズ"""
        while True:
            try:
                now = time.time()
                with self._sessions_lock:
                    victims = [
                        sid
                        for sid, s in list(self.sessions.items())
                        if now - s.get("last_seen", now) > self.ttl_sec
                    ]
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

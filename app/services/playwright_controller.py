# app/services/playwright_controller.py
from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from typing import Dict, Optional
from pathlib import Path
from datetime import datetime
import logging
import os
from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)


class _LDSession:
    """1セッション（=1ユーザーの仮登録～完成まで）を保持"""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.context = None
        self.page = None
        self.email = None
        self.nickname = None
        self.password = None
        self.desired_blog_id = None
        self.created_filename = None


class PlaywrightController:
    """
    専用スレッド上で単一の asyncio イベントループを常駐。
    すべての Playwright 操作はこのループに “run_coroutine_threadsafe” で投げる。
    """
    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready_evt = threading.Event()
        self._boot_evt = threading.Event()   # ブート完了イベント

        self._p = None           # async_playwright() の start() 結果
        self._browser = None     # 単一ブラウザ（セッションごとに context）
        self._sessions: Dict[str, _LDSession] = {}

        self._start_thread()

    # ---------- スレッド＆ループ常駐 ----------
    def _start_thread(self):
        def _runner():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.create_task(self._bootstrap())
            self._ready_evt.set()
            try:
                self._loop.run_forever()
            finally:
                try:
                    self._loop.run_until_complete(self._teardown())
                finally:
                    asyncio.set_event_loop(None)

        self._thread = threading.Thread(target=_runner, name="playwright-controller", daemon=True)
        self._thread.start()
        self._ready_evt.wait()

    async def _bootstrap(self):
        logger.info("[PW-Controller] bootstrap start")
        self._p = await async_playwright().start()
        # headless=True でOK。必要なら args を追加
        self._browser = await self._p.chromium.launch(headless=True, args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-infobars",
            "--disable-dev-shm-usage",
        ])
        logger.info("[PW-Controller] browser launched")
        # ブート完了を通知（スレッドセーフ）
        self._boot_evt.set()

    async def _teardown(self):
        try:
            for s in list(self._sessions.values()):
                try:
                    if s.context:
                        await s.context.close()
                except Exception:
                    pass
            self._sessions.clear()
        finally:
            try:
                if self._browser:
                    await self._browser.close()
            except Exception:
                pass
            try:
                if self._p:
                    await self._p.stop()
            except Exception:
                pass
            logger.info("[PW-Controller] teardown complete")

    # ---------- 外部から呼ぶ同期ラッパ ----------
    def _submit(self, coro) -> Future:
        if not self._loop:
            raise RuntimeError("Playwright controller loop is not ready")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    # ↓ prepare_captcha 相当：CAPTCHA画像を取得してファイル名を返す
    def start_session_sync(self, session_id: str, email: str, nickname: str, password: str,
                           desired_blog_id: Optional[str], timeout: float = 90.0) -> dict:
        # ブート完了を待つ（タイムアウトは同じ値を使う）
        if not self._boot_evt.wait(timeout=timeout):
            raise TimeoutError("Playwright controller bootstrap timed out")
        logger.info(f"[PW-Controller] start_session_sync called (session_id={session_id})")
        fut = self._submit(self._start_session(session_id, email, nickname, password, desired_blog_id))
        return fut.result(timeout=timeout)

    # ↓ submit_captcha 相当：同じ page で続行、APIキーまで取得して返す
    def submit_captcha_sync(self, session_id: str, captcha_text: str, token: str, site,
                            timeout: float = 240.0) -> dict:
        if not self._boot_evt.is_set():
            raise RuntimeError("Playwright controller is not bootstrapped")
        logger.info(f"[PW-Controller] submit_captcha_sync called (session_id={session_id})")
        fut = self._submit(self._submit_captcha(session_id, captcha_text, token, site))
        return fut.result(timeout=timeout)

    # 明示破棄
    def close_session_sync(self, session_id: str, timeout: float = 15.0) -> None:
        fut = self._submit(self._close_session(session_id))
        try:
            fut.result(timeout=timeout)
        except Exception:
            pass

    # ---------- 内部コルーチン本体 ----------
    async def _start_session(self, session_id: str, email: str, nickname: str, password: str,
                             desired_blog_id: Optional[str]) -> dict:
        # 保存先を APP_ROOT 配下に固定（未設定なら CWD）
        app_root = Path(os.environ.get("APP_ROOT", ".")).resolve()
        CAPTCHA_SAVE_DIR = app_root / "app" / "static" / "captchas"
        CAPTCHA_SAVE_DIR.mkdir(parents=True, exist_ok=True)

        s = _LDSession(session_id)
        # ★ recover_atompub_key で使うのでセッションに保持（現状コードに抜けていた点）
        s.email = email
        s.nickname = nickname
        s.password = password
        s.desired_blog_id = desired_blog_id

        logger.info(f"[PW-Controller] _start_session: new_context (session_id={session_id})")
        s.context = await self._browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"),
            locale="ja-JP",
        )
        logger.info(f"[PW-Controller] _start_session: add_init_script")
        await s.context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            window.navigator.chrome = { runtime: {} };
            Object.defineProperty(navigator, 'languages', { get: () => ['ja-JP', 'ja'] });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
        """)

        logger.info(f"[PW-Controller] _start_session: new_page")
        s.page = await s.context.new_page()
        try:
            logger.info(f"[PW-Controller] _start_session: goto register/input (session_id={session_id})")
            await s.page.goto("https://member.livedoor.com/register/input", wait_until="load")
            logger.info(f"[PW-Controller] _start_session: page loaded")

            # 入力欄がロードされていることを念のため待機
            await s.page.wait_for_selector('input[name="livedoor_id"]', timeout=15000)

            await s.page.fill('input[name="livedoor_id"]', nickname)
            await s.page.fill('input[name="password"]', password)
            await s.page.fill('input[name="password2"]', password)
            await s.page.fill('input[name="email"]', email)
            await s.page.click('input[value="ユーザー情報を登録"]')

            logger.info(f"[PW-Controller] _start_session: wait_for #captcha-img")
            await s.page.wait_for_selector("#captcha-img", timeout=10000)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captcha_{session_id}_{ts}.png"
            filepath = CAPTCHA_SAVE_DIR / filename

            await s.page.locator("#captcha-img").screenshot(path=str(filepath))
            logger.info(f"[PW-Controller] CAPTCHA saved: {filepath}")

            self._sessions[session_id] = s
            s.created_filename = filename
            return {"filename": filename}
        except Exception as e:
            # 例外時は HTML/PNG をダンプ
            try:
                html = await s.page.content()
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                eh = f"/tmp/ld_prepare_exception_{ts}.html"
                ep = f"/tmp/ld_prepare_exception_{ts}.png"
                Path(eh).write_text(html, encoding="utf-8")
                await s.page.screenshot(path=ep)
                logger.error(f"[PW-Controller] _start_session failed: {e} (dumped {eh}, {ep})")
            except Exception:
                logger.exception(f"[PW-Controller] _start_session failed (no dump): {e}")
            # 失敗時は context を閉じる
            try:
                await s.context.close()
            except Exception:
                pass
            raise

    async def _submit_captcha(self, session_id: str, captcha_text: str, token: str, site) -> dict:
        """
        同じ page で CAPTCHA を入力→メール確認→AtomPub回収
        """
        from app.services.mail_utils.mail_tm import poll_latest_link_tm_async as poll_latest_link_gw
        from app.services.blog_signup.livedoor_atompub_recover import recover_atompub_key
        import asyncio

        s = self._sessions.get(session_id)
        if not s or not s.page:
            raise RuntimeError("session is missing or expired")

        # CAPTCHA送信
        await s.page.wait_for_selector("#captcha", state="visible", timeout=10000)
        await s.page.fill("#captcha", captcha_text)
        await s.page.click('input[id="commit-button"]')
        await s.page.wait_for_timeout(1500)

        html = await s.page.content()
        cur = s.page.url
        if ("仮登録メール" not in html) and (not cur.endswith("/register/done")):
            # 失敗の証跡
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            err_html = f"/tmp/ld_captcha_failed_{ts}.html"
            err_png = f"/tmp/ld_captcha_failed_{ts}.png"
            Path(err_html).write_text(html, encoding="utf-8")
            await s.page.screenshot(path=err_png)
            logger.warning(f"[PW-Controller] CAPTCHA failed: {err_html}, {err_png}")
            return {"captcha_success": False, "html_path": err_html, "png_path": err_png}

        # メール到着待ち → 認証URL取得
        url = None
        for i in range(6):  # 最大 ~30秒待つ
            u = await poll_latest_link_gw(token)
            if u:
                url = u
                break
            await asyncio.sleep(5)

        if not url:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            err_html = f"/tmp/ld_email_link_fail_{ts}.html"
            err_png = f"/tmp/ld_email_link_fail_{ts}.png"
            Path(err_html).write_text(await s.page.content(), encoding="utf-8")
            await s.page.screenshot(path=err_png)
            logger.error(f"[PW-Controller] mail link not found: {err_html}, {err_png}")
            return {"captcha_success": False, "error": "メール認証に失敗"}

        await s.page.goto(url)
        try:
            await s.page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass

        # AtomPub回収（必要に応じて desired_blog_id を渡す）
        rec = await recover_atompub_key(
            s.page, s.nickname, s.email, s.password, site, desired_blog_id=s.desired_blog_id
        )
        if not rec.get("success"):
            return {
                "captcha_success": True,
                "error": rec.get("error", "AtomPub再取得に失敗"),
                "html_path": rec.get("html_path"),
                "png_path": rec.get("png_path"),
            }

        return {
            "captcha_success": True,
            "blog_id": rec["blog_id"],
            "api_key": rec["api_key"],
            "endpoint": rec.get("endpoint"),
        }

    async def _close_session(self, session_id: str):
        s = self._sessions.pop(session_id, None)
        if not s:
            return
        try:
            if s.context:
                await s.context.close()
        except Exception:
            pass


# シングルトン的に使う
controller = PlaywrightController()

# 利便のための関数
def start_session_sync(session_id: str, email: str, nickname: str, password: str,
                       desired_blog_id: Optional[str], timeout: float = 90.0) -> dict:
    return controller.start_session_sync(session_id, email, nickname, password, desired_blog_id, timeout)

def submit_captcha_sync(session_id: str, captcha_text: str, token: str, site,
                        timeout: float = 240.0) -> dict:
    return controller.submit_captcha_sync(session_id, captcha_text, token, site, timeout)

def close_session_sync(session_id: str, timeout: float = 15.0) -> None:
    return controller.close_session_sync(session_id, timeout)

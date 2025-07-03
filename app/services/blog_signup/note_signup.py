"""
NOTE の会員登録自動化
playwright==1.53.0 で動作確認
"""

from playwright.sync_api import Page, sync_playwright, TimeoutError as PWTimeout
import logging, random, string, time

SIGNUP_URL = "https://note.com/signup"

__all__ = ["signup_note_account"]


def _random_wait(a=0.8, b=1.6):
    time.sleep(random.uniform(a, b))


def signup_note_account(email: str, password: str) -> dict:
    """
    Note アカウントを新規登録し、成功すれば True を返す。
    Returns
    -------
    dict : {"ok": bool, "error": str | None}
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"]
            )
            ctx = browser.new_context(locale="ja-JP")
            page: Page = ctx.new_page()

            # 1️⃣ サインアップページへ
            page.goto(SIGNUP_URL, timeout=30_000)
            page.wait_for_load_state("networkidle")

            # 2️⃣ メール／パスワード入力
            page.get_by_placeholder("mail@example.com").fill(email)
            _random_wait()
            page.get_by_placeholder("パスワード").fill(password)
            _random_wait()

            # 3️⃣ 「同意して登録」ボタン
            page.get_by_role("button", name="同意して登録").click()

            # 4️⃣ 完了ページへ遷移するまで待機（60s）
            page.wait_for_url("**/signup/complete**", timeout=60_000)

            browser.close()
            return {"ok": True, "error": None}

    except PWTimeout as e:
        logging.error("[note_signup] Timeout: %s", e)
        return {"ok": False, "error": f"Timeout: {e}"}

    except Exception as e:  # noqa: BLE001
        logging.exception("[note_signup] Unexpected error")
        return {"ok": False, "error": str(e)}

# app/services/blog_signup/note_signup.py  ★全コード

import random, string, time, logging
from playwright.sync_api import (
    sync_playwright,
    TimeoutError as PWTimeout,
    Error as PWError
)

LANDING = "https://note.com/signup?signup_type=email"
FORM    = "https://note.com/signup/form?redirectPath=%2Fsignup"

__all__ = ["signup_note_account"]


def _wait(a=0.6, b=1.5):
    time.sleep(random.uniform(a, b))


def _rand_ua() -> str:
    # Chrome 117〜125 あたりをランダム生成
    ver = random.randint(117, 125)
    build = random.randint(0, 9999)
    patch = random.randint(0, 199)
    return (f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            f"AppleWebKit/537.36 (KHTML, like Gecko) "
            f"Chrome/{ver}.0.{build}.{patch} Safari/537.36")


def signup_note_account(email: str, password: str) -> dict:
    """
    Note にメール+PWで会員登録。成功なら {"ok": True}
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                ],
            )
            ctx = browser.new_context(
                locale="ja-JP",
                viewport={"width": 1280, "height": 960},
                user_agent=_rand_ua(),
            )
            page = ctx.new_page()

            # webdriver フラグ偽装
            page.add_init_script("Object.defineProperty(navigator,'webdriver',{get:() => undefined})")

            # -- 1. ランディング → フォーム ---------------------------------
            page.goto(LANDING, timeout=30_000)
            page.wait_for_load_state("networkidle")

            try:
                page.locator("text=メールで登録").first.click(timeout=5_000)
                page.wait_for_url("**/signup/form**", timeout=15_000)
            except PWTimeout:
                page.goto(FORM, timeout=15_000)

            # -- 2. メール & パスワード入力 ----------------------------------
            email_sel = 'input[type="email"]'
            pass_sel  = 'input[type="password"]'

            page.fill(email_sel, email, timeout=15_000)
            _wait()
            page.fill(pass_sel, password)
            _wait()

            # -- 3. reCAPTCHA スコアが上がるのを最大 15 s 待つ --------------
            btn = page.locator("button:has-text('同意して登録')")
            for _ in range(30):         # 30 × 0.5 ≒ 15 s
                if btn.is_enabled():
                    break
                time.sleep(0.5)
            else:
                logging.error("[note_signup] signup button never enabled (captcha)")
                return {"ok": False, "error": "reCAPTCHA score too low → button disabled"}

            # -- 4. クリック → 完了ページ -----------------------------------
            btn.click()
            page.wait_for_url("**/signup/complete**", timeout=60_000)
            browser.close()
            return {"ok": True, "error": None}

    except PWTimeout as e:
        logging.error("[note_signup] Timeout: %s", e)
        return {"ok": False, "error": f"Timeout: {e}"}

    except PWError as e:
        logging.error("[note_signup] Playwright error: %s", e)
        return {"ok": False, "error": str(e)}

    except Exception as e:          # noqa: BLE001
        logging.exception("[note_signup] Unexpected")
        return {"ok": False, "error": str(e)}

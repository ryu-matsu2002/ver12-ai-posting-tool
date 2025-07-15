# app/services/agent/action_executor.py

import logging
from app.services.agent.field_extractor import extract_form_fields
from app.services.captcha_solver import solve

logger = logging.getLogger(__name__)

async def execute_signup_actions(page, html: str, values: dict, user_id: int = None) -> bool:
    """
    推論結果を元に Playwright で自律的にフォーム入力・送信を行う。
    
    Args:
        page: Playwrightのページオブジェクト
        html: ページのHTML文字列
        values: 入力に使うデータ（email, password, nickname）
        user_id: GPTログ保存に使用（任意）

    Returns:
        bool: True=成功 / False=失敗
    """
    fields = extract_form_fields(html)  # GPT推論結果（同期）

    try:
        # 1. 入力フィールド処理
        if sel := fields.get("email"):
            await page.fill(sel, values["email"])
            logger.info(f"✅ 入力: email → {sel}")
        if sel := fields.get("password"):
            await page.fill(sel, values["password"])
            logger.info(f"✅ 入力: password → {sel}")
        if sel := fields.get("password2"):
            await page.fill(sel, values["password"])
            logger.info(f"✅ 入力: password2 → {sel}")
        if sel := fields.get("nickname"):
            await page.fill(sel, values["nickname"])
            logger.info(f"✅ 入力: nickname → {sel}")

        # 2. CAPTCHA処理（画像セレクタがあれば）
        if sel := fields.get("captcha"):
            elem = await page.query_selector(sel)
            if elem:
                img_path = "/tmp/gpt_captcha.png"
                await elem.screenshot(path=img_path)
                solved = solve(img_path)
                await page.fill("input[name='captcha']", solved)
                logger.info(f"🧠 CAPTCHA突破成功 → {solved}")
            else:
                logger.warning("⚠️ CAPTCHA要素が見つかりません")

        # 3. 送信ボタンをクリック
        if sel := fields.get("submit"):
            await page.click(sel)
            logger.info(f"🚀 登録ボタンをクリック → {sel}")
        else:
            logger.warning("⚠️ 登録ボタンセレクタが推論されませんでした")
            return False

        return True

    except Exception as e:
        logger.error(f"❌ 実行中にエラー発生: {e}")
        return False

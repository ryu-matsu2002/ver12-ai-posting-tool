# app/services/blog_signup/livedoor_login.py
from playwright.async_api import async_playwright

# 返り値: [{"name": "...", "value": "...", "domain": ".livedoor.com"}, ...]
async def get_livedoor_cookies(email: str, password: str):
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        # 1) 会員ログイン
        await page.goto("https://member.livedoor.com/login/", wait_until="domcontentloaded")

        # 入力候補（UI変更に備え複数トライ）
        selectors = [
            'input[name="livedoor_id"]',  # 旧
            'input[name="user_id"]',
            'input[type="email"]',
            '#login-email', '#user_id',
        ]
        filled = False
        for sel in selectors:
            try:
                if await page.locator(sel).count() > 0:
                    await page.fill(sel, email)
                    filled = True
                    break
            except: pass
        if not filled:
            raise RuntimeError("emailフィールドが見つかりません")

        pw_selectors = [
            'input[name="password"]',
            'input[type="password"]',
            '#login-pass', '#password',
        ]
        filled = False
        for sel in pw_selectors:
            try:
                if await page.locator(sel).count() > 0:
                    await page.fill(sel, password)
                    filled = True
                    break
            except: pass
        if not filled:
            raise RuntimeError("passwordフィールドが見つかりません")

        # 送信
        # ボタン候補を順にクリック
        for sel in ['button[type="submit"]', 'input[type="submit"]', '#login-btn']:
            try:
                if await page.locator(sel).count() > 0:
                    await page.click(sel)
                    break
            except: pass

        await page.wait_for_load_state("networkidle")

        # 2) 管理画面に一度アクセス（blogcms 側のCookieも取る）
        await page.goto("https://livedoor.blogcms.jp/member/", wait_until="load")

        # Cookie収集（両ドメイン）
        cookies = await context.cookies()
        await browser.close()

        # 必要なドメインに限定して返す
        wanted = []
        for c in cookies:
            if "livedoor.com" in c.get("domain", "") or "blogcms.jp" in c.get("domain", ""):
                wanted.append({
                    "name": c["name"],
                    "value": c["value"],
                    "domain": c["domain"],
                })
        return wanted

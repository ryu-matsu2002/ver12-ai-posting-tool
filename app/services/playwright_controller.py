# app/services/playwright_controller.py
captcha_sessions = {}

async def store_session(session_id, page):
    captcha_sessions[session_id] = page

async def get_session(session_id):
    return captcha_sessions.get(session_id)

async def delete_session(session_id):
    page = captcha_sessions.pop(session_id, None)
    if page:
        await page.close()

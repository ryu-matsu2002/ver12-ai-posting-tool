# helper/main.py
import threading, time, json, os, socket, uuid, tempfile, re
from pathlib import Path
from typing import Optional, Dict, Any

from flask import Flask, request, jsonify, make_response
import requests
from flask_cors import CORS

# Playwright（ローカルユーザー端末で実ブラウザを起動）
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

PORT = int(os.environ.get("HELPER_PORT", 17653))

# ジョブ状態（最小の同時実行制御）
jobs: Dict[str, Dict[str, Any]] = {}

CAPTCHA_TMP_DIR = Path(tempfile.gettempdir()) / "extseo_captcha"
CAPTCHA_TMP_DIR.mkdir(parents=True, exist_ok=True)

def _public_ip() -> Optional[str]:
    try:
        return requests.get("https://api.ipify.org", timeout=5).text.strip()
    except Exception:
        return None

def _post_callback(cb_url: Optional[str], data: dict):
    if not cb_url:
        return
    try:
        requests.post(cb_url, json=data, timeout=10)
    except Exception:
        pass

def _progress(job: dict, *, step: str, progress: int, extra: dict | None = None):
    data = {
        "ok": True,
        "status": "running",
        "step": step,
        "progress": max(0, min(100, int(progress))),
        "token": job.get("token"),
        "site_id": job.get("site_id"),
        "account_id": job.get("account_id"),
        "blog": job.get("blog") or "livedoor",
        "helper_host": socket.gethostname(),
        "helper_ip_public": _public_ip(),
    }
    if extra:
        data.update(extra)
    _post_callback(job.get("callback_url"), data)

# ---------------------- mail.tm 簡易クライアント ----------------------
MAILTM_BASE = "https://api.mail.tm"

def _mailtm_create_inbox() -> dict:
    doms = requests.get(f"{MAILTM_BASE}/domains", timeout=10).json()
    members = doms.get("hydra:member") or []
    domain = (members[0] or {}).get("domain") if members else "mail.tm"
    local = f"seo{int(time.time()*1000)}{uuid.uuid4().hex[:4]}"
    address = f"{local}@{domain}"
    password = uuid.uuid4().hex

    # アカウント作成（既にあってもOK）
    try:
        requests.post(f"{MAILTM_BASE}/accounts",
                      json={"address": address, "password": password},
                      timeout=10)
    except Exception:
        pass
    # トークン取得
    tok = requests.post(f"{MAILTM_BASE}/token",
                        json={"address": address, "password": password},
                        timeout=10).json().get("token")
    return {"address": address, "password": password, "token": tok}

def _mailtm_poll_verify_url(token: str, *, timeout_sec: int = 300) -> Optional[str]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    t0 = time.time()
    patt = re.compile(r"https://member\.livedoor\.com/verify/[A-Za-z0-9]+")
    seen_ids = set()
    while time.time() - t0 < timeout_sec:
        try:
            lst = requests.get(f"{MAILTM_BASE}/messages", headers=headers, timeout=10).json()
            for msg in (lst.get("hydra:member") or []):
                mid = msg.get("id")
                if not mid or mid in seen_ids:
                    continue
                seen_ids.add(mid)
                body = requests.get(f"{MAILTM_BASE}/messages/{mid}", headers=headers, timeout=10).json()
                text = f"{body.get('text','')}\n{body.get('html','')}"
                m = patt.search(text or "")
                if m:
                    return m.group(0)
        except Exception:
            pass
        time.sleep(5)
    return None

# ---------------------- Livedoor操作（同期Playwright） ----------------------

def _ld_prepare_captcha(page, email: str, livedoor_id: str, password: str) -> Path:
    page.goto("https://member.livedoor.com/register/input", wait_until="load")
    page.fill('input[name="livedoor_id"]', livedoor_id)
    page.fill('input[name="password"]', password)
    page.fill('input[name="password2"]', password)
    page.fill('input[name="email"]', email)
    # 送信してCAPTCHA表示
    page.click('input[type="submit"][value="ユーザー情報を登録"]')
    img = page.locator("#captcha-img").first
    try:
        img.wait_for(state="visible", timeout=20_000)
    except PWTimeoutError:
        img.wait_for(state="attached", timeout=5_000)
    ts = time.strftime("%Y%m%d_%H%M%S")
    p = CAPTCHA_TMP_DIR / f"captcha_{ts}_{uuid.uuid4().hex[:6]}.png"
    img.screenshot(path=str(p))
    return p

def _ld_submit_captcha_and_reach_done(page, captcha_text: str) -> bool:
    page.fill('input[name="captcha"]', (captcha_text or "").replace(" ", "").replace("　", ""))
    # 送信
    page.click('input[type="submit"]')
    try:
        page.wait_for_url("**/register/done", timeout=30_000)
        return True
    except PWTimeoutError:
        try:
            page.screenshot(path=str(CAPTCHA_TMP_DIR / f"failed_after_captcha_{int(time.time())}.png"),
                            full_page=True)
        except Exception:
            pass
        return False

def _ld_blog_create_and_fetch_api(page) -> dict:
    """
    前提：メール認証が完了済みで、livedoor側にログイン状態が伝播していること。
    できるだけ汎用セレクタでブログ作成→APIキー発行・取得まで行う。
    """
    # blogcms のセッションを確立
    page.goto("https://livedoor.blogcms.jp/member/", wait_until="load")
    try:
        page.wait_for_load_state("networkidle", timeout=10_000)
    except Exception:
        pass

    # 作成ページへ
    for _ in range(3):
        page.goto("https://livedoor.blogcms.jp/member/blog/create", wait_until="load")
        try:
            page.wait_for_load_state("networkidle", timeout=10_000)
        except Exception:
            pass
        # タイトル欄
        if page.locator('#blogTitle, input[name="title"]').first.count() > 0:
            break
        # 「ブログを作成」導線を踏む場合
        for sel in ['a:has-text("ブログを作成")',
                    'button:has-text("ブログを作成")',
                    'a.button:has-text("ブログを作成")',
                    'a[href*="/member/blog/create"]']:
            try:
                loc = page.locator(sel).first
                if loc.count() > 0 and loc.is_visible():
                    loc.click(timeout=4000)
                    try: page.wait_for_load_state("networkidle", timeout=8000)
                    except Exception: pass
                    break
            except Exception:
                pass

    # タイトル入力
    title_input = None
    for sel in ['#blogTitle', 'input[name="title"]', 'input#title', 'input[name="blog_title"]']:
        try:
            loc = page.locator(sel).first
            if loc.count() > 0:
                title_input = loc
                break
        except Exception:
            pass
    if not title_input:
        return {"success": False, "error": "title input not found"}

    title_input.click()
    try:
        title_input.fill("")
    except Exception:
        try:
            title_input.press("Control+A"); title_input.press("Delete")
        except Exception:
            pass
    title_input.fill("日々ブログ")

    # 作成ボタン
    btn = None
    for sel in [
        'input[type="submit"][value="ブログを作成する"]',
        'input[type="submit"][value*="ブログを作成"]',
        'button[type="submit"]',
        'button:has-text("ブログを作成")',
        'a.button:has-text("ブログを作成")',
    ]:
        try:
            loc = page.locator(sel).first
            if loc.count() > 0 and (loc.is_visible() if hasattr(loc, "is_visible") else True):
                btn = loc; break
        except Exception:
            pass
    if not btn:
        return {"success": False, "error": "create button not found"}
    try:
        btn.click(timeout=8000)
    except Exception:
        try:
            page.evaluate("(el)=>el.click()", btn)
        except Exception:
            pass

    try:
        page.wait_for_load_state("load", timeout=20_000)
        page.wait_for_load_state("networkidle", timeout=15_000)
    except Exception:
        pass

    # member へ戻り、ブログ設定→APIへ
    page.goto("https://livedoor.blogcms.jp/member/", wait_until="load")
    try:
        page.wait_for_load_state("networkidle", timeout=10_000)
    except Exception:
        pass

    # ブログ設定リンクから blog_id を抽出
    blog_id = None
    link = None
    for sel in [
        'a[title="ブログ設定"]',
        'a:has-text("ブログ設定")',
        'a[href^="/blog/"][href$="/config/"]',
        'a[href*="/config/"]'
    ]:
        try:
            loc = page.locator(sel).first
            if loc.count() > 0:
                link = loc
                href = loc.get_attribute("href")
                if href:
                    parts = href.split("/")
                    if len(parts) > 2:
                        blog_id = parts[2]
                break
        except Exception:
            pass
    if not blog_id:
        # URL から保険抽出
        if "/blog/" in page.url:
            try:
                blog_id = page.url.split("/blog/")[1].split("/")[0]
            except Exception:
                blog_id = "unknown"

    if link:
        try: link.click(timeout=6000)
        except Exception:
            try:
                page.goto(f"https://livedoor.blogcms.jp/blog/{blog_id}/config/", wait_until="load")
            except Exception:
                pass
    else:
        try:
            page.goto(f"https://livedoor.blogcms.jp/blog/{blog_id}/config/", wait_until="load")
        except Exception:
            pass

    try:
        page.wait_for_load_state("networkidle", timeout=10_000)
    except Exception:
        pass

    # API 発行ページへ
    api_link = None
    for sel in [
        'a.configIdxApi[title="API Keyの発行・確認"]',
        'a[title*="API Key"]',
        'a:has-text("API Key")',
        'a:has-text("API Keyの発行")',
    ]:
        try:
            loc = page.locator(sel).first
            if loc.count() > 0:
                api_link = loc; break
        except Exception:
            pass
    if api_link:
        try: api_link.click(timeout=6000)
        except Exception: pass
        try:
            page.wait_for_load_state("load", timeout=10000)
        except Exception:
            pass

    # API 発行
    try:
        page.wait_for_selector('input#apiKeyIssue', timeout=15000)
        page.locator('input#apiKeyIssue').first.click(timeout=6000)
        page.wait_for_selector('button:has-text("実行")', timeout=15000)
        page.locator('button:has-text("実行")').first.click(timeout=6000)
    except Exception:
        pass

    # 値の読取
    endpoint = ""
    for sel in [
        'input.input-xxlarge[readonly]',
        'input[readonly][name*="endpoint"]',
        'input[readonly][id*="endpoint"]',
    ]:
        try:
            page.wait_for_selector(sel, timeout=8000)
            endpoint = page.locator(sel).first.input_value()
            if endpoint:
                break
        except Exception:
            pass

    api_key = ""
    try:
        page.wait_for_selector('input#apiKey', timeout=15000)
        for _ in range(30):
            api_key = (page.locator('input#apiKey').input_value() or "").strip()
            if api_key:
                break
            time.sleep(0.5)
    except Exception:
        pass

    # 取得できなければ1回リトライ
    if not api_key:
        try:
            page.reload(wait_until="load")
            page.wait_for_selector('input#apiKeyIssue', timeout=15000)
            page.locator('input#apiKeyIssue').first.click(timeout=6000)
            page.wait_for_selector('button:has-text("実行")', timeout=15000)
            page.locator('button:has-text("実行")').first.click(timeout=6000)
            page.wait_for_selector('input#apiKey', timeout=15000)
            api_key = (page.locator('input#apiKey').input_value() or "").strip()
        except Exception:
            pass

    if not api_key:
        return {"success": False, "error": "api key empty", "blog_id": blog_id or ""}

    return {"success": True, "blog_id": blog_id or "", "api_key": api_key, "endpoint": endpoint}

# ---------------------- 仕事本体 ----------------------

def run_signup_job(job_id: str, payload: dict):
    """
    1) メール作成 (mail.tm)
    2) Chromium起動（ヘッドフル）＆仮登録入力
    3) CAPTCHAを撮影 → サーバへアップロード（/external-seo/prepare_captcha）
    4) 以降は /solve で続行
    """
    job = jobs[job_id]
    token = payload.get("token")
    site_id = payload.get("site_id")
    account_id = payload.get("account_id")
    callback_url = payload.get("callback_url")
    upload_url = payload.get("upload_url")
    blog = payload.get("blog") or "livedoor"

    # upload_url が来ていなければ callback_url を基に推定
    if not upload_url and callback_url:
        try:
            from urllib.parse import urlparse
            u = urlparse(callback_url)
            upload_url = f"{u.scheme}://{u.netloc}/external-seo/prepare_captcha"
        except Exception:
            pass

    # メール用意
    mail = _mailtm_create_inbox()
    email_addr = mail["address"]
    job.update({"mailtm": mail})

    # 認証情報
    livedoor_id = f"u{uuid.uuid4().hex[:8]}"
    password = uuid.uuid4().hex[:12]
    job.update({"email": email_addr, "livedoor_id": livedoor_id, "password": password})

    # Playwright開始（stopは /solve 側でまとめて行う）
    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()

    # 共有
    job.update({"_pw": pw, "browser": browser, "context": context, "page": page, "status": "running"})

    _progress(job, step="helper_started", progress=5)

    try:
        # CAPTCHA準備
        img_path = _ld_prepare_captcha(page, email_addr, livedoor_id, password)
        _progress(job, step="captcha_ready", progress=20)

        # サーバにアップロード（UIがこの画像を表示）
        files = {"file": open(str(img_path), "rb")}
        data = {"token": token or "", "site_id": site_id or "", "account_id": account_id or ""}
        if upload_url:
            try:
                requests.post(upload_url, files=files, data=data, timeout=15)
            except Exception:
                pass

        # 次の入力を待機
        job["done_waiting_captcha"] = False
        _progress(job, step="captcha_shown", progress=25)

    except Exception as e:
        _post_callback(callback_url, {
            "ok": False, "error": f"prepare failed: {e}",
            "token": token, "site_id": site_id, "account_id": account_id,
            "blog": blog, "step": "failed"
        })
        job["done"] = True
        try:
            browser.close(); pw.stop()
        except Exception:
            pass
        return

# ---------------------- Flask ルート ----------------------

@app.get("/ping")
def ping():
    resp = jsonify({"ok": True, "name": "signup-helper", "version": "1.0.0"})
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp

@app.route("/start", methods=["POST", "OPTIONS"])
def start():
    if request.method == "OPTIONS":
        resp = make_response("", 204)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return resp

    try:
        payload = request.get_json(force=True, silent=False) or {}
    except Exception:
        resp = jsonify({"ok": False, "error": "invalid json"})
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp, 400

    job_id = f'job-{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}'
    jobs[job_id] = {
        "done": False,
        "payload": payload,
        "token": payload.get("token"),
        "site_id": payload.get("site_id"),
        "account_id": payload.get("account_id"),
        "blog": payload.get("blog") or "livedoor",
        "callback_url": payload.get("callback_url"),
        "upload_url": payload.get("upload_url"),
    }

    th = threading.Thread(target=run_signup_job, args=(job_id, payload), daemon=True)
    th.start()
    resp = jsonify({"ok": True, "job_id": job_id})
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp

@app.post("/solve")
def solve():
    """
    期待するJSON:
    {
      "job_id": "...",
      "captcha_text": "abcd",
      // （任意）force_mailtm: true で必ずmail.tmを使う
    }
    """
    try:
        data = request.get_json(force=True, silent=False) or {}
    except Exception:
        data = request.form or {}

    job_id = data.get("job_id")
    captcha_text = (data.get("captcha_text") or "").strip()
    if not job_id or job_id not in jobs:
        return jsonify({"ok": False, "error": "invalid job_id"}), 400
    if not captcha_text:
        return jsonify({"ok": False, "error": "missing captcha_text"}), 400

    job = jobs[job_id]
    page = job.get("page")
    browser = job.get("browser")
    pw = job.get("_pw")
    if not page or not browser or not pw:
        return jsonify({"ok": False, "error": "page not ready"}), 400

    token = job.get("token"); site_id = job.get("site_id"); account_id = job.get("account_id")
    callback_url = job.get("callback_url")

    # 1) CAPTCHA送信
    ok = _ld_submit_captcha_and_reach_done(page, captcha_text)
    if not ok:
        _post_callback(callback_url, {
            "ok": True, "status": "captcha_failed",
            "token": token, "site_id": site_id, "account_id": account_id,
            "step": "captcha_failed", "progress": 100
        })
        try:
            browser.close(); pw.stop()
        except Exception:
            pass
        job["done"] = True
        return jsonify({"ok": True, "status": "captcha_failed"})

    _progress(job, step="captcha_success", progress=45)

    # 2) メール認証（mail.tm）
    mt = job.get("mailtm") or {}
    verify_url = _mailtm_poll_verify_url(mt.get("token"), timeout_sec=300)
    if not verify_url:
        _post_callback(callback_url, {
            "ok": True, "status": "recreate_required",
            "token": token, "site_id": site_id, "account_id": account_id,
            "step": "mail_timeout", "progress": 100
        })
        try:
            browser.close(); pw.stop()
        except Exception:
            pass
        job["done"] = True
        return jsonify({"ok": True, "status": "recreate_required"})

    # 認証URLへ
    page.goto(verify_url, wait_until="load")
    try:
        page.wait_for_load_state("networkidle", timeout=10_000)
    except Exception:
        pass
    _progress(job, step="mail_verified", progress=60)

    # 3) ブログ作成 → APIキー取得
    res = _ld_blog_create_and_fetch_api(page)
    if not res.get("success"):
        _post_callback(callback_url, {
            "ok": True, "status": "recreate_required",
            "token": token, "site_id": site_id, "account_id": account_id,
            "step": "create_failed", "progress": 100,
            "detail": res
        })
        try:
            browser.close(); pw.stop()
        except Exception:
            pass
        job["done"] = True
        return jsonify({"ok": True, "status": "recreate_required", "detail": res})

    blog_id = res.get("blog_id")
    api_key = res.get("api_key")
    endpoint = res.get("endpoint")

    # 完了通知
    _post_callback(callback_url, {
        "ok": True,
        "status": "api_key_received",
        "site_id": site_id,
        "account_id": account_id,
        "blog": job.get("blog") or "livedoor",
        "token": token,
        "progress": 100,
        "step": "apiKey_received",
        "helper_host": socket.gethostname(),
        "helper_ip_public": _public_ip(),
        "blog_id": blog_id,
        "endpoint": endpoint,
        "api_key": api_key
    })

    # 後片付け
    try:
        browser.close(); pw.stop()
    except Exception:
        pass
    job["done"] = True

    return jsonify({"ok": True, "status": "done", "blog_id": blog_id, "endpoint": endpoint, "api_key": api_key})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=PORT)

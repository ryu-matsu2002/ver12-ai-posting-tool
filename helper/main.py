# helper/main.py
import threading, time, json, os, socket, re, random, string, tempfile
from pathlib import Path
from typing import Optional, Tuple

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import requests

# Playwright (同期API)
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

# 設定
PORT = int(os.environ.get("HELPER_PORT", 17653))
HEADLESS = os.environ.get("PW_HEADLESS", "0") not in ("0", "false", "False", "no", "NO")
USER_DATA_DIR = os.environ.get("PW_USER_DATA_DIR")  # 任意: セッション維持したい場合
CAPTCHA_DIR = Path(tempfile.gettempdir()) / "signup_captchas"
CAPTCHA_DIR.mkdir(parents=True, exist_ok=True)

# 簡易メモリキュー（同時実行は最小限）
jobs = {}  # job_id -> dict

# ------------ ユーティリティ ------------

def rand_id(n=10):
    chars = string.ascii_lowercase + string.digits
    return ''.join(random.choice(chars) for _ in range(n))

def send_callback(job: dict, step: str = None, progress: Optional[int] = None, **extra):
    """サーバの /external-seo/callback に進捗・結果を送る"""
    url = job.get("callback_url")
    if not url:
        return False
    data = {
        "ok": True,
        "token": job.get("token"),
        "site_id": job.get("site_id"),
        "account_id": job.get("account_id"),
        "blog": job.get("blog") or "livedoor",
        "helper_host": socket.gethostname(),
        "helper_ip_public": get_public_ip_safe(),
    }
    if step:
        data["step"] = step
    if progress is not None:
        data["progress"] = max(0, min(100, int(progress)))
    data.update(extra)
    try:
        requests.post(url, json=data, timeout=15)
        return True
    except Exception:
        return False

def get_public_ip_safe() -> Optional[str]:
    try:
        return requests.get("https://api.ipify.org", timeout=5).text.strip()
    except Exception:
        return None

def upload_captcha(job: dict, image_path: str) -> bool:
    """
    サーバの /external-seo/prepare_captcha に画像を渡してUIに表示してもらう。
    期待: multipart/form-data, フィールド名 'file' を基本に送信（互換で 'captcha' も併送）
    """
    url = job.get("upload_url")
    if not url:
        return False
    fields = {
        "token": job.get("token", ""),
        "site_id": str(job.get("site_id") or ""),
        "account_id": str(job.get("account_id") or ""),
        "blog": job.get("blog") or "livedoor",
    }
    files = {}
    try:
        with open(image_path, "rb") as f:
            files["file"] = ("captcha.png", f, "image/png")
            # 互換: フィールド名 'captcha' でも受ける実装向けに同時送信（サーバ側はどちらか見る）
            with open(image_path, "rb") as f2:
                files["captcha"] = ("captcha.png", f2, "image/png")
                resp = requests.post(url, data=fields, files=files, timeout=30)
        return bool(resp.status_code // 100 == 2)
    except Exception:
        return False

# ---------- 1secmail（簡易メール） ----------
SECMAIL_API = "https://www.1secmail.com/api/v1/"

def gen_mailbox() -> Tuple[str, str, str]:
    r = requests.get(SECMAIL_API, params={"action": "genRandomMailbox", "count": 1}, timeout=15)
    r.raise_for_status()
    arr = r.json()
    email = arr[0]
    login, domain = email.split("@", 1)
    return email, login, domain

def poll_verify_link(login: str, domain: str, timeout_sec=180, interval=5) -> Optional[str]:
    pattern = re.compile(r"https://member\.livedoor\.com/verify/[A-Za-z0-9]+")
    deadline = time.time() + timeout_sec
    last_ids = set()
    while time.time() < deadline:
        try:
            r = requests.get(SECMAIL_API, params={"action":"getMessages","login":login,"domain":domain}, timeout=15)
            if r.status_code == 200:
                for msg in r.json():
                    mid = msg.get("id")
                    if mid in last_ids:
                        continue
                    last_ids.add(mid)
                    body = requests.get(SECMAIL_API, params={"action":"readMessage","login":login,"domain":domain,"id":mid}, timeout=15).json()
                    text = (body.get("textBody") or "") + "\n" + (body.get("htmlBody") or "")
                    m = pattern.search(text)
                    if m:
                        return m.group(0)
        except Exception:
            pass
        time.sleep(interval)
    return None

# ------------ Livedoorブラウザ操作 ------------

def livedoor_prepare_and_upload_captcha(pw, job: dict):
    """
    Chromiumを起動して入力→CAPTCHAまで到達し、画像をアップロード。page/bctxはjobに保持。
    """
    browser = None
    context = None
    page = None
    try:
        args = {
            "headless": HEADLESS,
            "args": ["--disable-dev-shm-usage"],
        }
        if USER_DATA_DIR:
            args["user_data_dir"] = USER_DATA_DIR
        browser = pw.chromium.launch_persistent_context(**args) if USER_DATA_DIR else pw.chromium.launch(headless=HEADLESS)
        context = browser if USER_DATA_DIR else browser.new_context()

        page = context.new_page()
        job["page"] = page
        job["context"] = context
        job["browser"] = browser

        # メール・ID・PW 準備（1secmail）
        email, login, domain = gen_mailbox()
        job["email"] = email
        job["sec_login"] = login
        job["sec_domain"] = domain

        # livedoor_id / password
        livedoor_id = job.get("livedoor_id") or ( "u" + rand_id(9) )
        password = job.get("password") or ( "P-" + rand_id(6) + "_" + rand_id(6) )
        job["livedoor_id"] = livedoor_id
        job["password"] = password

        send_callback(job, step="starting", progress=5)

        # 入力ページへ
        page.goto("https://member.livedoor.com/register/input", wait_until="load", timeout=30000)

        # 入力
        page.fill('input[name="livedoor_id"]', livedoor_id)
        page.fill('input[name="password"]', password)
        page.fill('input[name="password2"]', password)
        page.fill('input[name="email"]', email)

        # 送信
        page.click('input[type="submit"][value="ユーザー情報を登録"]', timeout=15000)

        # CAPTCHA要素
        img = page.locator("#captcha-img").first
        try:
            img.wait_for(state="visible", timeout=20000)
        except PWTimeout:
            # attachedだけでもトライ
            img.wait_for(state="attached", timeout=8000)

        # スクショ保存
        ts = time.strftime("%Y%m%d_%H%M%S")
        img_path = str(CAPTCHA_DIR / f"captcha_{job['job_id']}_{ts}.png")
        try:
            img.screenshot(path=img_path)
        except Exception:
            # 要素単体がダメな場合はページ全体
            page.screenshot(path=img_path, full_page=True)

        # サーバにCAPTCHA画像アップロード → UIに表示
        upload_ok = upload_captcha(job, img_path)

        # 進捗通知
        send_callback(job, step="captcha_required", progress=12)

        if not upload_ok:
            # 画像アップロード失敗でも /captcha_status が空のままなので、進捗は流し続ける
            pass

        return True
    except Exception as e:
        send_callback(job, step="failed", progress=1, error=str(e))
        raise
    finally:
        # ここではブラウザは開いたまま（solve待ち）。クリーンアップはジョブ完了時。
        pass

def livedoor_submit_captcha_and_verify(pw, job: dict, captcha_text: str) -> bool:
    """CAPTCHA送信 → /register/done → メール認証完了まで"""
    page = job.get("page")
    if not page:
        raise RuntimeError("page not found")

    try:
        # CAPTCHA送信
        page.fill('input[name="captcha"]', (captcha_text or "").replace(" ", "").replace("　", ""))
        page.click('input[type="submit"]', timeout=15000)
        try:
            page.wait_for_url("**/register/done", timeout=30000)
        except PWTimeout:
            # 失敗
            send_callback(job, step="failed", progress=20)
            return False

        send_callback(job, step="captcha_submitted", progress=45)

        # 直後に blogcms 側へ一度入ってセッションを温める
        try:
            page.goto("https://livedoor.blogcms.jp/member/", wait_until="load", timeout=30000)
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass

        # メールの認証リンクをポーリング
        send_callback(job, step="waiting_email", progress=55)
        link = poll_verify_link(job["sec_login"], job["sec_domain"], timeout_sec=240, interval=6)
        if not link:
            send_callback(job, step="failed", progress=60, error="email_verify_link_not_found")
            return False

        # 認証リンクを踏む
        page.goto(link, wait_until="load", timeout=30000)
        try:
            page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            pass
        send_callback(job, step="email_verified", progress=65)
        return True
    except Exception as e:
        send_callback(job, step="failed", progress=40, error=str(e))
        return False

def livedoor_create_blog_and_issue_key(job: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    ブログ作成 → AtomPub Key発行をやり切り、(blog_id, endpoint, api_key) を返す
    """
    page = job.get("page")
    if not page:
        raise RuntimeError("page not found")

    def maybe_close_overlays():
        sels = [
            'button#iubenda-cs-accept-btn', 'button#iubenda-cs-accept',
            'button:has-text("同意")', 'button:has-text("OK")'
        ]
        for sel in sels:
            try:
                loc = page.locator(sel).first
                if loc.count() > 0 and loc.is_visible():
                    loc.click(timeout=800)  # best-effort
            except Exception:
                pass

    try:
        send_callback(job, step="creating_blog", progress=78)

        # 作成ページへ
        for i in range(3):
            page.goto("https://livedoor.blogcms.jp/member/blog/create", wait_until="load", timeout=30000)
            try:
                page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                pass
            # 直接フォーム?
            if page.locator('#blogTitle, input[name="title"]').first.count() > 0:
                break
            # 「ブログを作成」導線を押す
            maybe_close_overlays()
            for sel in ['a:has-text("ブログを作成")', 'button:has-text("ブログを作成")', 'a[href*="/member/blog/create"]']:
                try:
                    loc = page.locator(sel).first
                    if loc.count() > 0 and loc.is_visible():
                        loc.click(timeout=3000)
                        page.wait_for_load_state("networkidle", timeout=8000)
                        if page.locator('#blogTitle, input[name="title"]').first.count() > 0:
                            break
                except Exception:
                    pass

        # タイトル入力
        title = job.get("blog_title") or "日々ブログ"
        el = page.locator('#blogTitle').first
        if el.count() == 0:
            el = page.locator('input[name="title"]').first
        if el.count() == 0:
            # UI変化の保険
            raise RuntimeError("blog title input not found")
        try:
            el.fill("")
        except Exception:
            try:
                el.click(); el.press("Control+A"); el.press("Delete")
            except Exception:
                pass
        el.fill(title)

        # 送信ボタン（候補）
        btn_sels = [
            'input[type="submit"][value="ブログを作成する"]',
            'input[type="submit"][value*="ブログを作成"]',
            'button[type="submit"]',
            'button:has-text("ブログを作成")',
            'a.button:has-text("ブログを作成")',
        ]
        clicked = False
        for sel in btn_sels:
            try:
                loc = page.locator(sel).first
                if loc.count() > 0 and loc.is_visible():
                    try:
                        loc.click(timeout=6000)
                    except Exception:
                        try:
                            loc.click(timeout=6000, force=True)
                        except Exception:
                            pass
                    clicked = True
                    break
            except Exception:
                pass
        if not clicked:
            raise RuntimeError("create button not found")

        # 遷移待ち
        try:
            page.wait_for_load_state("load", timeout=20000)
        except Exception:
            pass
        try:
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass

        # /welcome などの痕跡で作成成功判定の緩和
        ok = False
        try:
            page.wait_for_url(re.compile(r"/welcome($|[/?#])"), timeout=12000)
            ok = True
        except Exception:
            if page.locator('a:has-text("最初のブログを書く")').first.count() > 0:
                ok = True
        if not ok:
            # 失敗とみなし戻す（UI変更等）
            raise RuntimeError("blog create not confirmed")

        # member へ戻って blog_id 推定
        page.goto("https://livedoor.blogcms.jp/member/", wait_until="load", timeout=30000)
        try:
            page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            pass

        blog_id = None
        # a[href^="/blog/xxx/config"] を拾う
        links = page.locator('a[href*="/blog/"][href*="/config/"]').all()
        for a in links:
            try:
                href = a.get_attribute("href") or ""
                parts = href.split("/")
                i = parts.index("blog") if "blog" in parts else -1
                if i >= 0 and len(parts) > i+1:
                    blog_id = parts[i+1]
                    break
            except Exception:
                pass
        if not blog_id:
            # URLからの推測
            if "/blog/" in page.url:
                try:
                    blog_id = page.url.split("/blog/")[1].split("/")[0]
                except Exception:
                    pass
        if not blog_id:
            raise RuntimeError("blog_id not found")

        # 設定画面へ→API Key
        send_callback(job, step="issuing_api_key", progress=90, blog_id=blog_id)

        # API Key ページへ誘導リンクを探す（幅広く）
        api_nav = None
        for sel in [
            'a[title="API Keyの発行・確認"]',
            'a:has-text("API Key")',
            'a:has-text("API Keyの発行")',
        ]:
            try:
                loc = page.locator(sel).first
                if loc.count() > 0:
                    api_nav = loc; break
            except Exception:
                pass
        if api_nav:
            try:
                api_nav.click(timeout=6000)
            except Exception:
                api_nav.click(timeout=6000, force=True)
            try:
                page.wait_for_load_state("load", timeout=10000)
            except Exception:
                pass

        # 発行ボタン
        page.wait_for_selector('input#apiKeyIssue', timeout=15000)
        page.locator('input#apiKeyIssue').first.click(timeout=6000)
        page.wait_for_selector('button:has-text("実行")', timeout=15000)
        page.locator('button:has-text("実行")').first.click(timeout=6000)

        # 値を読む
        def read_endpoint_and_key():
            endpoint = ""
            for sel in ['input.input-xxlarge[readonly]', 'input[readonly][name*="endpoint"]', 'input[readonly][id*="endpoint"]']:
                try:
                    endpoint = page.locator(sel).first.input_value()
                    if endpoint:
                        break
                except Exception:
                    pass
            key = ""
            for _ in range(30):
                try:
                    key = (page.locator('input#apiKey').input_value() or "").strip()
                    if key:
                        break
                except Exception:
                    pass
                time.sleep(0.5)
            return endpoint, key

        endpoint, api_key = read_endpoint_and_key()
        if not api_key:
            # リロードして再発行
            try:
                page.reload(wait_until="load")
                page.wait_for_load_state("networkidle", timeout=8000)
            except Exception:
                pass
            page.wait_for_selector('input#apiKeyIssue', timeout=15000)
            page.locator('input#apiKeyIssue').first.click(timeout=6000)
            page.wait_for_selector('button:has-text("実行")', timeout=15000)
            page.locator('button:has-text("実行")').first.click(timeout=6000)
            endpoint, api_key = read_endpoint_and_key()

        if not api_key:
            raise RuntimeError("api_key not issued")

        # 完了コールバック
        send_callback(job, step="api_key_ok", progress=100, blog_id=blog_id, endpoint=endpoint, api_key=api_key)
        return blog_id, endpoint, api_key

    except Exception as e:
        send_callback(job, step="failed", progress=92, error=str(e))
        return None, None, None

# ------------ ジョブ本体 ------------

def run_signup_job(job_id: str, payload: dict):
    """
    方式A: クライアントのIPで実行。CAPTCHAアップロード→/solve待ち→メール認証→ブログ作成→APIキー回収→callback
    """
    job = {
        "job_id": job_id,
        "token": payload.get("token"),
        "site_id": payload.get("site_id"),
        "account_id": payload.get("account_id"),
        "blog": payload.get("blog") or "livedoor",
        "callback_url": payload.get("callback_url"),
        "upload_url": payload.get("upload_url"),
        "livedoor_id": payload.get("livedoor_id"),
        "password": payload.get("password"),
        "blog_title": payload.get("blog_title"),
        "solve_event": threading.Event(),
        "captcha_text": None,
        "done": False,
    }
    jobs[job_id] = job

    try:
        with sync_playwright() as pw:
            # 1) 入力→CAPTCHA提示→画像アップロード
            livedoor_prepare_and_upload_captcha(pw, job)

            # 2) /solve を待機
            send_callback(job, step="waiting_captcha_text", progress=25)
            solved = job["solve_event"].wait(timeout=5 * 60)  # 5分待ち
            if not solved or not job.get("captcha_text"):
                send_callback(job, step="failed", progress=30, error="captcha_timeout")
                return

            # 3) CAPTCHA送信→/register/done→メール認証
            ok = livedoor_submit_captcha_and_verify(pw, job, job["captcha_text"])
            if not ok:
                # /solve の返却用に覚えておく
                job["solve_result"] = "captcha_failed"
                return

            # 4) ブログ作成→APIキー発行
            b, ep, key = livedoor_create_blog_and_issue_key(job)
            if not key:
                job["solve_result"] = "recreate_required"
                return

            # 完了
            job["solve_result"] = "done"
    except Exception as e:
        send_callback(job, step="failed", progress=3, error=str(e))
    finally:
        # 後片付け
        try:
            page = job.get("page")
            context = job.get("context")
            browser = job.get("browser")
            if page:
                try: page.close()
                except Exception: pass
            if context and hasattr(context, "close"):
                try: context.close()
                except Exception: pass
            if browser and hasattr(browser, "close"):
                try:
                    # launch_persistent_context の場合は context==browser
                    if browser is not context:
                        browser.close()
                except Exception:
                    pass
        except Exception:
            pass
        job["done"] = True

# ------------ ルート ------------

@app.get("/ping")
def ping():
    resp = jsonify({"ok": True, "name": "signup-helper", "version": "1.0.0"})
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp

@app.post("/start")
def start():
    # PreflightはCORSでカバー済
    try:
        payload = request.get_json(force=True, silent=False) or {}
    except Exception:
        resp = jsonify({"ok": False, "error": "invalid json"})
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp, 400

    # 必須チェック
    for k in ("token", "site_id", "account_id", "callback_url", "upload_url"):
        if not payload.get(k):
            return jsonify({"ok": False, "error": f"missing {k}"}), 400

    job_id = f'job-{int(time.time()*1000)}-{rand_id(4)}'
    th = threading.Thread(target=run_signup_job, args=(job_id, payload), daemon=True)
    th.start()
    resp = jsonify({"ok": True, "job_id": job_id})
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp

@app.post("/solve")
def solve():
    """
    フロントのCAPTCHA解答から呼ばれる。job_idで特定して同一セッションに送る。
    """
    try:
        data = request.get_json(force=True, silent=False) or {}
    except Exception:
        return jsonify({"ok": False, "error": "invalid json"}), 400

    job_id = data.get("job_id")
    captcha_text = (data.get("captcha_text") or "").strip()
    if not job_id or not captcha_text:
        return jsonify({"ok": False, "error": "missing job_id or captcha_text"}), 400

    job = jobs.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "job not found"}), 404

    # セットして起こす
    job["captcha_text"] = captcha_text
    ev = job.get("solve_event")
    if ev:
        ev.set()

    # ここで即時の最終結果は分からないが、UXのために概況を返す
    # run_signup_job が進めて、完了時は /external-seo/callback でAPIキーが飛ぶ
    result = job.get("solve_result")
    if result == "captcha_failed":
        return jsonify({"ok": True, "status": "captcha_failed"})
    elif result == "recreate_required":
        return jsonify({"ok": True, "status": "recreate_required"})
    elif result == "done":
        return jsonify({"ok": True, "status": "done"})
    else:
        return jsonify({"ok": True, "status": "running"})

# ------------- エラールート(OPTION) -------------
@app.route("/start", methods=["OPTIONS"])
@app.route("/solve", methods=["OPTIONS"])
def opt():
    resp = make_response("", 204)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return resp

# ------------- エントリポイント -------------
if __name__ == "__main__":
    # ダブルクリック起動でも動くシンプル構成
    app.run(host="127.0.0.1", port=PORT)

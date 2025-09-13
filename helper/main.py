# helper/main.py
import threading
import time
import json
import os
import socket
from typing import Optional, Dict

from flask import Flask, request, jsonify, make_response
import requests
from flask_cors import CORS

# --- Playwright は任意依存（未導入でもダミーモードで動く） ---
PLAYWRIGHT_AVAILABLE = True
try:
    from playwright.sync_api import sync_playwright
except Exception:
    PLAYWRIGHT_AVAILABLE = False

app = Flask(__name__)
# すべて許可（localhost 間の疎通用）
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

# 設定
PORT = int(os.environ.get("HELPER_PORT", 17653))

# ジョブ状態（同時実行は最小限制御）
class JobState:
    def __init__(self):
        self.done: bool = False
        self.payload: Dict = {}
        self.captcha_text: Optional[str] = None
        self.wait_event: threading.Event = threading.Event()
        self.running: bool = False
        self.error: Optional[str] = None

jobs: Dict[str, JobState] = {}
jobs_lock = threading.Lock()


# ========== 共通ユーティリティ ==========
def _allow_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp

def post_progress(callback_url: Optional[str], job_id: str,
                  stage: str, message: str, pct: int = 0,
                  extra: Optional[Dict] = None):
    """サーバへ進捗をPOSTする（失敗しても握りつぶし）"""
    if not callback_url:
        return
    data = {
        "job_id": job_id,
        "stage": stage,
        "message": message,
        "pct": pct,
    }
    if extra:
        data.update(extra)
    try:
        requests.post(callback_url, json=data, timeout=10)
    except Exception:
        pass

def upload_captcha(upload_url: Optional[str], job_id: str, img_bytes: Optional[bytes]) -> Optional[str]:
    """
    サーバの /external-seo/prepare_captcha へ画像を送る。
    戻り: 画像URL（サーバ応答） or None
    """
    if not upload_url or not img_bytes:
        return None
    try:
        files = {"image": ("captcha.png", img_bytes, "image/png")}
        data = {"job_id": job_id}
        r = requests.post(upload_url, data=data, files=files, timeout=20)
        j = r.json() if r.ok else {}
        return j.get("url")
    except Exception:
        return None

def get_public_ip() -> Optional[str]:
    try:
        return requests.get("https://api.ipify.org", timeout=5).text.strip()
    except Exception:
        return None


# ========== API ==========
@app.get("/ping")
def ping():
    resp = jsonify({"ok": True, "name": "signup-helper", "version": "1.0.0"})
    return _allow_cors(resp)

@app.route("/start", methods=["POST", "OPTIONS"])
def start():
    # Preflight
    if request.method == "OPTIONS":
        resp = make_response("", 204)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return resp

    """
    期待するJSON（どれも必須ではないが、あると精度UP）:
    {
      "token": "...",                      # /external-seo/start の戻り値（サーバ照合用）
      "site_id": 123, "account_id": 456,
      "blog": "livedoor",
      "callback_url": "https://xxx/external-seo/callback",
      "upload_url": "https://xxx/external-seo/prepare_captcha"   # 任意: あるとCAPTCHA画像をサーバ側に保存
    }
    """
    try:
        payload = request.get_json(force=True, silent=False) or {}
    except Exception:
        return _allow_cors(jsonify({"ok": False, "error": "invalid json"})), 400

    job_id = f'job-{int(time.time() * 1000)}'
    st = JobState()
    st.payload = payload
    with jobs_lock:
        jobs[job_id] = st

    th = threading.Thread(target=run_signup_job, args=(job_id,), daemon=True)
    th.start()
    return _allow_cors(jsonify({"ok": True, "job_id": job_id}))

@app.route("/solve", methods=["POST", "OPTIONS"])
def solve():
    # Preflight
    if request.method == "OPTIONS":
        resp = make_response("", 204)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return resp

    """
    期待するJSON:
    {
      "job_id": "job-...",
      "text": "ABCD"   # ユーザーが入力したCAPTCHA文字
    }
    """
    try:
        body = request.get_json(force=True, silent=False) or {}
    except Exception:
        return _allow_cors(jsonify({"ok": False, "error": "invalid json"})), 400

    job_id = body.get("job_id")
    text = body.get("text")
    if not job_id or not text:
        return _allow_cors(jsonify({"ok": False, "error": "missing params"})), 400

    with jobs_lock:
        st = jobs.get(job_id)
    if not st:
        return _allow_cors(jsonify({"ok": False, "error": "unknown job"})), 404

    st.captcha_text = text
    st.wait_event.set()
    return _allow_cors(jsonify({"ok": True}))


# ========== メインジョブ ==========
def run_signup_job(job_id: str):
    """
    実ジョブ（ユーザーPC=この端末のIPでアクセス）:
    1) ブラウザ起動 → 2) Livedoor 仮登録 → 3) CAPTCHA到達→画像アップロード
    4) /solve でのユーザー入力待ち → 5) 送信 → 6) メール認証 → 7) APIキー取得
    8) サーバへ callback
    """
    with jobs_lock:
        st = jobs.get(job_id)
    if not st:
        return

    payload = st.payload
    callback_url = payload.get("callback_url")
    upload_url = payload.get("upload_url")
    blog = payload.get("blog", "livedoor")

    # 進捗: スタート
    post_progress(callback_url, job_id, "start", "ヘルパー開始", 1, {
        "blog": blog,
        "token": payload.get("token"),
        "site_id": payload.get("site_id"),
        "account_id": payload.get("account_id"),
        "helper_host": socket.gethostname(),
        "helper_ip_public": get_public_ip(),
    })

    # Playwright 未導入の場合はダミーフロー
    if not PLAYWRIGHT_AVAILABLE:
        dummy_flow(job_id, st)
        return finish_ok(callback_url, job_id, payload, api_key=None, note="playwright_not_installed")

    # ---- ここから Playwright 実ジョブ（スケルトン）----
    st.running = True
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)  # ヘッドフルで人間確認可
            context = browser.new_context()
            page = context.new_page()

            # 1) Livedoor へ遷移
            post_progress(callback_url, job_id, "navigate", "Livedoorへ遷移中", 5)
            # 例: 仮登録入力ページ（実サイトの最新パスに合わせて調整）
            page.goto("https://member.livedoor.com/register/input", timeout=120000)
            page.wait_for_load_state("domcontentloaded")

            # 2) 仮登録フォーム入力（<<< 実装者: 既存サーバ版のセレクタを移植 >>>）
            # page.fill("#mail", payload.get("signup_mail", "example@domain.test"))
            # page.click("button[type=submit]")

            # 3) CAPTCHA 到達待ち
            post_progress(callback_url, job_id, "wait_captcha", "CAPTCHA待機中", 15)
            captcha_el = page.wait_for_selector(
                "img[alt*=captcha], img[src*='captcha']",
                timeout=120000
            )

            # 4) 画像スクリーンショット取得→サーバへアップロード
            img_bytes = captcha_el.screenshot(type="png")
            post_progress(callback_url, job_id, "captcha_found", "CAPTCHA取得", 20)
            image_url = upload_captcha(upload_url, job_id, img_bytes)
            post_progress(callback_url, job_id, "captcha_uploaded",
                          "CAPTCHA画像アップロード済み。入力待ち", 25, {"image_url": image_url})

            # 5) /solve からの入力待ち（最大10分）
            st.wait_event.clear()
            solved = st.wait_event.wait(timeout=600)
            if not solved or not st.captcha_text:
                post_progress(callback_url, job_id, "captcha_timeout", "CAPTCHA入力がタイムアウト", 100, {"result": "timeout"})
                try:
                    browser.close()
                except Exception:
                    pass
                st.done = True
                return

            # 6) 入力して送信
            post_progress(callback_url, job_id, "captcha_submit", "CAPTCHA送信中", 35)
            # 例: input[name='captcha']（実サイトに合わせ調整）
            page.fill("input[name='captcha']", st.captcha_text)
            page.click("button[type=submit]")
            page.wait_for_load_state("load")

            # 7) メール認証 → APIキー回収（<<< 実装者: 既存手順のセレクタを移植 >>>）
            post_progress(callback_url, job_id, "mail_verify", "メール認証フロー進行中", 55)
            # ... ここにメール受信/リンク踏み/画面遷移の実処理 ...

            post_progress(callback_url, job_id, "api_key", "APIキー取得中", 80)
            # 例: APIキー抽出
            # api_key = page.text_content("code.api-key")
            api_key = "DUMMY-KEY-FROM-CLIENT"  # <<< 実装者: 実抽出に置換 >>>

            # 8) 完了
            try:
                browser.close()
            except Exception:
                pass

            st.done = True
            return finish_ok(callback_url, job_id, payload, api_key=api_key, note="ok")
    except Exception as e:
        st.error = str(e)
        post_progress(callback_url, job_id, "error", f"例外: {e}", 100, {"result": "error"})
        st.done = True
        return


# ========== 付帯関数 ==========
def dummy_flow(job_id: str, st: JobState):
    """Playwright 未導入時のダミー進捗（2〜3秒で完了）"""
    payload = st.payload
    callback_url = payload.get("callback_url")
    for pct in (10, 25, 50, 75, 90):
        post_progress(callback_url, job_id, "dummy", f"ダミー進行中 {pct}%", pct)
        time.sleep(0.4)
    st.done = True

def finish_ok(callback_url: Optional[str], job_id: str, payload: Dict,
              api_key: Optional[str], note: str):
    """最終完了通知（/external-seo/callback）"""
    data = {
        "ok": True,
        "status": "api_key_received" if api_key else "done",
        "site_id": payload.get("site_id"),
        "account_id": payload.get("account_id"),
        "blog": payload.get("blog", "livedoor"),
        "token": payload.get("token"),  # サーバ照合用（セッション extseo_token 等）
        "progress": 100,
        "step": "apiKey_received" if api_key else "finished",
        "helper_host": socket.gethostname(),
        "helper_ip_public": get_public_ip(),
    }
    if api_key:
        data["api_key"] = api_key
    post_progress(callback_url, job_id, "done", "完了", 100, data)


# ========== エントリポイント ==========
if __name__ == "__main__":
    # どのOSでもダブルクリックで立ち上がるようにシンプルに
    # Flask のデフォルトサーバでOK（localhost限定）
    app.run(host="127.0.0.1", port=PORT)

# helper/main.py
import threading, time, json, os, socket
from flask import Flask, request, jsonify
import requests

# 任意: Playwright を使う（後で本実装を差し込み）
# from playwright.sync_api import sync_playwright

app = Flask(__name__)

# 設定
PORT = int(os.environ.get("HELPER_PORT", 17653))

# 簡易メモリキュー（同時実行制御は最小限）
jobs = {}

@app.get("/ping")
def ping():
    return jsonify({"ok": True, "name": "signup-helper", "version": "0.1.0"})

def run_signup_job(job_id, payload):
    """
    実ジョブ（ユーザーのIP=この端末から実行）
    1) ブラウザ起動 → 2) 各サイトでアカウント作成 → 3) APIキー取得
    4) サーバへ callback
    """
    try:
        # --- ここから実装 ---
        # ※ まずは疎通の骨組み。Playwright 具体処理は後から差し込み。
        # with sync_playwright() as p:
        #     browser = p.chromium.launch(headless=False)  # <-- ユーザーIPでアクセス
        #     page = browser.new_page()
        #     page.goto("https://livedoor.blogcms.jp/member/blog/create")
        #     ... ここでフォーム入力/画像DL ... 最後に apiKey 取得 ...
        #     browser.close()

        # （後でここに Playwright の実処理を移植）
        # with sync_playwright() as p:
        #     browser = p.chromium.launch(headless=False)
        #     page = browser.new_page()
        #     ... 実処理 ...
        #     browser.close()

        # まずはダミー進捗（2〜3秒で完了通知）
        for _ in range(3):
            time.sleep(1)

        # 実行端末のパブリックIPを取得（検証用）
        try:
            public_ip = requests.get("https://api.ipify.org", timeout=5).text.strip()
        except Exception:
            public_ip = None

        callback_url = payload.get("callback_url")
        data = {
            "ok": True,
            "status": "api_key_received",
            "site_id": payload.get("site_id"),
            "account_id": payload.get("account_id"),
            "blog": payload.get("blog", "livedoor"),
            "token": payload.get("token"),      # サーバ照合用（セッション extseo_token）
            "progress": 100,
            "step": "apiKey_received",
            "helper_host": socket.gethostname(),
            "helper_ip_public": public_ip,
            # "api_key": "xxxxx"  # 実装後ここに取得キー
        }
        if callback_url:
            try:
                requests.post(callback_url, json=data, timeout=10)
            except Exception as e:
                print("callback failed:", e)
        jobs[job_id]["done"] = True
    except Exception as e:
        callback_url = payload.get("callback_url")
        if callback_url:
            try:
                requests.post(callback_url, json={
                    "ok": False, "error": str(e),
                    "site_id": payload.get("site_id"),
                    "account_id": payload.get("account_id"),
                    "blog": payload.get("blog", "livedoor"),
                    "token": payload.get("token"),
                    "step": "failed",
                }, timeout=10)
            except Exception:
                pass
        jobs[job_id]["done"] = True

@app.post("/start")
def start():
    """
    期待するJSON:
    {
      "token": "...",                      # /external-seo/start の戻り値
      "site_id": 123, "account_id": 456,
      "blog": "livedoor",
      "callback_url": "https://xxx/external-seo/callback"
    }
    """
    try:
        payload = request.get_json(force=True, silent=False) or {}
    except Exception:
        return jsonify({"ok": False, "error": "invalid json"}), 400

    job_id = f'job-{int(time.time()*1000)}'
    jobs[job_id] = {"done": False, "payload": payload}

    th = threading.Thread(target=run_signup_job, args=(job_id, payload), daemon=True)
    th.start()
    return jsonify({"ok": True, "job_id": job_id})

if __name__ == "__main__":
    # どのOSでもダブルクリックで立ち上がるようにシンプルに
    app.run(host="127.0.0.1", port=PORT)

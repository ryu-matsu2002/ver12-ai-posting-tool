import os
import requests
import logging
from datetime import datetime, date, timedelta

from google.oauth2 import service_account
from googleapiclient.discovery import build
from flask import current_app
from app.models import GSCAuthToken
from app import db

# ────── Service Account 認証情報の読み込み（主に管理者用途）──────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
KEY_PATH = os.path.join(BASE_DIR, "credentials", "service_account.json")
SCOPES = ["https://www.googleapis.com/auth/webmasters.readonly"]

def get_search_console_service():
    credentials = service_account.Credentials.from_service_account_file(
        KEY_PATH, scopes=SCOPES
    )
    service = build("searchconsole", "v1", credentials=credentials)
    return service

def fetch_search_queries(site_url: str, days: int = 7, row_limit: int = 20):
    service = get_search_console_service()

    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    request = {
        "startDate": start_date.isoformat(),
        "endDate": end_date.isoformat(),
        "dimensions": ["query"],
        "rowLimit": row_limit
    }

    response = service.searchanalytics().query(siteUrl=site_url, body=request).execute()
    return response.get("rows", [])

# ────── 各ユーザーごとのアクセストークンを使ったGSCデータ取得 ──────

def refresh_gsc_token(token_entry: GSCAuthToken) -> GSCAuthToken:
    logging.info(f"[GSC] Refresh token 実行中 user_id={token_entry.user_id}")

    token_url = "https://oauth2.googleapis.com/token"
    payload = {
        "client_id": current_app.config["GSC_CLIENT_ID"],
        "client_secret": current_app.config["GSC_CLIENT_SECRET"],
        "refresh_token": token_entry.refresh_token,
        "grant_type": "refresh_token"
    }

    res = requests.post(token_url, data=payload)
    if res.status_code != 200:
        logging.error(f"[GSC] トークンリフレッシュ失敗: {res.text}")
        raise Exception("アクセストークンの更新に失敗しました")

    tokens = res.json()
    token_entry.access_token = tokens["access_token"]
    token_entry.token_expiry = datetime.utcnow() + timedelta(seconds=tokens["expires_in"])
    db.session.commit()
    return token_entry

def fetch_search_queries_for_user(site_url: str, user_id: int, days: int = 7, row_limit: int = 40):
    token_entry = GSCAuthToken.query.filter_by(user_id=user_id).first()
    if not token_entry:
        raise Exception("GSCトークンが見つかりません")

    if not token_entry.access_token:
        raise Exception("アクセストークンが存在しません")

    if token_entry.token_expiry and token_entry.token_expiry < datetime.utcnow():
        logging.info(f"[GSC] アクセストークン期限切れ → refresh実行 user_id={user_id}")
        token_entry = refresh_gsc_token(token_entry)

    headers = {
        "Authorization": f"Bearer {token_entry.access_token}",
        "Content-Type": "application/json"
    }

    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    url = f"https://searchconsole.googleapis.com/v1/sites/{site_url}/searchAnalytics/query"
    payload = {
        "startDate": start_date.isoformat(),
        "endDate": end_date.isoformat(),
        "dimensions": ["query"],
        "rowLimit": row_limit
    }

    res = requests.post(url, headers=headers, json=payload)
    if res.status_code != 200:
        raise Exception(f"Search Console APIエラー: {res.status_code} - {res.text}")

    return res.json().get("rows", [])

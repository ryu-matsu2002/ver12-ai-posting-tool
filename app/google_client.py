import os
import requests
import logging
from datetime import datetime, date, timedelta

from google.oauth2 import service_account
from googleapiclient.discovery import build
from flask import current_app
from app.models import Site
from app import db

# ────── Service Account 認証情報の読み込み ──────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
KEY_PATH = os.path.join(BASE_DIR, "credentials", "service_account.json")
SCOPES = ["https://www.googleapis.com/auth/webmasters.readonly"]

def get_search_console_service():
    credentials = service_account.Credentials.from_service_account_file(
        KEY_PATH, scopes=SCOPES
    )
    service = build("searchconsole", "v1", credentials=credentials)
    return service

# ────── 🔍 Search Console からキーワード取得 ──────
def fetch_search_queries_for_site(site_url: str, days: int = 28, row_limit: int = 1000) -> list[str]:
    try:
        # ✅ 修正: URL末尾に / を補完（GSC APIは完全一致が必須）
        if not site_url.endswith("/"):
            site_url += "/"

        # ✅ 追加: クエリ取得ログ（事前）
        logging.info(f"[GSC] クエリ取得開始: {site_url}")

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
        rows = response.get("rows", [])

        # ✅ 追加: クエリ取得結果のログ
        logging.info(f"[GSC] {len(rows)} 件のクエリを取得: {site_url}")
        if not rows:
            logging.warning(f"[GSC] クエリが0件（空）で返却されました: {site_url}")

        return [row["keys"][0] for row in rows]

    except Exception as e:
        logging.error(f"[GSC取得失敗] site: {site_url} → {e}")
        return []

# ────── 🔄 メトリクス取得（クリック数・表示回数） ──────
def update_gsc_metrics(site: Site):
    if not site.gsc_connected:
        return

    try:
        # ✅ 修正: URL末尾に / を補完（GSC APIは完全一致が必須）
        site_url = site.url
        if not site_url.endswith("/"):
            site_url += "/"

        service = get_search_console_service()
        today = date.today()
        start_date = today - timedelta(days=30)

        request = {
            "startDate": start_date.isoformat(),
            "endDate": today.isoformat(),
            "dimensions": ["query"],
            "rowLimit": 25000
        }

        response = service.searchanalytics().query(siteUrl=site_url, body=request).execute()
        rows = response.get("rows", [])

        clicks = sum(row.get("clicks", 0) for row in rows)
        impressions = sum(row.get("impressions", 0) for row in rows)

        site.clicks = clicks
        site.impressions = impressions
        db.session.commit()

        logging.info(f"[GSC] ✅ メトリクス更新完了 - site: {site_url} | Clicks: {clicks}, Impressions: {impressions}")

    except Exception as e:
        logging.error(f"[GSC] メトリクス取得失敗 - {site.url} - {e}")

# ────── 🔁 全接続サイトを一括更新（スケジューラー用）──────
def update_all_gsc_sites():
    sites = Site.query.filter_by(gsc_connected=True).all()
    for site in sites:
        update_gsc_metrics(site)

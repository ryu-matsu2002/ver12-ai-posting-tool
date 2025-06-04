import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from datetime import date, timedelta

# 認証ファイルパス
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

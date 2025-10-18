import logging, time, os
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from app.models import GSCAuthToken

# Inspection API は Search Console API 内の "urlInspection.index.inspect"
# 参考: https://developers.google.com/webmaster-tools/v1/urlInspection.index/inspect

SCOPES = ["https://www.googleapis.com/auth/webmasters"]

def _build_inspection_service(access_token: str, refresh_token: Optional[str] = None, expiry: Optional[datetime] = None):
    creds = Credentials(
        token=access_token,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        scopes=SCOPES,
    )
    return build("searchconsole", "v1", credentials=creds, cache_discovery=False)

def inspect_url_with_token(site_id: int, url: str, resource_id: str) -> Dict[str, Any]:
    """
    ユーザーのGSCトークンを使って URL Inspection を実行。
    成功時: {'result': {...}} を返す
    失敗時: {'error': '...'} を返す
    """
    token: Optional[GSCAuthToken] = (
        GSCAuthToken.query.filter_by(site_id=site_id).order_by(GSCAuthToken.id.desc()).first()
    )
    if not token:
        return {"error": "no_token"}

    service = _build_inspection_service(token.access_token, token.refresh_token, token.token_expiry)
    body = {"inspectionUrl": url, "siteUrl": resource_id}

    # 指数バックオフ
    delay = 0.5
    for attempt in range(6):  # 最大 ~15秒程度
        try:
            req = service.urlInspection().index().inspect(body=body)
            resp = req.execute()
            return {"result": resp}
        except HttpError as e:
            status = getattr(e, "status_code", None) or getattr(e.resp, "status", None)
            if status in (429, 500, 502, 503, 504):
                time.sleep(delay)
                delay = min(delay * 2, 8.0)
                continue
            logging.warning(f"[INSPECT] http error {status} for {url}: {e}")
            return {"error": f"http_{status}"}
        except Exception as e:
            logging.exception(f"[INSPECT] unknown error for {url}: {e}")
            return {"error": "unknown"}
    return {"error": "retry_exhausted"}

def parse_inspection_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inspection API の生データから必要な要点を抽出。
    """
    try:
        idx = payload.get("result", {}).get("inspectionResult", {})
        coverage = idx.get("indexStatusResult", {})
        verdict = coverage.get("verdict")  # PASS/FAIL/NEUTRAL
        indexed = verdict == "PASS"
        return {
            "indexed": bool(indexed),
            "coverage_state": coverage.get("coverageState"),
            "verdict": verdict,
            "last_crawl_time": coverage.get("lastCrawlTime"),
            "robots_txt_state": coverage.get("robotsTxtState"),
            "page_fetch_state": coverage.get("pageFetchState"),
            "raw": payload.get("result", {}),
        }
    except Exception:
        return {"indexed": False, "coverage_state": None, "verdict": None, "raw": payload}

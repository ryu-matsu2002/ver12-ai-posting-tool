import os
import requests
import logging
from datetime import datetime, date, timedelta, timezone

from google.oauth2 import service_account
from googleapiclient.discovery import build
from flask import current_app
from app.models import Site, GSCMetric, GSCDailyTotal, GSCConfig  # ✅ 追加
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

# ────── ✅✅✅ 追加: GSCMetricとして保存する処理 ──────
def store_metrics_from_gsc_rows(rows, site, metric_date: date):
    for row in rows:
        query = row["keys"][0]
        impressions = row.get("impressions", 0)
        clicks = row.get("clicks", 0)
        ctr = row.get("ctr", 0.0)
        position = row.get("position", 0.0)

        metric = GSCMetric(
            site_id=site.id,
            user_id=site.user_id,
            date=metric_date,
            query=query,
            impressions=impressions,
            clicks=clicks,
            ctr=ctr,
            position=position,
        )
        db.session.add(metric)
    db.session.commit()
    logging.info(f"[GSCMetric] ✅ 保存完了: {site.name} ({len(rows)} 件)")

# ────── 🔍 Search Console からキーワード取得 ──────
def fetch_search_queries_for_site(site: Site, days: int = 28, row_limit: int = 1000) -> list[str]:
    try:
        # ✅ 修正: URL末尾に / を補完（GSC APIは完全一致が必須）
        site_url = site.url
        if not site_url.endswith("/"):
            site_url += "/"

        # ✅ クエリ取得ログ（事前）
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

        # ✅✅✅ GSCMetricに保存（今回の新機能）
        store_metrics_from_gsc_rows(rows, site, end_date)

        # ✅ 既存機能: 検索キーワードのリストを返す（記事生成用）
        return [row["keys"][0] for row in rows]

    except Exception as e:
        logging.error(f"[GSC取得失敗] site: {site.url} → {e}")
        return []

# ────── 🔄 メトリクス取得（クリック数・表示回数） ──────
def update_gsc_metrics(site: Site):
    """
    互換用：内部的には “日次合計の保存” に置き換え。
    （既存呼び出し元があっても壊さないために残しておく）
    """
    try:
        update_site_daily_totals(site, days=35)
    except Exception as e:
        logging.error(f"[GSC] update_gsc_metrics (compat) failed - {site.url} - {e}")

# ────── 🔁 全接続サイトを一括更新（スケジューラー用）──────
def update_all_gsc_sites():
    sites = Site.query.filter_by(gsc_connected=True).all()
    total_upsert = 0
    for site in sites:
        try:
            total_upsert += update_site_daily_totals(site, days=35)
        except Exception as e:
            logging.error(f"[GSC] site batch failed: {site.url} - {e}")
    logging.info(f"[GSC] batch done: upsert={total_upsert} rows")


# 追加：内部ヘルパー
def _site_url_norm(site: Site) -> str:
    return site.url if site.url.endswith("/") else site.url + "/"

# 追加：GSCのプロパティURIを決定（あれば GSCConfig 優先）
def _resolve_property_uri(site: Site) -> str:
    cfg = GSCConfig.query.filter_by(site_id=site.id).order_by(GSCConfig.id.desc()).first()
    if cfg and cfg.property_uri:
        return cfg.property_uri.strip()
    # fallback: URLプレフィックス
    return _site_url_norm(site)

def fetch_daily_totals_for_property(property_uri: str, start_d: date, end_d: date):
    """
    GSCから dimensions=['date'] で “日別合計” をそのまま取得する。
    返り値は Search Console API の rows（keys[0]が'YYYY-MM-DD'）。
    """
    service = get_search_console_service()
    body = {
        "startDate": start_d.isoformat(),
        "endDate": end_d.isoformat(),
        "dimensions": ["date"],
        "rowLimit": 25000
    }
    logging.info(f"[GSC] daily totals: {property_uri} {start_d}..{end_d}")
    resp = service.searchanalytics().query(siteUrl=property_uri, body=body).execute()
    rows = resp.get("rows", [])
    logging.info(f"[GSC] daily totals rows={len(rows)} {property_uri}")
    return rows

def upsert_gsc_daily_totals(site: Site, property_uri: str, rows: list[dict]) -> int:
    """
    rows（date次元の合計）を GSCDailyTotal に upsert。戻り値は upsert 件数。
    """
    if not rows:
        return 0

    # すでにある日付を一括で取得しておく（小さな期間なのでこれで十分高速）
    dates = [date.fromisoformat(r["keys"][0]) for r in rows if r.get("keys")]
    existing = {
        (x.date): x
        for x in GSCDailyTotal.query
            .filter(GSCDailyTotal.site_id == site.id,
                    GSCDailyTotal.property_uri == property_uri,
                    GSCDailyTotal.date.in_(dates))
            .all()
    }

    upsert_cnt = 0
    for r in rows:
        if not r.get("keys"):
            continue
        d = date.fromisoformat(r["keys"][0])
        clicks = int(r.get("clicks", 0) or 0)
        impressions = int(r.get("impressions", 0) or 0)

        row = existing.get(d)
        if row:
            # 値が違うときのみ更新
            if row.clicks != clicks or row.impressions != impressions:
                row.clicks = clicks
                row.impressions = impressions
                upsert_cnt += 1
        else:
            db.session.add(GSCDailyTotal(
                site_id=site.id,
                user_id=site.user_id,
                property_uri=property_uri,
                date=d,
                clicks=clicks,
                impressions=impressions
            ))
            upsert_cnt += 1

    if upsert_cnt:
        db.session.commit()
    return upsert_cnt

def update_site_daily_totals(site: Site, days: int = 35) -> int:
    """
    サイトの直近N日（デフォ35日）の “日次合計” を取得して DB に保存。
    GSC UIと一致させるため、JSTの「今日」基準で範囲を組む。
    戻り値は upsert件数。
    """
    if not site.gsc_connected:
        return 0

    JST = timezone(timedelta(hours=9))
    today_jst = datetime.utcnow().replace(tzinfo=timezone.utc).astimezone(JST).date()
    # 例：28日表示に完全一致させるには、start = today_jst - timedelta(days=27)
    span = max(1, int(days))
    start_d = today_jst - timedelta(days=span - 1)
    end_d = today_jst

    prop = _resolve_property_uri(site)
    rows = fetch_daily_totals_for_property(prop, start_d, end_d)
    upserted = upsert_gsc_daily_totals(site, prop, rows)
    logging.info(f"[GSC] upserted={upserted} site={site.name} prop={prop}")
    return upserted



def _run_search_analytics(site: Site, days: int, dimensions: list[str], row_limit: int,
                          order_by_impressions: bool = False):
    service = get_search_console_service()
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    body = {
        "startDate": start_date.isoformat(),
        "endDate": end_date.isoformat(),
        "dimensions": dimensions,
        "rowLimit": row_limit
    }
    if order_by_impressions:
        body["orderBy"] = [{"field": "impressions", "descending": True}]

    site_url = _site_url_norm(site)
    logging.info(f"[GSC] query dims={dimensions} limit={row_limit} {site_url}")
    resp = service.searchanalytics().query(siteUrl=site_url, body=body).execute()
    rows = resp.get("rows", [])
    logging.info(f"[GSC] rows={len(rows)} dims={dimensions} {site_url}")
    return rows

# 追加：上位クエリ40件（表示回数降順）
def fetch_top_queries_for_site(site: Site, days: int = 28, limit: int = 40) -> list[dict]:
    try:
        rows = _run_search_analytics(site, days, ["query"], limit, order_by_impressions=True)
        out = []
        for r in rows:
            out.append({
                "query": r["keys"][0],
                "impressions": int(r.get("impressions", 0)),
                "clicks": int(r.get("clicks", 0)),
                "ctr": float(r.get("ctr", 0.0)),
                "position": float(r.get("position", 0.0)),
            })
        return out
    except Exception as e:
        logging.error(f"[GSC] fetch_top_queries_for_site failed: {e}")
        return []

# 追加：上位ページ3件（表示回数降順）
def fetch_top_pages_for_site(site: Site, days: int = 28, limit: int = 3) -> list[str]:
    try:
        rows = _run_search_analytics(site, days, ["page"], limit, order_by_impressions=True)
        return [r["keys"][0] for r in rows if r.get("keys")]
    except Exception as e:
        logging.error(f"[GSC] fetch_top_pages_for_site failed: {e}")
        return []

# 追加：サイト内の全ページ（Search Consoleに出てくる範囲）
def fetch_all_pages_for_site(site: Site, days: int = 180, limit: int = 25000) -> list[str]:
    try:
        rows = _run_search_analytics(site, days, ["page"], limit, order_by_impressions=False)
        # 重複や空を除去
        pages = [r["keys"][0] for r in rows if r.get("keys") and r["keys"][0]]
        # 絶対URLのみ、末尾のスラッシュはそのまま（そのページURLを尊重）
        return list(dict.fromkeys(pages))  # 順序維持の重複排除
    except Exception as e:
        logging.error(f"[GSC] fetch_all_pages_for_site failed: {e}")
        return []

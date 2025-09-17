import os
import requests
import logging
from datetime import datetime, date, timedelta, timezone

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from flask import current_app
from app.models import Site, GSCMetric, GSCDailyTotal, GSCConfig  # ✅ 追加
from app import db
from typing import List, Dict, Tuple, Optional

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


# ─────────────────────────────────────────────────────────────────────
# 環境変数ヘルパー（値が無ければデフォルトにフォールバック）
def _get_env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except Exception:
        return default

def _jst_today() -> date:
    """JSTベースの今日（日付）を返す。"""
    JST = timezone(timedelta(hours=9))
    return datetime.utcnow().replace(tzinfo=timezone.utc).astimezone(JST).date()

def _jst_28d_window_end_start() -> Tuple[date, date]:
    """
    GSC UI と合わせて『昨日までの28日』を基本窓とする。
    end = 昨日(JST), start = end - 27 日
    """
    end_d = _jst_today() - timedelta(days=1)
    start_d = end_d - timedelta(days=27)
    return end_d, start_d

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
def fetch_search_queries_for_site(site: Site, days: int = 28, row_limit: int = 25000) -> list[str]:
    try:
        service = get_search_console_service()
        # JST準拠で“昨日まで”
        JST = timezone(timedelta(hours=9))
        today_jst = datetime.utcnow().replace(tzinfo=timezone.utc).astimezone(JST).date()
        end_date = today_jst - timedelta(days=1)  # 昨日まで
        start_date = end_date - timedelta(days=days)
        property_uri = _resolve_property_uri(site)

        logging.info(f"[GSC] クエリ取得開始: {property_uri}")

        rows = []
        start_row = 0
        while True:
            body = {
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
                "dimensions": ["query"],
                "rowLimit": row_limit,
                "startRow": start_row,
                "searchType": "web",
                "dataState": "FINAL",
            }
            resp = service.searchanalytics().query(siteUrl=property_uri, body=body).execute()
            chunk = resp.get("rows", []) or []
            rows.extend(chunk)
            logging.info(f"[GSC] pagination: fetched={len(chunk)} total={len(rows)} startRow={start_row}")
            if len(chunk) < row_limit:
                break
            start_row += row_limit

        # ✅ 追加: クエリ取得結果のログ
        logging.info(f"[GSC] {len(rows)} 件のクエリを取得: {property_uri}")
        if not rows:
            logging.warning(f"[GSC] クエリが0件（空）で返却されました: {property_uri}")

        # ✅✅✅ GSCMetricに保存（今回の新機能）
        # ※注意：この保存は“期間合算”を1日付に押し込む設計。
        # UI合計との一致性を重視するなら、dimensions=['query','date']で日次保存に改修推奨。
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
    """
    既に GSCConfig があればそれを返す。なければ URL プレフィックスを返す。
    （実際の呼び出し側で候補を試し、成功した URI を GSCConfig に保存する）
    """
    cfg = GSCConfig.query.filter_by(site_id=site.id).order_by(GSCConfig.id.desc()).first()
    if cfg and cfg.property_uri:
        return cfg.property_uri.strip()
    return _site_url_norm(site)

def _run_query_date_matrix(property_uri: str, start_d: date, end_d: date, row_limit: int = 25000) -> List[Dict]:
    """
    dimensions=['query','date'] のマトリクスを取得。
    28日合計などのアプリ側集計に使う。
    """
    service = get_search_console_service()
    body = {
        "startDate": start_d.isoformat(),
        "endDate": end_d.isoformat(),
        "dimensions": ["query", "date"],
        "rowLimit": row_limit,
        "searchType": "web",
        "dataState": "FINAL",
    }
    logging.info(f"[GSC] query-date matrix: {property_uri} {start_d}..{end_d}")
    resp = service.searchanalytics().query(siteUrl=property_uri, body=body).execute()
    rows = resp.get("rows", []) or []
    logging.info(f"[GSC] query-date rows={len(rows)} {property_uri}")
    return rows

def _aggregate_query_stats(rows: List[Dict]) -> Dict[str, Dict]:
    """
    rows（keys=[query, 'YYYY-MM-DD']）をクエリ単位に集計。
    - impressions_28d: 合計表示回数
    - first_seen_date: 最初に観測された日付（最小日付）
    """
    from datetime import date as _date
    stats: Dict[str, Dict] = {}
    for r in rows:
        if not r.get("keys"): 
            continue
        q, ds = r["keys"][0], r["keys"][1]
        try:
            d = _date.fromisoformat(ds)
        except Exception:
            continue
        imp = int(r.get("impressions", 0) or 0)
        s = stats.get(q)
        if not s:
            stats[q] = {"impressions_28d": imp, "first_seen_date": d}
        else:
            s["impressions_28d"] += imp
            if d < s["first_seen_date"]:
                s["first_seen_date"] = d
    return stats

# ─────────────────────────────────────────────────────────────────────
# ✅ 新着クエリ抽出（cutoff以降に初観測 ＆ 28日合計imprがENV以上）
def fetch_new_queries_since(
    site: Site,
    cutoff_date: Optional[date] = None,
    min_impressions: Optional[int] = None,
    row_limit: int = 25000,
) -> List[Dict]:
    """
    返り値: [{"query": str, "impressions_28d": int, "first_seen_date": date}, ...]
    - first_seen_date >= cutoff_date
    - impressions_28d >= min_impressions
    窓は JST基準で『昨日までの28日』に固定（UI整合性のため）。
    """
    # フォールバック
    if cutoff_date is None:
        cutoff_date = getattr(site, "gsc_autogen_since", None) or _jst_today()
    if min_impressions is None:
        min_impressions = _get_env_int("GSC_MIN_IMPRESSIONS", 20)

    # 集計窓（28日固定）
    end_d, start_d = _jst_28d_window_end_start()

    # siteUrl は検証済みの property を優先
    property_uri = _resolve_property_uri(site)

    try:
        rows = _run_query_date_matrix(property_uri, start_d, end_d, row_limit=row_limit)
        stats = _aggregate_query_stats(rows)

        # フィルタ適用
        picked: List[Dict] = []
        for q, s in stats.items():
            if s["first_seen_date"] >= cutoff_date and s["impressions_28d"] >= min_impressions:
                picked.append({
                    "query": q,
                    "impressions_28d": int(s["impressions_28d"]),
                    "first_seen_date": s["first_seen_date"],
                })

        logging.info(
            "[GSC-AUTOGEN] pick_new_queries site_id=%s prop=%s start=%s end=%s cutoff=%s picked=%s (rows=%s, min_impr=%s)",
            site.id, property_uri, start_d, end_d, cutoff_date, len(picked), len(rows), min_impressions
        )
        return picked
    except HttpError as e:
        status = getattr(e, "status_code", None) or getattr(e.resp, "status", None)
        logging.warning(f"[GSC-AUTOGEN] query-date failed ({status}) site={site.id} prop={property_uri}")
        return []
    except Exception as e:
        logging.exception(f"[GSC-AUTOGEN] unexpected error site={site.id} prop={property_uri}: {e}")
        return []

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
        "rowLimit": 25000,
        "searchType": "web",
        "dataState": "FINAL",
    }
    logging.info(f"[GSC] daily totals: {property_uri} {start_d}..{end_d}")
    try:
        resp = service.searchanalytics().query(siteUrl=property_uri, body=body).execute()
        rows = resp.get("rows", [])
    except HttpError as e:
        # 呼び出し元でハンドリングしたいのでそのまま投げ直す
        raise
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
    # ✅ GSC UI に合わせて「昨日まで」
    end_d = today_jst - timedelta(days=1)
    start_d = end_d - timedelta(days=span - 1)

    # まず既知の URI（設定済みがあればそれ）から試行
    first = _resolve_property_uri(site)
    candidates = [first]
    # バリアントを追加（http/https, www 有無, sc-domain）
    try:
        from urllib.parse import urlparse
        p = urlparse(_site_url_norm(site))
        host = p.hostname or ""
        bare = host.replace("www.", "")
        candidates += [
            f"https://{host}/",
            f"http://{host}/",
            f"https://www.{bare}/",
            f"http://www.{bare}/",
            f"sc-domain:{bare}",
        ]
    except Exception:
        pass
    # 重複除去
    seen = set()
    candidates = [x for x in candidates if not (x in seen or seen.add(x))]

    last_err = None
    for prop in candidates:
        try:
            rows = fetch_daily_totals_for_property(prop, start_d, end_d)
            upserted = upsert_gsc_daily_totals(site, prop, rows)
            logging.info(f"[GSC] upserted={upserted} site={site.name} prop={prop}")
            # 成功した prop を学習保存（次回以降ダイレクトに使う）
            cfg = GSCConfig(site_id=site.id, user_id=site.user_id, property_uri=prop)
            db.session.add(cfg)
            db.session.commit()
            return upserted
        except HttpError as e:
            # 403 や 404 など → 次の候補へ
            status = getattr(e, "status_code", None) or getattr(e.resp, "status", None)
            logging.warning(f"[GSC] try prop failed ({status}): {prop}")
            last_err = e
            continue
        except Exception as e:
            logging.exception(f"[GSC] unexpected error for prop={prop}: {e}")
            last_err = e
            continue
    # すべて失敗：403ならスキップ、その他はraise
    if last_err:
        if isinstance(last_err, HttpError):
            status = getattr(last_err, "status_code", None) or getattr(last_err.resp, "status", None)
            if status == 403:
                logging.warning(f"[GSC] skip site (403 forbidden): site_id=%s url=%s", site.id, site.url)
                return 0
        raise last_err


def _run_search_analytics(site: Site, days: int, dimensions: list[str], row_limit: int,
                          order_by_impressions: bool = False):
    service = get_search_console_service()
    # JST準拠で“昨日まで”の窓に統一
    JST = timezone(timedelta(hours=9))
    today_jst = datetime.utcnow().replace(tzinfo=timezone.utc).astimezone(JST).date()
    end_date = today_jst - timedelta(days=1)  # 昨日まで
    start_date = end_date - timedelta(days=days)

    body = {
        "startDate": start_date.isoformat(),
        "endDate": end_date.isoformat(),
        "dimensions": dimensions,
        "rowLimit": row_limit,
        "searchType": "web",
        "dataState": "FINAL",
    }
    if order_by_impressions:
        body["orderBy"] = [{"field": "impressions", "descending": True}]

    property_uri = _resolve_property_uri(site)
    logging.info(f"[GSC] query dims={dimensions} limit={row_limit} prop={property_uri} body={body}")
    resp = service.searchanalytics().query(siteUrl=property_uri, body=body).execute()
    rows = resp.get("rows", [])
    logging.info(f"[GSC] rows={len(rows)} dims={dimensions} prop={property_uri}")
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

# app/google_client.py に追加（必要なときだけ呼ぶ）
def fetch_totals_direct(property_uri: str, start_d: date, end_d: date) -> dict:
    """
    GSC UIの28日合計に寄せるため、dimensionsなしで直接合計を取得。
    dataState='FINAL' を指定。
    """
    service = get_search_console_service()
    body = {
        "startDate": start_d.isoformat(),
        "endDate": end_d.isoformat(),
        # 無次元は避けたいが互換のため残す。APIがtotalsを返さない場合あり
        "dimensions": [],
        "dataState": "FINAL",
        "searchType": "web",
        "rowLimit": 1,
    }
    resp = service.searchanalytics().query(siteUrl=property_uri, body=body).execute()
    # rowsベース or top-level totals ベースの双方を探る
    if "rows" in resp and resp["rows"]:
        r0 = resp["rows"][0]
        return {
            "clicks": int(r0.get("clicks", 0) or 0),
            "impressions": int(r0.get("impressions", 0) or 0),
        }
    return {
        "clicks": int(resp.get("clicks", 0) or 0),
        "impressions": int(resp.get("impressions", 0) or 0),
    }

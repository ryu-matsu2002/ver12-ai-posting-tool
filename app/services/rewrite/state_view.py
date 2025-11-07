# -*- coding: utf-8 -*-
"""
統一集計ビュー vw_rewrite_state を読み取って、
管理UIで必要な集計を返す“読み取り専用”サービス。

このモジュールは DB データを変更しない（SELECT のみ）。
全ての画面はこのモジュール経由で数字を出すことで、
「待機／実行中／成功／失敗」の定義と一致を保つ。
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from sqlalchemy import text
from flask import current_app
from app import db


# ------------------------------------------------------------
# 内部ユーティリティ
# ------------------------------------------------------------

# UIでの表示順を固定したいときのための並び
_BUCKET_ORDER = ("waiting", "running", "success", "failed", "other")


def _sort_bucket_rows(rows: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """('bucket', cnt) の配列を UI 既定順で並び替える。"""
    order = {b: i for i, b in enumerate(_BUCKET_ORDER)}
    return sorted(rows, key=lambda r: order.get(r[0], 999))


# ------------------------------------------------------------
# 公開API：集計系
# ------------------------------------------------------------

def fetch_user_totals(user_id: int) -> Dict[str, int]:
    """
    ユーザー単位の最終バケット合計（waiting/running/success/failed/other）を返す。
    """
    sql = text("""
        SELECT final_bucket, COUNT(*) AS cnt
        FROM vw_rewrite_state
        WHERE user_id = :uid
        GROUP BY final_bucket
    """)
    rows = db.session.execute(sql, {"uid": user_id}).fetchall()
    # Dict[str, int] に整形（未出現バケットは0に）
    zeroed = {b: 0 for b in _BUCKET_ORDER}
    for b, cnt in rows:
        zeroed[b] = int(cnt)
    return zeroed


def fetch_user_site_breakdown(user_id: int) -> List[Dict]:
    """
    ユーザー配下のサイト別 × バケット件数の一覧を返す。
    返り値例: [{"site_id": 69, "waiting": 35, "running": 5, "success": 417, "failed": 56, "other": 67}, ...]
    """
    sql = text("""
        SELECT site_id, final_bucket, COUNT(*) AS cnt
        FROM vw_rewrite_state
        WHERE user_id = :uid
        GROUP BY site_id, final_bucket
    """)
    rows = db.session.execute(sql, {"uid": user_id}).fetchall()

    per_site: Dict[int, Dict[str, int]] = defaultdict(lambda: {b: 0 for b in _BUCKET_ORDER})
    for site_id, bucket, cnt in rows:
        per_site[int(site_id)][bucket] = int(cnt)

    result = []
    for site_id, counts in per_site.items():
        item = {"site_id": int(site_id)}
        item.update(counts)
        result.append(item)
    # site_id昇順
    result.sort(key=lambda x: x["site_id"])
    return result


def fetch_site_totals(user_id: int, site_id: int) -> Dict[str, int]:
    """
    サイト単位の最終バケット合計を返す。
    """
    sql = text("""
        SELECT final_bucket, COUNT(*) AS cnt
        FROM vw_rewrite_state
        WHERE user_id = :uid AND site_id = :sid
        GROUP BY final_bucket
    """)
    rows = db.session.execute(sql, {"uid": user_id, "sid": site_id}).fetchall()
    zeroed = {b: 0 for b in _BUCKET_ORDER}
    for b, cnt in rows:
        zeroed[b] = int(cnt)
    return zeroed


# ------------------------------------------------------------
# 公開API：記事一覧（サイト詳細でのテーブル）
# ------------------------------------------------------------

def fetch_site_articles(
    user_id: int,
    site_id: int,
    bucket: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict]:
    """
    サイト配下の記事一覧を返す。bucket で waiting/running/success/failed/other を絞り込める。
    UIの「リライト済み記事一覧（成功のみ）」等のテーブル源泉にできる。

    返り値: [{"article_id": int, "user_id": int, "site_id": int,
              "final_bucket": str, "plan_status": Optional[str],
              "scheduled_at": Optional[datetime], "started_at": Optional[datetime],
              "finished_at": Optional[datetime], "is_active": Optional[bool],
              "log_status": Optional[str], "log_executed_at": Optional[datetime]} ...]
    """
    where = ["user_id = :uid", "site_id = :sid"]
    params = {"uid": user_id, "sid": site_id, "limit": limit, "offset": offset}
    if bucket:
        where.append("final_bucket = :bucket")
        params["bucket"] = bucket

    sql = text(f"""
        SELECT
          article_id, user_id, site_id,
          plan_status, scheduled_at, started_at, finished_at, is_active,
          plan_created_at, log_status, log_executed_at, final_bucket
        FROM vw_rewrite_state
        WHERE {" AND ".join(where)}
        ORDER BY
          -- 成功/失敗は新しいログ優先、ほかは plan の新しさ
          COALESCE(log_executed_at, plan_created_at) DESC NULLS LAST,
          article_id DESC
        LIMIT :limit OFFSET :offset
    """)

    rows = db.session.execute(sql, params).mappings().all()
    return [dict(r) for r in rows]

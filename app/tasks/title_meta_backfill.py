# app/tasks/title_meta_backfill.py

from datetime import datetime

from typing import Any, Dict, Optional, Tuple
from sqlalchemy import func, or_

from app import db
from app.models import Article
from flask import current_app

# --- ユーティリティ ----------------------------------------------------------

def _now():
    # timezone=True のカラムに合わせて naive UTC を避ける（DB側でUTC管理）
    return datetime.utcnow()

def _coerce_limit(v: Optional[int], default: int = 200, max_cap: int = 1000) -> int:
    try:
        i = int(v or 0)
    except Exception:
        i = 0
    if i <= 0:
        i = default
    return max(1, min(i, max_cap))

def _as_bool(v: Any, default: bool=False) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1","true","yes","on"):
        return True
    if s in ("0","false","no","off"):
        return False
    return default

# --- メタ説明の生成ロジック（最小実装） --------------------------------------
# ここはあなたの本番生成器に差し替え可。
# ない場合でも“最低限の動作”として本文冒頭からサマライズする。
def _make_meta_description(title: str, body: Optional[str], limit_chars: int = 180) -> str:
    if (body or "").strip():
        text = body.strip().replace("\r", " ").replace("\n", " ")
        text = " ".join(text.split())
        cand = text[: limit_chars].strip()
        if len(cand) < 40 and title:
            cand = f"{title} – {cand}"
        return cand[: limit_chars]
    # body が空ならタイトルで埋める（WP側で上書き可）
    return (title or "")[: limit_chars]

def _judge_quality(meta_desc: str) -> str:
    l = len(meta_desc or "")
    if l == 0:
        return "empty"
    if l < 60:
        return "too_short"
    if l > 180:
        return "too_long"
    return "ok"

def _push_meta_to_wp_if_needed(article: Article) -> str:
    """投稿済み記事のメタをWP側へ反映し、結果ステータスを返す。
    戻り値: 'ok' | 'unresolved' | 'failed'
    """
    try:
        # 既存記事のメタ更新専用（既存APIを流用）
        from app.wp_client import update_post_meta, resolve_wp_post_id

        wp_post_id = getattr(article, "wp_post_id", None)
        site = getattr(article, "site", None)
        if not site:
            current_app.logger.info("[title-meta] skip WP meta sync (missing site) article_id=%s", article.id)
            return "unresolved"

        # ★ 追加：wp_post_id が空なら、内部SEOと同手法で「スキップ前に解決」
        if not wp_post_id:
            wp_post_id = resolve_wp_post_id(site=site, art=article, save=True) or None
            if wp_post_id:
                current_app.logger.info("[title-meta] resolved wp_post_id=%s article_id=%s", wp_post_id, article.id)

        # 解決できたときのみWPへ反映
        if wp_post_id:
            ok = update_post_meta(
                site=site,
                wp_post_id=wp_post_id,
                meta_description=article.meta_description or "",
            )
            return "ok" if ok else "failed"
        else:
            current_app.logger.info(
                "[title-meta] unresolved wp_post_id; meta not pushed article_id=%s", article.id
            )
            return "unresolved"
    except Exception as e:
        current_app.logger.exception(
            "[title-meta] WP meta sync failed article_id=%s err=%s", article.id, e
        )
        return "failed"

def _auto_post_if_needed(article: Article) -> None:
    """未投稿(done)記事をWPへ新規投稿し、DBのposted系フィールドを更新。"""
    try:
        from app.wp_client import post_to_wp
        result = post_to_wp(article)  # 戻り値の仕様に依存せず、後続は別ジョブでもよい
        # 可能ならここで article.status/posted_at/posted_url/wp_post_id を更新
        # post_to_wpが内部で更新する場合は何もしない
    except Exception as e:
        current_app.logger.exception("[title-meta] auto-post failed article_id=%s err=%s", article.id, e)

# --- 本体 --------------------------------------------------------------------

def run_title_meta_backfill(
    site_id: Optional[int] = None,
    user_id: Optional[int] = None,
    limit: Optional[int] = 200,
    dryrun: Optional[bool] = False,
    after_id: Optional[int] = None,
    push_to_wp: Optional[bool] = False,
) -> Dict[str, Any]:
    """
    Title/Meta バックフィル実行。
    - フィルタ: is_manual_meta=False AND status IN ('done','posted')
             AND meta_desc_quality IN ('empty','too_short','too_long','duplicate')
    - カーソル: id > after_id の昇順
    - 返却: {ok, updated, cursor, done}
    """
    dryrun = _as_bool(dryrun, False)
    push_to_wp = _as_bool(push_to_wp, False)
    limit_int = _coerce_limit(limit, default=200, max_cap=1000)

    qualities = ("empty", "too_short", "too_long", "duplicate")
    statuses  = ("done", "posted")

    q = db.session.query(Article).filter(
        Article.is_manual_meta == False,             # noqa: E712
        Article.status.in_(statuses),
        # ★ ここが肝：品質未評価(NULL) も対象に含める
        or_(Article.meta_desc_quality.in_(qualities),
            Article.meta_desc_quality.is_(None))
    )

    if user_id:
        q = q.filter(Article.user_id == int(user_id))
    if site_id:
        q = q.filter(Article.site_id == int(site_id))
    if after_id:
        q = q.filter(Article.id > int(after_id))

    q = q.order_by(Article.id.asc()).limit(limit_int)

    rows = q.all()
    if not rows:
        return {
            "ok": True,
            "updated": 0,
            "cursor": after_id,
            "done": True,
            # 進捗可視化用の追加フィールド（空でもキーは返す）
            "wp_target_total": 0,      # 分母候補：WP反映対象（postedかつpush_to_wp=True）の件数
            "wp_synced_ok": 0,         # 分子：実際にWPへ反映できた件数
            "wp_unresolved": 0,        # wp_post_id未解決などで未反映
            "wp_failed": 0,            # API等の失敗で未反映
        }

    updated = 0
    last_id = after_id or 0
    # 進捗カウンタ（UI用）
    wp_target_total = 0
    wp_synced_ok = 0
    wp_unresolved = 0
    wp_failed = 0

    for art in rows:
        last_id = art.id
        # 既存メタを再評価（short/long/dup なども再生成）
        new_meta = _make_meta_description(art.title or "", art.body or "", limit_chars=180)
        new_quality = _judge_quality(new_meta)

        # dryrun ならDB変更しない
        if dryrun:
            continue

        # 変更がある時だけ更新
        changed = False
        if (art.meta_description or "") != new_meta:
            art.meta_description = new_meta
            changed = True

        if (art.meta_desc_quality or "") != new_quality:
            art.meta_desc_quality = new_quality
            changed = True

        if changed:
            art.meta_desc_last_updated_at = _now()
            updated += 1

        # WP反映：要件に合わせて分岐
        if push_to_wp:
            if art.status == "posted":
                # 既に公開済み → メタをWP側へ更新（ここがWP分子のカウント対象）
                wp_target_total += 1
                result = _push_meta_to_wp_if_needed(art)
                if result == "ok":
                    wp_synced_ok += 1
                elif result == "unresolved":
                    wp_unresolved += 1
                else:
                    wp_failed += 1
            elif art.status == "done":
                # 未投稿 → 投稿せず、メタタグだけDB更新。
                # スケジューラーの通常投稿処理でWPへ反映される。
                current_app.logger.info("[title-meta] kept done article_id=%s for scheduler to post later", art.id)

    if not dryrun:
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            current_app.logger.exception("[title-meta] commit failed after_id=%s err=%s", after_id, e)
            return {"ok": False, "error": f"commit failed: {e}"}

    done = (len(rows) < limit_int)
    return {
        "ok": True,
        "updated": updated,
        "cursor": last_id,
        "done": done,
        # 進捗可視化用の追加フィールド
        "wp_target_total": wp_target_total,
        "wp_synced_ok": wp_synced_ok,
        "wp_unresolved": wp_unresolved,
        "wp_failed": wp_failed,
    }

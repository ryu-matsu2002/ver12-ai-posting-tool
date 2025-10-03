# app/tasks.py

import logging
from datetime import datetime
import pytz
import time 
from flask import current_app
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import text  # ★ 追加

from . import db
from .models import Article
from .wp_client import post_to_wp  # 統一された WordPress 投稿関数
from sqlalchemy.orm import selectinload

# ✅ GSCクリック・表示回数の毎日更新ジョブ用
from app.google_client import update_all_gsc_sites, fetch_new_queries_since

# 既存 import の下あたりに追加
from concurrent.futures import ThreadPoolExecutor
from .models import (Site, Keyword, ExternalSEOJob,
                     BlogType, ExternalBlogAccount, ExternalArticleSchedule)
from app.models import GSCAutogenDaily  # ★ 追加：日次サマリ

# app/tasks.py （インポートセクションの BlogType などの下あたり）
from app.services.blog_signup.livedoor_signup import signup as livedoor_signup
# 既存 import 群の下に追加
from app.external_seo_generator import generate_and_schedule_external_articles

from app.external_seo_generator import (
    TITLE_PROMPT as EXT_TITLE_PROMPT,
    BODY_PROMPT  as EXT_BODY_PROMPT,
)
from app.models import PromptTemplate
from app.article_generator import _generate


# === 内部SEO 自動化 で使う import ===
from app.models import InternalSeoRun
from app.utils.db_retry import with_db_retry
from app.services.internal_seo.indexer import sync_site_content_index
from app.services.internal_seo.link_graph import build_link_graph_for_site
from app.services.internal_seo.planner import plan_links_for_site
from app.services.internal_seo.applier import apply_actions_for_site, apply_actions_for_user
from app.services.internal_seo import user_scheduler  # 🆕 追加
import os
from math import inf
from typing import List, Dict, Set, Tuple, Optional
import json
from app.models import InternalLinkAction  # 🆕 refill 集計で使用
from app.models import InternalSeoUserSchedule  # 🆕 ユーザースケジュール確認用
from sqlalchemy import func  # 🆕 集計で使用
from app.services.internal_seo.enqueue import enqueue_refill_for_site  # 🆕 refill投入API

# ────────────────────────────────────────────────
# APScheduler ＋ スレッドプール
# ────────────────────────────────────────────────
# グローバルな APScheduler インスタンス（__init__.py で start されています）
scheduler = BackgroundScheduler(timezone="UTC")
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="extseo")


# ────────────────────────────────────────────────
# GSCオートジェン：サイトごとの軽量ロック（PostgreSQL advisory lock）
# ────────────────────────────────────────────────
def _lock_key(site_id: int) -> int:
    # 適当な名前空間キー（衝突回避用に固定係数）
    return 91337_00000 + int(site_id)

def _try_lock_site(site_id: int) -> bool:
    k = _lock_key(site_id)
    try:
        got = db.session.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": k}).scalar()
        return bool(got)
    except Exception:
        # DBがPostgreSQL以外でも落ちないようフォールバック（ロック無しで進行）
        current_app.logger.warning("[GSC-AUTOGEN] advisory lock unsupported; continue without lock (site=%s)", site_id)
        return True

def _unlock_site(site_id: int) -> None:
    k = _lock_key(site_id)
    try:
        db.session.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": k})
    except Exception:
        pass

# ────────────────────────────────────────────────
# 内部SEO：ユーザーごとの軽量ロック（PostgreSQL advisory lock）
# ────────────────────────────────────────────────
def _user_lock_key(user_id: int) -> int:
    # 別名前空間（site用と衝突しない係数）
    return 91338_00000 + int(user_id)

def _try_lock_user(user_id: int) -> bool:
    k = _user_lock_key(user_id)
    try:
        got = db.session.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": k}).scalar()
        return bool(got)
    except Exception:
        # 非PostgreSQLでも落ちないよう、ロック無しで前進
        current_app.logger.warning("[ISEO-USER] advisory lock unsupported; continue without lock (user=%s)", user_id)
        return True

def _unlock_user(user_id: int) -> None:
    k = _user_lock_key(user_id)
    try:
        db.session.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": k})
    except Exception:
        pass    

# --------------------------------------------------------------------------- #
# 1) WordPress 自動投稿ジョブ
# --------------------------------------------------------------------------- #
def _auto_post_job(app):
    with app.app_context():
        start = time.time()
        current_app.logger.info("Running auto_post_job")
        now = datetime.now(pytz.utc)

        try:
            pending = (
                db.session.query(Article)
                .filter(Article.status == "done", Article.scheduled_at <= now,Article.source != "external",)
                .options(selectinload(Article.site))
                .order_by(Article.scheduled_at.asc())
                .limit(20)
                .all()
            )

            for art in pending:
                if not art.site:
                    current_app.logger.warning(f"記事 {art.id} の投稿先サイト未設定")
                    continue

                try:
                    site = db.session.query(Site).get(art.site_id)
                    current_app.logger.info(f"投稿処理開始: Article {art.id}, User ID: {art.user_id}, Site ID: {art.site_id}")
                    url = post_to_wp(site, art)
                    art.posted_at = now
                    art.status = "posted"
                    art.posted_url = url 
                    db.session.commit()
                    current_app.logger.info(f"Auto-posted Article {art.id} -> {url}")

                except Exception as e:
                    db.session.rollback()
                    current_app.logger.error(f"初回投稿失敗: Article {art.id}, User ID: {art.user_id}, Site ID: {art.site_id}, エラー: {e}")
                    art.status = "error"  # 投稿失敗時にステータスを "error" に変更
                    db.session.commit()  # エラー状態として保存

        finally:
            db.session.close()
            end = time.time()
            current_app.logger.info(f"✅ [AutoPost] 自動投稿ジョブ終了（所要時間: {end - start:.1f}秒）")

# --------------------------------------------------------------------------- #
# 2) GSC メトリクス毎日更新
# --------------------------------------------------------------------------- #
def _gsc_metrics_job(app):
    """
    ✅ GSCクリック・表示回数の毎日更新ジョブ
    """
    with app.app_context():
        current_app.logger.info("🔄 GSCメトリクス更新ジョブを開始します")
        try:
            update_all_gsc_sites()
            current_app.logger.info("✅ GSCメトリクス更新完了")
        except Exception as e:
            current_app.logger.error(f"❌ GSCメトリクス更新失敗: {str(e)}")

# --------------------------------------------------------------------------- #
# 3) GSC 連携サイト向け 1000 記事ループ生成
# --------------------------------------------------------------------------- #
def gsc_loop_generate(site):
    """
    🔁 GSCからのクエリで1000記事未満なら通常記事フローで生成する（修正済）
    - 新規クエリを登録
    - 既存の未生成（pending/error）キーワードもすべて enqueue
    """
    from app import db
    from app.google_client import fetch_search_queries_for_site
    from app.models import Keyword, PromptTemplate
    from app.article_generator import enqueue_generation
    from flask import current_app

    if not site.gsc_connected:
        current_app.logger.info(f"[GSC LOOP] スキップ：未接続サイト {site.name}")
        return

    total_keywords = Keyword.query.filter_by(site_id=site.id).count()
    if total_keywords >= 1000:
        current_app.logger.info(f"[GSC LOOP] {site.name} は既に1000記事に到達済み")
        return

    # ✅ GSCクエリを取得
    try:
        queries = fetch_search_queries_for_site(site, days=28)
    except Exception as e:
        current_app.logger.warning(f"[GSC LOOP] クエリ取得失敗 - {site.url}: {e}")
        return

    # ✅ 新規キーワードを抽出して追加
    existing_keywords = set(
        k.keyword for k in Keyword.query.filter_by(site_id=site.id).all()
    )
    new_keywords = [q for q in queries if q not in existing_keywords]

    for kw in new_keywords:
        db.session.add(Keyword(
            keyword=kw,
            site_id=site.id,
            user_id=site.user_id,
            source='gsc',
            status='pending',
            used=False
        ))

    if new_keywords:
        current_app.logger.info(f"[GSC LOOP] {site.name} に新規キーワード {len(new_keywords)} 件登録")

    db.session.commit()

    # ✅ プロンプトを取得（なければ空）
    prompt = PromptTemplate.query.filter_by(user_id=site.user_id).order_by(PromptTemplate.id.desc()).first()
    title_prompt = prompt.title_pt if prompt else ""
    body_prompt  = prompt.body_pt  if prompt else ""

    # ✅ 修正：未生成（pending または error）をすべてキューに流す
    from sqlalchemy import or_

    ungenerated_keywords = (
        Keyword.query
        .filter(
            Keyword.site_id == site.id,
            Keyword.source == "gsc",
            Keyword.status.in_(["pending", "error"])
        )
        .order_by(Keyword.id.asc())
        .all()
    )

    if not ungenerated_keywords:
        current_app.logger.info(f"[GSC LOOP] {site.name} に生成待ちキーワードなし")
        return

    BATCH = 40
    for i in range(0, len(ungenerated_keywords), BATCH):
        batch_keywords = ungenerated_keywords[i:i+BATCH]
        keyword_strings = [k.keyword for k in batch_keywords]

        enqueue_generation(
            user_id      = site.user_id,
            site_id      = site.id,
            keywords     = keyword_strings,
            title_prompt = title_prompt,
            body_prompt  = body_prompt,
            format       = "html",
            self_review  = False,
        )

        # ✅ 修正：キュー投入済みとして status を更新
        for k in batch_keywords:
            k.status = "queued"

        db.session.commit()

        current_app.logger.info(
            f"[GSC LOOP] {site.name} → batch {i//BATCH+1}: {len(batch_keywords)} 件キュー投入"
        )


def _gsc_generation_job(app):
    """
    ✅ GSC記事自動生成ジョブ
    """
    with app.app_context():
        from app.models import Site

        current_app.logger.info("📝 GSC記事生成ジョブを開始します")
        sites = Site.query.filter_by(gsc_connected=True).all()

        for site in sites:
            try:
                gsc_loop_generate(site)
            except Exception as e:
                current_app.logger.warning(f"[GSC自動生成] 失敗 - {site.url}: {e}")

        current_app.logger.info("✅ GSC記事生成ジョブが完了しました")


# ──────────────────────────────────────────
# 🆕 GSCオートジェン（日次・新着限定・上限・DRYRUN・見える化）
# ──────────────────────────────────────────
def gsc_autogen_daily_job(app):
    """
    ENVのUTC時刻に日次で起動：
      - 対象：gsc_connected=True & gsc_generation_started=True
      - 新着抽出：fetch_new_queries_since(site)
      - 事前フィルタ：Keyword/Article 既存排除
      - 上限：ENV GSC_AUTOGEN_LIMIT
      - DRYRUN：ENV GSC_AUTOGEN_DRYRUN=1 なら投入せずカウントのみ
      - サマリ保存：GSCAutogenDaily（run_date=JST）
    """
    from app.models import PromptTemplate  # 局所 import（循環回避）
    from app.article_generator import enqueue_generation
    JST = pytz.timezone("Asia/Tokyo")
    jst_today = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(JST).date()

    limit_per_site = int(os.getenv("GSC_AUTOGEN_LIMIT", "50"))
    dryrun = os.getenv("GSC_AUTOGEN_DRYRUN", "1") == "1"

    with app.app_context():
        # 日次オートジェン有効フラグ（migration: gsc_autogen_daily）を使用
        sites = Site.query.filter_by(gsc_connected=True, gsc_autogen_daily=True).all()
        current_app.logger.info("[GSC-AUTOGEN] start: targets=%s limit=%s dryrun=%s", len(sites), limit_per_site, int(dryrun))

        for site in sites:
            if not _try_lock_site(site.id):
                current_app.logger.info("[GSC-AUTOGEN] skip (locked) site=%s", site.id)
                continue
            started_at = datetime.utcnow()
            error_msg: Optional[str] = None
            try:
                # 1) 新着抽出（cutoff 以降・28d impr しきい値は関数内でENV反映）
                rows = fetch_new_queries_since(site)
                candidate_keywords = [r["query"] for r in rows]
                picked_cnt = len(candidate_keywords)

                # 2) 事前フィルタ（重複や既存記事の除外）
                filt = filter_autogen_candidates(site.id, candidate_keywords)
                deduped = filt["deduped"]
                dup_cnt = len(filt["dup_keywords"]) + len(filt["art_dup_keywords"])

                # 3) 上限
                allowed = deduped[: max(0, limit_per_site)]
                limit_skipped = max(0, len(deduped) - len(allowed))

                queued_cnt = 0
                sample = allowed[:10]

                # 4) DRYRUN or 実投入
                if dryrun or not allowed:
                    pass  # 何もしない（カウントのみ）
                else:
                    # 4-1) Keyword を作成（存在しないはずだが念のため重複排除）
                    existing = {
                        r[0]
                        for r in db.session.query(Keyword.keyword)
                        .filter(Keyword.site_id == site.id, Keyword.keyword.in_(allowed))
                        .all()
                    }
                    to_insert = [kw for kw in allowed if kw not in existing]
                    for kw in to_insert:
                        db.session.add(Keyword(
                            keyword=kw,
                            site_id=site.id,
                            user_id=site.user_id,
                            source="gsc",
                            status="pending",
                            used=False
                        ))
                    if to_insert:
                        db.session.commit()

                    # 4-2) プロンプト取得
                    prompt = (PromptTemplate.query
                              .filter_by(user_id=site.user_id)
                              .order_by(PromptTemplate.id.desc())
                              .first())
                    title_pt = prompt.title_pt if prompt else ""
                    body_pt  = prompt.body_pt  if prompt else ""

                    # 4-3) キュー投入（既存の enqueue_generation を利用）
                    enqueue_generation(
                        user_id=site.user_id,
                        site_id=site.id,
                        keywords=allowed,
                        title_prompt=title_pt,
                        body_prompt=body_pt,
                        format="html",
                        self_review=False,
                        source="gsc",
                    )
                    queued_cnt = len(allowed)

                # 5) サマリ保存（upsert）
                rec = GSCAutogenDaily.query.filter_by(site_id=site.id, run_date=jst_today).first()
                if not rec:
                    rec = GSCAutogenDaily(site_id=site.id, run_date=jst_today)
                rec.picked = int(picked_cnt)
                rec.queued = int(queued_cnt)
                rec.dup = int(dup_cnt)
                rec.limit_skipped = int(limit_skipped)
                rec.dryrun = int(dryrun)
                rec.sample_keywords_json = json.dumps(sample, ensure_ascii=False)
                rec.started_at = rec.started_at or started_at
                rec.finished_at = datetime.utcnow()
                rec.error = None
                db.session.add(rec)
                db.session.commit()

                current_app.logger.info(
                    "[GSC-AUTOGEN] site=%s pick=%s queued=%s dup=%s limit=%s dryrun=%s",
                    site.id, picked_cnt, queued_cnt, dup_cnt, limit_skipped, int(dryrun)
                )
            except Exception as e:
                db.session.rollback()
                error_msg = str(e)
                # サマリにもエラーを残す
                try:
                    rec = GSCAutogenDaily.query.filter_by(site_id=site.id, run_date=jst_today).first()
                    if not rec:
                        rec = GSCAutogenDaily(site_id=site.id, run_date=jst_today)
                    rec.started_at = rec.started_at or started_at
                    rec.finished_at = datetime.utcnow()
                    rec.error = error_msg
                    db.session.add(rec)
                    db.session.commit()
                except Exception:
                    pass
                current_app.logger.exception("[GSC-AUTOGEN] failed site=%s: %s", site.id, error_msg)
            finally:
                _unlock_site(site.id)


# --------------------------------------------------------------------------- #
# 🆕 Pending Regenerator Job（通常 & 外部SEO）— 手動再生成と同じフローを自動で実行
# --------------------------------------------------------------------------- #
def _pending_regenerator_job(app):
    """
    40分おきに実行:
      - 通常記事（source <> 'external'）で status IN ('pending','gen') を再生成
        * ユーザーごとに直近の PromptTemplate を使って、_generate() を呼ぶ
      - 外部SEO（source='external'）で status IN ('pending','gen') も再生成
        * external_seo_generator の固定プロンプトを使って、_generate() を呼ぶ
    既存のスケジュール/投稿フローには一切手を加えない（副作用なし）。
    """
    with app.app_context():
        try:
            # ======== 環境変数による上限（安全） =========
            normal_per_run      = int(os.getenv("PENDING_REGEN_NORMAL_PER_RUN", "200"))
            normal_per_user_max = int(os.getenv("PENDING_REGEN_NORMAL_PER_USER", "60"))
            ext_per_run         = int(os.getenv("PENDING_REGEN_EXT_PER_RUN", "20"))
            normal_workers      = int(os.getenv("PENDING_REGEN_WORKERS", "10"))
            ext_workers         = int(os.getenv("PENDING_REGEN_EXT_WORKERS", "4"))

            # ------------------------------
            # 1) 通常記事（source <> 'external'）
            # ------------------------------
            if normal_per_run > 0:
                # pending/gen を持つユーザーを多い順に抽出
                user_rows = db.session.execute(text("""
                    WITH u_has_prompt AS (
                      SELECT user_id, 1 AS has_prompt
                      FROM prompt_template
                      GROUP BY user_id
                    )
                    SELECT
                      a.user_id,
                      COUNT(*) AS pending_cnt,
                      COALESCE(MAX(u.has_prompt), 0) AS has_prompt
                    FROM articles a
                    LEFT JOIN u_has_prompt u ON u.user_id = a.user_id
                    WHERE a.status IN ('pending','gen') AND (a.source IS NULL OR a.source <> 'external')
                    GROUP BY a.user_id
                    ORDER BY pending_cnt DESC
                """)).mappings().all()

                picked_normal = []
                for row in user_rows:
                    if len(picked_normal) >= normal_per_run:
                        break
                    uid = int(row["user_id"])
                    # プロンプトが無いユーザーはスキップ（手動再生成の仕様に合わせる）
                    prompt = (PromptTemplate.query
                              .filter_by(user_id=uid)
                              .order_by(PromptTemplate.id.desc())
                              .first())
                    if not prompt:
                        current_app.logger.info(f"[pending-regenerator] skip user {uid}: no PromptTemplate")
                        continue

                    remain = normal_per_run - len(picked_normal)
                    take   = min(normal_per_user_max, remain)
                    if take <= 0:
                        break

                    arts = (Article.query
                            .filter(Article.user_id == uid,
                                    Article.status.in_(["pending","gen"]),
                                    (Article.source == None) | (Article.source != "external"))  # noqa: E711
                            .order_by(Article.created_at.asc())
                            .limit(take)
                            .all())
                    for a in arts:
                        # 既に posted/done などに誤って混入していないか保険
                        if a.status not in ("pending","gen"):
                            continue
                        picked_normal.append((a.id, uid, prompt.title_pt or "", prompt.body_pt or ""))
                        if len(picked_normal) >= normal_per_run:
                            break

                if picked_normal:
                    current_app.logger.info(f"[pending-regenerator] normal picked={len(picked_normal)} users={len(user_rows)}")
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    with ThreadPoolExecutor(max_workers=normal_workers) as ex:
                        futs = [
                            ex.submit(_generate, app, aid, tpt, bpt, "html", False, user_id=uid)
                            for (aid, uid, tpt, bpt) in picked_normal
                        ]
                        for f in as_completed(futs):
                            try:
                                f.result()
                            except Exception as e:
                                current_app.logger.exception(f"[pending-regenerator] normal generate error: {e}")

            # ------------------------------
            # 2) 外部SEO（source = 'external'）
            # ------------------------------
            if ext_per_run > 0:
                ext_articles = (Article.query
                                .filter(Article.status.in_(["pending","gen"]),
                                        Article.source == "external")
                                .order_by(Article.created_at.asc())
                                .limit(ext_per_run)
                                .all())
                if ext_articles:
                    current_app.logger.info(f"[pending-regenerator] external picked={len(ext_articles)}")
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    with ThreadPoolExecutor(max_workers=ext_workers) as ex:
                        futs = [
                            # external は固定プロンプトで再生成（既存の自動生成と同一ロジックの核を使用）
                            ex.submit(_generate, app, art.id, EXT_TITLE_PROMPT, EXT_BODY_PROMPT, "html", False, user_id=art.user_id)
                            for art in ext_articles
                        ]
                        for f in as_completed(futs):
                            try:
                                f.result()
                            except Exception as e:
                                current_app.logger.exception(f"[pending-regenerator] external generate error: {e}")

        except Exception as e:
            current_app.logger.exception(f"[pending-regenerator] job failed: {e}")

# --------------------------------------------------------------------------- #
# 🆕 Internal SEO Refill Job（A案）— ユーザー別に“弾（pending）”を補充するだけ
# --------------------------------------------------------------------------- #
def _internal_seo_user_refill_job(app):
    """
    目的:
      - ユーザー単位で pending（InternalLinkAction.status='pending'）の“distinct post_id 数”を集計
      - しきい値(INTERNAL_SEO_REFILL_TARGET)を下回るユーザーに対して、
        そのユーザーのサイトから重複無く internal_seo_job_queue へ job_kind='refill' を投入
      - refill ジョブは limit_posts=0 を強制し、Applier を実質スキップ（＝補給専用）
      - 適用（apply）は user_scheduler 側が回す想定
    """
    with app.app_context():
        try:
            # ENV（安全な既定値付き）
            enabled = os.getenv("INTERNAL_SEO_REFILL_ENABLED", "1") != "0"
            if not enabled:
                current_app.logger.info("[refill] disabled by env")
                return
            target = int(os.getenv("INTERNAL_SEO_REFILL_TARGET", "50"))  # ユーザーごとの目標 pending 記事数
            per_user_cap = int(os.getenv("INTERNAL_SEO_REFILL_MAX_ENQUEUE_PER_USER", "2"))  # 1tickあたり投入上限/ユーザー

            # Planner用のボリューム（未指定時は既存ENVに素直に従う）
            pages         = int(os.getenv("INTERNAL_SEO_REFILL_PAGES",         os.getenv("INTERNAL_SEO_PAGES", "10")))
            per_page      = int(os.getenv("INTERNAL_SEO_REFILL_PER_PAGE",      os.getenv("INTERNAL_SEO_PER_PAGE", "100")))
            min_score     = float(os.getenv("INTERNAL_SEO_REFILL_MIN_SCORE",   os.getenv("INTERNAL_SEO_MIN_SCORE", "0.05")))
            max_k         = int(os.getenv("INTERNAL_SEO_REFILL_MAX_K",         os.getenv("INTERNAL_SEO_MAX_K", "80")))
            limit_sources = int(os.getenv("INTERNAL_SEO_REFILL_LIMIT_SOURCES", os.getenv("INTERNAL_SEO_LIMIT_SOURCES", "200")))
            # refill は Applier を回さないため 0 強制
            limit_posts   = 0

            # 1) ユーザーごとの pending 記事数（distinct post_id）を集計
            #    FROM internal_link_actions → JOIN site （安全に明示）
            pend_rows = (
                db.session.query(
                    Site.user_id.label("user_id"),
                    func.count(func.distinct(InternalLinkAction.post_id)).label("pending_posts"),
                )
                .select_from(InternalLinkAction)
                .join(Site, Site.id == InternalLinkAction.site_id)
                .filter(InternalLinkAction.status == "pending")
                .group_by(Site.user_id)
                .all()
            )
            pending_map = {int(r.user_id): int(r.pending_posts) for r in pend_rows}

            # 2) 全ユーザーを列挙（pending が 0 のユーザーも対象にするため Site テーブルから）
            user_rows = db.session.query(Site.user_id).group_by(Site.user_id).all()
            all_user_ids = [int(u[0]) for u in user_rows]
            if not all_user_ids:
                current_app.logger.info("[refill] no users found (no sites)")
                return

            # 3) しきい値を下回るユーザーを対象に、未キューのサイトへ 'refill' を投入
            enq_total = 0
            skipped_locked = 0
            for uid in all_user_ids:
                # 🛡 ユーザースケジュールの有効性チェック（開始ボタン未押下/一時停止/スケジュール無しは補給しない）
                sched = InternalSeoUserSchedule.query.filter_by(user_id=uid).one_or_none()
                if not sched:
                    current_app.logger.info("[refill] skip uid=%s (no user schedule)", uid)
                    continue
                if not sched.is_enabled:
                    current_app.logger.info("[refill] skip uid=%s (user schedule disabled)", uid)
                    continue
                if getattr(sched, "status", None) == "paused":
                    current_app.logger.info("[refill] skip uid=%s (user schedule paused)", uid)
                    continue
                cur = pending_map.get(uid, 0)
                if cur >= target:
                    continue  # 目標に達している

                # このユーザーの対象サイト（既に queued/running が無いサイトを抽出）
                rows = db.session.execute(text("""
                    SELECT s.id
                      FROM site s
                 LEFT JOIN internal_seo_job_queue q
                        ON q.site_id = s.id AND q.status IN ('queued','running')
                     WHERE s.user_id = :uid
                       AND q.site_id IS NULL
                """), {"uid": uid}).fetchall()
                site_ids = [int(r[0]) for r in rows]
                if not site_ids:
                    continue

                need = min(per_user_cap, max(1, (target - cur + 1) // 2))  # 欠乏度に応じて控えめに投入
                picked = site_ids[:need]

                # 4) enqueue API を使用（内部で重複チェック＆commit 済み）
                for sid in picked:
                    res = enqueue_refill_for_site(
                        sid,
                        pages=pages,
                        per_page=per_page,
                        min_score=min_score,
                        max_k=max_k,
                        limit_sources=limit_sources,
                        incremental=True,
                        job_kind="refill",
                    )
                    if res.get("enqueued"):
                        enq_total += 1
                    else:
                        # 既に queued/running など
                        if res.get("reason") == "already-queued-or-running":
                            skipped_locked += 1

            current_app.logger.info(
                f"[refill] enqueued={enq_total} skipped_locked={skipped_locked} "
                f"target={target} cap/u={per_user_cap} params={{pages:{pages}, per_page:{per_page}, "
                f"min_score:{min_score}, max_k:{max_k}, limit_sources:{limit_sources}}}"
            )
        except Exception as e:
            db.session.rollback()
            current_app.logger.exception(f"[refill] job failed: {e}")



# app/tasks.py どこでも OK ですが _run_external_seo_job の直前あたりが読みやすい
def _run_livedoor_signup(app, site_id: int) -> None:
    """
    1) livedoor_signup.signup() を呼び出し ExternalBlogAccount を新規作成
       - 成功時は ExternalSEOJob を step='generate' で作成 / 更新
       - 失敗時は status='error' のジョブを残す
    2) API キーなど詳細ログは ExternalBlogAccount のカラムに保存済み
    """
    with app.app_context():
        try:
            # ① Site オブジェクトを取得して存在確認
            site = Site.query.get(site_id)
            if not site:
                raise ValueError(f"Site id={site_id} not found")

            # ② livedoor アカウント自動登録
            acc = livedoor_signup(site)   # ← Site オブジェクトを渡す
            current_app.logger.info(
                "[LD-Signup] success: site=%s account_id=%s", site_id, acc.id
            )

            # ③ ジョブ行を running → generate へ
            job = ExternalSEOJob(
                site_id     = site_id,
                blog_type   = BlogType.LIVEDOOR,
                status      = "running",
                step        = "generate",
                article_cnt = 0,
                message     = "signup OK",
            )
            db.session.add(job)
            db.session.commit()

            # TODO: enqueue_generation(job.id) などで記事生成タスクを投入する場合はここ

        except Exception as e:
            # ④ 失敗時：エラージョブを残す & ログ
            current_app.logger.exception("[LD-Signup] failed: %s", e)

            job = ExternalSEOJob(
                site_id   = site_id,
                blog_type = BlogType.LIVEDOOR,
                status    = "error",
                step      = "error",
                message   = str(e),
            )
            db.session.add(job)
            db.session.commit()

            
def enqueue_livedoor_signup(site_id: int):
    """
    外部SEO開始ボタン → この関数を呼ぶ
    """
    app = current_app._get_current_object()
    executor.submit(_run_livedoor_signup, app, site_id)


def _run_generate_and_schedule(app, user_id: int, site_id: int, blog_account_id: int,
                               count: int = 100, per_day: int = 10, start_day_jst=None):
    """
    外部SEO 100本生成＋スケジューリングを“アプリコンテキスト内で”実行するワーカー本体
    """
    with app.app_context():
        try:
            created = generate_and_schedule_external_articles(
                user_id=user_id,
                site_id=site_id,
                blog_account_id=blog_account_id,
                count=count,
                per_day=per_day,
                start_day_jst=start_day_jst,
            )
            current_app.logger.info(
                "[external-seo] generate+schedule done: site=%s acct=%s created=%s",
                site_id, blog_account_id, created
            )
        except Exception as e:
            current_app.logger.exception("[external-seo] generate+schedule failed: %s", e)


def enqueue_generate_and_schedule(user_id: int, site_id: int, blog_account_id: int,
                                  count: int = 100, per_day: int = 10, start_day_jst=None):
    """
    ルートから呼ぶ軽量関数（スレッドプールで非同期実行）
    """
    app = current_app._get_current_object()
    executor.submit(
        _run_generate_and_schedule, app,
        user_id, site_id, blog_account_id, count, per_day, start_day_jst
    )


# --------------------------------------------------------------------------- #
# 4) 外部SEO ① キュー作成ジョブ
# --------------------------------------------------------------------------- #



# --------------------------------------------------------------------------- #
# 5) 外部SEO ② 投稿ジョブ
# --------------------------------------------------------------------------- #

def _finalize_external_job(job_id: int):
    """同一サイトの“このジョブ以降に作られた”スケジュールの完了を集計して、全部 posted なら success にする"""
    job = ExternalSEOJob.query.get(job_id)
    if not job:
        return

    # このジョブ作成以降に生成されたスケジュールだけを対象にする（同一サイトの別バッチと混ざらないため）
    q_base = (ExternalArticleSchedule.query
              .join(ExternalBlogAccount,
                    ExternalArticleSchedule.blog_account_id == ExternalBlogAccount.id)
              .filter(ExternalBlogAccount.site_id == job.site_id,
                      ExternalArticleSchedule.created_at >= job.created_at))

    total = q_base.count()
    posted = q_base.filter(ExternalArticleSchedule.status == "posted").count()

    # total が 0 の間は判定しない
    if total and posted >= total:
        job.step = "done"
        job.status = "success"
        db.session.commit()
        

# ──────────────────────────────────────────
# 外部ブログ投稿ジョブ（外部SEO・キーワード厳密紐付け版）
# ──────────────────────────────────────────
def _run_external_post_job(app, schedule_id: int | None = None):
    """
    外部SEO記事を外部ブログに投稿するジョブ
    - ExternalArticleSchedule の pending を対象
    - ExternalBlogAccount が LIVEDOOR のみ対応
    - ✅ Article は sched.article_id でダイレクト取得（keyword 照合は廃止）
    - 二重投稿防止：Article.posted_url が空のもののみ
    """
    from datetime import datetime
    from flask import current_app
    from sqlalchemy import or_
    from app import db
    from app.models import (
        ExternalArticleSchedule,
        ExternalBlogAccount,
        ExternalSEOJob,
        BlogType,
        Article,
    )
    from app.services.blog_post.livedoor_post import post_livedoor_article

    with app.app_context():
        now = datetime.utcnow()
        current_app.logger.info(f"[ExtPost] Job start at {now}")

        # ← ここを rows_q にして、その後も rows_q を使う
        rows_q = (
            db.session.query(ExternalArticleSchedule, ExternalBlogAccount, Article)
            .join(ExternalBlogAccount, ExternalArticleSchedule.blog_account_id == ExternalBlogAccount.id)
            .join(Article, ExternalArticleSchedule.article_id == Article.id)
            .filter(
                ExternalArticleSchedule.status == "pending",
                ExternalArticleSchedule.scheduled_date <= now,
                Article.source == "external",
                or_(Article.posted_url == None, Article.posted_url == ""),  # noqa: E711
                Article.status == "done",
            )
        )

        if schedule_id is not None:
            rows_q = rows_q.filter(ExternalArticleSchedule.id == schedule_id)

        rows = (
            rows_q
            .order_by(ExternalArticleSchedule.scheduled_date.asc())
            .limit(1 if schedule_id is not None else 12)
            .all()
        )

        current_app.logger.info(f"[ExtPost] Found {len(rows)} pending schedules")
        if not rows:
            return

        for sched, acct, art in rows:
            try:
                # ブログ種別チェック
                if acct.blog_type != BlogType.LIVEDOOR:
                    sched.status = "error"
                    sched.message = f"未対応ブログタイプ: {acct.blog_type}"
                    db.session.commit()
                    current_app.logger.warning(
                        f"[ExtPost] Unsupported blog type {acct.blog_type} (sched_id={sched.id})"
                    )
                    continue

                # 追加の安全チェック：記事のサイトとアカウントのサイトが一致しているか
                if art.site_id != acct.site_id:
                    sched.status = "error"
                    sched.message = f"site mismatch: article.site_id={art.site_id} / account.site_id={acct.site_id}"
                    db.session.commit()
                    current_app.logger.error(
                        f"[ExtPost] Site mismatch (sched_id={sched.id}, art_id={art.id})"
                    )
                    continue

                current_app.logger.info(
                    f"[ExtPost] Posting article_id={art.id} (kw='{art.keyword}') to Livedoor..."
                )
                res = post_livedoor_article(acct, art.title, art.body)

                if res.get("ok"):
                    # スケジュール更新
                    sched.status = "posted"
                    sched.posted_url = res.get("url")
                    sched.posted_at = res.get("posted_at")

                    # 記事側も posted
                    art.status = "posted"
                    if res.get("url"):
                        art.posted_url = res["url"]
                    art.posted_at = res.get("posted_at") or now

                    # アカウントの投稿カウント
                    acct.posted_cnt = (acct.posted_cnt or 0) + 1

                    db.session.commit()
                    current_app.logger.info(f"[ExtPost] Success: {res.get('url')}")

                    # ジョブ完了判定（最新ジョブのみ）
                    latest_job = (
                        ExternalSEOJob.query.filter_by(site_id=acct.site_id)
                        .order_by(ExternalSEOJob.id.desc())
                        .first()
                    )
                    if latest_job:
                        _finalize_external_job(latest_job.id)

                else:
                    # 投稿失敗
                    err = res.get("error") or "unknown error"
                    sched.status = "error"
                    sched.message = err
                    db.session.commit()
                    current_app.logger.error(f"[ExtPost] Failed: {err}")

            except Exception as e:
                current_app.logger.exception(f"[ExtPost] Exception during posting: {e}")
                sched.status = "error"
                sched.message = str(e)
                db.session.commit()




# --------------------------------------------------------------------------- #
# 6) 外部SEO バッチ監視ジョブ（100 本完了ごとに次バッチ）
# --------------------------------------------------------------------------- #


def init_scheduler(app):
    """
    Flask アプリ起動時に呼び出して:
      1) APScheduler に自動投稿ジョブを登録
      2) 3分間隔で _auto_post_job を実行するようスケジュール
      - GSC記事生成ジョブ：10分間隔（←ここ修正）
    """
    scheduler.add_job(
        func=_auto_post_job,
        trigger="interval",
        minutes=3,
        args=[app],
        id="auto_post_job",
        replace_existing=True,
        max_instances=5
    )


    # 🆕 Pending 再生成ジョブ（40分おき）
    scheduler.add_job(
        func=_pending_regenerator_job,
        trigger="interval",
        minutes=40,
        args=[app],
        id="pending_regenerator_job",
        replace_existing=True,
        max_instances=1,
    )

    # ✅ GSCクリック・表示回数を毎日0時に自動更新するジョブ
    scheduler.add_job(
        func=_gsc_metrics_job,
        trigger="cron",
        hour=0,
        minute=0,
        args=[app],
        id="gsc_metrics_job",
        replace_existing=True,
        max_instances=1
    )

    # 🆕 ✅ GSCオートジェン（日次・新着限定）
    gsc_utc_hour = int(os.getenv("GSC_AUTOGEN_UTC_HOUR", "18"))
    gsc_utc_min  = int(os.getenv("GSC_AUTOGEN_UTC_MIN", "0"))
    scheduler.add_job(
        func=gsc_autogen_daily_job,
        trigger="cron",
        hour=gsc_utc_hour,
        minute=gsc_utc_min,
        args=[app],
        id="gsc_autogen_daily_job",
        replace_existing=True,
        max_instances=1
    )

    # ✅ 外部ブログ投稿ジョブ（10分おき）
    scheduler.add_job(
        func=_run_external_post_job,
        trigger="interval",
        minutes=10,
        args=[app],
        id="external_post_job",
        replace_existing=True,
        max_instances=1
    )

    # ✅ 内部SEO ナイトリー実行（環境変数でON/OFF可能／レガシー運用）
    #   - デフォルト: 毎日 18:15 UTC = JST 03:15
    if os.getenv("INTERNAL_SEO_ENABLED", "1") == "1":
        utc_hour = int(os.getenv("INTERNAL_SEO_UTC_HOUR", "18"))
        utc_min  = int(os.getenv("INTERNAL_SEO_UTC_MIN",  "15"))
        scheduler.add_job(
            func=_internal_seo_nightly_job,
            trigger="cron",
            hour=utc_hour,
            minute=utc_min,
            args=[app],
            id="internal_seo_job",
            replace_existing=True,
            max_instances=1,
        )
        app.logger.info(f"Scheduler started: internal_seo_job daily at {utc_hour:02d}:{utc_min:02d} UTC")
    else:
        # 明示的にレガシーナイトリーを停止していることを起動ログに残す
        app.logger.info("legacy internal-seo nightly OFF (INTERNAL_SEO_ENABLED!=1): skipping internal_seo_job")

    # ✅ 内部SEO ワーカー（キュー消化）※毎分
    if os.getenv("INTERNAL_SEO_WORKER_ENABLED", "1") == "1":
        scheduler.add_job(
            func=_internal_seo_worker_tick,
            trigger="interval",
            minutes=int(os.getenv("INTERNAL_SEO_WORKER_INTERVAL_MIN", "1")),
            args=[app],
            id="internal_seo_worker_tick",
            replace_existing=True,
            max_instances=1,
        )
        app.logger.info("Scheduler started: internal_seo_worker_tick every minute")
    else:
        app.logger.info("Scheduler skipped: internal_seo_worker_tick (INTERNAL_SEO_WORKER_ENABLED!=1)")

    # ✅ 内部SEO ユーザーごとの自動ジョブ（ENVでON/OFF可能）
    if os.getenv("INTERNAL_SEO_USER_ENABLED", "1") == "1":
        scheduler.add_job(
            func=user_scheduler.user_scheduler_tick,
            trigger="interval",
            minutes=int(os.getenv("INTERNAL_SEO_USER_INTERVAL_MIN", "1")),
            args=[app],
            id="internal_seo_user_scheduler_job",
            replace_existing=True,
            max_instances=1,
        )
        app.logger.info("Scheduler started: internal_seo_user_scheduler_job (user-scope tick)")
    else:
        app.logger.info("Scheduler skipped: internal_seo_user_scheduler_job (INTERNAL_SEO_USER_ENABLED!=1)")

    # 🆕 内部SEO ユーザー適用ループ（“開始ボタンを押したユーザーだけ”適用を回す）
    if os.getenv("INTERNAL_SEO_USER_APPLY_ENABLED", "1") == "1":
        scheduler.add_job(
            func=_internal_seo_user_apply_tick,
            trigger="interval",
            minutes=int(os.getenv("INTERNAL_SEO_USER_APPLY_INTERVAL_MIN", "3")),
            args=[app],
            id="internal_seo_user_apply_tick",
            replace_existing=True,
            max_instances=1,
        )
        app.logger.info("Scheduler started: internal_seo_user_apply_tick (user apply loop)")
    else:
        app.logger.info("Scheduler skipped: internal_seo_user_apply_tick (INTERNAL_SEO_USER_APPLY_ENABLED!=1)")            

    # 🆕 Internal SEO Refill Job（ユーザー別の“弾補給”専用）※ENVでON/OFF
    if os.getenv("INTERNAL_SEO_REFILL_ENABLED", "1") == "1":
        scheduler.add_job(
            func=_internal_seo_user_refill_job,
            trigger="interval",
            minutes=int(os.getenv("INTERNAL_SEO_REFILL_INTERVAL_MIN", "10")),
            args=[app],
            id="internal_seo_user_refill_job",
            replace_existing=True,
            max_instances=1,
        )
        app.logger.info("Scheduler started: internal_seo_user_refill_job (user refill)")
    else:
        app.logger.info("Scheduler skipped: internal_seo_user_refill_job (INTERNAL_SEO_REFILL_ENABLED!=1)")    
 
    scheduler.start()
    app.logger.info("Scheduler started: auto_post_job every 3 minutes")
    app.logger.info("Scheduler started: external_post_job every 10 minutes")
    app.logger.info("Scheduler started: gsc_metrics_job daily at 0:00")
    app.logger.info(f"Scheduler started: gsc_autogen_daily_job daily at {gsc_utc_hour:02d}:{gsc_utc_min:02d} UTC")
    app.logger.info("Scheduler started: pending_regenerator_job every 40 minutes")
    app.logger.info("Scheduler maybe started: internal_seo_user_refill_job (see env)")

    
# ────────────────────────────────────────────────
# GSCオートジェン：事前フィルタ用ユーティリティ（タスク5で利用）
# ────────────────────────────────────────────────
def _build_dup_sets(site_id: int, candidates: List[str]) -> Tuple[Set[str], Set[str]]:
    """
    事前フィルタのための“既存集合”を用意。
    戻り値:
      (gsc_keywords_set, article_dup_set)
        - gsc_keywords_set … Keyword(source='gsc') として既に存在するキーワード
        - article_dup_set  … Article の pending/gen/done/posted に既存のキーワード
    """
    from app.models import Keyword, Article
    gsc_keywords_set: Set[str] = {
        r[0] for r in db.session.query(Keyword.keyword)
        .filter(
            Keyword.site_id == site_id,
            Keyword.source == "gsc",
            Keyword.keyword.in_(candidates)
        ).all()
    }
    article_dup_set: Set[str] = {
        r[0] for r in db.session.query(Article.keyword)
        .filter(
            Article.site_id == site_id,
            Article.keyword.in_(candidates),
            Article.status.in_(["pending", "gen", "done", "posted"])
        ).all()
    }
    return gsc_keywords_set, article_dup_set

def filter_autogen_candidates(site_id: int, candidates: List[str]) -> Dict[str, List[str]]:
    """
    新規投入前の“事前フィルタ”：重複や衝突のある候補を除外。
    戻り値:
      {
        "deduped": [...],           # 投入候補（重複除外後）
        "dup_keywords": [...],      # 既存 Keyword(source='gsc') 由来の除外
        "art_dup_keywords": [...],  # 既存 Article 由来の除外
      }
    """
    if not candidates:
        return {"deduped": [], "dup_keywords": [], "art_dup_keywords": []}
    gsc_dup, art_dup = _build_dup_sets(site_id, candidates)
    dup_keywords = sorted(list(gsc_dup))
    art_dup_keywords = sorted(list(art_dup))
    blocked = gsc_dup.union(art_dup)
    deduped = [kw for kw in candidates if kw not in blocked]
    return {
        "deduped": deduped,
        "dup_keywords": dup_keywords,
        "art_dup_keywords": art_dup_keywords,
    }    

# ────────────────────────────────────────────────
# 内部SEO 自動化ジョブ
# ────────────────────────────────────────────────
@with_db_retry(max_retries=3, backoff=1.8)
def _internal_seo_run_one(site_id: int,
                          pages: int,
                          per_page: int,
                          min_score: float,
                          max_k: int,
                          limit_sources: int,
                          limit_posts: int,
                          incremental: bool,
                          job_kind: str = "scheduler"):
    """
    1サイト分の内部SEOパイプラインを実行し、InternalSeoRun に記録する。
    CLIの実装と同じステップ/同じ統計キーで保存する。
    """
    # ランレコードを作成（running）
    run = InternalSeoRun(
        site_id=site_id,
        job_kind=job_kind,
        status="running",
        started_at=datetime.utcnow(),
        stats={},
    )
    db.session.add(run)
    db.session.commit()

    t0 = time.perf_counter()
    try:
        # Indexer
        current_app.logger.info(f"[Indexer] site={site_id} incremental={incremental} pages={pages} per_page={per_page}")
        stats_idx = sync_site_content_index(site_id, per_page=per_page, max_pages=pages, incremental=incremental)
        current_app.logger.info(f"[Indexer] -> {stats_idx}")

        # LinkGraph
        current_app.logger.info(f"[LinkGraph] site={site_id} max_k={max_k} min_score={min_score}")
        stats_graph = build_link_graph_for_site(site_id, max_targets_per_source=max_k, min_score=min_score)
        current_app.logger.info(f"[LinkGraph] -> {stats_graph}")

        # Planner
        current_app.logger.info(f"[Planner] site={site_id} limit_sources={limit_sources} max_candidates={max_k} min_score={min_score}")
        stats_plan = plan_links_for_site(site_id, limit_sources=limit_sources, mode_swap_check=True,
                                         min_score=min_score, max_candidates=max_k)
        current_app.logger.info(f"[Planner] -> {stats_plan}")

        # Applier（refill など limit_posts<=0 の場合は完全スキップ）
        if int(limit_posts) <= 0:
            res_apply = {"applied": 0, "swapped": 0, "skipped": 0, "processed_posts": 0, "note": "applier skipped (limit_posts<=0)"}
            current_app.logger.info(f"[Applier] skipped (limit_posts<=0) site={site_id}")
        else:
            current_app.logger.info(f"[Applier] site={site_id} limit_posts={limit_posts}")
            res_apply = apply_actions_for_site(site_id, limit_posts=limit_posts, dry_run=False)
            current_app.logger.info(f"[Applier] -> {res_apply}")

        # 成功で確定
        run.status = "success"
        run.ended_at = datetime.utcnow()
        run.duration_ms = int((time.perf_counter() - t0) * 1000)
        run.stats = {
            "indexer": stats_idx,
            "link_graph": stats_graph,
            "planner": stats_plan,
            "applier": res_apply,
            "params": {
                "incremental": incremental,
                "pages": pages,
                "per_page": per_page,
                "min_score": min_score,
                "max_k": max_k,
                "limit_sources": limit_sources,
                "limit_posts": limit_posts,
                "job_kind": job_kind,
            },
        }
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        run.status = "error"
        run.ended_at = datetime.utcnow()
        run.duration_ms = int((time.perf_counter() - t0) * 1000)
        run.stats = (run.stats or {})
        run.stats["error"] = {"type": e.__class__.__name__, "message": str(e)}
        db.session.add(run)
        db.session.commit()
        current_app.logger.exception(f"[internal-seo] failed for site {site_id}: {e}")

@with_db_retry(max_retries=3, backoff=1.8)
def _internal_seo_worker_tick(app):
    """
    internal_seo_job_queue から 'queued' を安全に取り出し、
    同時実行上限を守りつつ _internal_seo_run_one を回す。
    """
    with app.app_context():
        # 同時実行上限（ENV で調整）
        max_parallel = int(os.getenv("INTERNAL_SEO_WORKER_PARALLELISM", "3"))
        # いま走っている run 数
        running_cnt = db.session.execute(text("""
            SELECT COUNT(*) FROM internal_seo_runs WHERE status='running'
        """)).scalar_one()
        available = max(0, max_parallel - int(running_cnt or 0))
        if available <= 0:
            current_app.logger.info(f"[internal-seo worker] saturated: running={running_cnt} / max={max_parallel}")
            return

        # 取り出し件数は上限に控えめ（念のため 1 サイト=1 run 想定）
        take = min(available, int(os.getenv("INTERNAL_SEO_WORKER_TAKE", "2")))

        # queued をロックして running に更新（SKIP LOCKED で多重取得回避）
        rows = db.session.execute(text(f"""
            WITH picked AS (
              SELECT id
              FROM internal_seo_job_queue
              WHERE status='queued'
              ORDER BY created_at ASC, id ASC
              FOR UPDATE SKIP LOCKED
              LIMIT :take
            )
            UPDATE internal_seo_job_queue q
               SET status='running', started_at=now()
              FROM picked p
             WHERE q.id = p.id
          RETURNING q.id, q.site_id, q.pages, q.per_page, q.min_score, q.max_k,
                    q.limit_sources, q.limit_posts, q.incremental, q.job_kind;
        """), {"take": take}).mappings().all()
        db.session.commit()

        if not rows:
            current_app.logger.info("[internal-seo worker] no queued jobs")
            return

        current_app.logger.info(f"[internal-seo worker] picked {len(rows)} job(s)")

        # 1件ずつ実行（同プロセス内で逐次。必要なら ThreadPoolExecutor で並列化も可）
        for r in rows:
            j_id = r["id"]
            try:
                _internal_seo_run_one(
                    site_id       = int(r["site_id"]),
                    pages         = int(r["pages"] or os.getenv("INTERNAL_SEO_PAGES", 10)),
                    per_page      = int(r["per_page"] or os.getenv("INTERNAL_SEO_PER_PAGE", 100)),
                    min_score     = float(r["min_score"] or os.getenv("INTERNAL_SEO_MIN_SCORE", 0.05)),
                    max_k         = int(r["max_k"] or os.getenv("INTERNAL_SEO_MAX_K", 80)),
                    limit_sources = int(r["limit_sources"] or os.getenv("INTERNAL_SEO_LIMIT_SOURCES", 200)),
                    limit_posts   = int(r["limit_posts"] or os.getenv("INTERNAL_SEO_LIMIT_POSTS", 50)),
                    incremental   = bool(r["incremental"]),
                    job_kind      = r["job_kind"] or "worker",
                )
                db.session.execute(text("""
                    UPDATE internal_seo_job_queue
                       SET status='done', ended_at=now()
                     WHERE id=:id
                """), {"id": j_id})
                db.session.commit()
            except Exception as e:
                current_app.logger.exception(f"[internal-seo worker] job {j_id} failed: {e}")
                db.session.execute(text("""
                    UPDATE internal_seo_job_queue
                       SET status='error', ended_at=now(), message=:msg
                     WHERE id=:id
                """), {"id": j_id, "msg": str(e)})
                db.session.commit()

def _internal_seo_nightly_job(app):
    """
    すべてのサイト（または ENV 指定のサイト群）について、
    “実行本体” はワーカーに任せるため、ここではキューに積むだけ。
    既に queued/running のサイトは除外して重複投入を防ぐ。
    """
    with app.app_context():
        pages          = int(os.getenv("INTERNAL_SEO_PAGES", "10"))
        per_page       = int(os.getenv("INTERNAL_SEO_PER_PAGE", "100"))
        min_score      = float(os.getenv("INTERNAL_SEO_MIN_SCORE", "0.05"))
        max_k          = int(os.getenv("INTERNAL_SEO_MAX_K", "80"))
        limit_sources  = int(os.getenv("INTERNAL_SEO_LIMIT_SOURCES", "200"))
        limit_posts    = int(os.getenv("INTERNAL_SEO_LIMIT_POSTS", "50"))
        incremental    = os.getenv("INTERNAL_SEO_INCREMENTAL", "1") == "1"
        job_kind       = os.getenv("INTERNAL_SEO_JOB_KIND", "nightly-enqueue")

        only_ids = os.getenv("INTERNAL_SEO_SITE_IDS")
        if only_ids:
            ids = [int(x) for x in only_ids.split(",") if x.strip().isdigit()]
            site_predicate = "WHERE s.id = ANY(:ids)"
            params = {
                "ids": ids,
                "pages": pages, "per_page": per_page, "min_score": min_score, "max_k": max_k,
                "limit_sources": limit_sources, "limit_posts": limit_posts,
                "incremental": incremental, "job_kind": job_kind,
            }
        else:
            site_predicate = ""
            params = {
                "pages": pages, "per_page": per_page, "min_score": min_score, "max_k": max_k,
                "limit_sources": limit_sources, "limit_posts": limit_posts,
                "incremental": incremental, "job_kind": job_kind,
            }

        # 既に queued/running のサイトは除外したうえで一括INSERT
        sql = text(f"""
            INSERT INTO internal_seo_job_queue
              (site_id, pages, per_page, min_score, max_k, limit_sources, limit_posts,
               incremental, job_kind, status, created_at)
            SELECT
              s.id, :pages, :per_page, :min_score, :max_k, :limit_sources, :limit_posts,
              :incremental, :job_kind, 'queued', now()
            FROM site s
            LEFT JOIN internal_seo_job_queue q
                   ON q.site_id = s.id
                  AND q.status IN ('queued','running')
            {site_predicate}
            WHERE q.site_id IS NULL
        """)

        res = db.session.execute(sql, params)
        db.session.commit()
        inserted = res.rowcount or 0
        # 参考までの総サイト数（絞り込み時は ids の長さ）
        total_sites = len(params["ids"]) if only_ids else (db.session.execute(text("SELECT COUNT(*) FROM site")).scalar() or 0)

        current_app.logger.info(
            f"[internal-seo nightly] enqueued {inserted}/{total_sites} "
            f"params={{pages:{pages}, per_page:{per_page}, min_score:{min_score}, "
            f"max_k:{max_k}, limit_sources:{limit_sources}, limit_posts:{limit_posts}, "
            f"incremental:{incremental}, job_kind:{job_kind}}}"
        )

# ────────────────────────────────────────────────
# 🆕 内部SEO ユーザー適用ティック（巡回ループ本体）
#   - 条件: InternalSeoUserSchedule.is_enabled=True かつ status<>'paused'
#   - そのユーザーに対して apply_actions_for_user() を実行
#   - 1ティックの総予算＆ユーザー上限は ENV で制御
# ────────────────────────────────────────────────
def _internal_seo_user_apply_tick(app):
    with app.app_context():
        try:
            if os.getenv("INTERNAL_SEO_USER_APPLY_ENABLED", "1") != "1":
                return

            # ✅ 1ティックの総処理記事数（全ユーザー合算の上限）
            total_budget = int(os.getenv("INTERNAL_SEO_USER_APPLY_BUDGET", "200"))
            # ✅ 1ユーザーあたりの上限
            per_user_cap = int(os.getenv("INTERNAL_SEO_USER_APPLY_PER_USER", "50"))
            if total_budget <= 0 or per_user_cap <= 0:
                current_app.logger.info("[ISEO-USER] apply disabled by zero budget/cap")
                return

            # 対象ユーザー（開始ボタンON & 一時停止でない）
            rows = (InternalSeoUserSchedule.query
                    .filter(InternalSeoUserSchedule.is_enabled == True)  # noqa: E712
                    .filter((InternalSeoUserSchedule.status.is_(None)) | (InternalSeoUserSchedule.status != "paused"))
                    .order_by(InternalSeoUserSchedule.user_id.asc())
                    .all())
            if not rows:
                current_app.logger.info("[ISEO-USER] no eligible users")
                return

            remaining = total_budget
            picked = 0
            for sched in rows:
                if remaining <= 0:
                    break
                uid = int(sched.user_id)
                # ユーザーロック（多重実行防止）
                if not _try_lock_user(uid):
                    current_app.logger.info("[ISEO-USER] skip (locked) user=%s", uid)
                    continue
                try:
                    quota = min(per_user_cap, remaining)
                    if quota <= 0:
                        break
                    # applier: ユーザーの全サイトに均等配分して pending を実反映
                    res = apply_actions_for_user(user_id=uid, limit_posts=quota, dry_run=False)
                    # 実際に処理できた記事数で予算を減らす
                    processed = int(res.get("processed_posts", 0) or 0)
                    remaining -= max(0, processed)
                    picked += 1
                    current_app.logger.info(
                        "[ISEO-USER] uid=%s processed=%s applied=%s swapped=%s skipped=%s pending_total=%s remaining_budget=%s",
                        uid,
                        processed,
                        int(res.get("applied", 0) or 0),
                        int(res.get("swapped", 0) or 0),
                        int(res.get("skipped", 0) or 0),
                        int(res.get("pending_total", 0) or 0),
                        remaining
                    )
                except Exception as e:
                    current_app.logger.exception("[ISEO-USER] apply failed uid=%s: %s", uid, e)
                finally:
                    _unlock_user(uid)

            current_app.logger.info("[ISEO-USER] tick done: users=%s total_budget=%s remaining=%s",
                                    picked, total_budget, remaining)
        except Exception as e:
            current_app.logger.exception("[ISEO-USER] tick error: %s", e)
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
from app.google_client import update_all_gsc_sites

# 既存 import の下あたりに追加
from concurrent.futures import ThreadPoolExecutor
from .models import (Site, Keyword, ExternalSEOJob,
                     BlogType, ExternalBlogAccount, ExternalArticleSchedule)

# app/tasks.py （インポートセクションの BlogType などの下あたり）
from app.services.blog_signup.livedoor_signup import signup as livedoor_signup
# 既存 import 群の下に追加
from app.external_seo_generator import generate_and_schedule_external_articles

# === 内部SEO 自動化 で使う import ===
from app.models import InternalSeoRun
from app.utils.db_retry import with_db_retry
from app.services.internal_seo.indexer import sync_site_content_index
from app.services.internal_seo.link_graph import build_link_graph_for_site
from app.services.internal_seo.planner import plan_links_for_site
from app.services.internal_seo.applier import apply_actions_for_site
import os
from math import inf

# ────────────────────────────────────────────────
# APScheduler ＋ スレッドプール
# ────────────────────────────────────────────────
# グローバルな APScheduler インスタンス（__init__.py で start されています）
scheduler = BackgroundScheduler(timezone="UTC")
executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="extseo")

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
            .limit(1 if schedule_id is not None else 10)
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

    # ✅ GSC記事生成ジョブ
    scheduler.add_job(
        func=_gsc_generation_job,
        trigger="interval",
        minutes=20,
        args=[app],
        id="gsc_generation_job",
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

    # ✅ 内部SEO ナイトリー実行（環境変数でON/OFF可能）
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
        app.logger.info("Scheduler skipped: internal_seo_job (INTERNAL_SEO_ENABLED!=1)")

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
 


    scheduler.start()
    app.logger.info("Scheduler started: auto_post_job every 3 minutes")
    app.logger.info("Scheduler started: gsc_metrics_job daily at 0:00")
    app.logger.info("Scheduler started: gsc_generation_job every 20 minutes")
    app.logger.info("Scheduler started: external_post_job every 10 minutes")

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

        # Applier
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
            sites = Site.query.filter(Site.id.in_(ids)).all()
        else:
            sites = Site.query.order_by(Site.id.asc()).all()

        enq = text("""
            INSERT INTO internal_seo_job_queue
              (site_id, pages, per_page, min_score, max_k, limit_sources, limit_posts,
               incremental, job_kind, status, created_at)
            VALUES
              (:site_id, :pages, :per_page, :min_score, :max_k, :limit_sources, :limit_posts,
               :incremental, :job_kind, 'queued', now())
        """)

        enqueued = 0
        for s in sites:
            try:
                db.session.execute(enq, dict(
                    site_id=s.id, pages=pages, per_page=per_page, min_score=min_score,
                    max_k=max_k, limit_sources=limit_sources, limit_posts=limit_posts,
                    incremental=incremental, job_kind=job_kind,
                ))
                enqueued += 1
            except Exception as e:
                current_app.logger.exception(f"[internal-seo nightly] enqueue failed site={s.id}: {e}")

        db.session.commit()
        current_app.logger.info(
            f"[internal-seo nightly] enqueued {enqueued}/{len(sites)} jobs "
            f"params={{pages:{pages}, per_page:{per_page}, min_score:{min_score}, "
            f"max_k:{max_k}, limit_sources:{limit_sources}, limit_posts:{limit_posts}, "
            f"incremental:{incremental}, job_kind:{job_kind}}}"
        )

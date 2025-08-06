# app/tasks.py

import logging
from datetime import datetime
import pytz
import time 
from flask import current_app
from apscheduler.schedulers.background import BackgroundScheduler

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



# ────────────────────────────────────────────────
# APScheduler ＋ スレッドプール
# ────────────────────────────────────────────────
# グローバルな APScheduler インスタンス（__init__.py で start されています）
scheduler = BackgroundScheduler(timezone="UTC")
executor = ThreadPoolExecutor(max_workers=1)  # ✅ 外部SEOでは同時1件まで

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
                .filter(Article.status == "done", Article.scheduled_at <= now)
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

# --------------------------------------------------------------------------- #
# 4) 外部SEO ① キュー作成ジョブ
# --------------------------------------------------------------------------- #



# --------------------------------------------------------------------------- #
# 5) 外部SEO ② 投稿ジョブ
# --------------------------------------------------------------------------- #
def _finalize_external_job(job_id: int):
    """すべて posted になったら job を完了にする"""
    job = ExternalSEOJob.query.get(job_id)
    if not job:
        return

    total = job.article_cnt
    posted = (ExternalArticleSchedule.query
              .join(ExternalBlogAccount,
                    ExternalArticleSchedule.blog_account_id == ExternalBlogAccount.id)
              .filter(ExternalBlogAccount.site_id == job.site_id,
                      ExternalArticleSchedule.status == "posted")
              .count())
    if posted >= total:
        job.step   = "done"
        job.status = "success"
        db.session.commit()           

# ──────────────────────────────────────────
# 外部ブログ投稿ジョブ（簡略・安定版）
# ──────────────────────────────────────────
def _run_external_post_job(app):
    """
    外部SEO記事を外部ブログに投稿するジョブ（簡略版）
    - status="done" の外部SEO記事を投稿
    - Keyword一致条件を排除し、site_id/source/status で判定
    """
    from app.models import ExternalArticleSchedule, ExternalBlogAccount, BlogType, Article, ExternalSEOJob
    from datetime import datetime
    from app.services.blog_post.livedoor_post import post_livedoor_article

    with app.app_context():
        now = datetime.utcnow()
        current_app.logger.info(f"[ExtPost] Job start at {now}")

        # 投稿対象スケジュール取得
        rows = (
            db.session.query(ExternalArticleSchedule, ExternalBlogAccount)
            .join(ExternalBlogAccount, ExternalArticleSchedule.blog_account_id == ExternalBlogAccount.id)
            .filter(
                ExternalArticleSchedule.status == "pending",
                ExternalArticleSchedule.scheduled_date <= now
            )
            .order_by(ExternalArticleSchedule.scheduled_date.asc())
            .limit(10)
            .all()
        )

        current_app.logger.info(f"[ExtPost] Found {len(rows)} pending schedules")
        if not rows:
            return

        for sched, acct in rows:
            current_app.logger.info(
                f"[ExtPost] Processing sched_id={sched.id}, scheduled_date={sched.scheduled_date}, account={acct.username}"
            )

            try:
                # ブログタイプチェック
                if acct.blog_type != BlogType.LIVEDOOR:
                    sched.status = "error"
                    sched.message = f"未対応ブログタイプ: {acct.blog_type}"
                    db.session.commit()
                    continue

                # 外部SEO専用の記事取得
                art = (
                    db.session.query(Article)
                    .filter(
                        Article.site_id == acct.site_id,
                        Article.status == "done",
                        Article.source == "external"  # 外部SEOのみ
                    )
                    .order_by(Article.id.asc())
                    .first()
                )

                if not art:
                    sched.status = "error"
                    sched.message = "記事未生成"
                    db.session.commit()
                    current_app.logger.warning(f"[ExtPost] Article not found for site_id={acct.site_id}")
                    continue

                # 投稿処理
                current_app.logger.info(f"[ExtPost] Posting article_id={art.id} to Livedoor...")
                res = post_livedoor_article(acct, art.title, art.body)

                if res.get("ok"):
                    sched.status = "posted"
                    sched.posted_url = res["url"]
                    sched.posted_at = res.get("posted_at")
                    art.status = "posted"
                    acct.posted_cnt += 1
                    db.session.commit()
                    current_app.logger.info(f"[ExtPost] Success: {res['url']}")

                    # ジョブ完了判定
                    latest_job = (
                        ExternalSEOJob.query
                        .filter_by(site_id=acct.site_id)
                        .order_by(ExternalSEOJob.id.desc())
                        .first()
                    )
                    if latest_job:
                        _finalize_external_job(latest_job.id)
                else:
                    sched.status = "error"
                    sched.message = res.get("error")
                    db.session.commit()
                    current_app.logger.error(f"[ExtPost] Failed: {res.get('error')}")

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


    scheduler.start()
    app.logger.info("Scheduler started: auto_post_job every 3 minutes")
    app.logger.info("Scheduler started: gsc_metrics_job daily at 0:00")
    app.logger.info("Scheduler started: gsc_generation_job every 20 minutes")
    app.logger.info("Scheduler started: external_post_job every 10 minutes")
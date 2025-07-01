# app/tasks.py

import logging
from datetime import datetime
import pytz
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
from app.services.blog_signup import register_blog_account
from app.article_generator import enqueue_generation  # 既存非同期記事生成キュー



# グローバルな APScheduler インスタンス（__init__.py で start されています）
scheduler = BackgroundScheduler(timezone="UTC")
executor = ThreadPoolExecutor(max_workers=4)  # 🆕 外部SEOジョブ用


def _auto_post_job(app):
    with app.app_context():
        now = datetime.now(pytz.utc)

        try:
            pending = (
                db.session.query(Article)
                .filter(Article.status == "done", Article.scheduled_at <= now)
                .options(selectinload(Article.site))
                .order_by(Article.scheduled_at.asc())
                .limit(300)
                .all()
            )

            for art in pending:
                if not art.site:
                    current_app.logger.warning(f"記事 {art.id} の投稿先サイト未設定")
                    continue

                try:
                    url = post_to_wp(art.site, art)
                    art.posted_at = now
                    art.status = "posted"
                    db.session.commit()
                    current_app.logger.info(f"Auto-posted Article {art.id} -> {url}")

                except Exception as e:
                    db.session.rollback()
                    current_app.logger.warning(f"初回投稿失敗: Article {art.id} {e}")

                    retry_attempts = 3
                    for attempt in range(retry_attempts):
                        try:
                            url = post_to_wp(art.site, art)
                            art.posted_at = now
                            art.status = "posted"
                            db.session.commit()
                            current_app.logger.info(f"Retry Success: Article {art.id} -> {url}")
                            break
                        except Exception as retry_exception:
                            db.session.rollback()
                            current_app.logger.warning(
                                f"Retry {attempt + 1} failed for Article {art.id}: {retry_exception}"
                            )

        finally:
            db.session.close()

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
    ✅ GSC記事自動生成ジョブ（毎日）
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

def _run_external_seo_job(app, site_id: int):
    """
    1) ExternalSEOJob を running に
    2) Note アカウントを自動登録
    3) 上位キーワード100件で記事生成をキューに流し
    4) ExternalArticleSchedule を作成
    """
    with app.app_context():
        from sqlalchemy.exc import SQLAlchemyError

        # ── 1. ジョブ行を作成 ──────────────────
        job = ExternalSEOJob(site_id=site_id, status="running", step="signup")
        db.session.add(job)
        db.session.commit()

        try:
            # ── 2. アカウント自動登録 ───────────
            account = register_blog_account(site_id, BlogType.NOTE)
            job.step = "generating"
            db.session.commit()

            # ── 3. 上位キーワード100件抽出 ────────
            top_kws = (
                Keyword.query.filter_by(site_id=site_id, status="done")
                .order_by(Keyword.times_used.desc())
                .limit(100)
                .all()
            )
            if not top_kws:
                raise ValueError("上位キーワードがありません")

            # キュー投入 & スケジュール生成
            schedules = []
            for kw in top_kws:
                # 既存の非同期記事生成キューを使用
                enqueue_generation(
                    user_id=kw.user_id,
                    site_id=site_id,
                    keywords=[kw.keyword],
                    format="html",
                    self_review=False,
                )

                sched = ExternalArticleSchedule(
                    blog_account_id=account.id,
                    keyword_id=kw.id,
                    scheduled_date=datetime.utcnow(),  # ★あとで間隔制御可
                )
                schedules.append(sched)

            db.session.bulk_save_objects(schedules)
            job.article_cnt = len(schedules)
            job.step = "finished"
            job.status = "success"
            db.session.commit()

        except Exception as e:
            db.session.rollback()
            job.status = "error"
            job.message = str(e)
            db.session.commit()
            current_app.logger.error(f"[外部SEO] 失敗: {e}")

def enqueue_external_seo(site_id: int):
    """
    外部SEOジョブをバックグラウンドスレッドに投入。
    ルート側から `enqueue_external_seo(site_id)` を呼ぶだけでOK。
    """
    app = current_app._get_current_object()
    executor.submit(_run_external_seo_job, app, site_id)



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
        max_instances=1
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


    scheduler.start()
    app.logger.info("Scheduler started: auto_post_job every 3 minutes")
    app.logger.info("Scheduler started: gsc_metrics_job daily at 0:00")
    app.logger.info("Scheduler started: gsc_generation_job every 20 minutes")
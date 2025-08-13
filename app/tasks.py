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
# 既存 import 群の下に追加
from app.external_seo_generator import generate_and_schedule_external_articles



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


    scheduler.start()
    app.logger.info("Scheduler started: auto_post_job every 3 minutes")
    app.logger.info("Scheduler started: gsc_metrics_job daily at 0:00")
    app.logger.info("Scheduler started: gsc_generation_job every 20 minutes")
    app.logger.info("Scheduler started: external_post_job every 10 minutes")
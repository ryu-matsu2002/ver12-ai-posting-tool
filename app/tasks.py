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


# グローバルな APScheduler インスタンス（__init__.py で start されています）
scheduler = BackgroundScheduler(timezone="UTC")


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
    🔁 GSCからのクエリで1000記事未満なら記事生成をループ継続
    """
    from app import db
    from app.google_client import fetch_search_queries_for_site
    from app.models import Keyword
    from app.article_generator import enqueue_generation
    from flask import current_app

    # ✅ GSC接続されていないサイトはスキップ
    if not site.gsc_connected:
        current_app.logger.info(f"[GSC LOOP] スキップ：未接続サイト {site.name}")
        return

    # ✅ すでに1000記事以上生成済みならスキップ
    total_keywords = Keyword.query.filter_by(site_id=site.id).count()
    if total_keywords >= 1000:
        current_app.logger.info(f"[GSC LOOP] {site.name} は既に1000記事に到達済み")
        return

    # ✅ GSCからクエリ取得
    try:
        queries = fetch_search_queries_for_site(site, days=28)
    except Exception as e:
        current_app.logger.warning(f"[GSC LOOP] クエリ取得失敗 - {site.url}: {e}")
        return

    # ✅ 🔽 既存キーワードを status="done" 以外含めて重複排除対象とする
    existing_keywords = set(
        k.keyword for k in Keyword.query.filter_by(site_id=site.id).all()
    )

    # ✅ 新しいキーワードのみ抽出（DBにまだ存在しないもの）
    new_keywords = [q for q in queries if q not in existing_keywords]

    if not new_keywords:
        current_app.logger.info(f"[GSC LOOP] {site.name} に新規キーワードなし")
        return

    # ✅ DB保存（source='gsc', status='unprocessed'）＋記事生成キュー追加
    for kw in new_keywords:
        db.session.add(Keyword(
            keyword=kw,
            site_id=site.id,
            user_id=site.user_id,
            source='gsc',
            status='unprocessed'  # ✅🔧 ステータス初期化を明示
        ))

    db.session.commit()

    # ✅ 🔽 最大40件の未処理キーワードを "generating" にロック
    targets = Keyword.query.filter_by(
        site_id=site.id,
        user_id=site.user_id,
        source='gsc',
        status='unprocessed'
    ).limit(40).all()

    if not targets:
        current_app.logger.info(f"[GSC LOOP] {site.name} に未処理キーワードなし")
        return

    for kw in targets:
        kw.status = 'generating'
    db.session.commit()

    # ✅ 修正箇所：キーワードIDで渡すように変更
    keyword_ids = [k.id for k in targets]
    from app.article_generator import submit_article_job
    submit_article_job(site.id, keyword_ids)


    current_app.logger.info(f"[GSC LOOP] {site.name} に {len(keyword_ids)} 件生成キュー投入")

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
        minutes=10,
        args=[app],
        id="gsc_generation_job",
        replace_existing=True,
        max_instances=1
    )


    scheduler.start()
    app.logger.info("Scheduler started: auto_post_job every 3 minutes")
    app.logger.info("Scheduler started: gsc_metrics_job daily at 0:00")
    app.logger.info("Scheduler started: gsc_generation_job every 10 minutes")
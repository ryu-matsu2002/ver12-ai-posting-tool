import click
import time
from app import db
from datetime import datetime
from app.models import Site, InternalSeoRun
from app.utils.db_retry import with_db_retry
from app.services.internal_seo.indexer import sync_site_content_index
from app.services.internal_seo.link_graph import build_link_graph_for_site
from app.services.internal_seo.planner import plan_links_for_site
from app.services.internal_seo.applier import apply_actions_for_site


def register_cli(app):
    @app.cli.command("internal-seo-run")
    @click.option("--site-id", required=True, type=int, help="対象 Site.id")
    @click.option("--pages", default=10, show_default=True, help="indexer の page 数")
    @click.option("--per-page", default=100, show_default=True, help="WP REST per_page")
    @click.option("--min-score", default=0.05, show_default=True, type=float, help="link graph/plan の下限スコア")
    @click.option("--max-k", default=80, show_default=True, type=int, help="各ソースの最大候補数")
    @click.option("--limit-sources", default=200, show_default=True, type=int, help="planner のソース数上限")
    @click.option("--limit-posts", default=50, show_default=True, type=int, help="applier が更新する投稿数上限")
    @click.option("--incremental/--full", default=True, show_default=True, help="indexer の増分/フル")
    @click.option("--job-kind", default="manual-cli", show_default=True, help="実行種別のラベル（ログ用）")
    def internal_seo_run(site_id, pages, per_page, min_score, max_k, limit_sources, limit_posts, incremental, job_kind):
        run_pipeline_with_log(
            site_id=site_id,
            pages=pages,
            per_page=per_page,
            min_score=min_score,
            max_k=max_k,
            limit_sources=limit_sources,
            limit_posts=limit_posts,
            incremental=incremental,
            job_kind=job_kind,
        )

    @with_db_retry(max_retries=4, backoff=1.6)
    def run_pipeline(site_id, pages, per_page, min_score, max_k, limit_sources, limit_posts, incremental):
        site = Site.query.get(site_id)
        if not site:
            click.echo(f"[ERROR] Site {site_id} not found")
            return

        click.echo(f"[Indexer] site={site_id} incremental={incremental} pages={pages} per_page={per_page}")
        stats_idx = sync_site_content_index(site_id, per_page=per_page, max_pages=pages, incremental=incremental)
        click.echo(f"  -> {stats_idx}")

        click.echo(f"[LinkGraph] site={site_id} max_k={max_k} min_score={min_score}")
        stats_graph = build_link_graph_for_site(site_id, max_targets_per_source=max_k, min_score=min_score)
        click.echo(f"  -> {stats_graph}")

        click.echo(f"[Planner] site={site_id} limit_sources={limit_sources} max_candidates={max_k} min_score={min_score}")
        stats_plan = plan_links_for_site(site_id, limit_sources=limit_sources, mode_swap_check=True,
                                         min_score=min_score, max_candidates=max_k)
        click.echo(f"  -> {stats_plan}")

        click.echo(f"[Applier] site={site_id} limit_posts={limit_posts}")
        res = apply_actions_for_site(site_id, limit_posts=limit_posts, dry_run=False)
        click.echo(f"  -> {res}")

    @with_db_retry(max_retries=4, backoff=1.6)
    def run_pipeline_with_log(site_id, pages, per_page, min_score, max_k, limit_sources, limit_posts, incremental, job_kind):
        """
        1ラン全体を InternalSeoRun に記録するラッパ。
        成功/失敗、各ステージの統計、所要時間を1レコードで残す。
        """
        # ❶ ラン行を作成（running）
        run = InternalSeoRun(
            site_id=site_id,
            job_kind=job_kind,
            status="running",
            started_at=datetime.utcnow(),
            stats={},  # 後で段階的に詰める
        )
        db.session.add(run)
        db.session.commit()  # id を確定させる
        click.echo(f"[Run] started id={run.id} site={site_id} kind={job_kind}")

        t0 = time.perf_counter()
        try:
            # ❷ 実処理（既存関数をそのまま使用）
            click.echo(f"[Indexer] site={site_id} incremental={incremental} pages={pages} per_page={per_page}")
            stats_idx = sync_site_content_index(site_id, per_page=per_page, max_pages=pages, incremental=incremental)
            click.echo(f"  -> {stats_idx}")

            click.echo(f"[LinkGraph] site={site_id} max_k={max_k} min_score={min_score}")
            stats_graph = build_link_graph_for_site(site_id, max_targets_per_source=max_k, min_score=min_score)
            click.echo(f"  -> {stats_graph}")

            click.echo(f"[Planner] site={site_id} limit_sources={limit_sources} max_candidates={max_k} min_score={min_score}")
            stats_plan = plan_links_for_site(site_id, limit_sources=limit_sources, mode_swap_check=True,
                                             min_score=min_score, max_candidates=max_k)
            click.echo(f"  -> {stats_plan}")

            click.echo(f"[Applier] site={site_id} limit_posts={limit_posts}")
            res_apply = apply_actions_for_site(site_id, limit_posts=limit_posts, dry_run=False)
            click.echo(f"  -> {res_apply}")

            # ❸ 結果を集約して成功で確定
            duration_ms = int((time.perf_counter() - t0) * 1000)
            run.status = "success"
            run.ended_at = datetime.utcnow()
            run.duration_ms = duration_ms
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
            click.echo(f"[Run] success id={run.id} duration_ms={duration_ms}")
        except Exception as e:
            # ❹ 失敗時のロギング
            duration_ms = int((time.perf_counter() - t0) * 1000)
            db.session.rollback()
            run.status = "error"
            run.ended_at = datetime.utcnow()
            run.duration_ms = duration_ms
            # 既に入った stats（途中まで）を壊さないように dict に追加
            run.stats = (run.stats or {})
            run.stats["error"] = {"type": e.__class__.__name__, "message": str(e)}
            db.session.add(run)
            db.session.commit()
            click.echo(f"[Run] error id={run.id} duration_ms={duration_ms} err={e}")
            # エラーでも CLI 自体は終了させず戻る（cron運用を想定）
            return    

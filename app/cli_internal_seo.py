import click
from app import db
from app.models import Site
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
    def internal_seo_run(site_id, pages, per_page, min_score, max_k, limit_sources, limit_posts, incremental):
        run_pipeline(site_id, pages, per_page, min_score, max_k, limit_sources, limit_posts, incremental)

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

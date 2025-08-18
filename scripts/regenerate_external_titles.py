# scripts/regenerate_external_titles.py

import sys
import click
import logging
from flask import Flask
from app import create_app, db
from app.models import Article, Site
from app.article_generator import _unique_title
from app.external_seo_generator import TITLE_PROMPT

@click.command()
@click.option("--site_id", required=True, type=int, help="対象のサイトID")
def regenerate(site_id):
    """指定サイトの記事タイトルを全件一括で再生成して上書き"""
    app: Flask = create_app()
    with app.app_context():
        site = Site.query.get(site_id)
        if not site:
            click.echo(f"❌ Site not found: {site_id}")
            sys.exit(1)

        # done or 記事生成済み の記事を全件取得
        articles = (
            Article.query
            .filter(Article.site_id == site_id)
            .filter(Article.status.in_(["done", "記事生成済み"]))
            .all()
        )

        click.echo(f"✅ {len(articles)} 件の記事を処理開始 (site_id={site_id})")

        for art in articles:
            try:
                new_title = _unique_title(art.keyword, TITLE_PROMPT)
                if new_title and new_title != art.title:
                    click.echo(f"更新: {art.id}: '{art.title}' → '{new_title}'")
                    art.title = new_title
                    db.session.add(art)
            except Exception as e:
                logging.exception(f"[失敗] Article ID={art.id}: {e}")

        db.session.commit()
        click.echo("🎉 全件タイトル更新完了")

if __name__ == "__main__":
    regenerate()

# scripts/regenerate_external_titles.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import click
import logging
from flask import Flask
from app import create_app, db
from app.models import Article, Site, ExternalArticleSchedule
from app.article_generator import _unique_title
from app.external_seo_generator import TITLE_PROMPT

@click.command()
@click.option("--blog_account_id", required=True, type=int, help="対象の外部SEOブログアカウントID")
def regenerate(blog_account_id):
    """指定の外部SEOブログアカウントの記事タイトルを一括で再生成"""
    app: Flask = create_app()
    with app.app_context():
        # ExternalArticleSchedule 経由で記事を特定
        scheds = (
            ExternalArticleSchedule.query
            .filter_by(blog_account_id=blog_account_id)
            .all()
        )
        article_ids = [s.article_id for s in scheds]

        articles = (
            db.session.query(Article)
            .filter(Article.id.in_(article_ids))
            .filter(Article.status.in_(["done", "記事生成済み"]))
            .all()
        )

        click.echo(f"✅ {len(articles)} 件の外部SEO記事を処理開始 (blog_account_id={blog_account_id})")

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
        click.echo("🎉 外部SEO記事のタイトル更新完了")

if __name__ == "__main__":
    regenerate()

# scripts/backfill_posted_url.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import requests
import base64
from app import create_app, db
from app.models import Article, Site
from sqlalchemy import and_

app = create_app()

def get_wp_posts(site: Site):
    url = site.url.rstrip('/') + "/wp-json/wp/v2/posts?per_page=100"
    token = base64.b64encode(f"{site.username}:{site.app_pass}".encode("utf-8")).decode("utf-8")
    headers = {
        "Authorization": f"Basic {token}",
        "Accept": "application/json",
    }
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            return res.json()
        else:
            print(f"Failed to fetch posts for site {site.name}: {res.status_code}")
    except Exception as e:
        print(f"Error fetching posts: {e}")
    return []

def backfill():
    with app.app_context():
        articles = Article.query.filter(
            and_(Article.status == "posted", Article.posted_url == None)
        ).all()

        print(f"🔍 未登録の posted_url 記事数: {len(articles)}")

        for art in articles:
            site = art.site
            wp_posts = get_wp_posts(site)
            matched = next(
                (p for p in wp_posts if p.get("title", {}).get("rendered") == art.title),
                None
            )
            if matched:
                art.posted_url = matched.get("link")
                db.session.commit()
                print(f"✅ 記事 {art.id} → URL: {art.posted_url}")
            else:
                print(f"⚠️ 記事 {art.id} - '{art.title}' に一致する投稿が見つかりません")

if __name__ == "__main__":
    backfill()

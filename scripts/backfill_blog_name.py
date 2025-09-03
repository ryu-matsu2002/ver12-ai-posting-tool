# app/scripts/backfill_blog_name.py
import re
import requests
from bs4 import BeautifulSoup
from app import create_app, db
from app.models import ExternalBlogAccount
from app.enums import BlogType

# livedoorタイトル末尾の「 - livedoor Blog（ブログ）」等を除去
_SUFFIXES = [
    r"\s*-\s*livedoor\s*Blog（ブログ）\s*$",
    r"\s*-\s*livedoor\s*Blog\s*$",
    r"\s*\|\s*livedoor\s*Blog（ブログ）\s*$",
    r"\s*\|\s*livedoor\s*Blog\s*$",
]
SUFFIX_RE = re.compile("|".join(_SUFFIXES), re.IGNORECASE)

def _clean_title(t: str) -> str:
    t = (t or "").strip()
    t = SUFFIX_RE.sub("", t)
    return t.strip()

def _fetch_ld_title(blog_id: str) -> str | None:
    # 代表: blog.livedoor.jp / たまに blog.ldblog.jp
    urls = [
        f"https://blog.livedoor.jp/{blog_id}/",
        f"https://blog.ldblog.jp/{blog_id}/",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200 and "<title" in r.text.lower():
                soup = BeautifulSoup(r.text, "html.parser")
                tag = soup.find("title")
                if tag and tag.text:
                    name = _clean_title(tag.text)
                    if name:
                        return name[:200]
        except Exception:
            pass
    return None

def _derive_id_from_endpoint(endpoint: str) -> str | None:
    # 例: https://livedoor.blogcms.jp/atom/blog/<blogid>/entry
    m = re.search(r"/atom/(?:blog/)?([^/]+)/", endpoint or "")
    return m.group(1) if m else None

def run():
    app = create_app()
    with app.app_context():
        q = (
            db.session.query(ExternalBlogAccount)
            .filter(ExternalBlogAccount.blog_type == BlogType.LIVEDOOR)
            .filter((ExternalBlogAccount.blog_name == None) | (ExternalBlogAccount.blog_name == ""))  # noqa: E711
        )

        total = updated = skipped = 0
        for a in q.yield_per(100):
            total += 1
            blog_id = (a.livedoor_blog_id or "").strip()

            if not blog_id:
                # endpoint から拾えるなら補完
                endpoint = getattr(a, "atompub_endpoint", None) or getattr(a, "livedoor_endpoint", None)
                blog_id = _derive_id_from_endpoint(endpoint or "") or ""

            if not blog_id:
                skipped += 1
                continue

            title = _fetch_ld_title(blog_id)
            if title:
                a.blog_name = title
                db.session.add(a)
                updated += 1
        if updated:
            db.session.commit()
        print(f"[backfill_blog_name] total={total} updated={updated} skipped={skipped}")

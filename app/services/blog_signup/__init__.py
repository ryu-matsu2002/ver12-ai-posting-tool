# app/services/blog_signup/__init__.py
from .note_signup import register_note_account
from app.models import BlogType

def register_blog_account(site_id: int, blog_type: BlogType):
    if blog_type == BlogType.NOTE:
        return register_note_account(site_id)
    # 今後追加するなら:
    # elif blog_type == BlogType.HATENA:
    #     return register_hatena_account(site_id)
    else:
        raise ValueError(f"Unsupported blog type: {blog_type}")

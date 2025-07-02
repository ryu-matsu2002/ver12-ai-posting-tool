# app/services/blog_signup/__init__.py
from .note_signup import register_note_account
from app.models import BlogType

def register_blog_account(site_id: int, blog_type: BlogType):
    if blog_type == BlogType.NOTE:
        return register_note_account(site_id)
    raise ValueError(f"Blog type {blog_type} not supported yet.")


from .note_signup import signup_note_account

__all__ = ["signup_note_account"]

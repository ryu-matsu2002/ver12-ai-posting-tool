# app/services/blog_signup/note_signup.py
from app.models import ExternalBlogAccount, BlogType, db
from .base_signup import generate_random_email, generate_username, generate_password

def register_note_account(site_id: int) -> ExternalBlogAccount:
    email = generate_random_email()
    username = generate_username("note")
    password = generate_password()

    # ★ ここで Note のAPIまたは自動操作で実際の登録を行う（仮）

    # 登録されたアカウントを保存
    account = ExternalBlogAccount(
        site_id=site_id,
        blog_type=BlogType.NOTE,
        email=email,
        username=username,
        password=password,
        status="active"
    )
    db.session.add(account)
    db.session.commit()

    return account

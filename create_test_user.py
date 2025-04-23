from app import create_app, db
from app.models import User
from werkzeug.security import generate_password_hash

app = create_app()
app.app_context().push()

if not User.query.filter_by(email="test@example.com").first():
    db.session.add(User(
        email="test@example.com",
        password=generate_password_hash("password")
    ))
    db.session.commit()
    print("✅  test@example.com / password でログインできます")
else:
    print("✔  テストユーザーは既に存在します")

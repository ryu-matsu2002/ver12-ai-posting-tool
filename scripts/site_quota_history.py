# scripts/site_quota_history.py

from app import create_app, db
from app.models import User, SiteQuotaLog

app = create_app()

with app.app_context():
    users = User.query.filter_by(is_admin=False).all()

    for user in users:
        logs = (
            SiteQuotaLog.query
            .filter_by(user_id=user.id)
            .order_by(SiteQuotaLog.created_at.asc())
            .all()
        )
        if not logs:
            continue

        print(f"\n🧑 {user.last_name} {user.first_name}（user_id={user.id}）の枠履歴:")
        cumulative = 0
        for log in logs:
            cumulative += log.site_count
            print(f" - {log.created_at.strftime('%Y-%m-%d %H:%M:%S')} | {log.site_count:+} 件 | 計 {cumulative} 件 | {log.plan_type} | 理由：{log.reason}")

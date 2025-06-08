from __future__ import annotations
from datetime import timedelta

from flask import (
    Blueprint, render_template, redirect, url_for,
    flash, request, abort, g, jsonify, current_app, send_from_directory, session
)
from flask_login import (
    login_user, logout_user, login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from pytz import timezone
from sqlalchemy import asc, nulls_last

from . import db
from .models import User, Article, PromptTemplate, Site, Keyword
from .forms import (
    LoginForm, RegisterForm,
    GenerateForm, PromptForm, ArticleForm, SiteForm, 
    ProfileForm
)
from .article_generator import enqueue_generation
from .wp_client import post_to_wp, _decorate_html

# --- 既存の import の下に追加 ---
import re
import os
import logging
import openai
import threading
import datetime
from .image_utils import fetch_featured_image  # ← ✅ 正しい
from collections import defaultdict


from .article_generator import (
    _unique_title,
    _compose_body,
    _generate,
)
from app.forms import EditKeywordForm
from .forms import KeywordForm
from app.image_utils import _is_image_url

JST = timezone("Asia/Tokyo")
bp = Blueprint("main", __name__)

# 必要なら app/__init__.py で admin_bp を登録
admin_bp = Blueprint("admin", __name__)


@bp.route('/robots.txt')
def robots_txt():
    return send_from_directory('static', 'robots.txt')

@bp.route("/api/chat", methods=["POST"])
def chat_api():
    data = request.get_json()
    user_msg = data.get("message", "")
    username = data.get("username", "ユーザー")
    history = data.get("history", [])

    if not user_msg:
        return jsonify({"reply": "メッセージが空です。"})

    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # 履歴付きメッセージ
        messages = [{"role": "system", "content": f"あなたはVER12.AI-posting-tool『site craft』専属のAIアシスタントです。ユーザー（{username}さん）を名前で呼びながら、親しみやすくサポートしてください。"}]

        # 履歴追加
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            messages.append({"role": role, "content": content})

        # 今回の発言を最後に追加（すでにJS側で送ってるので重複しないよう注意）
        if not history or history[-1].get("content") != user_msg:
            messages.append({"role": "user", "content": user_msg})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=300,
            temperature=0.7
        )

        reply = response.choices[0].message.content.strip()
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"エラー：{str(e)}"})

import stripe
from app import db
from app.models import User, UserSiteQuota, PaymentLog

stripe_webhook_bp = Blueprint('stripe_webhook', __name__)

# ────────────── Webhook ハンドラ（通常購入／特別プラン両対応）
@stripe_webhook_bp.route("/stripe/webhook", methods=["POST"])
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get("stripe-signature")
    webhook_secret = current_app.config["STRIPE_WEBHOOK_SECRET"]

    # ✅ 追加: Webhookのペイロードをログ出力
    current_app.logger.info("📩 Stripe Webhook Received")
    current_app.logger.info(payload.decode("utf-8"))  # JSON形式で出力

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
    except stripe.error.SignatureVerificationError:
        current_app.logger.error("❌ Webhook signature verification failed")
        return "Webhook signature verification failed", 400
    except Exception as e:
        current_app.logger.error(f"❌ Error parsing webhook: {str(e)}")
        return f"Error parsing webhook: {str(e)}", 400

    # ✅ Stripe Checkout支払い完了（通常プラン）
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        metadata = session.get("metadata", {})
        user_id = metadata.get("user_id")
        site_count = int(metadata.get("site_count", 1))
        plan_type = metadata.get("plan_type", "affiliate")
        stripe_payment_id = session.get("payment_intent")

        # ✅ 重複チェック（最優先）
        existing = PaymentLog.query.filter_by(stripe_payment_id=stripe_payment_id).first()
        if existing:
            current_app.logger.warning("⚠️ Checkout Webhook: この支払いはすでに処理済みです")
            return jsonify({"message": "Already processed"}), 200

        # ✅ Quota 加算処理
        if user_id:
            user = User.query.get(int(user_id))
            if user:
                quota = UserSiteQuota.query.filter_by(user_id=user.id).first()
                if not quota:
                    quota = UserSiteQuota(
                        user_id=user.id, total_quota=0, used_quota=0, plan_type=plan_type
                    )
                    db.session.add(quota)

                quota.total_quota += site_count
                quota.plan_type = plan_type
                db.session.commit()

                current_app.logger.info(
                    f"✅ Checkout Webhook: user_id={user.id}, plan={plan_type}, site_count={site_count}"
                )
            else:
                current_app.logger.warning(f"⚠️ Checkout Webhook: user_id={user_id} のユーザーが見つかりません")
        else:
            current_app.logger.warning("⚠️ Checkout Webhook: metadata に user_id が含まれていません")

        # ✅ PaymentLog 保存処理
        email = session.get("customer_email")
        amount = session.get("amount_total") // 100  # セント → 円

        intent = stripe.PaymentIntent.retrieve(stripe_payment_id)
        charge_id = intent.get("latest_charge")
        charge = stripe.Charge.retrieve(charge_id)

        balance_tx_id = charge.get("balance_transaction")
        balance_tx = stripe.BalanceTransaction.retrieve(balance_tx_id)

        fee = balance_tx.fee // 100
        net = balance_tx.net // 100

        log = PaymentLog(
            user_id=user.id if user else None,
            email=email,
            amount=amount,
            fee=fee,
            net_income=net,
            plan_type=plan_type,
            stripe_payment_id=stripe_payment_id,
            status="succeeded"
        )
        db.session.add(log)
        db.session.commit()
        current_app.logger.info(f"💰 PaymentLog 保存（checkout）：{email} ¥{amount}")

    # ✅ special_purchase 成功時
    elif event["type"] == "payment_intent.succeeded":
        intent = event["data"]["object"]
        metadata = intent.get("metadata", {})
        user_id = metadata.get("user_id")
        site_count = int(metadata.get("site_count", 1))
        plan_type = metadata.get("plan_type", "affiliate")
        special = metadata.get("special", "no")
        stripe_payment_id = intent.get("id")

        # ✅ 重複チェック（最優先）
        existing = PaymentLog.query.filter_by(stripe_payment_id=stripe_payment_id).first()
        if existing:
            current_app.logger.warning("⚠️ PaymentIntent Webhook: この支払いはすでに処理済みです")
            return jsonify({"message": "Already processed"}), 200

        # ✅ Quota 加算処理
        if user_id:
            user = User.query.get(int(user_id))
            if user:
                quota = UserSiteQuota.query.filter_by(user_id=user.id).first()
                if not quota:
                    quota = UserSiteQuota(
                        user_id=user.id, total_quota=0, used_quota=0, plan_type=plan_type
                    )
                    db.session.add(quota)

                quota.total_quota += site_count
                quota.plan_type = plan_type
                db.session.commit()

                current_app.logger.info(
                    f"✅ Webhook: user_id={user.id}, plan={plan_type}, special={special}, site_count={site_count}"
                )
            else:
                current_app.logger.warning(f"⚠️ Webhook: user_id={user_id} のユーザーが見つかりません")
        else:
            current_app.logger.warning("⚠️ Webhook: metadata に user_id が含まれていません")

        # ✅ PaymentLog 保存処理
        amount = intent.get("amount") // 100
        email = intent.get("receipt_email") or intent.get("customer_email")

        charge_id = intent.get("latest_charge")
        charge = stripe.Charge.retrieve(charge_id)

        balance_tx_id = charge.get("balance_transaction")
        balance_tx = stripe.BalanceTransaction.retrieve(balance_tx_id)

        if not email and user:
            email = user.email

        fee = balance_tx.fee // 100
        net = balance_tx.net // 100

        log = PaymentLog(
            user_id=user.id if user else None,
            email=email,
            amount=amount,
            fee=fee,
            net_income=net,
            plan_type=plan_type,
            stripe_payment_id=stripe_payment_id,
            status="succeeded"
        )
        db.session.add(log)
        db.session.commit()
        current_app.logger.info(f"💰 PaymentLog 保存（special）：{email} ¥{amount}")

    return jsonify(success=True)


# Stripe APIキーを読み込み
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

# ────────────── create-payment-intent
@bp.route("/create-payment-intent", methods=["POST"])
def create_payment_intent():
    try:
        data = request.get_json()
        plan_type = data.get("plan_type", "affiliate")
        site_count = int(data.get("site_count", 1))
        user_id = int(data.get("user_id"))  # 必須
        special = data.get("special", "no")  # 特別プラン

        # 🔸 特別プランかどうかで価格を設定
        if special == "yes":
            unit_price = 1000
        else:
            unit_price = 3000 if plan_type == "affiliate" else 20000

        total_amount = unit_price * site_count

        # ✅ 修正：confirmation_method は削除！
        intent = stripe.PaymentIntent.create(
            amount=total_amount,
            currency="jpy",
            automatic_payment_methods={"enabled": True},
            payment_method_options={
                "card": {
                    "request_three_d_secure": "automatic"
                }
            },
            metadata={
                "user_id": str(user_id),
                "plan_type": plan_type,
                "site_count": str(site_count),
                "special": special
            }
        )

        return jsonify({"clientSecret": intent.client_secret})

    except Exception as e:
        current_app.logger.error(f"[create-payment-intent エラー] {e}")
        return jsonify(error=str(e)), 400


# ────────────── 通常購入ページ
@bp.route("/purchase", methods=["GET", "POST"])
@login_required
def purchase():
    if request.method == "POST":
        plan_type = request.form.get("plan_type")
        site_count = int(request.form.get("site_count", 1))

        if plan_type == "affiliate":
            price_id = os.getenv("STRIPE_PRICE_ID_AFFILIATE")
        elif plan_type == "business":
            price_id = os.getenv("STRIPE_PRICE_ID_BUSINESS")
        else:
            price_id = None

        if not price_id:
            flash("不正なプランが選択されました。", "error")
            return redirect(url_for("main.purchase"))

        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            customer_email=current_user.email,
            line_items=[{
                "price": price_id,
                "quantity": site_count,
            }],
            mode="payment" if plan_type == "affiliate" else "subscription",
            success_url=url_for("main.purchase", _external=True) + "?success=true",
            cancel_url=url_for("main.purchase", _external=True) + "?canceled=true",
            metadata={
                "user_id": current_user.id,
                "plan_type": plan_type,
                "site_count": site_count
            }
        )
        return redirect(session.url, code=303)

    return render_template("purchase.html")


# ────────────── 特別プランページ（テンプレート表示）
@bp.route("/<username>/special-purchase", methods=["GET"])
@login_required
def special_purchase(username):
    if current_user.username != username:
        abort(403)

    if not getattr(current_user, "is_special_access", False):
        flash("このページにはアクセスできません。", "danger")
        return redirect(url_for("main.dashboard", username=username))

    return render_template(
        "special_purchase.html",
        stripe_public_key=os.getenv("STRIPE_PUBLIC_KEY"),
        username=username
    )

# ────────────── 管理者ダッシュボード（セクション） ──────────────

from app.models import Article, User, PromptTemplate, Site
from os.path import exists, getsize

@admin_bp.route("/admin")
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash("このページにはアクセスできません。", "error")
        return redirect(url_for("main.dashboard", username=current_user.username))

    user_count    = User.query.count()
    site_count    = Site.query.count()
    prompt_count  = PromptTemplate.query.count()
    article_count = Article.query.count()
    users = User.query.all()

    missing_count_map = {}

    for user in users:
        articles = Article.query.filter(
            Article.user_id == user.id,
            Article.status.in_(["done", "posted", "error"]),
        ).all()

        missing = []
        for a in articles:
            url = a.image_url

            if not url or url.strip() in ["", "None"]:
                missing.append(a)

            elif url.startswith("/static/images/"):
                fname = url.replace("/static/images/", "")
                path = os.path.abspath(os.path.join("app", "static", "images", fname))
                if not fname or not exists(path) or getsize(path) == 0:
                    missing.append(a)

            elif url.startswith("http"):
                # 外部URLは期限切れの可能性があるため復元対象とする（HEADリクエストは行わない）
                missing.append(a)

        # 全ユーザーを記録（missing=0でも）
        missing_count_map[user.id] = len(missing)

    return render_template(
        "admin/dashboard.html",
        user_count=user_count,
        site_count=site_count,
        prompt_count=prompt_count,
        article_count=article_count,
        users=users,
        missing_count_map=missing_count_map
    )

# --- ダッシュボード強化系ルート ---

# 📊 統計サマリ（既存）
@admin_bp.route('/admin/dashboard')
@login_required
def admin_summary():
    return render_template("admin/dashboard.html")

# 🔄 処理中ジョブ一覧
@admin_bp.route("/admin/job-status")
@login_required
def job_status():
    processing_articles = Article.query.filter_by(status="gen").order_by(Article.created_at.desc()).all()
    return render_template("admin/job_status.html", articles=processing_articles)


# 🧠 API使用量／トークン分析
@admin_bp.route("/admin/api-usage")
@login_required
def api_usage():
    from app.models import TokenUsageLog, User
    from sqlalchemy import func

    # 日別集計（過去30日）
    today = datetime.utcnow().date()
    date_30_days_ago = today - timedelta(days=29)

    daily_data = (
        db.session.query(
            func.date(TokenUsageLog.created_at).label("date"),
            func.sum(TokenUsageLog.total_tokens).label("total_tokens")
        )
        .filter(TokenUsageLog.created_at >= date_30_days_ago)
        .group_by("date")
        .order_by("date")
        .all()
    )

    # ユーザー別集計（過去30日）
    user_data = (
        db.session.query(
            User.email,
            func.sum(TokenUsageLog.total_tokens).label("total_tokens")
        )
        .join(TokenUsageLog, TokenUsageLog.user_id == User.id)
        .filter(TokenUsageLog.created_at >= date_30_days_ago)
        .group_by(User.email)
        .order_by(func.sum(TokenUsageLog.total_tokens).desc())
        .all()
    )

    return render_template(
        "admin/api_usage.html",
        daily_data=daily_data,
        user_data=user_data
    )


# 💰 今月の売上＆取り分サマリ
@admin_bp.route("/admin/revenue-summary")
@login_required
def revenue_summary():
    from app.models import PaymentLog, User
    from sqlalchemy import func

    # 今月の開始日を取得（UTC）
    today = datetime.utcnow()
    first_day = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    # 今月の売上（成功した決済のみ）
    logs = (
        db.session.query(
            User.email,
            func.sum(PaymentLog.amount).label("total_amount"),
            func.count(PaymentLog.id).label("count")
        )
        .join(User, PaymentLog.user_id == User.id)
        .filter(PaymentLog.status == "succeeded")
        .filter(PaymentLog.created_at >= first_day)
        .group_by(User.email)
        .order_by(func.sum(PaymentLog.amount).desc())
        .all()
    )

    # 総売上
    total = sum(row.total_amount for row in logs)

    return render_template(
        "admin/revenue_summary.html",
        logs=logs,
        total=total
    )


# 📈 売上推移グラフ＋CSVダウンロード
# 📈 月別売上グラフ + CSVダウンロード
@admin_bp.route("/admin/revenue-graph")
@login_required
def revenue_graph():
    from app.models import PaymentLog
    from sqlalchemy import func
    from datetime import datetime, timedelta

    # 過去12ヶ月分の月次集計
    today = datetime.utcnow()
    first_day = today.replace(day=1) - timedelta(days=365)

    monthly_data = (
        db.session.query(
            func.to_char(PaymentLog.created_at, 'YYYY-MM').label("month"),
            func.sum(PaymentLog.amount).label("total")
        )
        .filter(PaymentLog.status == "succeeded")
        .filter(PaymentLog.created_at >= first_day)
        .group_by("month")
        .order_by("month")
        .all()
    )

    return render_template("admin/revenue_graph.html", monthly_data=monthly_data)

# 📥 CSVダウンロードルート
@admin_bp.route("/admin/download-revenue-log")
@login_required
def download_revenue_log():
    from app.models import PaymentLog, User
    import csv
    from io import StringIO
    from flask import Response

    logs = (
        db.session.query(
            PaymentLog.id,
            PaymentLog.amount,
            PaymentLog.status,
            PaymentLog.created_at,
            User.email
        )
        .join(User, PaymentLog.user_id == User.id)
        .order_by(PaymentLog.created_at.desc())
        .all()
    )

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Email", "金額（円）", "ステータス", "日時"])

    for log_id, amount, status, created_at, email in logs:
        writer.writerow([log_id, email, amount // 100, status, created_at.strftime("%Y-%m-%d %H:%M:%S")])

    output.seek(0)
    return Response(
        output,
        mimetype='text/csv',
        headers={"Content-Disposition": "attachment;filename=revenue_log.csv"}
    )



@admin_bp.route("/admin/users")
@login_required
def admin_users():
    if not current_user.is_admin:
        abort(403)

    users = User.query.order_by(User.id).all()

    # 統計情報の取得
    site_count    = Site.query.count()
    prompt_count  = PromptTemplate.query.count()
    article_count = Article.query.count()

    return render_template(
        "admin/users.html",
        users=users,
        site_count=site_count,
        prompt_count=prompt_count,
        article_count=article_count
    )
@admin_bp.route("/admin/user/<int:uid>")
@login_required
def admin_user_detail(uid):
    if not current_user.is_admin:
        abort(403)

    user = User.query.get_or_404(uid)

    return render_template("admin/user_detail.html", user=user)




@admin_bp.post("/admin/user/<int:uid>/toggle-special")
@login_required
def toggle_special_access(uid):
    # 管理者のみ許可
    if not current_user.is_admin:
        abort(403)

    # 対象ユーザー取得
    user = User.query.get_or_404(uid)

    # is_special_access をトグル（ON ⇔ OFF）
    user.is_special_access = not user.is_special_access
    db.session.commit()

    flash(f"{user.email} の特別アクセスを {'✅ 有効化' if user.is_special_access else '❌ 無効化'} しました。", "success")
    return redirect(url_for("admin.admin_users"))


@admin_bp.route("/admin/sites")
@login_required
def admin_sites():
    if not current_user.is_admin:
        flash("このページにはアクセスできません。", "error")
        return redirect(url_for("main.dashboard", username=current_user.username))

    from sqlalchemy import func
    from app.models import Site, Article, User
    from sqlalchemy import case

    # サイト情報と記事ステータス集計を取得
    result = (
    db.session.query(
        Site.id,
        Site.name,
        Site.url,
        User.email.label("user_email"),
        func.count(Article.id).label("total"),
        func.sum(case((Article.status == "done", 1), else_=0)).label("done"),
        func.sum(case((Article.status == "posted", 1), else_=0)).label("posted"),
        func.sum(case((Article.status == "error", 1), else_=0)).label("error"),
    )
    .join(User, Site.user_id == User.id)
    .outerjoin(Article, Site.id == Article.site_id)
    .group_by(Site.id, User.email)
    .all()
)

    return render_template("admin/sites.html", sites=result)

@admin_bp.post("/admin/delete-stuck-articles")
@login_required
def delete_stuck_articles():
    if not current_user.is_admin:
        abort(403)

    stuck = Article.query.filter(Article.status.in_(["pending", "gen"])).all()

    deleted_count = len(stuck)
    for a in stuck:
        db.session.delete(a)
    db.session.commit()

    flash(f"{deleted_count} 件の途中停止記事を削除しました", "success")
    return redirect(url_for("admin.admin_dashboard"))

@admin_bp.route("/admin/accounting", methods=["GET", "POST"])
@login_required
def accounting():
    if not current_user.is_admin:
        abort(403)

    from datetime import datetime
    from app.models import PaymentLog

    now = datetime.utcnow()
    now_year = now.year

    # パラメータ取得
    year_param = request.args.get("year", str(now.year))
    month_param = request.args.get("month", str(now.month))

    if year_param == "all":
        logs = PaymentLog.query.order_by(PaymentLog.created_at.desc()).all()
        selected_year = "all"
        selected_month = None
    else:
        year = int(year_param)
        month = int(month_param)
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1)
        else:
            end_date = datetime(year, month + 1, 1)

        logs = PaymentLog.query.filter(
            PaymentLog.created_at >= start_date,
            PaymentLog.created_at < end_date
        ).order_by(PaymentLog.created_at.desc()).all()

        selected_year = year
        selected_month = month

    # ✅ Stripeから取得済みの値をそのまま使用
    total_amount = sum(log.amount or 0 for log in logs)
    total_fee = sum(log.fee or 0 for log in logs)
    total_net = sum(log.net_income or 0 for log in logs)

    return render_template("admin/accounting.html",
        logs=logs,
        total_amount=total_amount,
        total_fee=total_fee,
        total_net=total_net,
        ryu_total=0,         # ← 分配しない場合は0
        take_total=0,
        expense_total=0,
        selected_year=selected_year,
        selected_month=selected_month,
        now_year=now_year
    )


@admin_bp.route("/admin/user/<int:uid>/articles")
@login_required
def user_articles(uid):
    if not current_user.is_admin:
        abort(403)
    user = User.query.get_or_404(uid)
    articles = Article.query.filter_by(user_id=uid).order_by(Article.created_at.desc()).all()

    # ✅ pending/gen の件数をカウント
    stuck_count = Article.query.filter(
        Article.user_id == uid,
        Article.status.in_(["pending", "gen"])
    ).count()

    return render_template("admin/user_articles.html", user=user, articles=articles, stuck_count=stuck_count)


@admin_bp.post("/admin/user/<int:uid>/delete-stuck")
@login_required
def delete_user_stuck_articles(uid):
    if not current_user.is_admin:
        abort(403)

    user = User.query.get_or_404(uid)

    stuck_articles = Article.query.filter(
        Article.user_id == uid,
        Article.status.in_(["pending", "gen"])
    ).all()

    count = len(stuck_articles)
    for art in stuck_articles:
        db.session.delete(art)
    db.session.commit()

    flash(f"{count} 件の途中停止記事を削除しました", "success")
    return redirect(url_for("admin.user_articles", uid=uid))

@admin_bp.post("/admin/login-as/<int:user_id>")
@login_required
def admin_login_as(user_id):
    if not current_user.is_admin:
        abort(403)

    user = User.query.get_or_404(user_id)
    login_user(user)
    flash(f"{user.email} としてログインしました", "info")
    return redirect(url_for("main.dashboard", username=current_user.username))


@admin_bp.post("/admin/fix-missing-images")
@login_required
def fix_missing_images():
    if not current_user.is_admin:
        abort(403)

    from app.image_utils import fetch_featured_image
    import re, os

    updated = 0
    articles = Article.query.filter(
        Article.status.in_(["done", "posted"])
    ).all()

    for art in articles:
        url = art.image_url or ""
        is_missing = (
            not url
            or url == "None"
            or not url.strip()
            or url.endswith("/")  # 不完全URL
            or ("/static/images/" in url and not os.path.exists(f"app{url}"))  # ローカルにファイルなし
            or ("/static/images/" in url and os.path.getsize(f"app{url}") == 0)  # サイズ0の破損画像
        )

        if not is_missing:
            continue

        match = re.search(r"<h2\b[^>]*>(.*?)</h2>", art.body or "", re.IGNORECASE)
        first_h2 = match.group(1) if match else ""
        query = f"{art.keyword} {first_h2}".strip()
        title = art.title or art.keyword or "記事"
        try:
            art.image_url = fetch_featured_image(query, title=title, body=art.body)
            updated += 1
        except Exception as e:
            current_app.logger.warning(f"[画像復元失敗] Article ID: {art.id}, Error: {e}")

    db.session.commit()
    flash(f"{updated} 件のアイキャッチ画像を復元しました。", "success")
    return redirect(url_for("admin.admin_dashboard"))



@admin_bp.route("/admin/delete_user/<int:user_id>", methods=["POST"])
@login_required
def delete_user(user_id):
    if not current_user.is_admin:
        abort(403)
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    flash("ユーザーを削除しました。", "success")
    return redirect(url_for("admin.admin_users"))



# ─────────── 管理者専用：アイキャッチ一括復元（ユーザー単位）
@admin_bp.route("/refresh-images/<int:user_id>")
@login_required
def refresh_images(user_id):
    if not current_user.is_admin:
        flash("管理者権限が必要です。", "danger")
        return redirect(url_for("main.dashboard", username=current_user.username))

    import re
    from app.image_utils import fetch_featured_image, DEFAULT_IMAGE_URL
    from app import db

    restored = 0
    failed = 0

    articles = Article.query.filter(
        Article.user_id == user_id,
        Article.status.in_(["done", "posted"]),
        (Article.image_url.is_(None)) | (Article.image_url == "") | (Article.image_url == "None")
    ).all()

    print(f"=== 対象記事数: {len(articles)}")

    for art in articles:
        try:
            match = re.search(r"<h2[^>]*>(.*?)</h2>", art.body or "", re.IGNORECASE)
            first_h2 = match.group(1) if match else ""
            query = f"{art.keyword} {first_h2}".strip() or art.title or art.keyword or "記事 アイキャッチ"
            title = art.title or art.keyword or "記事"

            print(f"🟡 記事ID={art.id}, クエリ='{query}'")

            new_url = fetch_featured_image(query, title=title)

            if new_url and new_url != DEFAULT_IMAGE_URL:
                art.image_url = new_url
                restored += 1
                print(f"✅ 復元成功 → {new_url}")
            else:
                failed += 1
                print(f"❌ 復元失敗（DEFAULT_IMAGE_URL）")

        except Exception as e:
            failed += 1
            print(f"🔥 Exception: {e}")
            continue

    db.session.commit()
    flash(f"✅ 復元完了: {restored} 件 / ❌ 失敗: {failed} 件", "info")
    return redirect(url_for("admin.admin_dashboard"))


@admin_bp.post("/admin/user/<int:uid>/regenerate-stuck")
@login_required
def regenerate_user_stuck_articles(uid):
    if not current_user.is_admin:
        abort(403)

    stuck_articles = Article.query.filter(
        Article.user_id == uid,
        Article.status.in_(["pending", "gen"])
    ).all()

    app = current_app._get_current_object()

    def _background_regeneration():
        with app.app_context():
            prompt = PromptTemplate.query.filter_by(user_id=uid).first()
            if not prompt:
                logging.warning(f"[再生成中止] user_id={uid} にプロンプトがありません")
                return

            for art in stuck_articles:
                try:
                    threading.Thread(
                        target=_generate,
                        args=(app, art.id, prompt.title_pt, prompt.body_pt),
                        daemon=True
                    ).start()
                except Exception as e:
                    logging.exception(f"[管理者再生成失敗] article_id={art.id} error={e}")

    threading.Thread(target=_background_regeneration, daemon=True).start()

    flash(f"{len(stuck_articles)} 件の途中停止記事を再生成キューに登録しました（バックグラウンド処理）", "success")
    return redirect(url_for("admin.user_articles", uid=uid))


# ────────────── キーワード ──────────────

@bp.route("/<username>/keywords", methods=["GET", "POST"])
@login_required
def keywords(username):
    if current_user.username != username:
        abort(403)

    form = KeywordForm()

    user_sites = Site.query.filter_by(user_id=current_user.id).all()
    form.site_id.choices = [(0, "―― サイトを選択 ――")] + [(s.id, s.name) for s in user_sites]

    if form.validate_on_submit():
        site_id = form.site_id.data
        if site_id == 0:
            flash("サイトを選択してください。", "danger")
            return redirect(url_for("main.keywords", username=username))

        lines = [line.strip() for line in form.keywords.data.splitlines() if line.strip()]
        for word in lines:
            keyword = Keyword(
                keyword=word,
                user_id=current_user.id,
                site_id=site_id
            )
            db.session.add(keyword)
        db.session.commit()
        flash(f"{len(lines)} 件のキーワードを追加しました", "success")
        return redirect(url_for("main.keywords", username=username, site_id=site_id))

    selected_site_id = request.args.get("site_id", type=int)
    status_filter = request.args.get("status")
    selected_site = Site.query.get(selected_site_id) if selected_site_id else None

    base_query = Keyword.query.filter_by(user_id=current_user.id)
    if selected_site_id:
        base_query = base_query.filter_by(site_id=selected_site_id)
    if status_filter == "used":
        base_query = base_query.filter_by(used=True)
    elif status_filter == "unused":
        base_query = base_query.filter_by(used=False)

    all_keywords = base_query.order_by(Keyword.site_id, Keyword.id.desc()).all()
    site_map = {s.id: s.name for s in user_sites}
    grouped_keywords = defaultdict(lambda: {"site_name": "", "keywords": [], "status_filter": status_filter})

    for kw in all_keywords:
        grouped_keywords[kw.site_id]["site_name"] = site_map.get(kw.site_id, "未設定")
        grouped_keywords[kw.site_id]["keywords"].append(kw)

    return render_template(
        "keywords.html",
        form=form,
        selected_site=selected_site,
        sites=user_sites,
        grouped_keywords=grouped_keywords,
        site_map=site_map
    )


@bp.route("/api/keywords/<int:site_id>")
@login_required
def api_unused_keywords(site_id):
    offset = request.args.get("offset", 0, type=int)
    limit = 40
    keywords = Keyword.query.filter_by(user_id=current_user.id, site_id=site_id, used=False)\
        .order_by(Keyword.id.desc()).offset(offset).limit(limit).all()
    return jsonify([{"id": k.id, "keyword": k.keyword, "used": k.used} for k in keywords])


@bp.route("/api/keywords/all/<int:site_id>")
@login_required
def api_all_keywords(site_id):
    keywords = Keyword.query.filter_by(user_id=current_user.id, site_id=site_id)\
        .order_by(Keyword.id.desc()).limit(1000).all()
    return jsonify([{"id": k.id, "keyword": k.keyword} for k in keywords])


@bp.route("/<username>/keywords/edit/<int:keyword_id>", methods=["GET", "POST"])
@login_required
def edit_keyword(username, keyword_id):
    if current_user.username != username:
        abort(403)
    keyword = Keyword.query.get_or_404(keyword_id)
    if keyword.user_id != current_user.id:
        abort(403)

    form = EditKeywordForm(obj=keyword)
    form.site_id.choices = [(s.id, s.name) for s in Site.query.filter_by(user_id=current_user.id).all()]

    if form.validate_on_submit():
        keyword.keyword = form.keyword.data.strip()
        keyword.site_id = form.site_id.data
        db.session.commit()
        flash("キーワードを更新しました", "success")
        return redirect(url_for("main.keywords", username=username))

    return render_template("edit_keyword.html", form=form)


@bp.route("/<username>/keywords/view/<int:keyword_id>")
@login_required
def view_keyword(username, keyword_id):
    if current_user.username != username:
        abort(403)
    keyword = Keyword.query.get_or_404(keyword_id)
    if keyword.user_id != current_user.id:
        abort(403)
    return render_template("view_keyword.html", keyword=keyword)


@bp.route("/<username>/keywords/delete/<int:keyword_id>")
@login_required
def delete_keyword(username, keyword_id):
    if current_user.username != username:
        abort(403)
    keyword = Keyword.query.get_or_404(keyword_id)
    if keyword.user_id != current_user.id:
        abort(403)
    db.session.delete(keyword)
    db.session.commit()
    flash("キーワードを削除しました。", "success")
    return redirect(url_for("main.keywords", username=username))


@bp.post("/<username>/keywords/bulk-action")
@login_required
def bulk_action_keywords(username):
    if current_user.username != username:
        abort(403)
    action = request.form.get("action")
    keyword_ids = request.form.getlist("keyword_ids")

    if not keyword_ids:
        flash("対象のキーワードが選択されていません。", "warning")
        return redirect(request.referrer or url_for("main.keywords", username=username))

    if action == "delete":
        Keyword.query.filter(
            Keyword.id.in_(keyword_ids),
            Keyword.user_id == current_user.id
        ).delete(synchronize_session=False)
        db.session.commit()
        flash("選択されたキーワードを削除しました。", "success")

    return redirect(request.referrer or url_for("main.keywords", username=username))

# ────────────── chatgpt ──────────────

@bp.route("/<username>/chatgpt")
@login_required
def chatgpt(username):
    if current_user.username != username:
        abort(403)
    return render_template("chatgpt.html")




# ─────────── 認証
@bp.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        identifier = form.identifier.data
        password = form.password.data

        user = User.query.filter(
            (User.email == identifier) | (User.username == identifier)
        ).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            flash("ログイン成功！", "success")
            return redirect(url_for("main.dashboard", username=user.username))
        else:
            flash("ログイン情報が正しくありません。", "danger")

    return render_template("login.html", form=form)



@bp.route("/register", methods=["GET", "POST"])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        # 登録用パスワードが正しいか
        if form.register_key.data != "tcctool":
            flash("登録専用パスワードが間違っています。", "danger")
            return render_template("register.html", form=form)

        # メールアドレスの重複チェック
        if User.query.filter_by(email=form.email.data).first():
            flash("このメールアドレスは既に登録されています。", "danger")
            return render_template("register.html", form=form)

        # ユーザー名（username）の重複チェック
        if User.query.filter_by(username=form.username.data).first():
            flash("このユーザー名はすでに使われています。", "danger")
            return render_template("register.html", form=form)

        # ユーザー作成・保存
        new_user = User(
            email=form.email.data,
            password=generate_password_hash(form.password.data),
            username=form.username.data,
            user_type=form.user_type.data,
            company_name=form.company_name.data,
            company_kana=form.company_kana.data,
            last_name=form.last_name.data,
            first_name=form.first_name.data,
            last_kana=form.last_kana.data,
            first_kana=form.first_kana.data,
            postal_code=form.postal_code.data,
            address=form.address.data,
            phone=form.phone.data
        )
        db.session.add(new_user)
        db.session.commit()

        flash("登録が完了しました。ログインしてください。", "success")
        return redirect(url_for(".login"))

    return render_template("register.html", form=form)



@bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for(".login"))

# ────────────── プロフィール ──────────────

@bp.route("/<username>/profile", methods=["GET", "POST"])
@login_required
def profile(username):
    if current_user.username != username:
        abort(403)

    form = ProfileForm(obj=current_user)

    if form.validate_on_submit():
        # ユーザー名が仮名（user123）のままで、変更された場合のみ許可
        if current_user.username.startswith("user") and form.username.data != current_user.username:
            # 重複チェック
            if User.query.filter_by(username=form.username.data).first():
                flash("このユーザー名はすでに使われています。", "danger")
                return render_template("profile.html", form=form)
            else:
                current_user.username = form.username.data

        # 基本情報の更新
        current_user.last_name  = form.last_name.data
        current_user.first_name = form.first_name.data
        current_user.last_kana  = form.last_kana.data
        current_user.first_kana = form.first_kana.data
        current_user.phone      = form.phone.data
        current_user.postal_code = form.postal_code.data
        current_user.address    = form.address.data  # ← 統合された住所フィールドに対応

        db.session.commit()
        flash("プロフィールを更新しました。", "success")

        return redirect(url_for("main.profile", username=current_user.username))

    return render_template("profile.html", form=form)


# ────────────── ツール本体コード ──────────────

@bp.route("/")
@login_required
def root_redirect():
    return redirect(url_for("main.dashboard", username=current_user.username))


# ─────────── Dashboard
from app.models import UserSiteQuota, Article  # 追加

@bp.route("/<username>/dashboard")
@login_required
def dashboard(username):
    if current_user.username != username:
        abort(403)

    # 記事統計
    g.total_articles = Article.query.filter_by(user_id=current_user.id).count()
    g.generating     = Article.query.filter(
        Article.user_id == current_user.id,
        Article.status.in_(["pending", "gen"])
    ).count()
    g.done   = Article.query.filter_by(user_id=current_user.id, status="done").count()
    g.posted = Article.query.filter_by(user_id=current_user.id, status="posted").count()
    g.error  = Article.query.filter_by(user_id=current_user.id, status="error").count()

    # ✅ user / quota 情報取得
    user = current_user
    quota = UserSiteQuota.query.filter_by(user_id=user.id).first()

    # ✅ 存在しない場合でも安全に表示
    plan_type   = quota.plan_type if quota else "未契約"
    total_quota = quota.total_quota if quota else 0
    used_quota  = len(user.sites)

    # ✅ 残り枠を明示（テンプレート側表示用に追加）
    remaining_quota = total_quota - used_quota

    return render_template(
    "dashboard.html",
    plan_type=plan_type,
    total_quota=total_quota,
    used_quota=used_quota,
    remaining_quota=remaining_quota,  # ← ✅ 追加
    total_articles=g.total_articles,
    done=g.done,
    posted=g.posted,
    error=g.error
    )

# ─────────── プロンプト CRUD（新規登録のみ）
@bp.route("/<username>/prompts", methods=["GET", "POST"])
@login_required
def prompts(username):
    if current_user.username != username:
        abort(403)

    form = PromptForm()

    if form.validate_on_submit():
        db.session.add(PromptTemplate(
            genre    = form.genre.data,
            title_pt = form.title_pt.data,
            body_pt  = form.body_pt.data,
            user_id  = current_user.id
        ))
        db.session.commit()
        flash("プロンプトを保存しました", "success")
        return redirect(url_for(".prompts", username=username))

    plist = PromptTemplate.query.filter_by(user_id=current_user.id).all()
    return render_template("prompts.html", form=form, prompts=plist)



# ─────────── プロンプト編集ページ（専用ページ）
@bp.route("/prompt/edit/<int:pid>", methods=["GET", "POST"])
@login_required
def edit_prompt(pid: int):
    pt = PromptTemplate.query.get_or_404(pid)
    if pt.user_id != current_user.id:
        abort(403)

    form = PromptForm(obj=pt)
    if form.validate_on_submit():
        pt.genre    = form.genre.data
        pt.title_pt = form.title_pt.data
        pt.body_pt  = form.body_pt.data
        db.session.commit()
        flash("プロンプトを更新しました", "success")
        return redirect(url_for(".prompts", username=current_user.username))

    return render_template("prompt_edit.html", form=form, prompt=pt)


# ─────────── プロンプト削除
@bp.post("/prompts/delete/<int:pid>")
@login_required
def delete_prompt(pid: int):
    pt = PromptTemplate.query.get_or_404(pid)
    if pt.user_id != current_user.id:
        abort(403)
    db.session.delete(pt)
    db.session.commit()
    flash("削除しました", "success")
    return redirect(url_for(".prompts", username=current_user.username))


# ─────────── プロンプト取得API（記事生成用）
@bp.route("/api/prompt/<int:pid>")
@login_required
def api_prompt(pid: int):
    pt = PromptTemplate.query.get_or_404(pid)
    if pt.user_id != current_user.id:
        abort(403)
    return jsonify({
        "title_pt": pt.title_pt,
        "body_pt": pt.body_pt
    })

@bp.route("/purchase-history")
@login_required
def purchase_history():
    user = current_user

    # SiteQuotaLogから登録枠履歴（すべてのプラン分）
    logs = SiteQuotaLog.query.filter_by(user_id=user.id).order_by(SiteQuotaLog.created_at.desc()).all()

    return render_template("purchase_history.html", logs=logs)

# ────────────── 登録サイト管理 ──────────────

from os import getenv
from app.forms import SiteForm
from app.models import SiteQuotaLog
from . import bp

@bp.route("/<username>/sites", methods=["GET", "POST"])
@login_required
def sites(username):
    if current_user.username != username:
        abort(403)

    form = SiteForm()
    user = current_user

    # 🔹 現在の登録済みサイト一覧と件数
    site_list = Site.query.filter_by(user_id=user.id).all()
    site_count = len(site_list)

    # 🔹 ユーザーが持つ全プランの登録枠データ
    quotas = UserSiteQuota.query.filter_by(user_id=user.id).all()

    # 🔹 プランごとの使用状況と履歴ログを集計
    quota_by_plan = {}
    for q in quotas:
        plan = q.plan_type
        used = q.used_quota or 0
        total = q.total_quota or 0
        remaining = max(total - used, 0)
        logs = SiteQuotaLog.query.filter_by(user_id=user.id, plan_type=plan).order_by(SiteQuotaLog.created_at.desc()).all()

        quota_by_plan[plan] = {
            "total": total,
            "used": used,
            "remaining": remaining,
            "logs": logs
        }

    # 🔹 全体としての合計上限
    total_quota = sum([q.total_quota for q in quotas])

    if form.validate_on_submit():
        if site_count >= total_quota:
            flash("サイト登録上限に達しています。追加するには課金が必要です。", "danger")
            return redirect(url_for("main.sites", username=username))

        # 🔹 選択されたプランを取得
        selected_plan = form.plan_type.data

        # 🔹 対象のプランの使用枠を1つ加算
        quota = UserSiteQuota.query.filter_by(user_id=user.id, plan_type=selected_plan).first()
        if quota:
            quota.used_quota += 1
        else:
            flash("プラン情報が見つかりません。", "danger")
            return redirect(url_for("main.sites", username=username))

        # 🔹 サイト登録（plan_typeを保存）
        db.session.add(Site(
            name       = form.name.data,
            url        = form.url.data.rstrip("/"),
            username   = form.username.data,
            app_pass   = form.app_pass.data,
            user_id    = user.id,
            plan_type  = selected_plan
        ))

        db.session.commit()
        flash("サイトを登録しました", "success")
        return redirect(url_for("main.sites", username=username))

    # 🔹 Stripeでの支払い履歴（※登録枠履歴とは別）
    history_logs = PaymentLog.query.filter_by(user_id=user.id).order_by(PaymentLog.created_at.desc()).all()

    return render_template(
        "sites.html",
        form=form,
        sites=site_list,
        plans=quota_by_plan,
        remaining_quota=total_quota - site_count,
        total_quota=total_quota,
        used_quota=site_count,
        history_logs=history_logs,
        stripe_public_key=getenv("STRIPE_PUBLIC_KEY")
    )

@bp.post("/<username>/sites/<int:sid>/delete")
@login_required
def delete_site(username, sid: int):
    if current_user.username != username:
        abort(403)

    site = Site.query.get_or_404(sid)
    if site.user_id != current_user.id:
        abort(403)

    db.session.delete(site)
    db.session.commit()
    flash("サイトを削除しました", "success")
    return redirect(url_for("main.sites", username=username))


@bp.route("/<username>/sites/<int:sid>/edit", methods=["GET", "POST"])
@login_required
def edit_site(username, sid: int):
    if current_user.username != username:
        abort(403)

    site = Site.query.get_or_404(sid)
    if site.user_id != current_user.id:
        abort(403)

    form = SiteForm(obj=site)

    if form.validate_on_submit():
        site.name     = form.name.data
        site.url      = form.url.data.rstrip("/")
        site.username = form.username.data
        site.app_pass = form.app_pass.data
        db.session.commit()
        flash("サイト情報を更新しました", "success")
        return redirect(url_for("main.log_sites", username=username))

    return render_template("site_edit.html", form=form, site=site)




# ─────────── 記事生成（ユーザー別）

@bp.route("/<username>/generate", methods=["GET", "POST"])
@login_required
def generate(username):
    if current_user.username != username:
        abort(403)

    form = GenerateForm()

    # ▼ プロンプトとサイトの選択肢をセット
    form.genre_select.choices = [(0, "― 使わない ―")] + [
        (p.id, p.genre)
        for p in PromptTemplate.query.filter_by(user_id=current_user.id)
    ]
    form.site_select.choices = [(0, "―― 選択 ――")] + [
        (s.id, s.name)
        for s in Site.query.filter_by(user_id=current_user.id)
    ]

    # ▼ クエリパラメータから事前選択されたsite_idとstatusを取得
    selected_site_id = request.args.get("site_id", type=int)
    status_filter = request.args.get("status")  # "used" / "unused" / None

    if request.method == "GET" and selected_site_id:
        form.site_select.data = selected_site_id

    # ▼ POST処理（記事生成）
    if form.validate_on_submit():
        kws = [k.strip() for k in form.keywords.data.splitlines() if k.strip()]
        site_id = form.site_select.data or None
        enqueue_generation(
            current_user.id,
            kws,
            form.title_prompt.data,
            form.body_prompt.data,
            site_id
        )
        flash(f"{len(kws)} 件をキューに登録しました", "success")
        return redirect(url_for("main.log_sites", username=username))

    # ▼ 表示するキーワード一覧を取得（statusフィルタも考慮）
    keyword_choices = []
    selected_site = None
    site_name = None

    if form.site_select.data:
        selected_site_id = form.site_select.data
        selected_site = Site.query.get(selected_site_id)
        site_name = selected_site.name if selected_site else ""

        keyword_query = Keyword.query.filter_by(
            user_id=current_user.id,
            site_id=selected_site_id
        )

        if status_filter == "used":
            keyword_query = keyword_query.filter_by(used=True)
        elif status_filter == "unused":
            keyword_query = keyword_query.filter_by(used=False)

        keyword_choices = keyword_query.order_by(Keyword.id.desc()).limit(1000).all()

    return render_template(
        "generate.html",
        form=form,
        keyword_choices=keyword_choices,
        selected_site=selected_site,
        site_name=site_name,
        status_filter=status_filter
    )

# ─────────── GSCルートコード

from app.google_client import fetch_search_queries
from app.models import Keyword  # 🔁 既存キーワード参照のため追加
from app.article_generator import enqueue_generation  # 🔁 忘れずに

@bp.route("/generate_from_gsc/<int:site_id>", methods=["GET", "POST"])
@login_required
def generate_from_gsc(site_id):
    site = Site.query.get_or_404(site_id)
    if site.user_id != current_user.id:
        abort(403)

    # ✅ GSC未接続のガード
    if not site.gsc_connected:
        flash("このサイトはまだSearch Consoleと接続されていません。", "danger")
        return redirect(url_for("main.gsc_connect"))

    try:
        rows = fetch_search_queries(site.url, days=7, row_limit=40)
        keywords = [row["keys"][0] for row in rows if "keys" in row]
    except Exception as e:
        flash(f"Search Consoleからキーワードの取得に失敗しました: {e}", "danger")
        return redirect(url_for("main.keywords", username=current_user.username))

    if not keywords:
        flash("検索クエリが見つかりませんでした。", "warning")
        return redirect(url_for("main.keywords", username=current_user.username))

    # ✅ 既存キーワードの重複チェック
    existing_keywords = set(
        k.keyword for k in Keyword.query.filter_by(site_id=site.id).all()
    )
    new_keywords = [kw for kw in keywords if kw not in existing_keywords]

    if not new_keywords:
        flash("すべてのキーワードが既に登録されています。", "info")
        return redirect(url_for("main.keywords", username=current_user.username))

    # ✅ GSC由来のキーワードとしてDBに追加
    for kw in new_keywords:
        db.session.add(Keyword(
            keyword=kw,
            site_id=site.id,
            user_id=current_user.id,
            source='gsc'
        ))

    # ✅ GSC接続状態を保存（初回のみ）※保険として残す
    if not site.gsc_connected:
        site.gsc_connected = True

    db.session.commit()

    # ✅ 記事生成キューへ
    enqueue_generation(new_keywords, site.id, current_user.id)

    flash(f"{len(new_keywords)}件のキーワードから記事生成を開始しました", "success")
    return redirect(url_for("main.keywords", username=current_user.username))


@bp.route("/gsc_generate", methods=["GET"])
@login_required
def gsc_generate():
    from .models import Site, Keyword

    # ✅ GSC連携済みサイトだけ取得
    gsc_sites = Site.query.filter_by(user_id=current_user.id, gsc_connected=True).all()

    # 初期状態：site_id がクエリにあるときだけ処理
    site_id = request.args.get("site_id", type=int)
    selected_site = None
    gsc_keywords = []

    if site_id:
        selected_site = Site.query.get_or_404(site_id)
        if selected_site.user_id != current_user.id:
            abort(403)

        # ✅ GSC由来キーワードのみ取得（source='gsc'）
        gsc_keywords = Keyword.query.filter_by(site_id=site_id, source='gsc').order_by(Keyword.id.desc()).all()

    return render_template(
        "gsc_generate.html",
        gsc_sites=gsc_sites,
        selected_site=selected_site,
        gsc_keywords=gsc_keywords
    )


# --- 既存インポートの下に追加（必要に応じて） ---
from flask import render_template, redirect, url_for, flash
from flask_login import login_required, current_user
from app.models import Site, db

# ✅ /gsc-connect: GSC連携ページの表示
# ✅ /gsc-connect: GSC連携ページの表示（トークンの有無で判定）
@bp.route("/gsc-connect")
@login_required
def gsc_connect():
    sites = Site.query.filter_by(user_id=current_user.id).all()

    # トークン取得
    from app.models import GSCAuthToken
    tokens = {
        token.site_id: token for token in GSCAuthToken.query.filter_by(user_id=current_user.id).all()
    }

    # 各サイトごとに is_token_connected フラグを付ける
    for site in sites:
        site.is_token_connected = site.id in tokens

    return render_template("gsc_connect.html", sites=sites)

# ✅ /connect_gsc/<site_id>: GSC連携フラグ設定（ダミー版・今後は不要）
@bp.route("/connect_gsc/<int:site_id>")
@login_required
def connect_gsc(site_id):
    site = Site.query.get_or_404(site_id)
    if site.user_id != current_user.id:
        flash("アクセス権がありません。", "danger")
        return redirect(url_for("main.gsc_connect", username=current_user.username))

    # （注意）本番では不要 → 本来は /authorize_gsc → /oauth2callback で接続
    site.gsc_connected = True
    db.session.commit()

    flash(f"サイト「{site.name}」とGoogleサーチコンソールの連携が完了しました。", "success")
    return redirect(url_for("main.gsc_connect", username=current_user.username))




# ─────────── 生成ログ
@bp.route("/<username>/log/site/<int:site_id>")
@login_required
def log(username, site_id):
    if current_user.username != username:
        abort(403)

    from collections import defaultdict
    from .article_generator import _generate_slots_per_site

    # ステータスフィルタを取得
    status = request.args.get("status")

    # 未スケジュール記事の slot を自動割当
    unscheduled = Article.query.filter(
        Article.user_id == current_user.id,
        Article.scheduled_at.is_(None),
    ).all()

    if unscheduled:
        site_map = defaultdict(list)
        for art in unscheduled:
            if art.site_id:
                site_map[art.site_id].append(art)

        for sid, articles in site_map.items():
            slots = iter(_generate_slots_per_site(current_app, sid, len(articles)))
            for art in articles:
                art.scheduled_at = next(slots)
        db.session.commit()

    # 記事取得クエリ
    q = Article.query.filter_by(user_id=current_user.id, site_id=site_id)
    if status:
        q = q.filter_by(status=status)
    q = q.order_by(
        nulls_last(asc(Article.scheduled_at)),
        Article.created_at.desc(),
    )

    site = Site.query.get_or_404(site_id)

    return render_template(
        "log.html",
        articles=q.all(),
        site=site,
        status=status,
        jst=JST
    )


# ─────────── ログ：サイト選択ページ（ユーザー別）
@bp.route("/<username>/log/sites")
@login_required
def log_sites(username):
    if current_user.username != username:
        abort(403)

    from sqlalchemy import func, case

    # サイトごとの記事集計
    result = (
        db.session.query(
            Site.id,
            Site.name,
            Site.url,
            func.count(Article.id).label("total"),
            func.sum(case((Article.status == "done", 1), else_=0)).label("done"),
            func.sum(case((Article.status == "posted", 1), else_=0)).label("posted"),
            func.sum(case((Article.status == "error", 1), else_=0)).label("error"),
        )
        .outerjoin(Article, Site.id == Article.site_id)
        .filter(Site.user_id == current_user.id)
        .group_by(Site.id)
        .all()
    )

    return render_template("log_sites.html", sites=result)



# ─────────── プレビュー
@bp.route("/preview/<int:article_id>")
@login_required
def preview(article_id: int):
    art = Article.query.get_or_404(article_id)
    if art.user_id != current_user.id:
        abort(403)
    styled = _decorate_html(art.body or "")
    return render_template("preview.html", article=art, styled_body=styled)


# ─────────── WordPress 即時投稿
@bp.post("/article/<int:id>/post")
@login_required
def post_article(id):
    art = Article.query.get_or_404(id)
    if art.user_id != current_user.id:
        abort(403)
    if not art.site:
        flash("投稿先サイトが設定されていません", "danger")
        return redirect(url_for(".log", site_id=art.site_id))

    try:
        url = post_to_wp(art.site, art)
        art.posted_at = datetime.datetime.utcnow()
        art.status = "posted"
        db.session.commit()
        flash(f"WordPress へ投稿しました: {url}", "success")
    except Exception as e:
        current_app.logger.exception("即時投稿失敗: %s", e)
        db.session.rollback()
        flash(f"投稿失敗: {e}", "danger")

    return redirect(url_for(".log", username=current_user.username, site_id=art.site_id))


# ─────────── 記事編集・削除・再試行
@bp.route("/article/<int:id>/edit", methods=["GET", "POST"])
@login_required
def edit_article(id):
    art = Article.query.get_or_404(id)
    if art.user_id != current_user.id:
        abort(403)
    form = ArticleForm(obj=art)
    if form.validate_on_submit():
        art.title = form.title.data
        art.body  = form.body.data
        db.session.commit()
        flash("記事を更新しました", "success")
        return redirect(url_for(".log", username=current_user.username, site_id=art.site_id))
    return render_template("edit_article.html", form=form, article=art)

@bp.post("/article/<int:id>/delete")
@login_required
def delete_article(id):
    art = Article.query.get_or_404(id)
    if art.user_id != current_user.id:
        abort(403)
    db.session.delete(art)
    db.session.commit()
    flash("記事を削除しました", "success")
    return redirect(url_for(".log", username=current_user.username, site_id=art.site_id))

@bp.post("/article/<int:id>/retry")
@login_required
def retry_article(id: int):
    art = Article.query.get_or_404(id)
    if art.user_id != current_user.id:
        abort(403)

    # ユーザーに紐づくプロンプトを取得（最初の1件）
    prompt = PromptTemplate.query.filter_by(user_id=current_user.id).first()
    if not prompt:
        flash("プロンプトテンプレートが見つかりません。先にプロンプトを作成してください。", "danger")
        return redirect(url_for(".prompts"))

    try:
        # ✅ タイトル生成（失敗時に例外を出すよう _unique_title() 側で調整済み）
        title = _unique_title(art.keyword, prompt.title_pt)
        if not title or title.strip() == "":
            raise ValueError("タイトルの生成に失敗しました")

        # ✅ アウトライン + 本文生成
        body = _compose_body(art.keyword, prompt.body_pt)
        if not body or body.strip() == "":
            raise ValueError("本文の生成に失敗しました")

        # ✅ アイキャッチ画像（optional）
        match = re.search(r"<h2\b[^>]*>(.*?)</h2>", body or "", re.IGNORECASE)
        first_h2 = match.group(1) if match else ""
        query = f"{art.keyword} {first_h2}".strip()
        image_url = fetch_featured_image(query)

        # ✅ 正常なデータとして保存
        art.title = title
        art.body = body
        art.image_url = image_url
        art.status = "done"
        art.progress = 100
        art.updated_at = datetime.utcnow()
        db.session.commit()

        flash("記事の再生成が完了しました", "success")

    except Exception as e:
        db.session.rollback()
        logging.exception(f"[再生成失敗] article_id={id} keyword={art.keyword} error={e}")
        flash("記事の再生成に失敗しました", "danger")

    return redirect(url_for(".log", site_id=art.site_id))

@bp.post("/articles/bulk-delete")
@login_required
def bulk_delete_articles():
    ids = request.form.getlist("selected_ids")
    if not ids:
        flash("削除する記事が選択されていません", "warning")
        return redirect(request.referrer or url_for(".dashboard"))

    for aid in ids:
        article = Article.query.get(int(aid))
        if article and article.user_id == current_user.id:
            db.session.delete(article)

    db.session.commit()
    flash(f"{len(ids)}件の記事を削除しました", "success")
    return redirect(request.referrer or url_for(".dashboard"))



@bp.route("/debug-post/<int:aid>")
@login_required
def debug_post(aid):
    art = Article.query.get_or_404(aid)
    if art.user_id != current_user.id:
        abort(403)
    try:
        url = post_to_wp(art.site, art)
        return f"SUCCESS: {url}"
    except Exception as e:
        return f"ERROR: {e}", 500
    
import requests
from app.models import GSCAuthToken, db
import datetime

# Google OAuth2 設定
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI")
GOOGLE_SCOPE = "https://www.googleapis.com/auth/webmasters.readonly"
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"


@bp.route("/authorize_gsc/<int:site_id>")
@login_required
def authorize_gsc(site_id):
    session["gsc_site_id"] = site_id  # 後でcallbackで参照するため保存
    auth_url = (
        f"{GOOGLE_AUTH_URL}?"
        f"response_type=code&client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={GOOGLE_REDIRECT_URI}"
        f"&scope={GOOGLE_SCOPE}&access_type=offline&prompt=consent"
    )
    return redirect(auth_url)


@bp.route("/oauth2callback")
@login_required
def oauth2callback():
    from app.models import Site

    code = request.args.get("code")
    if not code:
        flash("Google認証に失敗しました。", "danger")
        return redirect(url_for("main.gsc_connect"))

    site_id = session.get("gsc_site_id")
    site = Site.query.get_or_404(site_id)

    # トークン交換リクエスト
    data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code",
    }
    response = requests.post(GOOGLE_TOKEN_URL, data=data)
    if response.status_code != 200:
        flash("トークンの取得に失敗しました。", "danger")
        return redirect(url_for("main.gsc_connect"))

    tokens = response.json()
    access_token = tokens["access_token"]
    refresh_token = tokens.get("refresh_token")
    expires_in = tokens.get("expires_in", 3600)
    expiry = datetime.datetime.utcnow() + datetime.timedelta(seconds=expires_in)

    # 保存（存在する場合は更新）
    existing = GSCAuthToken.query.filter_by(site_id=site.id, user_id=current_user.id).first()
    if existing:
        existing.access_token = access_token
        existing.refresh_token = refresh_token
        existing.token_expiry = expiry
    else:
        new_token = GSCAuthToken(
            site_id=site.id,
            user_id=current_user.id,
            access_token=access_token,
            refresh_token=refresh_token,
            token_expiry=expiry,
        )
        db.session.add(new_token)

    site.gsc_connected = True
    db.session.commit()

    flash(f"サイト「{site.name}」とGoogle Search Consoleの接続に成功しました。", "success")
    return redirect(url_for("main.gsc_connect"))

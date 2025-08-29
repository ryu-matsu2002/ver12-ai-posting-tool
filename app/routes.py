from __future__ import annotations
from datetime import timedelta
import logging
logger = logging.getLogger(__name__)

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
from sqlalchemy.orm import selectinload

from app.extensions import func

from . import db
from .models import User, Article, PromptTemplate, Site, Keyword, Genre
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

# routes.py または api.py 内

from app.models import User, ChatLog, GSCConfig
from datetime import datetime

@bp.route("/api/chat", methods=["POST"])
def chat_api():
    data = request.get_json()
    user_msg = data.get("message", "").strip()
    username = data.get("username", "ユーザー")

    if not user_msg:
        return jsonify({"reply": "メッセージが空です。"})

    try:
        user = User.query.filter_by(username=username).first()
        if not user:
            return jsonify({"reply": "ユーザーが見つかりません。"})

        # 過去の履歴（最新10件）
        logs = ChatLog.query.filter_by(user_id=user.id).order_by(ChatLog.timestamp.desc()).limit(10).all()
        logs = list(reversed(logs))  # 時系列順にする

        # 会話履歴を構成
        messages = [
            {
                "role": "system",
                "content": f"あなたはVER12.AI-posting-tool『site craft』専属のAIアシスタントです。ユーザー（{username}さん）を名前で呼びながら、親しみやすくサポートしてください。"
            }
        ]
        for log in logs:
            messages.append({"role": log.role, "content": log.content})

        messages.append({"role": "user", "content": user_msg})

        # OpenAI呼び出し
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=300,
            temperature=0.7
        )
        reply = response.choices[0].message.content.strip()

        # ✅ DBに保存
        db.session.add(ChatLog(user_id=user.id, role="user", content=user_msg))
        db.session.add(ChatLog(user_id=user.id, role="assistant", content=reply))
        db.session.commit()

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"reply": f"エラー：{str(e)}"})


import stripe
from app import db
from app.models import User, UserSiteQuota, PaymentLog

stripe_webhook_bp = Blueprint('stripe_webhook', __name__)

@stripe_webhook_bp.route("/stripe/webhook", methods=["POST"])
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get("stripe-signature")
    webhook_secret = current_app.config["STRIPE_WEBHOOK_SECRET"]

    # ログ出力：受信記録
    current_app.logger.info("📩 Stripe Webhook Received")
    current_app.logger.info(payload.decode("utf-8"))

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
    except stripe.error.SignatureVerificationError:
        current_app.logger.error("❌ Webhook signature verification failed")
        return "Webhook signature verification failed", 400
    except Exception as e:
        current_app.logger.error(f"❌ Error parsing webhook: {str(e)}")
        return f"Error parsing webhook: {str(e)}", 400

    # PaymentIntent（通常購入も特別購入もここで処理）
    if event["type"] == "payment_intent.succeeded":
        intent = event["data"]["object"]
        metadata = intent.get("metadata", {})

        user_id = metadata.get("user_id")
        site_count = int(metadata.get("site_count", 1))
        plan_type = metadata.get("plan_type", "affiliate")
        special = metadata.get("special", "no")
        stripe_payment_id = intent.get("id")

        # 値のチェック
        if special not in ["yes", "no"]:
            current_app.logger.warning(f"⚠️ 無効な special フラグ：{special}")
            return jsonify({"message": "Invalid special flag"}), 400

        if not user_id:
            current_app.logger.warning("⚠️ metadata に user_id が含まれていません")
            return jsonify({"message": "Missing user_id"}), 400

        # SiteQuotaLogでの冪等性チェック（重複処理防止）
        existing_quota_log = SiteQuotaLog.query.filter_by(stripe_payment_id=stripe_payment_id).first()
        if existing_quota_log:
            current_app.logger.warning("⚠️ この支払いはすでにQuotaに反映済みです")
            return jsonify({"message": "Quota already granted"}), 200

        user = User.query.get(int(user_id))
        if not user:
            current_app.logger.warning(f"⚠️ user_id={user_id} のユーザーが見つかりません")
            return jsonify({"message": "User not found"}), 400

        # Quota加算処理
        quota = UserSiteQuota.query.filter_by(user_id=user.id).first()
        if not quota:
            quota = UserSiteQuota(user_id=user.id, total_quota=0, used_quota=0, plan_type=plan_type)
            db.session.add(quota)

        quota.total_quota += site_count
        quota.plan_type = plan_type
        db.session.commit()

        current_app.logger.info(
            f"✅ Quota加算: user_id={user.id}, plan={plan_type}, site_count={site_count}, special={special}"
        )

        # SiteQuotaLogに履歴を保存
        quota_log = SiteQuotaLog(
            user_id=user.id,
            stripe_payment_id=stripe_payment_id,
            site_count=site_count,
            reason="Stripe支払い"
        )
        db.session.add(quota_log)
        db.session.commit()

        # PaymentLog保存処理
        amount = intent.get("amount")   # ✅ 正確な小数で保持
        email = intent.get("receipt_email") or intent.get("customer_email")

        charge_id = intent.get("latest_charge")
        charge = stripe.Charge.retrieve(charge_id)
        balance_tx_id = charge.get("balance_transaction")
        balance_tx = stripe.BalanceTransaction.retrieve(balance_tx_id)

        if not email:
            email = user.email

        fee = balance_tx.fee # ✅ 小数で保持
        net = balance_tx.net

        log = PaymentLog(
            user_id=user.id,
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

        current_app.logger.info(f"💰 PaymentLog 保存：{email} ¥{amount}")

    return jsonify(success=True)


# Stripe APIキーを読み込み
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

# ────────────── create-payment-intent
@bp.route("/create-payment-intent", methods=["POST"])
def create_payment_intent():
    try:
        data = request.get_json()

        # ✅ 値の取得とバリデーション
        user_id = data.get("user_id")
        if user_id is None:
            raise ValueError("user_id is required")

        plan_type = data.get("plan_type", "affiliate")
        site_count = int(data.get("site_count", 1))
        special = data.get("special", "no")

        # ✅ special, plan_type のバリデーション
        if special not in ["yes", "no"]:
            raise ValueError(f"Invalid special value: {special}")
        if plan_type not in ["affiliate", "business"]:
            raise ValueError(f"Invalid plan_type: {plan_type}")

        user_id = int(user_id)  # ✅ int変換は後にする（エラー対処のため）

        # 🔸 特別プランかどうかで価格を設定
        if special == "yes":
            unit_price = 1000
        else:
            unit_price = 3000 if plan_type == "affiliate" else 20000

        total_amount = unit_price * site_count

        # ✅ Stripe PaymentIntent を作成
        intent = stripe.PaymentIntent.create(
            amount=total_amount,
            currency="jpy",
            automatic_payment_methods={"enabled": True},
            payment_method_options={
                "card": {
                    "request_three_d_secure": "any"
                }
            },
            metadata={  # ✅ Webhookで必要な情報をすべて埋め込む
                "user_id": str(user_id),
                "plan_type": plan_type,
                "site_count": str(site_count),
                "special": special
            }
        )

        # ✅ 成功ログ（デバッグしやすく）
        current_app.logger.info(
            f"✅ PaymentIntent 作成: user_id={user_id}, plan_type={plan_type}, site_count={site_count}, special={special}, amount={total_amount}"
        )

        return jsonify({"clientSecret": intent.client_secret})

    except Exception as e:
        import traceback
        current_app.logger.error(f"[create-payment-intent エラー] {e}")
        current_app.logger.error(traceback.format_exc())
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



import traceback

@admin_bp.route("/admin/sync-stripe-payments", methods=["POST"])
@login_required
def sync_stripe_payments():
    if not current_user.is_admin:
        abort(403)

    try:
        response = stripe.PaymentIntent.list(limit=100)
        data = response.data
        print(f"🔍 取得した決済件数: {len(data)}")

        for pi in data:
            payment_id = pi.id
            amount = pi.amount
            created_at = datetime.datetime.fromtimestamp(pi.created).strftime("%Y-%m-%d %H:%M")
            charge_id = pi.latest_charge

            email = (
                pi.get("receipt_email")
                or pi.get("customer_email")
                or "不明"
            )

            print(f"🧾 {created_at} | ¥{amount} | {payment_id} | email: {email} | チャージID: {charge_id}")

        return jsonify({"message": f"{len(data)} 件の決済を取得しました。ログを確認してください。"})

    except Exception as e:
        print("❌ エラー:", e)
        traceback.print_exc()
        return jsonify({"error": "処理中にエラーが発生しました"}), 500



@admin_bp.route("/admin/update-fee", methods=["POST"])
@login_required
def update_manual_fee():
    try:
        data = request.get_json()
        log_id = data.get("log_id")
        fee = data.get("manual_fee")

        if log_id is None or fee is None:
            return jsonify({"error": "不正なリクエスト"}), 400

        log = PaymentLog.query.get(log_id)
        if not log:
            return jsonify({"error": "該当するログが見つかりません"}), 404

        fee_int = int(fee)
        log.manual_fee = fee_int
        log.net_income = log.amount - fee_int  # ✅ 純利益を更新

        db.session.commit()

        return jsonify({"message": "手数料を保存しました"})
    except Exception as e:
        print("❌ 手数料保存中にエラー:", e)
        return jsonify({"error": "サーバーエラー"}), 500



# ────────────── 管理者ダッシュボード（セクション） ──────────────

from app.models import Article, User, PromptTemplate, Site
from os.path import exists, getsize

@admin_bp.route("/admin")
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash("このページにはアクセスできません。", "error")
        return redirect(url_for("main.dashboard", username=current_user.username))

    # ✅ 重い画像チェック処理を削除して即リダイレクト
    return redirect(url_for("admin.admin_users"))


@admin_bp.route("/admin/prompts")
@login_required
def admin_prompt_list():
    if not current_user.is_admin:
        abort(403)

    users = User.query.order_by(User.last_name, User.first_name).all()
    return render_template("admin/prompts.html", users=users)


@admin_bp.route("/admin/keywords")
@login_required
def admin_keyword_list():
    if not current_user.is_admin:
        abort(403)

    # 全ユーザー取得（first_name/last_name順で表示順が安定）
    users = User.query.order_by(User.last_name, User.first_name).all()
    return render_template("admin/keywords.html", users=users)


@admin_bp.route("/admin/gsc-status")
@login_required
def admin_gsc_status():
    if not current_user.is_admin:
        abort(403)

    from app.models import Site, Article, User, GSCConfig
    from sqlalchemy import case

    # 各サイトの投稿数・GSC設定を取得
    results = (
        db.session.query(
            Site.id,
            Site.name,
            Site.url,
            Site.plan_type,
            User.name.label("user_name"),
            func.count(Article.id).label("article_count"),
            func.max(GSCConfig.id).label("gsc_configured")
        )
        .join(User, Site.user_id == User.id)
        .outerjoin(Article, Article.site_id == Site.id)
        .outerjoin(GSCConfig, GSCConfig.site_id == Site.id)
        .group_by(Site.id, User.id)
        .order_by(func.count(Article.id).desc())
        .all()
    )

    return render_template("admin/gsc_status.html", results=results)


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

import subprocess
from flask import jsonify

@admin_bp.route("/admin/log-stream")
@login_required
def log_stream():
    """最新の system.log を読み込んでJSONで返す（最大30行）"""
    try:
        from app.utils.log_utils import parse_logs
        log_path = os.path.join("logs", "system.log")

        # ログファイルの末尾から最大30行を取得
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-30:]

        # 1行ごとに整形
        logs = parse_logs(lines)
        return jsonify({"logs": logs})

    except Exception as e:
        import traceback
        print("❌ log_stream failed:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)})




# 🧠 API使用量／トークン分析
@admin_bp.route("/admin/api-usage")
@login_required
def api_usage():
    from app.models import TokenUsageLog, User
    from datetime import datetime
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
    from datetime import datetime
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


# ─────────── 管理者：ジャンル管理（ユーザーごとのジャンル表示）
@admin_bp.route("/admin/genres", methods=["GET"])
@login_required
def manage_genres():
    if not current_user.is_admin:
        abort(403)

    from app.models import User  # 念のためUserをインポート
    users = User.query.order_by(User.last_name, User.first_name).all()

    return render_template("admin/genres.html", users=users)


@admin_bp.route("/admin/genres/delete/<int:genre_id>", methods=["POST"])
@login_required
def delete_genre(genre_id):
    if not current_user.is_admin:
        abort(403)

    genre = Genre.query.get_or_404(genre_id)
    db.session.delete(genre)
    db.session.commit()
    flash("ジャンルを削除しました", "info")
    return redirect(url_for("admin.manage_genres"))


@admin_bp.route("/admin/users", methods=["GET", "POST"])  # ✅ POST対応を追加
@login_required
def admin_users():
    if not current_user.is_admin:
        abort(403)

    # ✅ サイト枠追加リクエスト処理（POSTで来たときのみ）
    if request.method == "POST":
        if request.form.get("action") == "increase_quota":
            user_id = int(request.form.get("user_id"))
            plan_type = request.form.get("plan_type")

            # ✅ 該当ユーザー＆プランの枠を取得 or 作成
            quota = UserSiteQuota.query.filter_by(user_id=user_id, plan_type=plan_type).first()
            if quota:
                quota.total_quota += 1
            else:
                quota = UserSiteQuota(user_id=user_id, plan_type=plan_type, total_quota=1)
                db.session.add(quota)

            db.session.commit()
            flash("サイト枠を +1 しました", "success")

            return redirect(url_for("admin.admin_users"))

    # ✅ 必要最低限のユーザー情報のみ取得（→ Row形式 → dict形式に変換）
    raw_users = db.session.query(
        User.id,
        User.first_name,
        User.last_name,
        User.email,
        User.is_admin,
        User.is_special_access,
        User.created_at
    ).order_by(User.id).all()

    users = [
        {
            "id": u.id,
            "first_name": u.first_name,
            "last_name": u.last_name,
            "email": u.email,
            "is_admin": u.is_admin,
            "is_special_access": u.is_special_access,
            "created_at": u.created_at.strftime("%Y-%m-%d %H:%M") if u.created_at else "不明"
        }
        for u in raw_users
    ]

    site_count    = Site.query.count()
    prompt_count  = PromptTemplate.query.count()
    article_count = Article.query.count()

    # ✅ ここに追加
    stuck_counts = dict(
        db.session.query(
            Article.user_id,
            func.count()
        ).filter(
            Article.status.in_(["pending", "gen"])
        ).group_by(Article.user_id).all()
    )

    return render_template(
        "admin/users.html",
        users=users,  # ← JSONシリアライズ可能な形式に修正済み
        site_count=site_count,
        prompt_count=prompt_count,
        article_count=article_count,
        user_count=len(users),
        stuck_counts=stuck_counts
    )


@admin_bp.route("/api/admin/user_stats/<int:user_id>")
@login_required
def api_user_stats(user_id):
    if not current_user.is_admin:
        return jsonify({"error": "管理者権限が必要です"}), 403

    from collections import defaultdict

    # 🔸 記事数
    total_articles = db.session.query(func.count(Article.id)).filter_by(user_id=user_id).scalar()

    # 🔸 途中記事（pending / gen）
    stuck_articles = db.session.query(func.count(Article.id)).filter(
        Article.user_id == user_id,
        Article.status.in_(["pending", "gen"])
    ).scalar()

    # 🔸 サイト枠（UserSiteQuota と Site 使用数の差）
    quota_rows = db.session.query(
        UserSiteQuota.plan_type,
        UserSiteQuota.total_quota
    ).filter_by(user_id=user_id).all()

    site_counts = db.session.query(
        Site.plan_type,
        func.count(Site.id)
    ).filter_by(user_id=user_id).group_by(Site.plan_type).all()

    # 整形：plan_type → { used, total, remaining }
    summary = {}
    used_map = {pt: c for pt, c in site_counts}

    for plan_type, total_quota in quota_rows:
        used = used_map.get(plan_type, 0)
        remaining = max(total_quota - used, 0)
        summary[plan_type] = {
            "used": used,
            "total": total_quota,
            "remaining": remaining
        }

    return jsonify({
        "article_count": total_articles,
        "stuck_count": stuck_articles,
        "quota_summary": summary
    })



@admin_bp.route("/admin/user/<int:uid>")
@login_required
def admin_user_detail(uid):
    if not current_user.is_admin:
        abort(403)

    user = User.query.get_or_404(uid)

    # 関連情報をすべて取得
    sites = Site.query.filter_by(user_id=uid).all()
    prompts = PromptTemplate.query.filter_by(user_id=uid).all()
    keywords = Keyword.query.filter_by(user_id=uid).all()
    articles = Article.query.filter_by(user_id=uid).order_by(Article.created_at.desc()).limit(20).all()
    payments = PaymentLog.query.filter_by(user_id=uid).order_by(PaymentLog.created_at.desc()).all()

    return render_template(
        "admin/user_detail.html",
        user=user,
        sites=sites,
        prompts=prompts,
        keywords=keywords,
        articles=articles,
        payments=payments
    )


from app.forms import QuotaUpdateForm

@admin_bp.route("/admin/quota-edit/<int:uid>", methods=["GET", "POST"])
@login_required
def admin_quota_edit(uid):
    if not current_user.is_admin:
        abort(403)

    user = User.query.get_or_404(uid)
    form = QuotaUpdateForm()

    if form.validate_on_submit():
        plan_type = form.plan_type.data
        count = form.count.data

        # クォータ取得 or 作成
        quota = UserSiteQuota.query.filter_by(user_id=user.id, plan_type=plan_type).first()
        if not quota:
            quota = UserSiteQuota(user_id=user.id, plan_type=plan_type, total_quota=0, used_quota=0)
            db.session.add(quota)

        quota.total_quota += count

        log = SiteQuotaLog(
            user_id=user.id,
            plan_type=plan_type,
            site_count=count,
            reason="管理者手動追加",
            created_at = datetime.utcnow()
        )
        db.session.add(log)
        db.session.commit()

        flash(f"✅ {plan_type}プランに{count}枠追加しました", "success")
        return redirect(url_for("admin.admin_users"))

    return render_template("admin/quota_edit.html", user=user, form=form)



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

    from sqlalchemy import case, literal
    from app.models import Site, Article, User, Genre, GSCConfig
    from collections import defaultdict

    # 🔹 ジャンルID→ジャンル名の辞書を事前取得
    genre_dict = {g.id: g.name for g in Genre.query.all()}

    # 🔹 サイトごとの統計情報（投稿数など）＋GSC接続状態を取得
    raw = (
        db.session.query(
            Site.id,
            Site.name,
            Site.url,
            Site.plan_type,
            Site.genre_id,
            Site.user_id,
            func.concat(User.last_name, literal(" "), User.first_name).label("user_name"),
            func.count(Article.id).label("total"),
            func.sum(case((Article.status == "done", 1), else_=0)).label("done"),
            func.sum(case((Article.status == "posted", 1), else_=0)).label("posted"),
            func.sum(case((Article.status == "error", 1), else_=0)).label("error"),
            func.coalesce(Site.clicks, 0).label("clicks"),
            func.coalesce(Site.impressions, 0).label("impressions"),
            func.max(GSCConfig.id).isnot(None).label("gsc_connected")
        )
        .join(User, Site.user_id == User.id)
        .outerjoin(Article, Site.id == Article.site_id)
        .outerjoin(GSCConfig, Site.id == GSCConfig.site_id)
        .group_by(Site.id, User.id)
        .order_by(User.id, Site.id.desc())
        .all()
    )

    # 🔹 ユーザー単位でまとめてテンプレートに渡すための構造を構築
    sites_by_user = defaultdict(lambda: {"user_id": None, "sites": [], "genres": set()})

    for row in raw:
        user_name = row.user_name
        genre_id = row.genre_id
        genre_name = genre_dict.get(genre_id, "") if genre_id else ""

        # 各サイトの情報
        site_info = {
            "id": row.id,  # ← ✅ この行を追加してください
            "name": row.name,
            "url": row.url,
            "plan_type": row.plan_type,
            "total": row.total or 0,
            "done": row.done or 0,
            "posted": row.posted or 0,
            "error": row.error or 0,
            "clicks": row.clicks or 0,
            "impressions": row.impressions or 0,
            "genre": genre_name,
            "gsc_connected": bool(row.gsc_connected)  # ← GSC接続ラベルに正しく対応
        }

        # 初回時のみ user_id を登録
        if sites_by_user[user_name]["user_id"] is None:
            sites_by_user[user_name]["user_id"] = row.user_id

        # サイト情報を格納
        sites_by_user[user_name]["sites"].append(site_info)

        # ジャンル名があれば追加（重複回避のため set）
        if genre_name:
            sites_by_user[user_name]["genres"].add(genre_name)

    # 🔹 最終的に genres をソートされたリストに変換（select要素用）
    for user_data in sites_by_user.values():
        user_data["genres"] = sorted(user_data["genres"])

    # 🔹 テンプレートに渡す
    return render_template("admin/sites.html", sites_by_user=sites_by_user)

@admin_bp.route('/admin/delete_site/<int:site_id>', methods=['POST'])
@login_required
def delete_site(site_id):
    if not current_user.is_admin:
        abort(403)

    site = Site.query.get_or_404(site_id)

    # ✅ 関連記事削除
    Article.query.filter_by(site_id=site.id).delete()

    # ✅ 関連キーワード削除
    Keyword.query.filter_by(site_id=site.id).delete()

    # ✅ GSC 認証トークン削除
    GSCAuthToken.query.filter_by(site_id=site.id).delete()

    # ✅ GSC 設定データ削除
    GSCConfig.query.filter_by(site_id=site.id).delete()

    # ❌ アイキャッチ画像ファイルは残す（/static/images/...）

    # ❌ StripeやTokenログ等は削除しない（監査用）

    # ✅ 最後にサイト本体を削除
    db.session.delete(site)
    db.session.commit()

    flash('サイトと関連データ（記事・キーワード・GSC情報）を削除しました。', 'success')
    return redirect(url_for('admin.admin_sites'))  # ✅ 修正済み



@admin_bp.route("/admin/user/<int:uid>/bulk-delete", methods=["POST"])
@login_required
def bulk_delete_articles(uid):
    if not current_user.is_admin:
        abort(403)

    # pending または gen 状態の記事を一括削除
    Article.query.filter(
        Article.user_id == uid,
        Article.status.in_(["pending", "gen"])
    ).delete()

    db.session.commit()
    flash("✅ 途中状態の記事を一括削除しました", "success")
    return redirect(url_for("admin.user_articles", uid=uid))



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


from flask import render_template, request, redirect, url_for, flash, abort, current_app
from flask_login import login_required, current_user
from app.forms import RyunosukeDepositForm
from app.models import User, RyunosukeDeposit, Site, SiteQuotaLog, db
from collections import defaultdict
from datetime import datetime
from sqlalchemy.orm import selectinload, load_only
from sqlalchemy import func, extract
import time

@admin_bp.route("/admin/accounting", methods=["GET", "POST"])
@login_required
def accounting():
    if not current_user.is_admin:
        abort(403)

    selected_month = request.args.get("month", "all")

    # ✅ 入金フォーム処理（POST）
    form = RyunosukeDepositForm()
    if form.validate_on_submit():
        new_deposit = RyunosukeDeposit(
            deposit_date=form.deposit_date.data,
            amount=form.amount.data,
            memo=form.memo.data
        )
        db.session.add(new_deposit)
        db.session.commit()
        flash("龍之介の入金記録を保存しました", "success")
        return redirect(url_for("admin.accounting"))

    # ── 計測開始
    t0 = time.perf_counter()

    # ✅ 入金合計と残高
    paid_total = db.session.query(
        db.func.coalesce(db.func.sum(RyunosukeDeposit.amount), 0)
    ).scalar()
    current_app.logger.info("[/admin/accounting] paid_total in %.3fs", time.perf_counter()-t0); t0=time.perf_counter()

    # ✅ 全ユーザー＆関連情報を一括取得（N+1回避・sitesはロードしない）
    users = (
        User.query
        .options(
            load_only(User.id, User.first_name, User.last_name, User.is_admin, User.is_special_access),
        )
        .filter(User.is_admin == False)
        .all()
    )
    current_app.logger.info("[/admin/accounting] load users(+relations) in %.3fs", time.perf_counter()-t0); t0=time.perf_counter()

    # ✅ ユーザー分類＆サイト枠合計
    tcc_1000_total = 0
    tcc_3000_total = 0
    business_total = 0

    student_users = []
    member_users = []
    business_users = []

    for user in users:
        quota = user.site_quota
        if not quota or quota.total_quota == 0:
            continue
        if quota.plan_type == "business":
            business_total += quota.total_quota
            business_users.append(user)
        elif user.is_special_access or user.id == 16:
            tcc_1000_total += quota.total_quota
            member_users.append(user)
        else:
            tcc_3000_total += quota.total_quota
            student_users.append(user)

    current_app.logger.info("[/admin/accounting] classify users in %.3fs", time.perf_counter()-t0); t0=time.perf_counter()

    # ✅ 集計結果（現状の構成は完全維持）
    breakdown = {
        "unpurchased": {
            "count": tcc_3000_total,
            "ryu": tcc_3000_total * 1000,
            "take": tcc_3000_total * 2000,
        },
        "purchased": {
            "count": tcc_1000_total,
            "ryu": 0,
            "take": tcc_1000_total * 1000,
        },
        "business": {
            "count": business_total,
            "ryu": business_total * 16000,
            "take": business_total * 4000,
        },
        "total": {
            "count": tcc_3000_total + tcc_1000_total + business_total,
            "ryu": tcc_3000_total * 1000 + business_total * 16000,
            "take": tcc_3000_total * 2000 + tcc_1000_total * 1000 + business_total * 4000,
        },
    }

    # ✅ サイト登録データを月別にSQLで直接集計（join最適化＋NULL除外）
    site_data_raw = (
        db.session.query(
            func.date_trunc("month", Site.created_at).label("month"),
            func.count(Site.id)
        )
        .join(User, Site.user_id == User.id, isouter=False)
        .filter(
            Site.created_at.isnot(None),
            User.is_admin == False,
            User.is_special_access == False  # ← TCC研究生（3,000円）のみ
        )
        .group_by(func.date_trunc("month", Site.created_at))
        .all()
    )
    current_app.logger.info("[/admin/accounting] monthly site agg in %.3fs", time.perf_counter()-t0); t0=time.perf_counter()

    site_data_by_month = {}
    all_months_set = set()

    for month_obj, count in site_data_raw:
        month_key = month_obj.strftime("%Y-%m")
        all_months_set.add(month_key)
        site_data_by_month[month_key] = {
            "site_count": count,
            "ryunosuke_income": count * 1000,
            "takeshi_income": count * 2000
        }

    # ✅ 選択月のみ表示 or 全表示
    filtered_data = (
        site_data_by_month if selected_month == "all"
        else {
            selected_month: site_data_by_month.get(selected_month, {
                "site_count": 0,
                "ryunosuke_income": 0,
                "takeshi_income": 0
            })
        }
    )

    # ✅ 入金履歴と月一覧（変化なし）
    deposit_logs = RyunosukeDeposit.query.order_by(RyunosukeDeposit.deposit_date.desc()).all()
    current_app.logger.info("[/admin/accounting] load deposit_logs in %.3fs", time.perf_counter()-t0); t0=time.perf_counter()
    all_months = sorted(all_months_set, reverse=True)

    # ✅ テンプレートへ渡す（現状維持）
    resp = render_template(
        "admin/accounting.html",
        form=form,
        paid_total=paid_total,
        remaining=breakdown["unpurchased"]["ryu"] - paid_total,
        site_data_by_month=dict(sorted(filtered_data.items())),
        selected_month=selected_month,
        all_months=all_months,
        deposit_logs=deposit_logs,
        breakdown=breakdown,
        student_users=student_users,
        member_users=member_users,
        business_users=business_users
    )
    current_app.logger.info("[/admin/accounting] render_template in %.3fs", time.perf_counter()-t0)
    return resp


@admin_bp.route("/admin/accounting/details", methods=["GET"])
@login_required
def accounting_details():
    if not current_user.is_admin:
        abort(403)

    selected_month = request.args.get("month", "all")

    # ✅ 月一覧を抽出（NULLを除外して高速に）
    all_months_raw = (
        db.session.query(func.date_trunc("month", SiteQuotaLog.created_at))
        .filter(SiteQuotaLog.created_at.isnot(None))
        .distinct()
        .all()
    )

    all_months = sorted(
        {month[0].strftime("%Y-%m") for month in all_months_raw},
        reverse=True
    )

    # ✅ 月フィルタに応じてログ抽出
    logs_query = SiteQuotaLog.query.filter(SiteQuotaLog.created_at.isnot(None))

    if selected_month != "all":
        try:
            year, month = selected_month.split("-")
            logs_query = logs_query.filter(
                extract("year", SiteQuotaLog.created_at) == int(year),
                extract("month", SiteQuotaLog.created_at) == int(month)
            )
        except Exception:
            flash("不正な月形式です", "error")
            return redirect(url_for("admin.accounting_details"))

    # ✅ 並び順（新しい順）
    logs = logs_query.order_by(SiteQuotaLog.created_at.desc()).all()

    # ✅ テンプレートへ渡す（変化なし）
    return render_template(
        "admin/accounting_details.html",
        logs=logs,
        selected_month=selected_month,
        all_months=all_months
    )


@admin_bp.route("/admin/accounting/adjust", methods=["POST"])
@login_required
def adjust_quota():
    if not current_user.is_admin:
        abort(403)

    from flask import request, jsonify

    data = request.get_json()

    try:
        uid = int(data.get("uid"))
        delta = int(data.get("delta", 0))
    except (ValueError, TypeError):
        return jsonify({"error": "uid または delta の形式が不正です"}), 400

    if delta == 0:
        return jsonify({"error": "delta は 0 以外で指定してください"}), 400

    user = User.query.filter_by(id=uid, is_admin=False).first()
    if not user or not user.site_quota:
        return jsonify({"error": "対象ユーザーが見つかりません"}), 404

    quota = user.site_quota
    quota.total_quota = max(quota.total_quota + delta, 0)

    # ログ記録
    log = SiteQuotaLog(
        user_id=user.id,
        plan_type=quota.plan_type,
        site_count=delta,
        reason="管理者手動調整",
        created_at=datetime.utcnow()
    )
    db.session.add(log)
    db.session.commit()

    # ✅ 集計再構築
    stu_cnt = mem_cnt = biz_cnt = 0
    for u in User.query.filter_by(is_admin=False).all():
        sq = u.site_quota
        if not sq or sq.total_quota == 0:
            continue
        if sq.plan_type == "business":
            biz_cnt += sq.total_quota
        elif u.is_special_access:
            mem_cnt += sq.total_quota
        else:
            stu_cnt += sq.total_quota

    PRICES = {
        "student":  {"ryu": 1000,  "take": 2000},
        "member":   {"ryu": 0,     "take": 1000},
        "business": {"ryu": 16000, "take": 4000},
    }

    def calc(cnt, key):
        return {
            "count": cnt,
            "ryu": cnt * PRICES[key]["ryu"],
            "take": cnt * PRICES[key]["take"],
        }

    res_student  = calc(stu_cnt, "student")
    res_member   = calc(mem_cnt, "member")
    res_business = calc(biz_cnt, "business")

    res_total = {
        "count": stu_cnt + mem_cnt + biz_cnt,
        "ryu":   res_student["ryu"] + res_member["ryu"] + res_business["ryu"],
        "take":  res_student["take"] + res_member["take"] + res_business["take"]
    }

    return jsonify({
        "student":  res_student,
        "member":   res_member,
        "business": res_business,
        "total":    res_total,
        "message": f"✅ {user.last_name} {user.first_name} に {delta:+} 件 調整しました"
    })



# --- 既存: ユーザー全記事表示 ---
@admin_bp.route("/admin/user/<int:uid>/articles")
@login_required
def user_articles(uid):
    if not current_user.is_admin:
        abort(403)

    from collections import defaultdict
    from app.article_generator import _generate_slots_per_site
    from app.models import User, Article, Site
    from sqlalchemy.orm import selectinload
    from sqlalchemy import asc, nulls_last

    user = User.query.get_or_404(uid)
    status = request.args.get("status")
    sort_key = request.args.get("sort", "scheduled_at")
    sort_order = request.args.get("order", "desc")
    source = request.args.get("source", "all")

    # 🔹 未スケジュールの記事にslot自動割当
    unscheduled = Article.query.filter(
        Article.user_id == user.id,
        Article.scheduled_at.is_(None)
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

    # 🔹 記事取得クエリ
    q = Article.query.filter_by(user_id=user.id)
    if status:
        q = q.filter_by(status=status)
    if source == "gsc":
        q = q.filter_by(source="gsc")

    q = q.options(selectinload(Article.site))
    q = q.order_by(nulls_last(asc(Article.scheduled_at)), Article.created_at.desc())
    articles = q.all()

    # 🔽 並び替え（Python側）
    if sort_key == "clicks":
        articles.sort(key=lambda a: a.site.clicks or 0, reverse=(sort_order == "desc"))
    elif sort_key == "impr":
        articles.sort(key=lambda a: a.site.impressions or 0, reverse=(sort_order == "desc"))

    return render_template(
        "admin/user_articles.html",
        articles=articles,
        site=None,
        user=user,
        status=status,
        sort_key=sort_key,
        sort_order=sort_order,
        selected_source=source,
        jst=JST
    )


# --- ✅ 追加: サイト単位の記事一覧表示 ---
@admin_bp.route("/admin/site/<int:site_id>/articles")
@login_required
def site_articles(site_id):
    if not current_user.is_admin:
        abort(403)

    from app.models import Site, Article, User
    from sqlalchemy.orm import selectinload
    from sqlalchemy import asc, nulls_last

    site = Site.query.options(selectinload(Site.user)).get_or_404(site_id)
    user = site.user  # ✅ ここで site に紐づく正しい user を取得

    status = request.args.get("status")
    source = request.args.get("source", "all")

    q = Article.query.filter_by(site_id=site.id)
    if status:
        q = q.filter_by(status=status)
    if source == "gsc":
        q = q.filter_by(source="gsc")

    q = q.options(selectinload(Article.site))
    q = q.order_by(nulls_last(asc(Article.scheduled_at)), Article.created_at.desc())
    articles = q.all()

    return render_template(
        "admin/user_articles.html",
        articles=articles,
        site=site,
        user=user,  # ✅ この user は site に紐づいたもの
        status=status,
        sort_key=None,
        sort_order=None,
        selected_source=source,
        jst=JST
    )



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




@admin_bp.route("/admin/delete_user/<int:user_id>", methods=["POST"])
@login_required
def delete_user(user_id):
    if not current_user.is_admin:
        abort(403)

    user = User.query.get_or_404(user_id)

    db.session.delete(user)
    db.session.commit()

    flash("✅ ユーザーと関連データをすべて削除しました。", "success")
    return redirect(url_for("admin.admin_users"))


# ──────────────── GSCサイト状況一覧（管理者）────────────────
@admin_bp.route("/admin/gsc_sites")
@login_required
def admin_gsc_sites():
    if not current_user.is_admin:
        abort(403)

    from sqlalchemy.orm import selectinload
    from collections import defaultdict
    from app.models import Site, User, Keyword, Article

    # 全サイトをユーザー単位で取得（リレーション付きで最適化）
    users = User.query.options(selectinload(User.sites)).all()

    user_site_data = []

    for user in users:
        site_infos = []
        for site in user.sites:
            if not site.gsc_connected:
                continue  # GSC未接続サイトは除外

            # GSCキーワード全件
            keywords = Keyword.query.filter_by(site_id=site.id, source="gsc").all()
            done        = sum(1 for k in keywords if k.status == "done")
            generating  = sum(1 for k in keywords if k.status == "generating")
            unprocessed = sum(1 for k in keywords if k.status == "unprocessed")

            # 最新取得・生成日
            latest_keyword_date = max([k.created_at for k in keywords], default=None)

            # GSC記事の最新生成日（Article参照）
            latest_article = Article.query.filter_by(site_id=site.id, source="gsc").order_by(Article.created_at.desc()).first()
            latest_article_date = latest_article.created_at if latest_article else None

            site_infos.append({
                "site": site,
                "done": done,
                "generating": generating,
                "unprocessed": unprocessed,
                "total": done + generating + unprocessed,
                "latest_keyword_date": latest_keyword_date,
                "latest_article_date": latest_article_date
            })

        if site_infos:
            user_site_data.append({
                "user": user,
                "sites": site_infos
            })

    return render_template("admin/gsc_sites.html", user_site_data=user_site_data)


@admin_bp.get("/admin/user/<int:uid>/stuck-articles")
@login_required
def stuck_articles(uid):
    if not current_user.is_admin:
        abort(403)

    user = User.query.get_or_404(uid)

    stuck_articles = Article.query.filter(
        Article.user_id == uid,
        Article.status.in_(["pending", "gen"])
    ).order_by(Article.created_at.desc()).all()

    return render_template("admin/stuck_articles.html", user=user, articles=stuck_articles)


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
    return redirect(url_for("admin.stuck_articles", uid=uid))



# 先頭の import を修正
# 既存の import に追加（上の方）
from flask import Blueprint, request, jsonify, Response, redirect, url_for, render_template
from flask_login import login_required, current_user
from sqlalchemy import func, desc, asc
from datetime import datetime, timedelta
from app import db
from app.models import User, Site, Article

# ← これを先頭の import セクションに追加
from app.utils.monitor import (
    get_memory_usage,
    get_cpu_load,
    get_latest_restart_log,
    get_last_restart_time,
)

# ※ admin_bp は既存の Blueprint を使用

@admin_bp.route("/api/admin/rankings")  # ← フロントのfetch先が /api/admin/rankings ならこのまま
@login_required
def admin_rankings():
    # 非管理者ガード（未ログイン時でも安全）
    if not getattr(current_user, "is_admin", False):
        return jsonify({"error": "管理者のみアクセス可能です"}), 403

    # クエリ
    rank_type = (request.args.get("type") or "site").lower()
    order = (request.args.get("order") or "desc").lower()
    period = (request.args.get("period") or "3m").lower()
    start_date_str = request.args.get("start_date")
    end_date_str = request.args.get("end_date")

    # 24h エイリアス対応
    if period == "24h":
        period = "1d"

    sort_func = asc if order == "asc" else desc

    # 現在時刻（← 修正：datetime.utcnow()）
    now = datetime.utcnow()

    predefined_periods = {
        "1d":  now - timedelta(days=1),
        "7d":  now - timedelta(days=7),
        "28d": now - timedelta(days=28),
        "3m":  now - timedelta(days=90),
        "6m":  now - timedelta(days=180),
        "12m": now - timedelta(days=365),
        "16m": now - timedelta(days=480),
        "all": None,
    }

    # 期間決定
    if period == "custom":
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d") if start_date_str else None
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d") if end_date_str else now
        except ValueError:
            return jsonify({"error": "日付形式が不正です (YYYY-MM-DD)"}), 400
    else:
        start_date = predefined_periods.get(period, now - timedelta(days=90))
        end_date = now

    try:
        # ── ランキング種別ごと ─────────────────────────────
        if rank_type == "site":
            subquery = (
                db.session.query(
                    User.id.label("user_id"),
                    User.last_name,
                    User.first_name,
                    func.count(Site.id).label("site_count")
                )
                .outerjoin(Site, Site.user_id == User.id)
                .group_by(User.id, User.last_name, User.first_name)
                .subquery()
            )

            results = (
                db.session.query(
                    subquery.c.last_name,
                    subquery.c.first_name,
                    subquery.c.site_count
                )
                .order_by(sort_func(subquery.c.site_count))
                .all()
            )

            data = [
                {
                    "last_name": row.last_name,
                    "first_name": row.first_name,
                    "site_count": row.site_count or 0,
                }
                for row in results
            ]
            return Response(json.dumps(data, ensure_ascii=False), mimetype="application/json")

        elif rank_type in ("impressions", "clicks"):
            metric_column = Site.impressions if rank_type == "impressions" else Site.clicks
            query = (
                db.session.query(
                    Site.name.label("site_name"),
                    Site.url.label("site_url"),
                    User.last_name,
                    User.first_name,
                    metric_column.label("value")
                )
                .join(User, Site.user_id == User.id)
                .filter(metric_column.isnot(None))
            )
            results = query.order_by(sort_func(metric_column)).all()
            data = [
                {
                    "site_name": row.site_name,
                    "site_url": row.site_url,
                    "user_name": f"{row.last_name} {row.first_name}",
                    "value": row.value or 0,
                }
                for row in results
            ]
            return Response(json.dumps(data, ensure_ascii=False), mimetype="application/json")

        elif rank_type == "posted_articles":
            query = (
                db.session.query(
                    Site.name.label("site_name"),
                    Site.url.label("site_url"),
                    User.last_name,
                    User.first_name,
                    func.count(Article.id).label("value")
                )
                .join(User, Site.user_id == User.id)
                .join(Article, Article.site_id == Site.id)
                .filter(Article.status == "posted")
            )
            if start_date:
                query = query.filter(Article.posted_at >= start_date)
            if end_date:
                query = query.filter(Article.posted_at <= end_date)

            results = (
                query.group_by(Site.id, Site.name, Site.url, User.last_name, User.first_name)
                     .order_by(sort_func(func.count(Article.id)))
                     .all()
            )
            data = [
                {
                    "site_name": row.site_name,
                    "site_url": row.site_url,
                    "user_name": f"{row.last_name} {row.first_name}",
                    "value": row.value or 0,
                }
                for row in results
            ]
            return Response(json.dumps(data, ensure_ascii=False), mimetype="application/json")

        else:
            return jsonify({"error": "不正なランキングタイプです"}), 400

    except Exception as e:
        # 失敗時もJSONで返す（Networkで原因を見える化）
        current_app.logger.exception("[admin_rankings] server error")
        return jsonify({"error": "server_error", "detail": str(e)}), 500


@admin_bp.route("/admin/ranking-page")
@login_required
def admin_ranking_page():
    if not getattr(current_user, "is_admin", False):
        return redirect(url_for("main.dashboard", username=current_user.username))
    return render_template("admin/ranking_page.html")



# 監視ページ
@admin_bp.route("/admin/monitoring")
@login_required
def admin_monitoring():
    if not current_user.is_admin:
        return "アクセス拒否", 403

    memory = get_memory_usage()
    cpu = get_cpu_load()
    restart_logs = get_latest_restart_log()
    last_restart = get_last_restart_time()

    return render_template("admin/monitoring.html",
                           memory=memory,
                           cpu=cpu,
                           restart_logs=restart_logs,
                           last_restart=last_restart)


@admin_bp.route("/admin/captcha-dataset", methods=["POST"])
@login_required
def admin_captcha_label_update():
    from pathlib import Path

    image_file = request.form.get("image_file")
    new_label = request.form.get("label", "").strip()

    if not image_file or not new_label:
        return "無効な入力", 400

    dataset_dir = Path("data/captcha_dataset")
    txt_path = dataset_dir / Path(image_file).with_suffix(".txt")

    try:
        txt_path.write_text(new_label, encoding="utf-8")
        flash(f"{image_file} のラベルを更新しました。", "success")
    except Exception as e:
        flash(f"ラベル更新失敗: {e}", "danger")

    return redirect(url_for("admin.admin_captcha_dataset"))

@admin_bp.route("/admin/captcha-dataset", methods=["GET"])
@login_required
def admin_captcha_dataset():
    from pathlib import Path
    from flask import render_template

    # ✅ 学習用データ
    dataset_dir = Path("data/captcha_dataset")
    dataset_entries = []
    for path in sorted(dataset_dir.glob("*.png")):
        label_path = path.with_suffix(".txt")
        label = label_path.read_text(encoding="utf-8").strip() if label_path.exists() else ""
        dataset_entries.append({
            "image_file": path.name,
            "image_url": url_for('static', filename=f"../data/captcha_dataset/{path.name}"),
            "label": label
        })

    # ✅ 本番保存失敗画像（app/static/captchas）
    captchas_dir = Path("app/static/captchas")
    captcha_entries = []
    for path in sorted(captchas_dir.glob("*.png")):
        captcha_entries.append({
            "image_file": path.name,
            "image_url": url_for('static', filename=f"captchas/{path.name}"),
            "label": "（未設定）"
        })

    # ✅ 結合してテンプレートへ渡す
    entries = dataset_entries + captcha_entries

    return render_template("admin/captcha_dataset.html", entries=entries)




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

# 既存 import に追加
from flask import render_template, request, redirect, url_for, flash, session, current_app
from werkzeug.security import generate_password_hash
import secrets, time, unicodedata
from app.models import User
from app import db
from sqlalchemy import func
from app.forms import RealNameEmailResetRequestForm, PasswordResetSimpleForm


def _norm_name(s: str) -> str:
    # 全角/半角のゆらぎ吸収 + 空白除去（半角/全角スペース両方）
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    return s.replace(" ", "").replace("\u3000", "")  # 半角/全角スペース除去

@bp.route("/forgot-password", methods=["GET", "POST"])
def forgot_password_username_only():
    form = RealNameEmailResetRequestForm()
    if form.validate_on_submit():
        ln = form.last_name.data.strip()
        fn = form.first_name.data.strip()
        email = form.email.data.strip().lower()

        # メール一致のユーザーを取得（メールは小文字比較）
        user = User.query.filter(func.lower(User.email) == email).first()

        # 本名一致をサーバ側で厳密チェック（表記ゆれを軽減）
        if user and _norm_name(user.last_name) == _norm_name(ln) and _norm_name(user.first_name) == _norm_name(fn):
            grant = secrets.token_urlsafe(16)
            session["pw_reset_grant"] = {"uid": user.id, "grant": grant, "ts": time.time()}
            return redirect(url_for("main.reset_password_username_only", grant=grant))

        flash("本名とメールアドレスの組み合わせが確認できませんでした。", "danger")
        return render_template("forgot_username_only.html", form=form), 400

    return render_template("forgot_username_only.html", form=form)


# ---- Step2: 新パスワード設定
@bp.route("/reset-password-simple", methods=["GET", "POST"])
def reset_password_username_only():
    # TTL（秒）…未設定なら10分
    ttl = int(current_app.config.get("USERNAME_ONLY_RESET_TTL", 600))
    grant = request.args.get("grant") or request.form.get("grant")

    data = session.get("pw_reset_grant")
    if not data or data.get("grant") != grant or (time.time() - data.get("ts", 0)) > ttl:
        flash("操作が無効または期限切れです。最初からやり直してください。", "danger")
        session.pop("pw_reset_grant", None)
        return redirect(url_for("main.forgot_password_username_only"))

    user = User.query.get(data["uid"])
    if not user:
        flash("ユーザーが見つかりません。", "danger")
        session.pop("pw_reset_grant", None)
        return redirect(url_for("main.forgot_password_username_only"))

    form = PasswordResetSimpleForm()
    # hidden に grant を入れる
    if request.method == "GET":
        form.grant.data = grant

    if form.validate_on_submit():
        # ここまで来ていれば EqualTo で一致検証済み
        new_pw = form.password.data
        user.password = generate_password_hash(new_pw, method="pbkdf2:sha256", salt_length=16)
        db.session.commit()
        session.pop("pw_reset_grant", None)
        flash("パスワードを更新しました。新しいパスワードでログインしてください。", "success")
        return redirect(url_for("main.login"))

    return render_template("reset_username_only.html",
                           form=form, username=user.username, grant=grant)


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
from app.models import UserSiteQuota, Article, SiteQuotaLog, Site, User  # ← User を追加
from sqlalchemy import case
from flask import g
from collections import defaultdict

@bp.route("/<username>/dashboard")
@login_required
def dashboard(username):
    if current_user.username != username:
        abort(403)

    user = current_user

    # 🔸 記事統計（SQL1）
    article_stats = db.session.query(
        func.count(Article.id),
        func.sum(case((Article.status == "done", 1), else_=0)),
        func.sum(case((Article.status == "posted", 1), else_=0)),
        func.sum(case((Article.status == "error", 1), else_=0)),
        func.sum(case((Article.status.in_(["pending", "gen"]), 1), else_=0)),
    ).filter(Article.user_id == user.id).first()

    g.total_articles = article_stats[0]
    g.done = article_stats[1]
    g.posted = article_stats[2]
    g.error = article_stats[3]
    g.generating = article_stats[4]

    # 🔸 プラン別クォータ取得（SQL2）
    quotas = UserSiteQuota.query.filter_by(user_id=user.id).all()

    # 🔸 サイト使用状況を一括取得（SQL3）
    site_counts = db.session.query(
        Site.plan_type,
        func.count(Site.id)
    ).filter(Site.user_id == user.id).group_by(Site.plan_type).all()
    site_count_map = dict(site_counts)

    # 🔸 ログを一括取得（SQL4）
    # 🔸 ログを軽量取得（各プラン最大10件まで）
    log_map = defaultdict(list)
    for plan in set([q.plan_type for q in quotas]):
        logs = SiteQuotaLog.query.filter_by(user_id=user.id, plan_type=plan) \
            .order_by(SiteQuotaLog.created_at.desc()) \
            .limit(10).all()
        log_map[plan] = logs


    # 🔸 プラン別構成
    plans = {}
    for q in quotas:
        used = site_count_map.get(q.plan_type, 0)
        total = q.total_quota or 0
        remaining = max(total - used, 0)
        plans[q.plan_type] = {
            "used": used,
            "total": total,
            "remaining": remaining,
            "logs": log_map[q.plan_type]
        }

    total_quota = sum(q.total_quota for q in quotas)
    used_quota = sum(site_count_map.get(q.plan_type, 0) for q in quotas)
    remaining_quota = max(total_quota - used_quota, 0)


    return render_template(
        "dashboard.html",
        plan_type=quotas[0].plan_type if quotas else "未契約",
        total_quota=total_quota,
        used_quota=used_quota,
        remaining_quota=remaining_quota,
        total_articles=g.total_articles,
        done=g.done,
        posted=g.posted,
        error=g.error,
        plans=plans,
    )


# ─────────── Error Details
from app.models import Error  # ← Error モデルを追加
from flask import render_template, request

@bp.route("/<username>/view_errors")
@login_required
def view_errors(username):
    if current_user.username != username:
        abort(403)

    # ページ番号を取得
    page = request.args.get('page', 1, type=int)
    
    # エラー情報の取得（ページネーション対応）
    errors = Error.query.filter_by(user_id=current_user.id).order_by(Error.created_at.desc()).paginate(page=page, per_page=10)

    return render_template(
        "view_errors.html",
        errors=errors  # エラー詳細のリストをテンプレートに渡す
    )



@bp.route("/api/rankings")
@login_required
def api_rankings():
    rank_type = request.args.get("type", "site")

    if rank_type != "site":
        return jsonify({"error": "This endpoint only supports site rankings."}), 400

    # ✅ ユーザー別：登録サイト数ランキング（ダッシュボード用）
    excluded_user_ids = [1, 2, 14]  # ← 除外したいID
    subquery = (
        db.session.query(
            User.id.label("user_id"),
            User.last_name,
            User.first_name,
            func.count(Site.id).label("site_count")
        )
        .filter(~User.id.in_(excluded_user_ids))  # 🔥 ここを追加
        .outerjoin(Site, Site.user_id == User.id)
        .group_by(User.id, User.last_name, User.first_name)
        .subquery()
    )

    results = (
        db.session.query(
            subquery.c.last_name,
            subquery.c.first_name,
            subquery.c.site_count
        )
        .order_by(subquery.c.site_count.desc())
        .limit(50)
        .all()
    )

    data = [
        {
            "last_name": row.last_name,
            "first_name": row.first_name,
            "site_count": row.site_count
        }
        for row in results
    ]
    return jsonify(data)


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

@bp.route("/<username>/sites", methods=["GET", "POST"])
@login_required
def sites(username):
    if current_user.username != username:
        abort(403)

    form = SiteForm()
    user = current_user

    # ✅ ジャンルの選択肢をセット（自分が追加したジャンルのみ）
    genre_list = Genre.query.filter_by(user_id=current_user.id).order_by(Genre.name).all()
    form.genre_id.choices = [(0, "ジャンル未選択")] + [(g.id, g.name) for g in genre_list]

    # 🔹 登録済みサイト一覧と件数
    site_list = Site.query.filter_by(user_id=user.id).all()

    # 🔹 プランごとのクォータデータ
    quotas = UserSiteQuota.query.filter_by(user_id=user.id).all()

    # 🔹 プランごとのリアルタイム使用状況と履歴ログ
    quota_by_plan = {}
    for q in quotas:
        plan = q.plan_type
        used = Site.query.filter_by(user_id=user.id, plan_type=plan).count()  # ← 🔄 used_quotaをリアルタイムで算出
        total = q.total_quota or 0
        remaining = max(total - used, 0)
        logs = SiteQuotaLog.query.filter_by(user_id=user.id, plan_type=plan).order_by(SiteQuotaLog.created_at.desc()).all()

        quota_by_plan[plan] = {
            "total": total,
            "used": used,
            "remaining": remaining,
            "logs": logs
        }

    # 🔹 全体のトータル数と残数もリアルタイムで統一
    total_quota = sum([q.total_quota for q in quotas])
    used_quota = sum([Site.query.filter_by(user_id=user.id, plan_type=q.plan_type).count() for q in quotas])
    remaining_quota = total_quota - used_quota

    if form.validate_on_submit():
        if used_quota >= total_quota:
            flash("サイト登録上限に達しています。追加するには課金が必要です。", "danger")
            return redirect(url_for("main.sites", username=username))

        selected_plan = form.plan_type.data
        quota = UserSiteQuota.query.filter_by(user_id=user.id, plan_type=selected_plan).first()
        if quota:
            quota.used_quota = Site.query.filter_by(user_id=user.id, plan_type=selected_plan).count() + 1  # 🔄更新（念のため）
        else:
            flash("プラン情報が見つかりません。", "danger")
            return redirect(url_for("main.sites", username=username))

        db.session.add(Site(
            name       = form.name.data,
            url        = form.url.data.rstrip("/"),
            username   = form.username.data,
            app_pass   = form.app_pass.data,
            user_id    = user.id,
            plan_type  = selected_plan,
            genre_id   = form.genre_id.data if form.genre_id.data != 0 else None  # ✅
        ))

        db.session.commit()
        flash("サイトを登録しました", "success")
        return redirect(url_for("main.sites", username=username))

    # 🔹 Stripe履歴（参考表示用）
    history_logs = PaymentLog.query.filter_by(user_id=user.id).order_by(PaymentLog.created_at.desc()).all()
# 🔍 最初に優先表示するプラン（business優先）
    # 例：affiliate を優先して初期表示にする
    default_plan = "affiliate" if "affiliate" in quota_by_plan else "business"


    return render_template(
        "sites.html",
        form=form,
        sites=site_list,
        plans=quota_by_plan,
        remaining_quota=remaining_quota,  # ✅ ← 左上の表示に使用
        total_quota=total_quota,
        used_quota=used_quota,
        history_logs=history_logs,
        stripe_public_key=os.getenv("STRIPE_PUBLIC_KEY"),
        default_plan=default_plan  # ← 🔥追加！
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

    # ✅ 自分のジャンルだけを選択肢に含める
    genre_list = Genre.query.filter_by(user_id=current_user.id).order_by(Genre.name).all()
    form.genre_id.choices = [(0, "ジャンル未選択")] + [(g.id, g.name) for g in genre_list]

    # ✅ 初期値は GET のときだけ設定（POST時に上書きしない！）
    if request.method == "GET":
        form.genre_id.data = site.genre_id if site.genre_id is not None else 0

    if form.validate_on_submit():
        site.name       = form.name.data
        site.url        = form.url.data.rstrip("/")
        site.username   = form.username.data
        site.app_pass   = form.app_pass.data
        site.plan_type  = form.plan_type.data
        site.genre_id   = form.genre_id.data if form.genre_id.data != 0 else None

        db.session.commit()
        flash("サイト情報を更新しました", "success")
        return redirect(url_for("main.log_sites", username=username))

    else:
        if request.method == "POST":
            print("❌ バリデーションエラー:", form.errors)
            print("📌 ジャンルID:", form.genre_id.data)

    return render_template("site_edit.html", form=form, site=site)

@bp.route('/add_genre', methods=['POST'])
@login_required
def add_genre():
    data = request.get_json()
    name = data.get('name')
    description = data.get('description', '')

    if not name:
        return jsonify(success=False, error='Name required'), 400

    new_genre = Genre(name=name, description=description, user_id=current_user.id)
    db.session.add(new_genre)
    db.session.commit()

    return jsonify(success=True, genre_id=new_genre.id, genre_name=new_genre.name)


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

    # ▼ 件数カウント用の変数を初期化（デフォルトは0）
    total_count = 0
    used_count = 0
    unused_count = 0

    if form.site_select.data:
        selected_site_id = form.site_select.data
        selected_site = Site.query.get(selected_site_id)
        site_name = selected_site.name if selected_site else ""

        keyword_query = Keyword.query.filter_by(
            user_id=current_user.id,
            site_id=selected_site_id
        )

        # ▼ 件数カウント（フィルタなしで取得）
        all_keywords = keyword_query.all()
        total_count = len(all_keywords)
        used_count = sum(1 for kw in all_keywords if kw.used)
        unused_count = total_count - used_count

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
        status_filter=status_filter,
        total_count=total_count,      # ← 全体件数
        used_count=used_count,        # ← 生成済み件数
        unused_count=unused_count     # ← 未生成件数
    )

# ─────────── GSCルートコード

from app.google_client import fetch_search_queries_for_site
from app.models import Keyword  # 🔁 既存キーワード参照のため追加
from app.article_generator import enqueue_generation  # 🔁 忘れずに

#@bp.route("/generate_from_gsc/<int:site_id>", methods=["GET", "POST"])
#@login_required
#def generate_from_gsc(site_id):
    #site = Site.query.get_or_404(site_id)
    #if site.user_id != current_user.id:
       # abort(403)

    # ✅ GSC未接続のガード
    #if not site.gsc_connected:
        #flash("このサイトはまだSearch Consoleと接続されていません。", "danger")
        #return redirect(url_for("main.gsc_connect"))

    #try:
        #rows = fetch_search_queries(site.url, days=7, row_limit=40)
        #keywords = [row["keys"][0] for row in rows if "keys" in row]
    #except Exception as e:
        #flash(f"Search Consoleからキーワードの取得に失敗しました: {e}", "danger")
        #return redirect(url_for("main.keywords", username=current_user.username))

    #if not keywords:
        #flash("検索クエリが見つかりませんでした。", "warning")
        #return redirect(url_for("main.keywords", username=current_user.username))

    # ✅ 既存キーワードの重複チェック
    #existing_keywords = set(
        #k.keyword for k in Keyword.query.filter_by(site_id=site.id).all()
    #)
    #new_keywords = [kw for kw in keywords if kw not in existing_keywords]

    #if not new_keywords:
        #flash("すべてのキーワードが既に登録されています。", "info")
        #return redirect(url_for("main.keywords", username=current_user.username))

    # ✅ GSC由来のキーワードとしてDBに追加
    #for kw in new_keywords:
        #db.session.add(Keyword(
            #keyword=kw,
            #site_id=site.id,
            #user_id=current_user.id,
            #source='gsc'
        #))

    # ✅ GSC接続状態を保存（初回のみ）※保険として残す
    #if not site.gsc_connected:
        #site.gsc_connected = True

    #db.session.commit()

    # ✅ 記事生成キューへ
    #enqueue_generation(new_keywords, site.id, current_user.id)

    #flash(f"{len(new_keywords)}件のキーワードから記事生成を開始しました", "success")
    #return redirect(url_for("main.keywords", username=current_user.username))


@bp.route("/gsc_generate", methods=["GET", "POST"])
@login_required
def gsc_generate():
    from app.google_client import fetch_search_queries_for_site
    from app.article_generator import enqueue_generation
    from app.models import Keyword, PromptTemplate

    # --- POST（記事生成処理） ---
    if request.method == "POST":
        site_id = request.form.get("site_id", type=int)
        site = Site.query.get_or_404(site_id)

        if site.user_id != current_user.id:
            abort(403)

        # ✅ 追加：すでにGSC生成が始まっている場合は中止
        if site.gsc_generation_started:
            flash("⚠️ このサイトではすでにGSC記事生成が開始されています。", "warning")
            return redirect(url_for("main.gsc_generate", site_id=site_id))

        # ✅ 初回生成フラグをTrueにする（1回限りの起動）
        site.gsc_generation_started = True
        db.session.commit()

        prompt_id = request.form.get("prompt_id", type=int)
        title_prompt = request.form.get("title_prompt", "").strip()
        body_prompt = request.form.get("body_prompt", "").strip()

        if not site.gsc_connected:
            flash("このサイトはまだGSCと接続されていません。", "danger")
            return redirect(url_for("main.gsc_connect"))


        # GSCクエリ取得
        try:
            queries = fetch_search_queries_for_site(site, days=28, row_limit=1000)

            # 🔧 追加: 取得件数ログ
            current_app.logger.info(f"[GSC] {len(queries)} 件のクエリを取得 - {site.url}")

        except Exception as e:
            flash(f"GSCからのクエリ取得に失敗しました: {e}", "danger")
            return redirect(url_for("main.log_sites", username=current_user.username))

        # 重複排除
        # ✅ 既存キーワードのうち、status="done" のものは再利用不可として除外
        existing = set(
            k.keyword
            for k in Keyword.query.filter_by(site_id=site.id, source="gsc")
            if k.status == "done"
        )
        new_keywords = [q for q in queries if q not in existing]

        # 🔧 追加: 空 or 全重複の分岐で別メッセージ
        if not new_keywords:
            if not queries:
                flash("⚠️ GSCからクエリを取得できませんでした。URL形式が一致していない可能性があります。", "warning")
                current_app.logger.warning(f"[GSC] クエリが0件でした - {site.url}")
            else:
                flash("すべてのクエリが既に登録されています。", "info")
                current_app.logger.info(f"[GSC] 全クエリが既存のため登録スキップ - {site.url}")
            return redirect(url_for("main.log_sites", username=current_user.username))

        # DBに登録（source='gsc'）
        for kw in new_keywords:
            keyword = Keyword(site_id=site.id, keyword=kw, user_id=current_user.id, source="gsc")
            db.session.add(keyword)
        db.session.commit()

        # 🔸プロンプト取得（保存済みを優先）
        if prompt_id:
            saved_prompt = PromptTemplate.query.filter_by(id=prompt_id, user_id=current_user.id).first()
            if saved_prompt:
                title_prompt = saved_prompt.title_pt
                body_prompt = saved_prompt.body_pt

        # 🔁 記事生成をキューに追加
        enqueue_generation(
            user_id=current_user.id,
            site_id=site.id,
            keywords=new_keywords,
            title_prompt=title_prompt,
            body_prompt=body_prompt,
        )

        flash(f"{len(new_keywords)}件のGSCキーワードから記事生成を開始しました", "success")
        current_app.logger.info(f"[GSC] ✅ {len(new_keywords)} 件の記事生成キューを追加 - {site.url}")
        return redirect(url_for("main.log_sites", username=current_user.username))

    # --- GET（フォーム表示） ---
    site_id = request.args.get("site_id", type=int)
    if not site_id:
        flash("サイトIDが指定されていません。", "danger")
        return redirect(url_for("main.log_sites", username=current_user.username))

    site = Site.query.get_or_404(site_id)
    if site.user_id != current_user.id:
        abort(403)

    if not site.gsc_connected:
        flash("このサイトはまだGSCと接続されていません。", "danger")
        return redirect(url_for("main.gsc_connect"))
    
    # ✅ 追加: ステータスでフィルタリング
    status_filter = request.args.get("status")
    query = Keyword.query.filter_by(site_id=site.id, source="gsc")

    if status_filter in ["done", "unprocessed"]:
        query = query.filter(Keyword.status == status_filter)

    from app.models import Article, Keyword

# ✅ GSC由来の記事数（Keyword.source="gsc" に紐づく Article）
    # ✅ GSC記事数（JOIN ON 条件を明示）
    gsc_done = Article.query.filter_by(site_id=site.id, source="gsc").count()

# ✅ 全記事数（すべての Article）
    all_done = Article.query.filter_by(site_id=site.id).count()

# ✅ 通常記事数 = 全体 - GSC
    manual_done = all_done - gsc_done

# ✅ 合計・残り（上限：1000）
    total_done = gsc_done + manual_done
    remaining = max(1000 - total_done, 0)
    
    
    # ✅ フィルター前に全GSCキーワードを取得（件数用）
    gsc_done_keywords = Keyword.query.filter_by(site_id=site.id, source="gsc", status="done").count()
    gsc_pending_keywords = Keyword.query.filter_by(site_id=site.id, source="gsc", status="unprocessed").count()
    gsc_total_keywords = gsc_done_keywords + gsc_pending_keywords  # 🔧 合計を追加

    # ✅ 表示リスト用に再フィルタリング
    query = Keyword.query.filter_by(site_id=site.id, source="gsc")
    if status_filter == "done":
        query = query.filter(Keyword.status == "done")
    elif status_filter == "unprocessed":
        query = query.filter(Keyword.status != "done")
    gsc_keywords = query.order_by(Keyword.created_at.desc()).all()


    # 保存済みプロンプト
    saved_prompts = PromptTemplate.query.filter_by(user_id=current_user.id).order_by(PromptTemplate.genre).all()

    return render_template(
        "gsc_generate.html",
        selected_site=site,
        gsc_keywords=gsc_keywords,
        saved_prompts=saved_prompts,
        title_prompt="",  # 初期値
        body_prompt="",   # 初期値
        request=request,   # ✅ テンプレートでセレクトボックス選択保持に使う
        gsc_done=gsc_done,
        manual_done=manual_done,
        total_done=total_done,
        remaining=remaining,
        gsc_done_keywords=gsc_done_keywords,         # ✅ 追加
        gsc_pending_keywords=gsc_pending_keywords,    # ✅ 追加
        gsc_total_keywords=gsc_total_keywords  # 🔧 追加
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
    filter_status = request.args.get("status")  # "connected", "unconnected", "all"
    search_query = request.args.get("query", "").strip().lower()
    order = request.args.get("order")  # "recent", "most_views", "least_views"

    # ✅ クエリ構築（全件ベースで始める）
    sites_query = Site.query.filter_by(user_id=current_user.id)

    # ✅ 並び替え条件
    if order == "most_views":
        sites_query = sites_query.order_by(Site.impressions.desc())
    elif order == "least_views":
        sites_query = sites_query.order_by(Site.impressions.asc())
    else:
        sites_query = sites_query.order_by(Site.created_at.desc())  # デフォルト：新しい順

    sites = sites_query.all()

    # トークン取得
    from app.models import GSCAuthToken
    tokens = {token.site_id: token for token in GSCAuthToken.query.filter_by(user_id=current_user.id).all()}

    # ステータスフラグ付与
    for site in sites:
        site.is_token_connected = site.id in tokens

    # ✅ ステータスフィルター（Python側で処理）
    if filter_status == "connected":
        sites = [s for s in sites if s.gsc_connected]
    elif filter_status == "unconnected":
        sites = [s for s in sites if not s.gsc_connected]

    # ✅ 検索フィルター
    if search_query:
        sites = [s for s in sites if search_query in s.name.lower() or search_query in s.url.lower()]

    return render_template(
        "gsc_connect.html",
        sites=sites,
        filter_status=filter_status,
        search_query=search_query,
        order=order,
    )



@bp.route("/connect_gsc/<int:site_id>", methods=["POST"])
@login_required
def connect_gsc(site_id):
    site = Site.query.get_or_404(site_id)
    if site.user_id != current_user.id:
        flash("アクセス権がありません。", "danger")
        return redirect(url_for("main.gsc_connect"))

    site.gsc_connected = True
    db.session.commit()

    flash(f"✅ サイト「{site.name}」とGoogleサーチコンソールの接続が完了しました。", "success")
    return redirect(url_for("main.gsc_connect"))

# app/routes.py（末尾に追加）

# ✅ 必要なインポート
from flask import request, render_template  # ← Flaskの標準関数
from app.models import GSCMetric, Site      # ← GSCMetricを使ってDBから集計
from flask_login import login_required, current_user
from datetime import datetime, timedelta

# ✅ GSCアクセス分析ルート（ユーザー名不要に統一）
@bp.route("/gsc/<int:site_id>")  # ← ✅ ここを使用ルートに統一
@login_required
def gsc_analysis(site_id):
    # ✅ 対象ユーザーのサイトか確認
    site = Site.query.filter_by(id=site_id, user_id=current_user.id).first_or_404()

    # ✅ 未連携サイトは警告表示
    if not site.gsc_connected:
        return render_template("gsc_analysis.html", site=site, error="このサイトはGSCと未連携です")

    # ✅ GETパラメータ取得（range または start/end）
    range_param = request.args.get("range", "28d")
    start_param = request.args.get("start")
    end_param = request.args.get("end")

    today = datetime.utcnow().date()

    # ✅ 日付範囲の決定ロジック
    if range_param == "1d":
        start_date = today - timedelta(days=1)
    elif range_param == "7d":
        start_date = today - timedelta(days=7)
    elif range_param == "28d":
        start_date = today - timedelta(days=28)
    elif range_param == "3m":
        start_date = today - timedelta(days=90)
    elif range_param == "6m":
        start_date = today - timedelta(days=180)
    elif range_param == "12m":
        start_date = today - timedelta(days=365)
    elif range_param == "16m":
        start_date = today - timedelta(days=480)
    elif range_param == "custom" and start_param and end_param:
        try:
            start_date = datetime.strptime(start_param, "%Y-%m-%d").date()
            today = datetime.strptime(end_param, "%Y-%m-%d").date()
        except ValueError:
            return render_template(
                "gsc_analysis.html",
                site=site,
                error="日付形式が不正です"
            )
    else:
        # ✅ デフォルト28日
        start_date = today - timedelta(days=28)

    # ✅ データベースから該当期間のGSCメトリクスを取得
    metrics = GSCMetric.query.filter(
        GSCMetric.site_id == site_id,
        GSCMetric.date >= start_date,
        GSCMetric.date <= today
    ).order_by(GSCMetric.date.asc()).all()

    # ✅ テンプレートへデータ送信
    return render_template(
        "gsc_analysis.html",
        site=site,
        metrics=metrics,
        start_date=start_date,
        end_date=today,
        selected_range=range_param
    )


# ─────────── 生成ログ
@bp.route("/<username>/log/site/<int:site_id>")
@login_required
def log(username, site_id):
    if current_user.username != username:
        abort(403)

    from collections import defaultdict
    from .article_generator import _generate_slots_per_site

    # ステータス & ソートキー取得
    status = request.args.get("status")
    sort_key = request.args.get("sort", "scheduled_at")
    sort_order = request.args.get("order", "desc")

    # ✅ GSC絞り込み用パラメータ取得
    source = request.args.get("source", "all")

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

    if source == "gsc":
        q = q.filter_by(source="gsc")  # ✅ GSC記事のみフィルタ

    # 必ず site 情報も preload（clicks/impressions用）
    q = q.options(selectinload(Article.site))

    # 初期並び順：投稿予定日時優先
    q = q.order_by(
        nulls_last(asc(Article.scheduled_at)),
        Article.created_at.desc(),
    )

    articles = q.all()

    # 🔽 並び替え（Python側）
    if sort_key == "clicks":
        articles.sort(key=lambda a: a.site.clicks or 0, reverse=(sort_order == "desc"))
    elif sort_key == "impr":
        articles.sort(key=lambda a: a.site.impressions or 0, reverse=(sort_order == "desc"))

    site = Site.query.get_or_404(site_id)

    return render_template(
        "log.html",
        articles=articles,
        site=site,
        status=status,
        sort_key=sort_key,
        sort_order=sort_order,
        selected_source=source,  # ✅ フィルタUIの状態保持用
        jst=JST
    )



# ─────────── ログ：サイト選択ページ（ユーザー別）
@bp.route("/<username>/log/sites")
@login_required
def log_sites(username):
    if current_user.username != username:
        abort(403)

    from sqlalchemy import case
    from app.models import Genre

    # GETパラメータ
    status_filter = request.args.get("plan_type", "all")
    search_query = request.args.get("query", "").strip().lower()
    sort_key = request.args.get("sort", "created")
    sort_order = request.args.get("order", "asc")
    genre_id = request.args.get("genre_id", "0")
    try:
        genre_id = int(genre_id)
    except ValueError:
        genre_id = 0

    # ---------- サブクエリ（集計） ----------
    subquery = (
        db.session.query(
            Site.id.label("id"),
            Site.name.label("name"),
            Site.url.label("url"),
            Site.plan_type.label("plan_type"),
            Site.clicks.label("clicks"),
            Site.impressions.label("impressions"),
            Site.gsc_connected.label("gsc_connected"),
            Site.created_at.label("created_at"),
            func.count(Article.id).label("total"),
            func.sum(case((Article.status == "done", 1), else_=0)).label("done"),
            func.sum(case((Article.status == "posted", 1), else_=0)).label("posted"),
            func.sum(case((Article.status == "error", 1), else_=0)).label("error"),
        )
        .outerjoin(Article, Site.id == Article.site_id)
        .filter(Site.user_id == current_user.id)
        .group_by(Site.id)
    ).subquery()

    # ---------- メインクエリ（フィルター＆並び替え） ----------
    query = db.session.query(subquery)

    if status_filter in ["affiliate", "business"]:
        query = query.filter(subquery.c.plan_type == status_filter)

    if genre_id > 0:
        query = query.join(Site, Site.id == subquery.c.id).filter(Site.genre_id == genre_id)

    if search_query:
        query = query.filter(
            func.lower(subquery.c.name).like(f"%{search_query}%") |
            func.lower(subquery.c.url).like(f"%{search_query}%")
        )

    # 並び順カラム設定
    sort_columns = {
        "created": subquery.c.created_at,
        "total": subquery.c.total,
        "done": subquery.c.done,
        "posted": subquery.c.posted,
        "clicks": subquery.c.clicks,
        "impressions": subquery.c.impressions
    }
    order_column = sort_columns.get(sort_key, subquery.c.created_at)

    if sort_order == "desc":
        query = query.order_by(order_column.desc())
    else:
        query = query.order_by(order_column.asc())

    result = query.all()

    # ジャンル一覧
    genre_list = Genre.query.filter_by(user_id=current_user.id).order_by(Genre.name).all()
    genre_choices = [(0, "すべてのジャンル")] + [(g.id, g.name) for g in genre_list]

    return render_template(
        "log_sites.html",
        sites=result,
        selected_status=status_filter,
        selected_genre_id=genre_id,
        genre_choices=genre_choices,
        search_query=search_query,
        sort_key=sort_key,
        sort_order=sort_order
    )


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

# app/routes.py

@bp.route("/<username>/articles/<int:id>/retry", methods=["POST"])
@login_required
def retry_article(username, id):
    art = Article.query.get_or_404(id)
    if art.user_id != current_user.id or username != current_user.username:
        abort(403)

    if not art.title_prompt or not art.body_prompt:
        flash("この記事は再生成できません（プロンプト未保存）", "error")
        return redirect(url_for("main.view_articles", username=username))

    art.status = "pending"
    art.progress = 0
    art.updated_at = datetime.utcnow()
    db.session.commit()

    # バックグラウンドで再生成
    from app.article_generator import _generate
    app = current_app._get_current_object()
    threading.Thread(
        target=_generate,
        args=(app, art.id, art.title_prompt, art.body_prompt),
        daemon=True
    ).start()

    flash("記事の再生成を開始しました。しばらくお待ちください。", "success")
    return redirect(url_for("main.view_articles", username=username))



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

from app.forms import GenreForm

# ─────────── ジャンル管理ページ
@bp.route("/<username>/genres", methods=["GET", "POST"])
@login_required
def manage_genres(username):
    if current_user.username != username:
        abort(403)

    form = GenreForm()
    if form.validate_on_submit():
        # 🔹 既存ジャンル名と重複しないようにチェック（同一ユーザー内）
        existing = Genre.query.filter_by(user_id=current_user.id, name=form.name.data.strip()).first()
        if existing:
            flash("同じ名前のジャンルが既に存在します。", "warning")
        else:
            genre = Genre(
                name=form.name.data.strip(),
                description=form.description.data.strip(),
                user_id=current_user.id
            )
            db.session.add(genre)
            db.session.commit()
            flash("ジャンルを追加しました。", "success")
        return redirect(url_for("main.manage_genres", username=username))

    genres = Genre.query.filter_by(user_id=current_user.id).order_by(Genre.name).all()
    return render_template("genres.html", form=form, genres=genres)


# ─────────── ジャンル編集
@bp.route("/<username>/genres/edit/<int:genre_id>", methods=["GET", "POST"])
@login_required
def edit_genre(username, genre_id):
    if current_user.username != username:
        abort(403)

    genre = Genre.query.filter_by(id=genre_id, user_id=current_user.id).first_or_404()
    form = GenreForm(obj=genre)

    if form.validate_on_submit():
        genre.name = form.name.data.strip()
        genre.description = form.description.data.strip()
        db.session.commit()
        flash("ジャンルを更新しました。", "success")
        return redirect(url_for("main.manage_genres", username=username))

    return render_template("genres.html", form=form, genres=[], edit_genre=genre)


# ─────────── ジャンル削除
@bp.route("/<username>/genres/delete/<int:genre_id>", methods=["POST"])
@login_required
def delete_genre(username, genre_id):
    if current_user.username != username:
        abort(403)

    genre = Genre.query.filter_by(id=genre_id, user_id=current_user.id).first_or_404()
    db.session.delete(genre)
    db.session.commit()
    flash("ジャンルを削除しました。", "info")
    return redirect(url_for("main.manage_genres", username=username))


# -----------------------------------------------------------------
#────────── 外部SEO関連ルート ──────────
# -----------------------------------------------------------------

@bp.route("/external/sites")
@login_required
def external_seo_sites():
    from app.models import (
        Site, ExternalSEOJob, ExternalArticleSchedule,
        ExternalBlogAccount, BlogType, ExternalSEOJobLog
    )
    from app import db
    from sqlalchemy.orm import selectinload
    from sqlalchemy import func

    sites = (
        Site.query
        .filter_by(user_id=current_user.id)
        .options(
            selectinload(Site.external_jobs),
            selectinload(Site.external_accounts),
        )
        .all()
    )

    job_map, key_set = {}, set()
    for s in sites:
        for job in s.external_jobs:
            if job.status == "archived":
                continue
            key = (s.id, job.blog_type)
            key_set.add(key)
            job_map[(s.id, job.blog_type.value.lower())] = job

    posted_counts = (
        db.session.query(
            ExternalBlogAccount.site_id,
            ExternalBlogAccount.blog_type,
            func.count(ExternalArticleSchedule.id),
        )
        .join(
            ExternalArticleSchedule,
            ExternalArticleSchedule.blog_account_id == ExternalBlogAccount.id,
        )
        .filter(
            ExternalArticleSchedule.status == "posted",
            ExternalBlogAccount.site_id.in_([sid for sid, _ in key_set]) if key_set else True,
            ExternalBlogAccount.blog_type.in_([bt for _, bt in key_set]) if key_set else True,
        )
        .group_by(ExternalBlogAccount.site_id, ExternalBlogAccount.blog_type)
        .all()
    )
    for site_id, blog_type, cnt in posted_counts:
        key = (site_id, blog_type.value.lower())
        if key in job_map:
            job_map[key].posted_cnt = cnt
    for job in job_map.values():
        if not hasattr(job, "posted_cnt"):
            job.posted_cnt = 0

    # メトリクス集計（アカウント単位）
    all_ld_account_ids = []
    for s in sites:
        for acc in (s.external_accounts or []):
            if acc.blog_type == BlogType.LIVEDOOR:
                all_ld_account_ids.append(acc.id)

    per_acc_total, per_acc_posted = {}, {}
    if all_ld_account_ids:
        for aid, cnt in (
            db.session.query(
                ExternalArticleSchedule.blog_account_id,
                func.count(ExternalArticleSchedule.id),
            )
            .filter(ExternalArticleSchedule.blog_account_id.in_(all_ld_account_ids))
            .group_by(ExternalArticleSchedule.blog_account_id)
            .all()
        ):
            per_acc_total[aid] = cnt

        for aid, cnt in (
            db.session.query(
                ExternalArticleSchedule.blog_account_id,
                func.count(ExternalArticleSchedule.id),
            )
            .filter(
                ExternalArticleSchedule.blog_account_id.in_(all_ld_account_ids),
                ExternalArticleSchedule.status == "posted",
            )
            .group_by(ExternalArticleSchedule.blog_account_id)
            .all()
        ):
            per_acc_posted[aid] = cnt

    for s in sites:
        livedoor_accounts = []
        for acc in (s.external_accounts or []):
            if acc.blog_type != BlogType.LIVEDOOR:
                continue

            setattr(acc, "captcha_done", bool(getattr(acc, "is_captcha_completed", False)))
            setattr(acc, "email_verified", bool(getattr(acc, "is_email_verified", False)))

            livedoor_blog_id = getattr(acc, "livedoor_blog_id", None)
            setattr(acc, "blog_created", bool(livedoor_blog_id))
            setattr(acc, "api_key", getattr(acc, "atompub_key_enc", None))

            title = (
                getattr(acc, "nickname", None)
                or getattr(acc, "username", None)
                or livedoor_blog_id
                or f"account#{acc.id}"
            )
            setattr(acc, "blog_title", title)

            public_url = getattr(acc, "public_url", None)
            if not public_url and livedoor_blog_id:
                public_url = f"https://{livedoor_blog_id}.livedoor.blog/"
            setattr(acc, "public_url", public_url)

            total  = per_acc_total.get(acc.id, 0)
            posted = per_acc_posted.get(acc.id, 0)
            generated = max(total - posted, 0)
            setattr(acc, "stat_total", total)
            setattr(acc, "stat_posted", posted)
            setattr(acc, "stat_generated", generated)
            setattr(acc, "has_activity", total > 0)

            livedoor_accounts.append(acc)

        # ▼ livedoor_blog_id が同じものは 1 件に統合（NULLは統合しない）
        dedup_map = {}
        for acc in livedoor_accounts:
            key = getattr(acc, "livedoor_blog_id", None)
            if key is None:
                # blog_id 未確定はそのまま別カードとして扱う
                dedup_map[f"__id__:{acc.id}"] = acc
                continue

            prev = dedup_map.get(key)
            if not prev:
                dedup_map[key] = acc
                continue

            # どちらを残すか：APIキー > CAPTCHA済み > id新しい
            def score(a):
                return (
                    2 if getattr(a, "api_key", None) else
                    1 if getattr(a, "captcha_done", False) else
                    0,
                    getattr(a, "id", 0)
                )

            if score(acc) > score(prev):
                dedup_map[key] = acc

        dedup_list = list(dedup_map.values())
        s.livedoor_accounts = dedup_list
        s.ld_count = len(dedup_list)  # ← この値をテンプレの (n) に使う

    return render_template(
        "external_sites.html",
        sites=sites,
        job_map=job_map,
        ExternalSEOJobLog=ExternalSEOJobLog,
    )


@bp.post("/external/start")
@login_required
def start_external_seo() -> "Response":
    """
    HTMX から送られてくる

        site_id=<数字>&blog=<文字列>

    を受け取り、GPTベースのAIエージェントでアカウント作成を即時実行する。
    - blog=note → run_note_signup()
    - blog=hatena → run_hatena_signup()
    - blog=livedoor → run_livedoor_signup()
    """
    from flask import request, abort, jsonify, render_template
    from app.models import Site
    from app.enums import BlogType  # BlogType Enum
    from app.services.blog_signup import (
        note_signup,
        hatena_signup,
        livedoor_signup,
    )

    site_id = request.form.get("site_id", type=int)
    blog = (request.form.get("blog") or "").lower()

    if not site_id or not blog:
        return "site_id と blog は必須です", 400

    # BlogType Enum変換（存在しないblogなら400）
    try:
        blog_type = BlogType(blog)
    except ValueError:
        return "不正なブログタイプ", 400

    # サイト取得と所有権チェック（管理者はスキップ）
    site = Site.query.get_or_404(site_id)
    if (not current_user.is_admin) and (site.user_id != current_user.id):
        abort(403)

    # --- 🎯 GPTエージェントの実行 ---
    try:
        if blog_type == BlogType.NOTE:
            note_signup.signup(site)
        elif blog_type == BlogType.HATENA:
            hatena_signup.signup(site)
        elif blog_type == BlogType.LIVEDOOR:
            livedoor_signup.signup(site)
        else:
            return f"未対応のブログ: {blog}", 400
    except Exception as e:
        return f"AIエージェント失敗: {str(e)}", 500

    # HTMX対応
    if request.headers.get("HX-Request"):
        return render_template(
            "_job_progress.html",
            site_id=site_id,
            blog=blog_type.value,
            job=None
        )
    return jsonify(status="success")



# -----------------------------------------------------------------
# 外部SEO: 進捗パネル HTMX 用
# -----------------------------------------------------------------
@bp.route("/external/jobs/<int:site_id>")
@login_required
def external_seo_job_status(site_id):
    from app.models import ExternalSEOJob

    job = (ExternalSEOJob.query
           .filter_by(site_id=site_id)
           .order_by(ExternalSEOJob.id.desc())
           .first())

    return render_template("_job_progress.html",
                           job=job,
                           site_id=site_id)

# ──────────────────────────────────────────
# 外部SEO: 投稿スケジュール一覧表示
# ──────────────────────────────────────────
@bp.route("/external/schedules/<int:site_id>")
@login_required
def external_schedules(site_id):
    from app.models import ExternalArticleSchedule, Keyword, ExternalBlogAccount

    # blog_account_id を site_id で絞る
    schedules = (
        db.session.query(ExternalArticleSchedule, Keyword, ExternalBlogAccount)
        .join(Keyword, ExternalArticleSchedule.keyword_id == Keyword.id)
        .join(ExternalBlogAccount, ExternalArticleSchedule.blog_account_id == ExternalBlogAccount.id)
        .filter(ExternalBlogAccount.site_id == site_id)
        .order_by(ExternalArticleSchedule.scheduled_date.asc())
        .all()
    )
    return render_template("external_schedules.html",
                           schedules=schedules,
                           site_id=site_id)

from flask import send_file, make_response
from .services.blog_signup.crypto_utils import decrypt
from app.models import ExternalBlogAccount, BlogType
import asyncio, json, time


# -----------------------------------------------------------
# ユーザー向け: 自分の外部ブログアカウント一覧（検索・絞込・ソート対応）
# -----------------------------------------------------------

@bp.route("/external/accounts")
@login_required
def external_accounts():
    from app.models import ExternalBlogAccount, Site, ExternalArticleSchedule, BlogType
    from app.services.blog_signup.crypto_utils import decrypt
    from sqlalchemy import or_, func, case
    from sqlalchemy.orm import aliased

    blog_type = request.args.get("blog_type")
    sort      = request.args.get("sort")
    search    = request.args.get("q", "").strip()
    site_id   = request.args.get("site_id", type=int)

    # ベース: ログインユーザーのサイトに属する（site_id が NULL でも可）
    base = (
        db.session.query(ExternalBlogAccount.id)
        .outerjoin(Site, ExternalBlogAccount.site_id == Site.id)
        .filter(
            (ExternalBlogAccount.site_id == None) |  # noqa: E711
            (Site.user_id == current_user.id)
        )
    )
    if site_id:
        base = base.filter(ExternalBlogAccount.site_id == site_id)
    if blog_type:
        # Enum の可能性に配慮（文字列でも Enum でも比較できるように）
        try:
            bt = BlogType(blog_type)  # 文字列→Enum
            base = base.filter(ExternalBlogAccount.blog_type == bt)
        except Exception:
            base = base.filter(ExternalBlogAccount.blog_type == blog_type)

    if search:
        base = base.filter(or_(
            ExternalBlogAccount.email.ilike(f"%{search}%"),
            ExternalBlogAccount.nickname.ilike(f"%{search}%"),
            ExternalBlogAccount.username.ilike(f"%{search}%"),
        ))

    # 集計に使う別名（※ JOIN は schedule のみ。Keyword/Article には JOIN しない）
    S = aliased(ExternalArticleSchedule)

    # 各アカウント行ごとの集計（1アカウント=1行）
    # - total_cnt    : COUNT(DISTINCT S.id)
    # - posted_cnt   : SUM(CASE WHEN S.status='posted' THEN 1 ELSE 0 END)
    # - generated_cnt: SUM(CASE WHEN S.article_id IS NOT NULL THEN 1 ELSE 0 END)
    per_acc_rows = (
        db.session.query(
            ExternalBlogAccount,
            func.count(func.distinct(S.id)).label("total_cnt"),
            func.sum(case((S.status == "posted", 1), else_=0)).label("posted_cnt"),
            func.sum(case((S.article_id != None, 1), else_=0)).label("generated_cnt")  # noqa: E711
        )
        .select_from(ExternalBlogAccount)
        .outerjoin(S, S.blog_account_id == ExternalBlogAccount.id)
        .outerjoin(Site, ExternalBlogAccount.site_id == Site.id)
        .filter(base.whereclause)  # ベースのフィルタを適用
        .group_by(ExternalBlogAccount.id)
        .all()
    )

    # （blog_type, blog_id）でユニーク化し、代表1件に集計を合算
    def score(acc):
        # 代表選定優先度: APIキー > CAPTCHA済み > id
        return (
            2 if getattr(acc, "atompub_key_enc", None) else
            1 if getattr(acc, "is_captcha_completed", False) else
            0,
            getattr(acc, "id", 0)
        )

    groups = {}  # key -> {"repr": acc, "total":int, "posted":int, "generated":int, "raw":[acc,...]}
    for acc, total_cnt, posted_cnt, generated_cnt in per_acc_rows:
        key_blog_id = acc.livedoor_blog_id or f"__id__:{acc.id}"
        key = (acc.blog_type, key_blog_id)
        g = groups.get(key)
        total_i     = int(total_cnt or 0)
        posted_i    = int(posted_cnt or 0)
        generated_i = int(generated_cnt or 0)

        if not g:
            groups[key] = {
                "repr": acc,
                "total": total_i,
                "posted": posted_i,
                "generated": generated_i,
                "raw": [acc],
            }
        else:
            # 代表を差し替える場合がある
            if score(acc) > score(g["repr"]):
                g["repr"] = acc
            # 集計は合算
            g["total"]     += total_i
            g["posted"]    += posted_i
            g["generated"] += generated_i
            g["raw"].append(acc)

    # 表示用リスト（代表 acc に合算済みの数値を持たせる）
    accts = []
    for _, g in groups.items():
        a = g["repr"]
        a.total_cnt     = g["total"]
        a.posted_cnt    = g["posted"]
        a.generated_cnt = g["generated"]
        a._raw_count    = len(g["raw"])  # 任意：統合件数（表示したければテンプレで参照）
        accts.append(a)

    # 並び替え（ユニーク化後の値で）
    def sort_key(a):
        if sort == "posted_asc":
            return (a.posted_cnt or 0, a.id)
        if sort == "posted_desc":
            return (-(a.posted_cnt or 0), a.id)
        if sort == "generated_asc":
            return (a.generated_cnt or 0, a.id)
        if sort == "generated_desc":
            return (-(a.generated_cnt or 0), a.id)
        if sort == "total_asc":
            return (a.total_cnt or 0, a.id)
        # default: total_desc
        return (-(a.total_cnt or 0), a.id)

    accts.sort(key=sort_key)

    all_sites = Site.query.filter_by(user_id=current_user.id).all()

    return render_template(
        "external_accounts.html",
        accts=accts,                 # ← ユニーク化後の代表たち
        all_sites=all_sites,
        decrypt=decrypt,
        site_id=site_id,
        selected_blog_type=blog_type,
        selected_sort=sort,
        search_query=search
    )




@bp.route("/external/account/<int:acct_id>/articles")
@login_required
def external_account_articles(acct_id):
    from app.models import ExternalBlogAccount, ExternalArticleSchedule, Keyword, Article

    acct = ExternalBlogAccount.query.get_or_404(acct_id)
    site = acct.site
    if site.user_id != current_user.id and not current_user.is_admin:
        abort(403)

    rows = (
        db.session.query(ExternalArticleSchedule, Keyword, Article)
        .join(Keyword, ExternalArticleSchedule.keyword_id == Keyword.id)
        .outerjoin(Article, Article.id == ExternalArticleSchedule.article_id)
        .filter(ExternalArticleSchedule.blog_account_id == acct_id)
        # ▼ 修正：古い順（ASC）＋ タイブレークに schedule.id
        .order_by(ExternalArticleSchedule.scheduled_date.asc(),
                  ExternalArticleSchedule.id.asc())
        .all()
    )

    return render_template(
        "external_articles.html",
        acct=acct, site=site, rows=rows
    )


@bp.route("/external/article/<int:article_id>/preview")
@login_required
def external_article_preview(article_id):
    from app.models import Article

    art = Article.query.get_or_404(article_id)

    if art.user_id != current_user.id and not current_user.is_admin:
        abort(403)

    return render_template("external_article_preview.html", article=art)



# 外部SEO記事 編集
@bp.route("/external/article/<int:article_id>/edit", methods=["GET", "POST"])
@login_required
def external_article_edit(article_id):
    from app.models import Article
    art = Article.query.get_or_404(article_id)

    if art.user_id != current_user.id and not current_user.is_admin:
        abort(403)

    if request.method == "POST":
        art.title = request.form.get("title", art.title)
        art.body = request.form.get("body", art.body)
        db.session.commit()
        flash("記事を更新しました", "success")
        # 確実に戻れるように
        return redirect(request.referrer or url_for("main.external_schedules", site_id=art.site_id))

    return render_template("external_article_edit.html", article=art)

# 外部SEO記事 削除
@bp.route("/external/article/<int:article_id>/delete", methods=["POST"])
@login_required
def external_article_delete(article_id):
    from app.models import Article, ExternalArticleSchedule, Keyword

    art = Article.query.get_or_404(article_id)
    if art.user_id != current_user.id and not current_user.is_admin:
        abort(403)

    # Article から Keyword.id を引く
    kw = Keyword.query.filter_by(site_id=art.site_id, keyword=art.keyword).first()

    if kw:
        schedules = ExternalArticleSchedule.query.filter_by(keyword_id=kw.id).all()
        for sched in schedules:
            db.session.delete(sched)

    db.session.delete(art)
    db.session.commit()
    flash("記事を削除しました", "success")
    # 元画面に戻す（acct_id が取れないので referrer 優先）
    return redirect(request.referrer or url_for("main.external_schedules", site_id=art.site_id))


# 外部SEO記事 即時投稿
@bp.route("/external/schedule/<int:schedule_id>/post_now", methods=["POST"])
@login_required
def external_schedule_post_now(schedule_id):
    from datetime import datetime
    from flask import current_app, request, redirect, url_for, flash, abort
    from flask_login import current_user
    from app import db
    from app.models import ExternalArticleSchedule
    from app.tasks import _run_external_post_job  # ← ここを修正

    sched = ExternalArticleSchedule.query.get_or_404(schedule_id)
    acct = sched.blog_account
    site = acct.site

    # 所有権チェック
    if site.user_id != current_user.id and not current_user.is_admin:
        abort(403)

    # 直ちに実行対象へ（UTC naive）
    sched.scheduled_date = datetime.utcnow()
    sched.status = "pending"
    db.session.commit()

    try:
        # pending を処理
        _run_external_post_job(current_app._get_current_object(), schedule_id=schedule_id)
        flash("即時投稿を開始しました。しばらくしてページを更新してください。", "success")
    except Exception as e:
        current_app.logger.exception("[external] post_now failed")
        flash(f"即時投稿に失敗しました: {e}", "danger")

    return redirect(request.referrer or url_for("main.external_account_articles", acct_id=acct.id))

# --- 一括削除: 外部ブログアカウント + 予約 +（安全条件下の）生成記事 ---
@bp.post("/external/account/<int:acct_id>/delete")
@login_required
def external_account_delete(acct_id):
    from app.models import (
        ExternalBlogAccount, ExternalArticleSchedule, Keyword, Article, Site
    )
    from sqlalchemy import exists, and_, select
    from app import db

    acct = ExternalBlogAccount.query.get_or_404(acct_id)
    site: Site = acct.site

    # 権限
    if not current_user.is_admin and site.user_id != current_user.id:
        return {"ok": False, "error": "権限がありません"}, 403

    # まず、このアカウントの全スケジュールを取得（Keywordも使うためIDを保持）
    schedules = (
        db.session.query(ExternalArticleSchedule)
        .filter(ExternalArticleSchedule.blog_account_id == acct.id)
        .all()
    )
    keyword_ids = [s.keyword_id for s in schedules if getattr(s, "keyword_id", None)]
    # ID→テキストを得る（Articleは keyword(テキスト) 基準で紐付けられているため）
    kw_texts = []
    if keyword_ids:
        kw_texts = [
            k.keyword for k in db.session.query(Keyword).filter(Keyword.id.in_(keyword_ids)).all()
        ]

    # このアカウント以外でも同じキーワードIDが使われているか（残す条件）
    # → Articleは「同じ keyword テキスト」を共有し得るので、
    #   “他アカウントの予約が同一KeywordIDを参照していない”記事のみ削除対象とする
    if kw_texts:
        # schedules テーブルで “同一 keyword_id かつ 別アカウント” が存在しないことを条件に Article を削除
        # Article は site_id と source='external' で限定
        subq_other = (
            db.session.query(ExternalArticleSchedule.id)
            .filter(
                ExternalArticleSchedule.keyword_id.in_(keyword_ids),
                ExternalArticleSchedule.blog_account_id != acct.id
            )
            .exists()
        )
        # 削除対象 Article の選別
        articles_q = (
            db.session.query(Article)
            .filter(
                Article.site_id == site.id,
                Article.source == "external",
                Article.keyword.in_(kw_texts),
                ~subq_other   # 他アカウントの予約が無いキーワードのみ
            )
        )
        deleted_articles = articles_q.delete(synchronize_session=False)
    else:
        deleted_articles = 0

    # スケジュール削除
    db.session.query(ExternalArticleSchedule)\
        .filter(ExternalArticleSchedule.blog_account_id == acct.id)\
        .delete(synchronize_session=False)

    # アカウント削除
    db.session.delete(acct)
    db.session.commit()

    return {"ok": True, "deleted_articles": int(deleted_articles)}



# -----------------------------------------------------------
# 管理者向け: 全ユーザーの外部ブログアカウント一覧
# -----------------------------------------------------------
@admin_bp.route("/admin/blog_accounts")
@login_required
def admin_blog_accounts():
    if not current_user.is_admin:
        abort(403)

    from app.models import ExternalBlogAccount
    from app.services.blog_signup.crypto_utils import decrypt

    accts = (ExternalBlogAccount
             .query.order_by(ExternalBlogAccount.created_at.desc())
             .all())

    # ★ パスを "admin/xxx.html" に変更
    return render_template(
        "admin/admin_blog_accounts.html",
        accts    = accts,
        decrypt  = decrypt,
    )


# ---------------------------------------------------------
# 🔐 管理者専用：ワンクリックで対象ブログへログインする中間ページ
# ---------------------------------------------------------
# app/routes.py など
from flask import Blueprint, request, abort, render_template_string
from flask_login import login_required, current_user
from app import db


# ---------------------------------------------------------
# 🔐 管理者専用：ワンクリック自動ログイン
# ---------------------------------------------------------
@admin_bp.route("/admin/blog_login", methods=["POST"])
@login_required
def admin_blog_login():
    """
    管理者が「ワンクリックログイン」を押した時に呼び出される。
    - 対応サービス (note / hatena …) は自動 POST
    - 未対応サービスは資格情報を表示
    """
    if not current_user.is_admin:
        abort(403)

    from app.models import ExternalBlogAccount
    from app.services.blog_signup.crypto_utils import decrypt

    acct_id = request.form.get("account_id", type=int)
    if not acct_id:
        abort(400, "account_id missing")

    acct: ExternalBlogAccount | None = ExternalBlogAccount.query.get(acct_id)
    if not acct:
        abort(404, "account not found")

    email    = decrypt(acct.email)
    password = decrypt(acct.password)
    username = acct.username

    login_map = {
        "note": {
            "url": "https://note.com/login",
            "user_field": "email",
            "pass_field": "password",
        },
        "hatena": {
            "url": "https://www.hatena.ne.jp/login",
            "user_field": "name",
            "pass_field": "password",
        },
        # ここに他ブログを追加
    }

    cfg = login_map.get(acct.blog_type.value)

    # --- 対応ブログ：自動 POST フォーム ----
    if cfg:
        return f"""
        <!doctype html><html lang="ja"><head><meta charset="utf-8">
        <title>auto-login</title></head><body>
          <p style="font-family:sans-serif;margin-top:2rem">
            {acct.blog_type.value} にリダイレクト中…
          </p>
          <form id="f" action="{cfg['url']}" method="post">
            <input type="hidden" name="{cfg['user_field']}" value="{email}">
            <input type="hidden" name="{cfg['pass_field']}" value="{password}">
          </form>
          <script>setTimeout(()=>document.getElementById('f').submit(), 300);</script>
        </body></html>
        """

    # --- 未対応ブログ：資格情報表示 ----
    return render_template_string("""
      <!doctype html><html lang="ja"><head><meta charset="utf-8">
      <title>資格情報</title></head><body style="font-family:sans-serif">
        <h2>手動ログインが必要です</h2>
        <ul>
          <li><b>サービス</b>: {{ blog }}</li>
          <li><b>ユーザー名</b>: {{ uname }}</li>
          <li><b>メール</b>: {{ mail }}</li>
          <li><b>パスワード</b>: {{ pwd }}</li>
        </ul>
      </body></html>
    """, blog=acct.blog_type.value, uname=username, mail=email, pwd=password)



# -----------------------------------------------------------
# ワンクリックログイン 
# -----------------------------------------------------------
@bp.route("/external/login/<int:acct_id>")
@login_required
def blog_one_click_login(acct_id):
    acct = ExternalBlogAccount.query.get_or_404(acct_id)
    if not (current_user.is_admin or acct.site.user_id == current_user.id):
        abort(403)

    from app.services.blog_signup.crypto_utils import decrypt

    if acct.blog_type == BlogType.NOTE:
        from app.services.blog_signup.note_login import get_note_cookies
        cookies = asyncio.run(get_note_cookies(decrypt(acct.email), decrypt(acct.password)))
        resp = make_response(redirect("https://note.com"))
        for c in cookies:
            resp.set_cookie(
                key=c["name"], value=c["value"],
                domain=".note.com", path="/", secure=True, httponly=True,
                samesite="Lax", expires=int(time.time()) + 60*60
            )
        return resp

    elif acct.blog_type == BlogType.LIVEDOOR:
        # ★ 追加：Livedoor対応
        from app.services.blog_signup.livedoor_login import get_livedoor_cookies
        cookies = asyncio.run(get_livedoor_cookies(decrypt(acct.email), decrypt(acct.password)))
        # 管理画面側に入れる
        resp = make_response(redirect("https://livedoor.blogcms.jp/member/"))
        for c in cookies:
            resp.set_cookie(
                key=c["name"], value=c["value"],
                domain=c.get("domain", ".livedoor.com"),
                path="/", secure=True, httponly=True,
                samesite="Lax", expires=int(time.time()) + 60*60
            )
        return resp

    else:
        abort(400, "Login not supported yet")



# ====== ルート先頭の import 付近に追記/置換 ======
from flask import session as flask_session  # 既にあればOK
from app.services.blog_signup.livedoor_signup import (
    generate_safe_id, generate_safe_password,
    prepare_captcha as ld_prepare_captcha,   # 新API名
    submit_captcha as ld_submit_captcha,     # 新API名
    suggest_livedoor_blog_id,
    poll_latest_link_gw,                     # メール認証リンク取得
)
from app.services.mail_utils.mail_gw import create_inbox
from app.services.blog_signup.livedoor_atompub_recover import recover_atompub_key
from app.services.pw_controller import pwctl  # セッションの明示クローズ用
# 既存 import 群の近くに追記
from flask import current_app  # submit_captcha で使っているため
from app.services.pw_session_store import (
    save as pw_save,
    get_cred as pw_get,
    clear as pw_clear,
)


# ====== /prepare_captcha ======
@bp.route("/prepare_captcha", methods=["POST"])
@login_required
def prepare_captcha():
    from app.models import Site, ExternalBlogAccount
    from app import db
    from flask import jsonify, request, url_for
    from uuid import uuid4
    from pathlib import Path
    import time as _time
    import logging
    logger = logging.getLogger(__name__)

    site_id    = request.form.get("site_id", type=int)
    blog       = request.form.get("blog")  # "livedoor"
    account_id = request.form.get("account_id", type=int)

    if not site_id or not blog:
        return jsonify({"captcha_url": None, "error": "site_id または blog が指定されていません",
                        "site_id": site_id, "account_id": account_id})

    site = Site.query.get(site_id)
    if not site or (not current_user.is_admin and site.user_id != current_user.id):
        return jsonify({"captcha_url": None, "error": "権限がありません",
                        "site_id": site_id, "account_id": account_id})

    # 所有アカウント検証（任意）
    acct = ExternalBlogAccount.query.get(account_id) if account_id else None
    if acct:
        if acct.site_id != site_id:
            return jsonify({"captcha_url": None, "error": "account_id が site_id に属していません",
                            "site_id": site_id, "account_id": account_id})
        if (not current_user.is_admin) and (acct.site.user_id != current_user.id):
            return jsonify({"captcha_url": None, "error": "権限がありません",
                            "site_id": site_id, "account_id": account_id})

    # 仮登録データ（メール & 候補 blog_id）
    email, token = create_inbox()
    livedoor_id  = generate_safe_id()
    password     = generate_safe_password()

    try:
        base_text = site.name or site.url or ""
        desired_blog_id = suggest_livedoor_blog_id(base_text, db.session)
    except Exception:
        desired_blog_id = None

    # ▶ 新API: Playwright セッションを作って CAPTCHA 画像を保存
    try:
        session_id, img_abs_path = ld_prepare_captcha(email, livedoor_id, password)
    except Exception:
        logger.exception("[prepare_captcha] CAPTCHA生成で例外が発生")
        return jsonify({"captcha_url": None, "error": "CAPTCHAの準備に失敗しました",
                        "site_id": site_id, "account_id": account_id})
    
    # ★ 追加：資格情報を sid 単位で保存（フォームや Flask セッションに依存しない）
    pw_save(session_id,
            email=email,
            password=password,
            livedoor_id=livedoor_id,
            token=token,
            site_id=site_id,
            account_id=account_id,
            desired_blog_id=desired_blog_id)

    # 画像URL化
    img_name = Path(img_abs_path).name
    ts = int(_time.time())
    captcha_url = url_for("static", filename=f"captchas/{img_name}", _external=True) + f"?v={ts}"

    # セッション保持（次の /submit_captcha 用）
    flask_session.update({
        "captcha_email": email,
        "captcha_nickname": livedoor_id,
        "captcha_password": password,
        "captcha_token": token,
        "captcha_site_id": site_id,
        "captcha_blog": blog,
        "captcha_image_filename": img_name,
        "captcha_session_id": session_id,
        "captcha_account_id": account_id,
        "captcha_desired_blog_id": desired_blog_id,
    })

    # 任意：DBログ
    if acct:
        acct.captcha_session_id = session_id
        acct.captcha_image_path = f"captchas/{img_name}"
        db.session.commit()

    return jsonify({
        "captcha_url": captcha_url,
        "site_id": site_id,
        "account_id": account_id
    })


# ====== /submit_captcha ======
@bp.route("/submit_captcha", methods=["POST"])
@login_required
def submit_captcha():
    from app.services.blog_signup.crypto_utils import encrypt
    from app.models import Site, ExternalBlogAccount
    from app.enums import BlogType
    from app.utils.captcha_dataset_utils import save_captcha_label_pair
    from app import db
    from flask import jsonify, session, request
    import logging, contextlib, asyncio
    import time

    logger = logging.getLogger(__name__)

    captcha_text = request.form.get("captcha_text")
    if not captcha_text:
        return jsonify({"status": "error", "message": "CAPTCHA文字列が入力されていません"}), 400

    # 学習用保存
    img_name = session.get("captcha_image_filename")
    if captcha_text and img_name:
        with contextlib.suppress(Exception):
            save_captcha_label_pair(img_name, captcha_text)

    account_id = request.form.get("account_id", type=int) or session.get("captcha_account_id")
    site_id    = session.get("captcha_site_id")
    session_id = session.get("captcha_session_id")

    # ★ 追加：サーバー側ストアから資格情報を復元（フォーム/Flaskセッションが空でもOK）
    cred = pw_get(session_id) if session_id else None

    email = (
        request.form.get("email")
        or session.get("captcha_email")
        or (cred and cred.get("email"))
    )
    password = (
        request.form.get("password")
        or session.get("captcha_password")
        or (cred and cred.get("password"))
    )
    livedoor_id = (
        request.form.get("livedoor_id")
        or session.get("captcha_nickname")
        or (cred and cred.get("livedoor_id"))
    )

    # URLサブドメイン=ユーザーID（希望値）。無ければ livedoor_id を使う
    desired_blog_id = (
        request.form.get("desired_blog_id")
        or request.form.get("blog_id")
        or request.form.get("sub")
        or session.get("captcha_desired_blog_id")
        or (cred and cred.get("desired_blog_id"))
        or livedoor_id
    )


    if not all([site_id, session_id, account_id]):
        return jsonify({"status": "error", "message": "セッション情報が不足しています"}), 400

    site = Site.query.get(site_id)
    acct = ExternalBlogAccount.query.get(account_id)
    if not site or not acct or acct.site_id != site_id:
        return jsonify({"status": "error", "message": "対象が不正です"}), 400
    if (not current_user.is_admin) and (site.user_id != current_user.id):
        return jsonify({"status": "error", "message": "権限がありません"}), 403

    ok = False
    try:
        # ▶ 新API: 同一セッションで CAPTCHA 送信 → /register/done を待機
        ok = ld_submit_captcha(session_id, captcha_text)
    except Exception:
        logger.exception("[submit_captcha] CAPTCHA送信で例外")
        # セッションは後で必ず破棄
        return jsonify({"status": "error", "message": "CAPTCHA送信に失敗しました"}), 500

    if not ok:
        # 失敗時は中間アカウントをクリーンアップ（あれば）
        try:
            if acct and not getattr(acct, "atompub_key_enc", None):
                db.session.delete(acct)
                db.session.commit()
        except Exception:
            db.session.rollback()
        finally:
            with contextlib.suppress(Exception):
                pwctl.close_session(session_id)
            # セッションキー掃除（captcha_status は残す）
            for k in list(session.keys()):
                if k.startswith("captcha_") and k != "captcha_status":
                    session.pop(k)
        return jsonify({
            "status": "recreate_required",
            "message": "CAPTCHA突破に失敗しました。もう一度お試しください。",
            "site_id": site_id,
        }), 200

    # --- ここから 既存の「メール認証→AtomPubキー回収」を継続 ---
    try:
        # メール確認リンク取得（最大 5 回 / 30 秒）
        token = session.get("captcha_token") or (cred and cred.get("token"))
        if not token:
            with contextlib.suppress(Exception):
                pwctl.close_session(session_id)
            return jsonify({
                "status": "recreate_required",
                "message": "確認メールのトークンが見つかりませんでした（セッション復元に失敗）",
                "site_id": site_id,
            }), 200
        activation_url = None  # ← これを追加
        for _ in range(5):
            with contextlib.suppress(Exception):
                activation_url = asyncio.run(poll_latest_link_gw(token))
            if activation_url:
                break
            time.sleep(6)

        if not activation_url:
            with contextlib.suppress(Exception):
                pwctl.close_session(session_id)
            return jsonify({
                "status": "recreate_required",
                "message": "確認メールリンクが取得できませんでした",
                "site_id": site_id,
            }), 200

        # Playwright セッションでそのまま認証URLへ遷移して final 入力を拾う
        # （recover_atompub_key はページを受け取って blog_id / api_key を抽出する実装）
        # reviveは基本不要だが、落ちていたら自動復帰
        page = pwctl.run(pwctl.get_page(session_id)) or pwctl.run(pwctl.revive(session_id))
        if not page:
            raise RuntimeError("Playwright セッションが消失しました")

        # 認証URLへ遷移（これも pwctl のループ上で）
        pwctl.run(page.goto(activation_url, wait_until="load"))

        # ★ ここを asyncio.run(...) ではなく pwctl.run(...) にするのがポイント
        # ★ 置換：recover で使う livedoor の user_id は、基本 livedoor_id を使う
        user_id = (
            request.form.get("livedoor_id")
            or request.form.get("user_id")
            or request.form.get("userid")
            or request.form.get("username")
            or request.form.get("account_id")
            or livedoor_id
        )
        if not user_id:
            current_app.logger.error("[submit_captcha] livedoor user_id is missing (sid=%s)", session_id)
            return jsonify({"ok": False, "error": "missing_user_id"}), 400

        
        # --- ここでフォーム値を集める（名称の揺れを吸収） ---
        nickname = (
            request.form.get("nickname")
            or request.form.get("display_name")
            or request.form.get("name")
        )

        # ここからは “既存値を優先し、未設定のときだけフォームから補完”
        email = (
            email
            or request.form.get("email")
            or request.form.get("livedoor_email")
            or request.form.get("mail")
        )

        password = (
            password
            or request.form.get("password")
            or request.form.get("livedoor_password")
            or request.form.get("pass")
        )

        # desired_blog_id は関数前半で cred/セッション/フォームから一度確定済み。
        # 後段で再計算・上書きしない（そのまま desired_blog_id を使う）。


        # 最低限のバリデーション（必要に応じて 400 を返す）
        if not email or not password:
            current_app.logger.error(
                "[submit_captcha] email/password missing (sid=%s, has_email=%s, has_pw=%s)",
                session_id, bool(email), bool(password)
            )
            return jsonify({"ok": False, "error": "missing_email_or_password"}), 400

        if not nickname:
            nickname = email.split("@")[0]  # フォールバック

        result = pwctl.run(recover_atompub_key(
            page,
            livedoor_id=user_id,
            nickname=nickname or (email.split("@")[0] if email else None),
            email=email,
            password=password,
            site=site,
            desired_blog_id=desired_blog_id,
        ))


        if not result or not result.get("success"):
            with contextlib.suppress(Exception):
                pwctl.close_session(session_id)
            return jsonify({
                "status": "recreate_required",
                "message": result.get("error", "APIキーの回収に失敗しました"),
                "site_id": site_id,
            }), 200

        new_blog_id  = (result.get("blog_id") or "").strip() or None
        new_api_key  = (result.get("api_key") or "").strip() or None
        new_endpoint = (result.get("endpoint") or "").strip() or None

        # 重複 blog_id があれば既存を優先
        dup = None
        if new_blog_id:
            dup = (ExternalBlogAccount.query
                   .filter(
                       ExternalBlogAccount.site_id == site_id,
                       ExternalBlogAccount.blog_type == (acct.blog_type or BlogType.LIVEDOOR),
                       ExternalBlogAccount.livedoor_blog_id == new_blog_id,
                       ExternalBlogAccount.id != account_id
                   )
                   .first())
        target = dup or acct

        if hasattr(target, "is_captcha_completed"):
            target.is_captcha_completed = True
        if new_blog_id and hasattr(target, "livedoor_blog_id"):
            target.livedoor_blog_id = new_blog_id
        if new_blog_id and hasattr(target, "username"):
            if not target.username or target.username.startswith("u-"):
                target.username = new_blog_id
        if new_endpoint and hasattr(target, "atompub_endpoint"):
            with contextlib.suppress(Exception):
                target.atompub_endpoint = new_endpoint
        if new_api_key and hasattr(target, "atompub_key_enc"):
            with contextlib.suppress(Exception):
                target.atompub_key_enc = encrypt(new_api_key)
            if hasattr(target, "api_post_enabled"):
                target.api_post_enabled = True

        db.session.commit()

        got_api = bool(new_api_key or getattr(target, "atompub_key_enc", None))
        resolved_account_id = target.id
        session["captcha_status"] = {
            "captcha_sent": True,
            "email_verified": True,
            "account_created": True,
            "api_key_received": got_api,
            "step": "既存アカウントに紐付け済み" if dup else "API取得完了",
            "site_id": site_id,
            "account_id": resolved_account_id,
        }

        return jsonify({
            "status": "captcha_success",
            "step": session["captcha_status"]["step"],
            "site_id": site_id,
            "account_id": resolved_account_id,
            "api_key_received": got_api,
            "next_cta": "ready_to_post" if got_api else "captcha_done"
        }), 200

    finally:
        with contextlib.suppress(Exception):
            pwctl.close_session(session_id)
        with contextlib.suppress(Exception):   # ★ 追加
            pw_clear(session_id)
        for key in list(session.keys()):
            if key.startswith("captcha_") and key != "captcha_status":
                session.pop(key)



@bp.route("/captcha_status", methods=["GET"])
@login_required
def get_captcha_status():
    from flask import session, jsonify, request
    # DBフォールバック用
    from app.models import ExternalBlogAccount

    status = session.get("captcha_status")

    # 任意：?account_id=... が来たら整合性チェック
    q_acc = request.args.get("account_id", type=int)

    # セッションがある場合の基本応答
    if status:
        if q_acc and status.get("account_id") and status["account_id"] != q_acc:
            # 別アカウントのステータスを見に来た場合は未開始扱い
            return jsonify({"status": "not_started", "step": "未開始"}), 200
        return jsonify(status), 200

    # ★ セッションが切れても、DBがAPI取得済なら「API取得完了」を返すフォールバック
    if q_acc:
        acct = ExternalBlogAccount.query.get(q_acc)
        if acct and getattr(acct, "atompub_key_enc", None):
            return jsonify({
                "captcha_sent": True,
                "email_verified": True,          # ここは推定（API取得済み前提）
                "account_created": True,         # 同上
                "api_key_received": True,
                "step": "API取得完了",
                "site_id": getattr(acct, "site_id", None),
                "account_id": q_acc
            }), 200

    # 何も情報がない
    return jsonify({"status": "not_started", "step": "未開始"}), 200

@bp.get("/generate")
@login_required
def external_seo_generate_get():
    from datetime import datetime, timezone
    from app import db
    from app.models import Site, ExternalBlogAccount, BlogType
    from app.tasks import enqueue_generate_and_schedule
    from sqlalchemy import and_

    site_id = request.args.get("site_id", type=int)
    account_id = request.args.get("account_id", type=int)
    blog_type_param = request.args.get("blog_type", default="livedoor").strip().lower() if request.args.get("blog_type") else "livedoor"

    if not site_id:
        flash("site_id が不足しています。", "danger")
        return redirect(url_for("main.external_seo_sites"))

    site = Site.query.get_or_404(site_id)
    if (not current_user.is_admin) and (site.user_id != current_user.id):
        abort(403)

    try:
        target_blog_type = getattr(BlogType, blog_type_param.upper())
    except Exception:
        target_blog_type = BlogType.LIVEDOOR

    # 対象アカウントの選定
    if account_id:
        acct = ExternalBlogAccount.query.get_or_404(account_id)
        if acct.site_id != site_id:
            flash("不正なアクセスです（サイト不一致）", "danger")
            return redirect(url_for("main.external_seo_sites"))
        if acct.blog_type != target_blog_type:
            flash("不正なアクセスです（プラットフォーム不一致）", "danger")
            return redirect(url_for("main.external_seo_sites"))
        if not acct.atompub_key_enc:
            flash("このアカウントはAPIキー未取得のため記事生成できません。", "danger")
            return redirect(url_for("main.external_seo_sites"))
        accounts_to_run = [acct]
    else:
        # まとめ実行：未ロック & API 取得済みのみ候補にする
        accounts_to_run = (
            ExternalBlogAccount.query
            .filter(
                and_(
                    ExternalBlogAccount.site_id == site_id,
                    ExternalBlogAccount.blog_type == target_blog_type,
                    ExternalBlogAccount.atompub_key_enc.isnot(None),
                    ExternalBlogAccount.generation_locked.is_(False),
                )
            )
            .all()
        )
        if not accounts_to_run:
            flash("実行可能なアカウントが見つかりません（API未取得 または 既にロック済み）。", "warning")
            return redirect(url_for("main.external_seo_sites"))

    ok, ng, skipped_locked = 0, 0, 0
    failed = []

    for acct in accounts_to_run:
        try:
            # ---- ここが恒久ロックの肝 ----
            # 行ロックを取り、二重実行を防ぐ
            row = (
                ExternalBlogAccount.query
                .with_for_update()           # SELECT ... FOR UPDATE
                .filter_by(id=acct.id)
                .first()
            )
            if not row:
                skipped_locked += 1
                continue

            # 既にロック済みならスキップ
            if row.generation_locked:
                skipped_locked += 1
                continue

            # ここで恒久ロックを立てて確定
            row.generation_locked = True
            row.generation_locked_at = datetime.now(timezone.utc)
            db.session.add(row)
            db.session.commit()             # 先に確定 → 以後は二重実行不可

            # ロック確定後にキュー投入
            enqueue_generate_and_schedule(
                user_id=current_user.id,
                site_id=site_id,
                blog_account_id=row.id,
                count=100,
                per_day=10,
                start_day_jst=None,   # 翌日開始（関数内のデフォルトで処理）
            )
            ok += 1

        except Exception as e:
            db.session.rollback()
            ng += 1
            failed.append((acct.id, str(e)))

    # フィードバック
    if ok and not ng:
        msg = f"{ok}件のアカウントで生成を開始"
        if skipped_locked:
            msg += f" ／ ロック済みスキップ {skipped_locked}件"
        flash(msg, "success")
    elif ok and ng:
        flash(f"{ok}件開始 / {ng}件失敗（ロック済みスキップ {skipped_locked}件）", "warning")
    else:
        # 1件も開始できなかった
        if skipped_locked:
            flash("すべての対象がロック済みのため実行されませんでした。", "warning")
        else:
            flash("記事生成の開始に失敗しました。", "danger")

    if failed:
        for aid, msg in failed[:3]:
            flash(f"account_id={aid}: {msg}", "danger")
        if len(failed) > 3:
            flash(f"…他 {len(failed)-3}件", "danger")

    return redirect(url_for("main.external_seo_sites"))


from flask import render_template, redirect, url_for, request, session, flash
from app.services.mail_utils.mail_tm import poll_latest_link_tm_async as poll_latest_link_gw
from app.services.blog_signup.livedoor_signup import extract_verification_url

@bp.route('/confirm_email_manual/<task_id>')
def confirm_email_manual(task_id):
    """
    CAPTCHA後、認証リンクをユーザーに手動で表示する画面。
    """
    # メール受信（最大30回ポーリング） ← 既存関数を再利用
    email_body = poll_latest_link_gw(task_id=task_id, max_attempts=30, interval=5)

    if email_body:
        # 認証URLを抽出
        verification_url = extract_verification_url(email_body)
        if verification_url:
            return render_template("confirm_email.html", verification_url=verification_url)
        else:
            flash("認証リンクが見つかりませんでした", "danger")
            return redirect(url_for('dashboard'))
    else:
        flash("認証メールを取得できませんでした", "danger")
        return redirect(url_for('dashboard'))

from flask import request, session, redirect, url_for, flash
from app.services.blog_signup.livedoor_signup import fetch_livedoor_credentials


@bp.route('/finish_signup/<task_id>', methods=['POST'])
def finish_signup(task_id):
    """
    メール認証が完了した後に呼ばれる処理。
    AtomPub API Keyを取得し、DB保存 or 表示に進む。
    """
    try:
        # すでに存在する task_id のセッションや保存情報から再開
        result = fetch_livedoor_credentials(task_id)

        if result and result.get("blog_id") and result.get("api_key"):
            # 必要に応じてDB保存 or セッションに保存（ここでは表示用）
            flash("🎉 AtomPub API情報を正常に取得しました", "success")
            flash(f"ブログID: {result['blog_id']}", "info")
            flash(f"API Key: {result['api_key']}", "info")
            return redirect(url_for('dashboard'))  # または account_details, etc.
        else:
            flash("API情報の取得に失敗しました", "danger")
            return redirect(url_for('dashboard'))

    except Exception as e:
        flash(f"エラーが発生しました: {str(e)}", "danger")
        return redirect(url_for('dashboard'))

from flask import render_template, abort
from app.services.blog_signup.livedoor_signup import fetch_livedoor_credentials

@bp.route("/external/livedoor/confirm/<task_id>")
def confirm_livedoor_email(task_id):
    creds = fetch_livedoor_credentials(task_id)
    if not creds:
        abort(404, description="認証情報が見つかりません")
    return render_template("confirm_email.html", blog_id=creds["blog_id"], api_key=creds["api_key"])

# ===============================
# 外部SEO記事生成ルート（新規追加）
# ===============================

from flask import Blueprint, request, redirect, url_for, flash
from flask_login import login_required, current_user
from app.models import ExternalBlogAccount
from app.tasks import _run_external_post_job

# 既存の
# @bp.route("/external-seo/generate/<int:site_id>/<int:blog_id>", methods=["POST"])
# def external_seo_generate(...):
# を丸ごと置き換え

@bp.route("/external-seo/generate/<int:site_id>/<int:blog_id>", methods=["POST"])
@login_required
def external_seo_generate(site_id, blog_id):
    """
    既存の /external-seo/generate/<site_id>/<blog_id> を温存したまま、
    生成＆スケジューリングの新ロジックに差し替え。
    - 100本生成
    - 1日10本
    - スケジュール開始は「生成開始の翌日」
    """
    from flask import redirect, url_for, flash
    from app.models import ExternalBlogAccount, Site, BlogType
    from app.external_seo_generator import generate_and_schedule_external_articles

    # アカウント取得
    acct = ExternalBlogAccount.query.get_or_404(blog_id)

    # site_id整合性
    if acct.site_id != site_id:
        flash("不正なアクセスです（サイト不一致）。", "danger")
        return redirect(url_for("main.external_seo_sites"))

    # 所有権チェック（管理者はスキップ）
    site = Site.query.get_or_404(site_id)
    if (not current_user.is_admin) and (site.user_id != current_user.id):
        abort(403)

    # APIキー必須
    if not getattr(acct, "atompub_key_enc", None):
        flash("APIキーが未取得のため記事生成できません。", "danger")
        return redirect(url_for("main.external_seo_sites"))

    try:
        # ※ start_day_jst を省略 → ジェネレータ側で「翌日開始」に自動化
        created = generate_and_schedule_external_articles(
            user_id=current_user.id,
            site_id=site_id,
            blog_account_id=acct.id,
            count=100,
            per_day=10,
            start_day_jst=None,
        )
        flash(f"外部SEO記事の生成を開始しました（{created}件、1日10本・翌日から投稿）。", "success")
    except Exception as e:
        current_app.logger.exception("[external-seo] generate (legacy route) failed")
        flash(f"記事生成開始に失敗しました: {e}", "danger")

    return redirect(url_for("main.external_seo_sites"))


# ===============================
# 外部SEO: 100本生成＋1日10本スケジューリング（新規）
# ===============================
from flask import request, jsonify, current_app
from flask_login import login_required, current_user
from datetime import datetime, timedelta, timezone
from app.models import ExternalBlogAccount, BlogType
from app.external_seo_generator import generate_and_schedule_external_articles


JST = timezone(timedelta(hours=9))

@bp.route("/external-seo/generate_and_schedule", methods=["POST"])
@login_required
def external_seo_generate_and_schedule():
    """
    外部SEO記事をまとめて生成し、1日10本（JST 10:00〜21:59の“切りの良くない分”）でスケジューリング。
    JSON/FORM:
      site_id: int (必須)
      blog_account_id: int (任意。未指定なら site_id に紐づく最新 Livedoor を自動選択)
      count: 生成本数（デフォルト100）
      per_day: 1日あたり本数（デフォルト10）
      start_date_jst: "YYYY-MM-DD"（JSTの開始日。省略時は当日）
    """
    # 入力パラメータ
    site_id = request.form.get("site_id", type=int) or (request.json or {}).get("site_id")
    count = request.form.get("count", type=int) or (request.json or {}).get("count", 100)
    per_day = request.form.get("per_day", type=int) or (request.json or {}).get("per_day", 10)
    start_date_s = request.form.get("start_date_jst") or (request.json or {}).get("start_date_jst")

    if not site_id:
        return jsonify({"ok": False, "error": "site_id is required"}), 400

    if start_date_s:
        try:
            y, m, d = map(int, start_date_s.split("-"))
            start_day_jst = datetime(y, m, d, tzinfo=JST)
        except Exception:
            return jsonify({"ok": False, "error": "start_date_jst must be YYYY-MM-DD"}), 400
    else:
        start_day_jst = datetime.now(JST).replace(hour=0, minute=0, second=0, microsecond=0)

    # 対象アカウント
    blog_account_id = request.form.get("blog_account_id") or (request.json or {}).get("blog_account_id")
    if blog_account_id:
        acct = ExternalBlogAccount.query.get(int(blog_account_id))
    else:
        acct = (ExternalBlogAccount.query
                .filter_by(site_id=site_id, blog_type=BlogType.LIVEDOOR)
                .order_by(ExternalBlogAccount.id.desc())
                .first())
    if not acct:
        return jsonify({"ok": False, "error": "Livedoorアカウントが見つかりません"}), 400

    # 所有権チェック（管理者はスキップ）
    if (not current_user.is_admin) and (acct.site.user_id != current_user.id):
        return jsonify({"ok": False, "error": "権限がありません"}), 403

    # 実行
    try:
        created = generate_and_schedule_external_articles(
            user_id=current_user.id,
            site_id=site_id,
            blog_account_id=acct.id,
            count=int(count),
            per_day=int(per_day),
            start_day_jst=start_day_jst,
        )
        return jsonify({"ok": True, "created": created})
    except Exception as e:
        current_app.logger.exception("[external-seo] generate_and_schedule failed")
        return jsonify({"ok": False, "error": str(e)}), 500

from sqlalchemy.exc import IntegrityError
import secrets, time  # ★ 追加
import re as _re
from urllib.parse import urlparse
try:
    from unidecode import unidecode  # あれば日本語→ローマ字化
except Exception:
    def unidecode(x): return x


@bp.route("/external-seo/new-account", methods=["POST"])
@bp.route("/external-seo/new-account/", methods=["POST"])
@login_required
def external_seo_new_account():
    """
    Livedoorの仮アカウントを1件作成（必須カラムは存在確認してからセット）。
    例外時も必ずJSONで返す。
    """
    from flask import request, jsonify
    from app.models import Site, ExternalBlogAccount, BlogType
    from app import db
    import logging
    from datetime import datetime

    logger = logging.getLogger(__name__)

    # ---- ユーティリティ -------------------------------------------------
    def _stub_email(site_id: int) -> str:
        """email UNIQUE対策：衝突しないダミーを毎回生成"""
        # 例: pending-12-1723358300123-a3f1@stub.local
        return f"pending-{site_id}-{int(time.time()*1000)}-{secrets.token_hex(2)}@stub.local"

    def _stub_name(prefix: str, site_id: int) -> str:
        """username 用のダミー（安全にユニーク寄りに）"""
        # 例: u-12-1723358300123-a
        return f"{prefix}-{site_id}-{int(time.time()*1000)}-{secrets.token_hex(1)}"

    def _slugify_from_site(site: "Site") -> str:
        """
        サイト名/URLから display 用の短いスラッグを生成（a-z0-9-、先頭は英字、最大20文字）
        外部アカウントのカード表示に使う。DBの一意制約には関与しない。
        """
        base = (site.name or "")[:60]
        if not base and getattr(site, "url", None):
            try:
                host = urlparse(site.url).hostname or ""
                base = host.split(".")[0] if host else ""
            except Exception:
                base = ""

        if not base:
            base = f"site-{site.id}"

        s = unidecode(str(base)).lower()
        s = s.replace("&", " and ")
        s = _re.sub(r"[^a-z0-9]+", "-", s)
        s = _re.sub(r"-{2,}", "-", s).strip("-")
        if not s:
            s = f"site-{site.id}"
        if s[0].isdigit():
            s = "blog-" + s
        s = s[:20]
        if len(s) < 3:
            s = (s + "-blog")[:20]
        return s
    # --------------------------------------------------------------------

    try:
        site_id = request.form.get("site_id", type=int)
        if not site_id:
            return jsonify({"ok": False, "error": "site_id がありません"}), 200

        site = Site.query.get(site_id)
        if not site:
            return jsonify({"ok": False, "error": "Site が見つかりません"}), 200
        if (not current_user.is_admin) and (site.user_id != current_user.id):
            return jsonify({"ok": False, "error": "権限がありません"}), 200

        # 表示用スラッグ（カードのタイトルに使う）
        display_slug = _slugify_from_site(site)

        # UNIQUE衝突に備えて数回だけリトライ
        attempts = 0
        while True:
            try:
                # まず最小限の必須だけでインスタンス化（存在しない列は触らない）
                acc = ExternalBlogAccount(
                    site_id=site.id,
                    blog_type=BlogType.LIVEDOOR,
                )

                # --- カラムが存在する場合のみ安全にセット ---
                # UNIQUE の可能性がある email は必ずユニークなダミーにする
                if hasattr(acc, "email"):
                    acc.email = _stub_email(site.id)

                # username はダミー、nickname は表示に近い値（サイト由来スラッグ）を入れておく
                if hasattr(acc, "username"):
                    acc.username = _stub_name("u", site.id)
                if hasattr(acc, "password"):
                    acc.password = ""  # 仮
                if hasattr(acc, "nickname"):
                    acc.nickname = display_slug  # ← ここをサイト由来に

                # 状態系（存在すれば）
                if hasattr(acc, "status"):                acc.status = "active"
                if hasattr(acc, "message"):               acc.message = None
                if hasattr(acc, "cookie_path"):           acc.cookie_path = None
                if hasattr(acc, "livedoor_blog_id"):      acc.livedoor_blog_id = None
                if hasattr(acc, "atompub_key_enc"):       acc.atompub_key_enc = None
                if hasattr(acc, "api_post_enabled"):      acc.api_post_enabled = False
                if hasattr(acc, "is_captcha_completed"):  acc.is_captcha_completed = False
                # is_email_verified は存在しない環境があるため触らない
                if hasattr(acc, "posted_cnt"):            acc.posted_cnt = 0
                if hasattr(acc, "next_batch_started"):    acc.next_batch_started = None
                if hasattr(acc, "created_at"):            acc.created_at = datetime.utcnow()

                db.session.add(acc)
                db.session.commit()
                break  # ← 成功

            except IntegrityError:
                # email（や他の一意制約）衝突時は再採番してリトライ
                db.session.rollback()
                attempts += 1
                if attempts >= 5:
                    logger.exception("[external_seo_new_account] integrity error (retries exceeded)")
                    return jsonify({"ok": False, "error": "DBの一意制約で作成に失敗しました。時間をおいて再試行してください。"}), 200
                # ループ先頭で新しいダミーを採番して再作成

        account_payload = {
            "id": acc.id,
            # 表示名はサイト名ベースのスラッグを使用（カードのタイトルが人間にわかりやすくなる）
            "blog_title": display_slug,
            "public_url": None,
            "api_key": None,
            "stat_total": 0,
            "stat_posted": 0,
        }
        return jsonify({"ok": True, "site_id": site.id, "account": account_payload}), 200

    except Exception as e:
        db.session.rollback()
        logger.exception("[external_seo_new_account] error")
        return jsonify({"ok": False, "error": f"サーバエラー: {str(e)}"}), 200

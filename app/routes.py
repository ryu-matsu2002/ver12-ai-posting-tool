# app/routes.py

from __future__ import annotations
from datetime import datetime

from flask import (
    Blueprint, render_template, redirect, url_for,
    flash, request, abort, g, jsonify
)
from flask_login import (
    login_user, logout_user, login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from pytz import timezone
from sqlalchemy import asc, nulls_last

from . import db
from .models import User, Article, PromptTemplate, Site
from .forms import (
    LoginForm, RegisterForm,
    GenerateForm, PromptForm, ArticleForm, SiteForm
)
from .article_generator import enqueue_generation
from .wp_client import post_to_wp

JST = timezone("Asia/Tokyo")
bp = Blueprint("main", __name__)

# ───────────────────────────── 認証
@bp.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(request.args.get("next") or url_for(".dashboard"))
        flash("ログイン失敗", "danger")
    return render_template("login.html", form=form)

@bp.route("/register", methods=["GET", "POST"])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        if User.query.filter_by(email=form.email.data).first():
            flash("既に登録されています", "danger")
        else:
            db.session.add(User(
                email=form.email.data,
                password=generate_password_hash(form.password.data)
            ))
            db.session.commit()
            flash("登録完了！ログインしてください", "success")
            return redirect(url_for(".login"))
    return render_template("register.html", form=form)

@bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for(".login"))


# ───────────────────────────── Dashboard
@bp.route("/")
@login_required
def dashboard():
    g.total_articles = Article.query.filter_by(user_id=current_user.id).count()
    g.generating     = Article.query.filter_by(user_id=current_user.id, status="gen").count()
    g.done           = Article.query.filter_by(user_id=current_user.id, status="done").count()
    g.posted         = Article.query.filter_by(user_id=current_user.id, status="posted").count()
    return render_template("dashboard.html")

# ───────────────────────────── プロンプト CRUD
@bp.route("/prompts", methods=["GET", "POST"])
@login_required
def prompts():
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
        return redirect(url_for(".prompts"))

    plist = PromptTemplate.query.filter_by(user_id=current_user.id).all()
    return render_template("prompts.html", form=form, prompts=plist)

@bp.post("/prompts/delete/<int:pid>")
@login_required
def delete_prompt(pid: int):
    pt = PromptTemplate.query.get_or_404(pid)
    if pt.user_id != current_user.id:
        abort(403)
    db.session.delete(pt)
    db.session.commit()
    flash("削除しました", "success")
    return redirect(url_for(".prompts"))

@bp.route("/api/prompt/<int:pid>")
@login_required
def api_prompt(pid: int):
    pt = PromptTemplate.query.get_or_404(pid)
    if pt.user_id != current_user.id:
        abort(403)
    return jsonify({"title_pt": pt.title_pt, "body_pt": pt.body_pt})

# ───────────────────────────── WP サイト CRUD
@bp.route("/sites", methods=["GET", "POST"])
@login_required
def sites():
    form = SiteForm()
    if form.validate_on_submit():
        db.session.add(Site(
            name     = form.name.data,
            url      = form.url.data.rstrip("/"),
            username = form.username.data,
            app_pass = form.app_pass.data,
            user_id  = current_user.id
        ))
        db.session.commit()
        flash("サイトを登録しました", "success")
        return redirect(url_for(".sites"))

    site_list = Site.query.filter_by(user_id=current_user.id).all()
    return render_template("sites.html", form=form, sites=site_list)

@bp.post("/sites/<int:sid>/delete")
@login_required
def delete_site(sid: int):
    site = Site.query.get_or_404(sid)
    if site.user_id != current_user.id:
        abort(403)
    db.session.delete(site)
    db.session.commit()
    flash("サイトを削除しました", "success")
    return redirect(url_for(".sites"))

# ───────────────────────────── 記事生成
@bp.route("/generate", methods=["GET", "POST"])
@login_required
def generate():
    form = GenerateForm()

    form.genre_select.choices = [(0, "― 使わない ―")] + [
        (p.id, p.genre)
        for p in PromptTemplate.query.filter_by(user_id=current_user.id)
    ]
    form.site_select.choices = [(0, "―― 選択 ――")] + [
        (s.id, s.name)
        for s in Site.query.filter_by(user_id=current_user.id)
    ]

    if form.validate_on_submit():
        kws     = [k.strip() for k in form.keywords.data.splitlines() if k.strip()]
        site_id = form.site_select.data or None
        enqueue_generation(
            current_user.id,
            kws,
            form.title_prompt.data,
            form.body_prompt.data,
            site_id
        )
        flash(f"{len(kws)} 件をキューに登録しました", "success")
        return redirect(url_for(".log"))

    return render_template("generate.html", form=form)

# ───────────────────────────── 生成ログ
@bp.route("/log")
@login_required
def log():
    site_id = request.args.get("site_id", type=int)
    q = Article.query.filter_by(user_id=current_user.id)
    if site_id:
        q = q.filter_by(site_id=site_id)
    # scheduled_at を昇順、null を末尾、作成日時降順
    q = q.order_by(
        nulls_last(asc(Article.scheduled_at)),
        Article.created_at.desc()
    )
    return render_template(
        "log.html",
        articles=q.all(),
        sites=Site.query.filter_by(user_id=current_user.id).all(),
        site_id=site_id,
        jst=JST
    )

# ───────────────────────────── プレビュー
@bp.route("/preview/<int:article_id>")
@login_required
def preview(article_id: int):
    art = Article.query.get_or_404(article_id)
    if art.user_id != current_user.id:
        abort(403)
    return render_template("preview.html", article=art)

# ───────────────────────────── WordPress 即時投稿
@bp.post("/article/<int:id>/post")
@login_required
def post_article(id):
    art = Article.query.get_or_404(id)
    if art.user_id != current_user.id:
        abort(403)
    if not art.site:
        flash("投稿先サイトが設定されていません", "danger")
        return redirect(url_for(".log"))

    try:
        url = post_to_wp(art.site, art)
        art.posted_at = datetime.utcnow()
        art.status    = "posted"
        db.session.commit()
        flash(f"WordPress へ投稿しました: {url}", "success")
    except Exception as e:
        flash(f"投稿失敗: {e}", "danger")
    return redirect(url_for(".log"))

# ───────────────────────────── 記事編集・削除・再試行
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
        return redirect(url_for(".log"))
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
    return redirect(url_for(".log"))

@bp.post("/article/<int:id>/retry")
@login_required
def retry_article(id: int):
    art = Article.query.get_or_404(id)
    if art.user_id != current_user.id:
        abort(403)
    art.status, art.progress = "pending", 0
    db.session.commit()
    enqueue_generation(
        current_user.id,
        [art.keyword],
        art.title_pt or "",
        art.body_pt  or "",
        art.site_id
    )
    flash("再試行のキューに登録しました", "success")
    return redirect(url_for(".log"))

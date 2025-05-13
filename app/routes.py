from __future__ import annotations
from datetime import datetime

from flask import (
    Blueprint, render_template, redirect, url_for,
    flash, request, abort, g, jsonify, current_app
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
from .wp_client import post_to_wp, _decorate_html

# --- 既存の import の下に追加 ---
import re
import logging
from datetime import datetime
from .image_utils import fetch_featured_image  # ← ✅ 正しい


from .article_generator import (
    _unique_title,
    _compose_body,
)


JST = timezone("Asia/Tokyo")
bp = Blueprint("main", __name__)


# ─────────── 認証
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


# ─────────── Dashboard
@bp.route("/")
@login_required
def dashboard():
    g.total_articles = Article.query.filter_by(user_id=current_user.id).count()
    g.generating     = Article.query.filter_by(user_id=current_user.id, status="gen").count()
    g.done           = Article.query.filter_by(user_id=current_user.id, status="done").count()
    g.posted         = Article.query.filter_by(user_id=current_user.id, status="posted").count()
    return render_template("dashboard.html")


# ─────────── プロンプト CRUD
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


# ─────────── WP サイト CRUD
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


# ─────────── 記事生成
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


# ─────────── 生成ログ
@bp.route("/log")
@login_required
def log():
    site_id = request.args.get("site_id", type=int)

    # 未スケジュール記事の slot をサイトごとに自動割当
    from collections import defaultdict
    from .article_generator import _generate_slots_per_site  # ← 必須

    unscheduled = Article.query.filter(
        Article.user_id == current_user.id,
        Article.scheduled_at.is_(None),
    ).all()

    if unscheduled:
        # サイトごとに分類
        site_map = defaultdict(list)
        for art in unscheduled:
            if art.site_id:  # site_id が None の記事は無視
                site_map[art.site_id].append(art)

        # 各サイトごとにスロットを生成して割当
        for site_id, articles in site_map.items():
            slots = iter(_generate_slots_per_site(current_app, site_id, len(articles)))
            for art in articles:
                art.scheduled_at = next(slots)

        db.session.commit()

    # 記事取得
    q = Article.query.filter_by(user_id=current_user.id)
    if site_id:
        q = q.filter_by(site_id=site_id)
    q = q.order_by(
        nulls_last(asc(Article.scheduled_at)),
        Article.created_at.desc(),
    )

    return render_template(
        "log.html",
        articles=q.all(),
        sites=Site.query.filter_by(user_id=current_user.id).all(),
        site_id=site_id,
        jst=JST,
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
        return redirect(url_for(".log"))

    try:
        url = post_to_wp(art.site, art)
        art.posted_at = datetime.utcnow()
        art.status = "posted"
        db.session.commit()
        flash(f"WordPress へ投稿しました: {url}", "success")
    except Exception as e:
        current_app.logger.exception("即時投稿失敗: %s", e)
        db.session.rollback()
        flash(f"投稿失敗: {e}", "danger")

    return redirect(url_for(".log"))


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

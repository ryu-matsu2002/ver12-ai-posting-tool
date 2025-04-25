# app/forms.py

from flask_wtf import FlaskForm
from wtforms import (
    StringField,
    PasswordField,
    TextAreaField,
    SubmitField,
    SelectField,
    IntegerField,         # ← 新規追加
)
from wtforms.validators import DataRequired, Email, Length, EqualTo, URL, Optional

class LoginForm(FlaskForm):
    email    = StringField("Email",    validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit   = SubmitField("ログイン")

class RegisterForm(FlaskForm):
    email    = StringField("Email",    validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired(), Length(min=6)])
    confirm  = PasswordField("Confirm",  validators=[DataRequired(), EqualTo("password")])
    submit   = SubmitField("登録")

class GenerateForm(FlaskForm):
    keywords      = TextAreaField(
        "キーワード (改行区切り: 最大 40)",
        validators=[DataRequired(), Length(max=4000)]
    )
    title_prompt  = TextAreaField(
        "タイトル生成プロンプト",
        validators=[DataRequired()]
    )
    body_prompt   = TextAreaField(
        "本文生成プロンプト",
        validators=[DataRequired()]
    )
    body_length   = IntegerField(
        "本文文字数下限 (字)",       # ← 新規追加
        default=2000,               # ← デフォルト 2000 字
        validators=[Optional()]
    )
    genre_select  = SelectField(
        "保存済みプロンプト",
        choices=[(0, "― 使わない ―")],
        coerce=int,
        validators=[Optional()]
    )
    site_select   = SelectField(
        "投稿先サイト",
        choices=[(0, "―― 選択 ――")],
        coerce=int,
        validators=[Optional()]
    )
    submit        = SubmitField("生成開始")

class PromptForm(FlaskForm):
    genre    = StringField("ジャンル名", validators=[DataRequired(), Length(max=100)])
    title_pt = TextAreaField("タイトル用プロンプト", validators=[DataRequired()])
    body_pt  = TextAreaField("本文用プロンプト",      validators=[DataRequired()])
    submit   = SubmitField("保存")

class ArticleForm(FlaskForm):
    title  = StringField("タイトル", validators=[DataRequired()])
    body   = TextAreaField("本文 (HTML形式)", validators=[DataRequired()])
    submit = SubmitField("更新")

class SiteForm(FlaskForm):
    name     = StringField("サイト名",          validators=[DataRequired()])
    url      = StringField("サイトURL",        validators=[DataRequired(), URL()])
    username = StringField("ユーザー名",        validators=[DataRequired()])
    app_pass = StringField("アプリパスワード",  validators=[DataRequired()])
    submit   = SubmitField("保存")

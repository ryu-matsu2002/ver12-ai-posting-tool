# ──────────────────────────────────────────────
# app/forms.py

from flask_wtf import FlaskForm
from wtforms import (
    StringField,
    PasswordField,
    TextAreaField,
    SubmitField,
    SelectField,
    IntegerField,   
    HiddenField,            # ← 新規追加
)
from wtforms.validators import DataRequired, Email, Length, EqualTo, URL, Optional, NumberRange, ValidationError, Regexp

class LoginForm(FlaskForm):
    identifier = StringField("メールアドレスまたはユーザー名", validators=[DataRequired()])
    password   = PasswordField("パスワード", validators=[DataRequired()])
    submit     = SubmitField("ログイン")


class RegisterForm(FlaskForm):

    # ユーザー名（先頭に追加）
    username = StringField(
        "ユーザー名",
        validators=[
            DataRequired(),
            Length(min=3, max=50, message="3～50文字で入力してください。")
        ]
    )
    # メールアドレス・パスワード
    email    = StringField("メールアドレス", validators=[DataRequired(), Email()])
    password = PasswordField("パスワード", validators=[DataRequired(), Length(min=6)])
    confirm  = PasswordField("パスワード確認", validators=[DataRequired(), EqualTo("password")])

    # 区分（個人 or 法人）
    user_type = SelectField(
        "区分", choices=[("personal", "個人"), ("corporate", "法人")],
        validators=[DataRequired()]
    )

    # 法人用：会社名、会社名（フリガナ）
    company_name = StringField("会社名", validators=[Optional(), Length(max=100)])
    company_kana = StringField("会社名（フリガナ）", validators=[Optional(), Length(max=100)])

    # 氏名（姓・名）
    last_name  = StringField("姓", validators=[DataRequired(), Length(max=50)])
    first_name = StringField("名", validators=[DataRequired(), Length(max=50)])

    # フリガナ（姓・名）
    last_kana  = StringField("セイ", validators=[DataRequired(), Length(max=50)])
    first_kana = StringField("メイ", validators=[DataRequired(), Length(max=50)])

    # 郵便番号
    postal_code = StringField("郵便番号", validators=[DataRequired(), Length(max=10)])

    # 住所（統合）
    address = StringField("住所", validators=[DataRequired(), Length(max=200)])

    # 携帯電話番号
    phone = StringField("携帯電話番号", validators=[DataRequired(), Length(max=20)])

    # 新規登録用コード
    register_key = PasswordField("新規登録用コード", validators=[DataRequired()])

    submit = SubmitField("登録")


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
        validators=[Optional(), NumberRange(min=1, message="文字数は1文字以上でなければなりません。")]
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

    def validate_keywords(self, field):
        # キーワード数が40個を超えていないかを検証
        keywords = field.data.splitlines()
        if len(keywords) > 40:
            raise ValueError("キーワードは最大40個までです。")

class PromptForm(FlaskForm):
    id       = HiddenField() 
    genre    = StringField("プロンプト名", validators=[DataRequired(), Length(max=100)])
    title_pt = TextAreaField("タイトル用プロンプト", validators=[DataRequired()])
    body_pt  = TextAreaField("本文用プロンプト",      validators=[DataRequired()])
    submit   = SubmitField("保存")

class ArticleForm(FlaskForm):
    title  = StringField("タイトル", validators=[DataRequired()])
    body   = TextAreaField("本文 (HTML形式)", validators=[DataRequired()])
    submit = SubmitField("更新")

class SiteForm(FlaskForm):
    name     = StringField("サイト名",          validators=[DataRequired()])
    url      = StringField("サイトURL", validators=[
        DataRequired(),
        Regexp(
            r'^https:\/\/[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}(\/)?$',
            message='正しいURL形式（例：https://example.com）で入力してください'
        )
    ])
    username = StringField("ユーザー名",        validators=[DataRequired()])
    app_pass = StringField("アプリケーションパスワード",  validators=[DataRequired()])
    plan_type = SelectField(
        "プラン種別",
        choices=[("affiliate", "アフィリエイト用プラン"), ("business", "事業用プラン")],
        validators=[DataRequired()]
    )
    genre_id = SelectField(
    "ジャンル", 
    coerce=int, 
    choices=[], 
    validators=[Optional()]  # ✅ これがポイント
    )
    submit   = SubmitField("保存")

# ✅ 新バージョンの KeywordForm（サイト選択＋一括保存対応）
class KeywordForm(FlaskForm):
    site_id = SelectField(
        "対象サイト",
        coerce=int,
        validators=[DataRequired()]
    )

    keywords = TextAreaField(
        "キーワード（1行につき1キーワード・最大1000キーワード保存可能）",
        validators=[DataRequired(), Length(max=10000)]
    )
    submit = SubmitField("キーワードを保存")

    def validate_keywords(self, field):
        lines = field.data.splitlines()
        if len(lines) > 1000:
            raise ValidationError("キーワードは最大1000行までです。")

class EditKeywordForm(FlaskForm):
    site_id = SelectField("対象サイト", coerce=int, validators=[DataRequired()])
    keyword = StringField("キーワード", validators=[DataRequired(), Length(max=255)])
    submit = SubmitField("更新")

class ProfileForm(FlaskForm):
    # ユーザー名（初回のみ編集可能にする、テンプレート側で制御）
    username = StringField("ユーザー名", validators=[
        DataRequired(),
        Length(min=3, max=50, message="3～50文字で入力してください。")
    ])
    last_name = StringField("姓", validators=[DataRequired()])
    first_name = StringField("名", validators=[DataRequired()])
    last_kana = StringField("セイ", validators=[DataRequired()])
    first_kana = StringField("メイ", validators=[DataRequired()])
    phone = StringField("携帯電話番号", validators=[DataRequired()])
    postal_code = StringField("郵便番号", validators=[DataRequired()])
    address = StringField("住所", validators=[DataRequired(), Length(max=200)])
    submit = SubmitField("更新する")

class QuotaUpdateForm(FlaskForm):
    plan_type = SelectField(
        "プラン種別",
        choices=[
            ("affiliate", "アフィリエイト用プラン"),
            ("business", "事業用プラン")
        ],
        validators=[DataRequired()]
    )
    count = IntegerField(
        "追加サイト数",
        validators=[
            DataRequired(),
            NumberRange(min=1, max=100, message="1～100の数値で入力してください。")
        ]
    )
    reason = StringField(
        "追加理由",
        validators=[
            DataRequired(),
            Length(max=255)
        ]
    )
    submit = SubmitField("枠を追加する")

# ──── ジャンル登録・編集用フォーム ────
class GenreForm(FlaskForm):
    name = StringField("ジャンル名", validators=[DataRequired(), Length(max=100)])
    description = TextAreaField("説明", validators=[Length(max=300)])
    submit = SubmitField("保存")

# app/forms.py の適切な場所に追加

from wtforms import DateField

class RyunosukeDepositForm(FlaskForm):
    deposit_date = DateField("入金日", validators=[DataRequired()])
    amount = IntegerField("入金金額", validators=[DataRequired(), NumberRange(min=1)])
    memo = StringField("備考（任意）")
    submit = SubmitField("保存する")

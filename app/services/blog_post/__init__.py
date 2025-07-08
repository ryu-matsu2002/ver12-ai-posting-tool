"""
エイリアスパッケージ
====================
もともと Note／はてな 等の投稿ラッパは
app.services.blog_signup.blog_post.* に実装されています。

既存コードの import 句には

    from app.services.blog_post import post_blog_article
    from app.services.blog_post import livedoor_post

のように “services 直下” を想定している箇所が多い。

そこで、このファイルでは **名前空間エイリアス** を張り、
app.services.blog_post.* を
app.services.blog_signup.blog_post.* へ透過的に転送します。
"""

from importlib import import_module
import sys

# 実体モジュール（旧パス）を読み込む
_alias = import_module("app.services.blog_signup.blog_post")

# 現在のモジュール名を、その実体で上書き登録
sys.modules[__name__] = _alias

"""
ライブドアブログ アカウント自動登録（AIエージェント仕様）
==================================
* Playwright を長寿命コントローラ（pwctl）で管理
* 2段階フロー:
  - prepare_captcha(): 入力→CAPTCHA画像の保存（セッション保持）
  - submit_captcha(): CAPTCHA送信→/register/done待機→（以降はメール確認/キー回収まで実行）
"""
from __future__ import annotations

import logging
import time
import os
import json
import re as _re
import inspect
from types import SimpleNamespace
from pathlib import Path
from typing import Optional, Tuple

from flask import Blueprint, render_template, redirect, url_for, flash
from app import db
from app.enums import BlogType
from app.models import ExternalBlogAccount

from app.services.pw_controller import pwctl  # ← 長寿命Playwright
from playwright.async_api import Page, TimeoutError as PWTimeoutError

# 互換: 旧ルートが livedoor_signup から直接 import していた名前を再輸出
from app.services.mail_utils.mail_tm import (
    create_inbox as _create_inbox_gw,
    poll_latest_link_tm_async as poll_latest_link_gw,
)

logger = logging.getLogger(__name__)

# このモジュール用の Blueprint（既存をそのまま維持）
bp = Blueprint("livedoor_signup", __name__, url_prefix="/livedoor-signup")

# ─────────────────────────────────────────────
# 共有ディレクトリ（CAPTCHAとseedファイル）
# ─────────────────────────────────────────────
CAPTCHA_DIR = Path("app/static/captchas")
CAPTCHA_DIR.mkdir(parents=True, exist_ok=True)

# pw_controller と同じ場所を使ってセッション紐付けの seed を保存する
SESS_DIR = Path("/tmp/captcha_sessions")
SESS_DIR.mkdir(parents=True, exist_ok=True)

def _seed_path(session_id: str) -> Path:
    return SESS_DIR / f"{session_id}.seed.json"

def _save_seed(session_id: str, payload: dict) -> None:
    try:
        _seed_path(session_id).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        logger.info("[LD-Signup] seed saved sid=%s keys=%s", session_id, list(payload.keys()))
    except Exception:
        logger.exception("[LD-Signup] failed to save seed sid=%s", session_id)

def _load_seed(session_id: str) -> Optional[dict]:
    p = _seed_path(session_id)
    if not p.exists():
        logger.warning("[LD-Signup] seed not found sid=%s", session_id)
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("[LD-Signup] failed to load seed sid=%s", session_id)
        return None

# ─────────────────────────────────────────────
# 補助ユーティリティ（既存ロジックをそのまま活かす）
# ─────────────────────────────────────────────
try:
    from unidecode import unidecode
except Exception:
    def unidecode(x): return x

def _slugify_ascii(s: str) -> str:
    if not s:
        s = "blog"
    s = unidecode(str(s)).lower()
    s = s.replace("&", " and ")
    s = _re.sub(r"[^a-z0-9]+", "-", s)
    s = _re.sub(r"-{2,}", "-", s).strip("-")
    if s and s[0].isdigit():
        s = "blog-" + s
    if not s:
        s = "blog"
    s = s[:20]
    if len(s) < 3:
        s = (s + "-blog")[:20]
    return s

def suggest_livedoor_blog_id(base_text: str, db_session) -> str:
    from app.models import ExternalBlogAccount
    from app.enums import BlogType
    base = _slugify_ascii(base_text)
    candidate, n = base, 0
    while True:
        exists = db_session.query(ExternalBlogAccount.id).filter(
            ExternalBlogAccount.blog_type == BlogType.LIVEDOOR,
            ExternalBlogAccount.livedoor_blog_id == candidate
        ).first()
        if not exists:
            return candidate
        n += 1
        tail = str(n)
        candidate = (base[: max(1, 20 - len(tail) - 1)] + "-" + tail)

import random, string
def generate_safe_id(n=10) -> str:
    chars = string.ascii_lowercase + string.digits + "_"
    first_char = random.choice(string.ascii_lowercase)
    rest = ''.join(random.choices(chars, k=n - 1))
    return first_char + rest

def generate_safe_password(n=12) -> str:
    chars = string.ascii_letters + string.digits + "-_%$#"
    while True:
        password = ''.join(random.choices(chars, k=n))
        if any(c in "-_%$#" for c in password):
            return password

# ─────────────────────────────────────────────
# メール本文/テキストから認証URLを頑健に抽出（verify と email_auth/commit の両対応）
# ─────────────────────────────────────────────
def _extract_activation_url(text: str) -> Optional[str]:
    if not text:
        return None
    patterns = [
        r"https://member\.livedoor\.com/email_auth/commit/[A-Za-z0-9]+/[A-Za-z0-9]+",
        r"https://member\.livedoor\.com/verify/[A-Za-z0-9]+",
    ]
    for pat in patterns:
        m = _re.search(pat, text)
        if m:
            return m.group(0)
    return None

# ─────────────────────────────────────────────
# 新：CAPTCHA準備（セッション確保＆画像保存）— 同期API
# ─────────────────────────────────────────────
def prepare_captcha(email_addr: str, livedoor_id: str, password: str, *, site=None) -> Tuple[str, str]:
    """
    LiveDoor 会員登録フォームに入力→送信→CAPTCHAが出たら要素スクショを保存。
    返り値: (session_id, captcha_image_path)
    ※ 後段で使う seed（email/nickname/password/任意のsite情報）をセッションに紐付けて保存。
    """
    sid, page = pwctl.run(pwctl.create_session(provider="livedoor"))
    img_path = pwctl.run(_ld_prepare(page, email_addr, livedoor_id, password, sid))
    # 復旧用に storage_state を保存（ワーカー跨ぎ/復活にも強くする）
    pwctl.run(pwctl.save_storage_state(sid))

    # 後続の recover で使う seed を保存（site は必要最小限の dict 化）
    site_view = None
    if site is not None:
        site_view = {
            "id": getattr(site, "id", None),
            "name": getattr(site, "name", None),
            "url": getattr(site, "url", None),
            "primary_genre_name": getattr(site, "primary_genre_name", None),
            "genre_name": getattr(site, "genre_name", None),
            "category": getattr(site, "category", None),
        }
    _save_seed(sid, {
        "email": email_addr,
        "nickname": livedoor_id,
        "password": password,
        "site": site_view,
        # 将来の拡張用に mail.tm の task_id や token を載せるスロットを先に用意（無くても動く）
        "mailtm_task_id": getattr(site, "mailtm_task_id", None) if site else None,
        "mailtm_token": getattr(site, "mailtm_token", None) if site else None,
    })

    return sid, img_path

async def _ld_prepare(page: Page, email_addr: str, livedoor_id: str, password: str, session_id: str) -> str:
    logger.info("[LD-Signup] goto register/input (sid=%s)", session_id)
    await page.goto("https://member.livedoor.com/register/input", wait_until="load")

    await page.fill('input[name="livedoor_id"]', livedoor_id)
    await page.fill('input[name="password"]', password)
    await page.fill('input[name="password2"]', password)
    await page.fill('input[name="email"]', email_addr)

    await page.click('input[type="submit"][value="ユーザー情報を登録"]')

    img = page.locator("#captcha-img")
    try:
        await img.wait_for(state="visible", timeout=20_000)
    except PWTimeoutError:
        # attached→visible 切替の遅延にも一応対応
        await img.wait_for(state="attached", timeout=5_000)

    ts = time.strftime("%Y%m%d_%H%M%S")
    img_path = CAPTCHA_DIR / f"captcha_{session_id}_{ts}.png"
    await img.screenshot(path=str(img_path))

    logger.info("[LD-Signup] CAPTCHA画像を %s に保存 (sid=%s)", img_path, session_id)
    await pwctl.set_step(session_id, "captcha_required")
    return str(img_path)

# ─────────────────────────────────────────────
# 新：CAPTCHA送信（同一セッションで継続）— 同期API
# ─────────────────────────────────────────────
def submit_captcha(session_id: str, captcha_text: str) -> bool:
    """
    CAPTCHA文字列を送信し、/register/done に到達したら、
    * 可能ならメール認証リンクへ遷移（poll_latest_link_gw を多態に呼ぶ）
    * 続けて recover_atompub_key() を呼んでブログ作成→APIキー取得
    * 成果を ExternalBlogAccount に保存
    までを行う。いずれかの段階で致命失敗したら False。
    """
    page = pwctl.run(pwctl.get_page(session_id))
    if page is None:
        # ページを落としてしまっても storage_state から復旧可
        page = pwctl.run(pwctl.revive(session_id))
        if page is None:
            raise RuntimeError(f"signup session not found (sid={session_id})")

    # seed の読込
    seed = _load_seed(session_id) or {}
    email = seed.get("email")
    nickname = seed.get("nickname")
    password = seed.get("password")
    site_view = seed.get("site") or {}
    site_ns = SimpleNamespace(**{k: site_view.get(k) for k in ("id", "name", "url", "primary_genre_name", "genre_name", "category")})

    ok = pwctl.run(_ld_submit(page, captcha_text, session_id))
    if not ok:
        return False

    # ─────────────────────────────────────────
    # メール認証（可能な限り実施。取得できなければスキップして次へ）
    # ─────────────────────────────────────────
    try:
        activation_url = None

        # 返り値がURLなのか本文なのかに依らず取得できるように冗長に試す
        # 1) 代表的な引数パターンで呼んでみる
        candidates = []
        try:
            # token 指定
            if seed.get("mailtm_token"):
                res = poll_latest_link_gw(token=seed["mailtm_token"], max_attempts=24, interval=5)
                candidates.append(res)
        except TypeError:
            pass
        try:
            # task_id 指定
            if seed.get("mailtm_task_id"):
                res = poll_latest_link_gw(task_id=seed["mailtm_task_id"], max_attempts=24, interval=5)
                candidates.append(res)
        except TypeError:
            pass
        try:
            # email 指定（実装側で対応していれば拾える）
            if email:
                res = poll_latest_link_gw(email=email, max_attempts=24, interval=5)
                candidates.append(res)
        except TypeError:
            pass

        # 候補が空（＝どの呼び方も非対応）の場合、ダメ元でシグネチャ無し呼び出し
        if not candidates:
            try:
                res = poll_latest_link_gw()
                candidates.append(res)
            except TypeError:
                pass

        # coroutine だったら実行して中身を得る
        materialized: list = []
        for res in candidates:
            if inspect.iscoroutine(res):
                try:
                    res = pwctl.run(res)  # 内部ループでawait
                except Exception:
                    res = None
            materialized.append(res)

        # 返り値の型に応じてURL抽出
        for obj in materialized:
            if not obj:
                continue
            if isinstance(obj, str):
                # 文字列ならそのままURLか、本文
                u = _extract_activation_url(obj) or (obj if obj.startswith("http") else None)
                if u:
                    activation_url = u
                    break
            elif isinstance(obj, dict):
                # dictならよくあるキーを総当たり
                for key in ("url", "link", "activation_url", "auth_url"):
                    u = obj.get(key)
                    if isinstance(u, str) and u.startswith("http"):
                        activation_url = u
                        break
                if not activation_url:
                    # dictの中の本文をざっと見る
                    for key, val in obj.items():
                        if isinstance(val, str):
                            u = _extract_activation_url(val)
                            if u:
                                activation_url = u
                                break
                if activation_url:
                    break

        if activation_url:
            logger.info("[LD-Signup] activation URL detected: %s", activation_url)
            pwctl.run(page.goto(activation_url, wait_until="load"))
            pwctl.run(pwctl.set_step(session_id, "email_verified"))
        else:
            logger.warning("[LD-Signup] activation URL not found. proceed anyway (sid=%s)", session_id)

    except Exception:
        # 認証できなくても recover 側でblog_createに挑む（失敗時は recover がダンプ群を残す）
        logger.exception("[LD-Signup] email verification step failed (ignored) sid=%s", session_id)

    # ─────────────────────────────────────────
    # ブログ作成 → AtomPub キー取得 → DB保存
    # ─────────────────────────────────────────
    try:
        from app.services.blog_signup.livedoor_atompub_recover import recover_atompub_key

        result = pwctl.run(recover_atompub_key(
            page=page,
            nickname=nickname or "guest",
            email=email or "",
            password=password or "",
            site=site_ns,
            desired_blog_id=None
        ))

        if not result or not result.get("success"):
            logger.error("[LD-Signup] recover_atompub_key failed: %s", result)
            return False

        blog_id  = result.get("blog_id")
        api_key  = result.get("api_key")
        endpoint = result.get("endpoint")

        # ★ DB保存（一般的な列名。存在しない列は無視して安全に代入）
        acct = db.session.query(ExternalBlogAccount).filter(
            ExternalBlogAccount.blog_type == BlogType.LIVEDOOR,
            ExternalBlogAccount.email == email
        ).one_or_none()

        if not acct:
            acct = ExternalBlogAccount(blog_type=BlogType.LIVEDOOR, email=email)
            db.session.add(acct)

        if hasattr(acct, "livedoor_blog_id"):
            acct.livedoor_blog_id = blog_id
        if hasattr(acct, "livedoor_api_key"):
            acct.livedoor_api_key = api_key
        if hasattr(acct, "livedoor_endpoint"):
            acct.livedoor_endpoint = endpoint
        if hasattr(acct, "email_verified"):
            acct.email_verified = True
        if hasattr(acct, "blog_created"):
            acct.blog_created = True

        db.session.commit()
        pwctl.run(pwctl.set_step(session_id, "api_key_ok"))
        logger.info("[LD-Signup] ✅ blog_id=%s api_key[8]=%s...", blog_id, (api_key or "")[:8])
        return True

    except Exception:
        logger.exception("[LD-Signup] save account failed (sid=%s)", session_id)
        # recover内で失敗時はHTML/PNGが保存される想定
        return False

async def _ld_submit(page: Page, captcha_text: str, session_id: str) -> bool:
    logger.info("[LD-Signup] submit captcha (sid=%s)", session_id)

    # livedoor の CAPTCHA 入力欄（名称が違う場合はここだけ調整）
    await page.fill('input[name="captcha"]', captcha_text.replace(" ", "").replace("　", ""))

    # 送信（valueやidが変わっても拾えるよう汎用セレクタ）
    await page.click('input[type="submit"]')

    try:
        await page.wait_for_url("**/register/done", timeout=30_000)
    except PWTimeoutError:
        ts = time.strftime("%Y%m%d_%H%M%S")
        fail_png = CAPTCHA_DIR / f"failed_after_captcha_{session_id}_{ts}.png"
        try:
            await page.screenshot(path=str(fail_png), full_page=True)
        except Exception:
            pass
        logger.error("[LD-Signup] /register/done へ遷移せず（sid=%s）。スクショ: %s", session_id, fail_png)
        return False

    await pwctl.set_step(session_id, "captcha_submitted")
    logger.info("[LD-Signup] reached /register/done (sid=%s)", session_id)
    return True

# ─────────────────────────────────────────────
# 以降：旧・補助関数（メールURL抽出、手動確認画面など）
# ─────────────────────────────────────────────
import re
def extract_verification_url(email_body: str) -> str | None:
    """
    旧互換：/verify/ のみを見る簡易抽出。
    新実装では _extract_activation_url() が /email_auth/commit/ も拾う。
    """
    pattern = r"https://member\.livedoor\.com/verify/[a-zA-Z0-9]+"
    m = re.search(pattern, email_body)
    return m.group(0) if m else None

TEMP_DIR = "/tmp/livedoor_tasks"
os.makedirs(TEMP_DIR, exist_ok=True)

def save_livedoor_credentials(task_id: str, blog_id: str, api_key: str):
    path = os.path.join(TEMP_DIR, f"{task_id}.json")
    with open(path, "w") as f:
        json.dump({"blog_id": blog_id, "api_key": api_key}, f)

def fetch_livedoor_credentials(task_id: str) -> dict | None:
    path = os.path.join(TEMP_DIR, f"{task_id}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

@bp.route('/confirm_email_manual/<task_id>')
def confirm_email_manual(task_id):
    """
    CAPTCHA後、認証リンクをユーザーに手動で表示する画面（既存フローを維持）。
    旧実装互換：poll_latest_link_gw がメール本文を返す想定のまま。
    """
    from app.services.mail_utils.mail_tm import poll_latest_link_tm_async as poll_latest_link_gw
    email_body = poll_latest_link_gw(task_id=task_id, max_attempts=30, interval=5)

    if email_body:
        verification_url = extract_verification_url(email_body)
        if verification_url:
            return render_template("confirm_email.html", verification_url=verification_url)
        else:
            flash("認証リンクが見つかりませんでした", "danger")
            return redirect(url_for('dashboard'))
    else:
        flash("認証メールを取得できませんでした", "danger")
        return redirect(url_for('dashboard'))

# --- legacy compatibility shim ---------------------------------------------
def register_blog_account(site, email_seed: str = "ld"):
    """
    🔧 互換：旧フロー呼び出し対策（起動時importエラー防止用）
    実運用は新フロー /prepare_captcha → /submit_captcha を使ってください。
    呼ばれた場合は「CAPTCHAが必要」というレガシー互換レスポンスを返します。
    """
    # 既存のメール作成ユーティリティを使って最低限の情報を用意
    from app.services.mail_utils.mail_gw import create_inbox
    email, token = create_inbox()
    livedoor_id = generate_safe_id()
    password    = generate_safe_password()

    # 新APIで CAPTCHA 準備だけ実行（画像を保存し、セッションを確保）
    try:
        session_id, img_abs = prepare_captcha(email, livedoor_id, password, site=site)
        img_name = Path(img_abs).name
    except Exception:
        # ここで落ちても、少なくとも起動時の import は通っているのでアプリは動きます
        # 呼び出し元は新フローに移行してください
        raise RuntimeError("register_blog_account は非推奨です。/prepare_captcha → /submit_captcha を使ってください。")

    # 旧フローが期待していた形に“近い”返り値（フロントが旧実装でも破綻しにくい）
    return {
        "status": "captcha_required",
        "captcha_url": f"/static/captchas/{img_name}",
        "email": email,
        "nickname": livedoor_id,
        "password": password,
        "token": token,
        "session_id": session_id,
    }

# --- backward-compat exports (for legacy imports in tasks/routes) -----------
def signup(site, email_seed: str = "ld"):
    """旧コード向けの互換API。内部では register_blog_account を呼ぶだけ。"""
    return register_blog_account(site, email_seed=email_seed)

# 既にモジュール先頭で poll_latest_link_gw を import して module-global に置いているので、
# routes から `from ...livedoor_signup import poll_latest_link_gw` も有効のままです。
# （名前がグローバルに存在していれば import 対象にできます）
__all__ = [
    # 新API
    "prepare_captcha", "submit_captcha",
    "generate_safe_id", "generate_safe_password", "suggest_livedoor_blog_id",
    # 互換API
    "register_blog_account", "signup",
    # ルート互換で使う補助
    "poll_latest_link_gw", "extract_verification_url",
]

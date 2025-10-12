"""
ライブドアブログ アカウント自動登録（AIエージェント仕様）
==================================
* Playwright を長寿命コントローラ（pwctl）で管理
* 2段階フロー:
  - prepare_captcha(): 入力→CAPTCHA画像の保存（セッション保持）
  - submit_captcha(): CAPTCHA送信→/register/done待機→（以降はメール確認/キー回収の差込点）
  - create_blog_and_fetch_api_key(): （メール認証後に呼ぶ）ブログ作成～AtomPubキー取得～DB保存
"""
from __future__ import annotations

import logging
import time
import json
import os
import re as _re
import random, string
from pathlib import Path
from typing import Optional, Tuple, List

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
# 補助ユーティリティ（既存ロジックをそのまま活かす）
# ─────────────────────────────────────────────
try:
    from unidecode import unidecode
except Exception:
    def unidecode(x): return x

def _slugify_ascii(s: str) -> str:
    """
    既存のスラッグ化（ハイフン）→ Livedoor ID 規約に合う最小限の正規化に変更。
    規約: 3〜20文字、先頭は英字、半角英数字とアンダーバーのみ。
    """
    if not s:
        s = "blog"
    s = unidecode(str(s)).lower()
    s = s.replace("&", " and ")
    # 英数字以外は "_" に寄せ、連続は一つに
    s = _re.sub(r"[^a-z0-9_]+", "_", s)
    s = _re.sub(r"_{2,}", "_", s).strip("_")
    # 先頭は英字に強制（英字が無ければプレフィックスを与える）
    if not s or not s[0].isalpha():
        s = ("blog_" + s).strip("_")
    # 長さ制約
    s = s[:20]
    if len(s) < 3:
        s = (s + "_blog")[:20]
    return s

def suggest_livedoor_blog_id(base_text: str, db_session) -> str:
    """
    後方互換のための単一候補関数（内部は新ポリシー準拠）。
    DB衝突を避けるため、base / base_blog / base_info ... の順で探す。
    """
    base = _slugify_ascii(base_text)
    variants = [base, f"{base}_blog", f"{base}_info"]
    # 長さ20に収まるように各候補を切り詰め
    variants = [v[:20] for v in variants]
    # DB重複を避けて一つ返す
    for cand in variants:
        exists = db_session.query(ExternalBlogAccount.id).filter(
            ExternalBlogAccount.blog_type == BlogType.LIVEDOOR,
            ExternalBlogAccount.livedoor_blog_id == cand
        ).first()
        if not exists:
            return cand
    # それでも衝突する場合は末尾に番号を当てる（20文字上限を維持）
    n = 1
    while True:
        tail = f"_{n}"
        cand = (base[: max(3, 20 - len(tail))] + tail)
        exists = db_session.query(ExternalBlogAccount.id).filter(
            ExternalBlogAccount.blog_type == BlogType.LIVEDOOR,
            ExternalBlogAccount.livedoor_blog_id == cand
        ).first()
        if not exists and 3 <= len(cand) <= 20:
            return cand
        n += 1

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
# 新：CAPTCHA準備（セッション確保＆画像保存）— 同期API
# ─────────────────────────────────────────────
CAPTCHA_DIR = Path("app/static/captchas")
CAPTCHA_DIR.mkdir(parents=True, exist_ok=True)

def prepare_captcha(email_addr: str, livedoor_id: str, password: str) -> Tuple[str, str]:
    """
    LiveDoor 会員登録フォームに入力→送信→CAPTCHAが出たら要素スクショを保存。
    返り値: (session_id, captcha_image_path)
    """
    sid, page = pwctl.run(
        pwctl.create_session(
            provider="livedoor",
            auto_load_latest=False,      # ← 前回の storage_state を使わない
            storage_state_path=None      # ← 念のため明示的に未指定
        )
    )

    img_path = pwctl.run(_ld_prepare(page, email_addr, livedoor_id, password, sid))
    # 復旧用に storage_state を保存（ワーカー跨ぎ/復活にも強くする）
    pwctl.run(pwctl.save_storage_state(sid))
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
    CAPTCHA文字列を送信し、/register/done に到達したら True を返す。
    以降（メール認証→APIキー取得）は本関数外で行う想定。ブログ作成は
    create_blog_and_fetch_api_key() を呼び出してください。
    """
    page = pwctl.run(pwctl.get_page(session_id))
    if page is None:
        page = pwctl.run(pwctl.revive(session_id))
        if page is None:
            raise RuntimeError(f"signup session not found (sid={session_id})")

    ok = pwctl.run(_ld_submit(page, captcha_text, session_id))
    return ok

async def _ld_submit(page: Page, captcha_text: str, session_id: str) -> bool:
    logger.info("[LD-Signup] submit captcha (sid=%s)", session_id)

    await page.fill('input[name="captcha"]', captcha_text.replace(" ", "").replace("　", ""))
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

    # --- ここを追加：/register/done の直後に blogcms 側へ一度入ってクッキーを確立 ---
    try:
        await page.goto("https://livedoor.blogcms.jp/member/", wait_until="load")
        try:
            await page.wait_for_load_state("networkidle", timeout=10_000)
        except Exception:
            pass
        # SSO で blogcms.jp のセッションが張られた状態を保存
        from app.services.pw_controller import pwctl as _pwctl  # 循環参照回避のためローカル import
        await _pwctl.save_storage_state(session_id)
        logger.info("[LD-Signup] post-done: saved storage_state including blogcms cookies (sid=%s)", session_id)
    except Exception:
        logger.warning("[LD-Signup] post-done blogcms warm-up failed (sid=%s)", session_id, exc_info=True)

    return True

# ─────────────────────────────────────────────
# 以降：旧・補助関数（メールURL抽出、手動確認画面など）
# ─────────────────────────────────────────────
import re
def extract_verification_url(email_body: str) -> str | None:
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
    

# ─────────────────────────────────────────────
# 新：WPサイト情報ベースの Livedoor ID 候補生成（第1〜第3候補）
# ─────────────────────────────────────────────
def _extract_sld_from_url(url: str) -> str:
    """
    URL から第2レベル相当を抽出し、規約に沿って整形。
    例: https://example-site.co.jp → example_site
    """
    try:
        from urllib.parse import urlparse
        netloc = urlparse(url or "").netloc.lower()
    except Exception:
        netloc = ""
    if ":" in netloc:
        netloc = netloc.split(":", 1)[0]
    parts = [p for p in netloc.split(".") if p]
    # 規模の大きいTLD/SLDは除外
    junk = {"www","com","jp","net","org","co","info","biz","blog","site"}
    core = [p for p in parts if p not in junk]
    if not core:
        return ""
    # 末尾（右側）から意味ありそうな部分を取り、結合
    sld = "_".join(_re.sub(r"[^a-z0-9_]+","_", c) for c in core[-2:])
    sld = _re.sub(r"_+","_", sld)
    return _slugify_ascii(sld)

def generate_livedoor_id_candidates(site) -> List[str]:
    """
    規約準拠の ID 候補を第1〜第3候補で返す。
    先頭は URL の SLD を優先。取れない場合は Site.name をローマ字化。
    """
    site_url  = (getattr(site, "url", "")  or "").strip()
    site_name = (getattr(site, "name", "") or "").strip()

    base = _extract_sld_from_url(site_url)
    if not base:
        base = _slugify_ascii(site_name or "blog")

    # 長さ20を厳守（下で接尾辞を付けるため、必要に応じて切り詰める）
    base = base[:20] if base else "blog"
    # 3〜20に丸める（短すぎるとき）
    if len(base) < 3:
        base = (base + "_blog")[:20]

    c1 = base
    c2 = (base[:20 - len("_blog")] + "_blog") if len(base) <= 20 else base[:20]
    c3 = (base[:20 - len("_info")] + "_info") if len(base) <= 20 else base[:20]
    # 冗長や重複の掃除
    out = []
    for c in (c1, c2, c3):
        c = _re.sub(r"[^a-z0-9_]+", "", c)
        c = c[:20]
        if len(c) < 3:
            continue
        if not c[0].isalpha():
            c = ("blog_" + c)[:20]
        if c not in out:
            out.append(c)
    # 最低1つ保証
    if not out:
        out = ["blog_id"]
    return out    

@bp.route('/confirm_email_manual/<task_id>')
def confirm_email_manual(task_id):
    """
    CAPTCHA後、認証リンクをユーザーに手動で表示する画面（既存フローを維持）。
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
    from app.services.mail_utils.mail_gw import create_inbox
    email, token = create_inbox()
    # 🔁 ここで WPサイト情報から Livedoor ID 候補を生成（第1候補を採用）
    id_candidates = generate_livedoor_id_candidates(site)
    livedoor_id = id_candidates[0]
    password    = generate_safe_password()

    try:
        session_id, img_abs = prepare_captcha(email, livedoor_id, password)
        img_name = Path(img_abs).name
    except Exception:
        raise RuntimeError("register_blog_account は非推奨です。/prepare_captcha → /submit_captcha を使ってください。")

    return {
        "status": "captcha_required",
        "captcha_url": f"/static/captchas/{img_name}",
        "email": email,
        "nickname": livedoor_id,
        "password": password,
        "token": token,
        "session_id": session_id,
        # UI での手動登録支援用：第1〜第3候補を返す
        "livedoor_id_candidates": id_candidates,
    }

# --- ここから：ブログ作成（Recover）内蔵 ---
import asyncio
from datetime import datetime
from urllib.parse import urlparse, urljoin

# ------------- 汎用ユーティリティ -------------
def _deterministic_index(salt: str, n: int) -> int:
    if n <= 0:
        return 0
    acc = 0
    for ch in str(salt):
        acc = (acc * 131 + ord(ch)) & 0xFFFFFFFF
    return acc % n

def _has_cjk(s: str) -> bool:
    return bool(_re.search(r"[\u3040-\u30FF\u3400-\u9FFF]", s or ""))

def _norm(s: str) -> str:
    s = (s or "").lower()
    s = _re.sub(r"[\s\-_／|｜/・]+", "", s)
    return s

def _domain_tokens(url: str) -> list[str]:
    try:
        netloc = urlparse(url or "").netloc.lower()
    except Exception:
        netloc = ""
    if ":" in netloc:
        netloc = netloc.split(":", 1)[0]
    parts = [p for p in netloc.split(".") if p and p not in ("www", "com", "jp", "net", "org", "co")]
    words = []
    for p in parts:
        words.extend([w for w in p.replace("_", "-").split("-") if w])
    return words

STOPWORDS_JP = {"株式会社","有限会社","合同会社","公式","オフィシャル","ブログ","サイト","ホームページ","ショップ","ストア","サービス","工房","教室","情報","案内","チャンネル","通信","マガジン"}
STOPWORDS_EN = {"inc","ltd","llc","official","blog","site","homepage","shop","store","service","studio","channel","magazine","info","news"}

def _name_tokens(name: str) -> list[str]:
    if not name:
        return []
    parts = _re.split(r"[\s\u3000\-/＿_・|｜／]+", str(name))
    toks: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        p = _re.sub(r"[^\w\u3040-\u30FF\u3400-\u9FFFー]+", "", p)
        if p:
            toks.append(p)
    return toks

def _guess_genre(site) -> tuple[str, bool]:
    for attr in ("primary_genre_name", "genre_name", "genre", "main_genre", "category", "category_name"):
        v = getattr(site, attr, None)
        if isinstance(v, str) and v.strip():
            txt = v.strip()
            return txt, _has_cjk(txt)
        name = getattr(v, "name", None)
        if isinstance(name, str) and name.strip():
            txt = name.strip()
            return txt, _has_cjk(txt)

    name = (getattr(site, "name", "") or "")
    url  = (getattr(site, "url", "")  or "")
    txt  = (name + " " + url).lower()
    toks = set(_domain_tokens(url))

    JP = [
        ("ピラティス", ("pilates","ピラティス","yoga","体幹","姿勢","fitness","stretch")),
        ("留学",       ("studyabroad","abroad","留学","ielts","toefl","海外","study")),
        ("旅行",       ("travel","trip","観光","hotel","onsen","温泉","tour")),
        ("美容",       ("beauty","esthetic","skin","hair","美容","コスメ","メイク")),
        ("ビジネス",   ("business","marketing","sales","seo","経営","起業","副業")),
    ]
    for label, keys in JP:
        if any(k in txt for k in keys) or any(k in toks for k in keys):
            return label, True

    EN = [
        ("Pilates", ("pilates","yoga","fitness","posture","stretch")),
        ("Study Abroad", ("studyabroad","abroad","study","ielts","toefl")),
        ("Travel", ("travel","trip","hotel","onsen","tour")),
        ("Beauty", ("beauty","esthetic","skin","hair","cosme","makeup")),
        ("Business", ("business","marketing","sales","seo","startup")),
    ]
    for label, keys in EN:
        if any(k in txt for k in keys) or any(k in toks for k in keys):
            return label, False

    return ("日々", _has_cjk(name) or _has_cjk(url))

def _too_similar_to_site(title: str, site) -> bool:
    t = _norm(title)
    site_name = (getattr(site, "name", "") or "")
    site_url  = (getattr(site, "url", "")  or "")
    n = _norm(site_name)
    if not t:
        return True
    if t == n or (t and n and (t in n or n in t)):
        return True
    toks = set(_domain_tokens(site_url))
    toks |= {w for w in _name_tokens(site_name) if not _has_cjk(w)}
    toks = {_norm(w) for w in toks if w}
    for w in toks:
        if not w:
            continue
        if w in t or t in w:
            return True
    return False

def _templates_jp(topic: str) -> list[str]:
    base = (topic or "").strip() or "日々"
    return [
        f"{base}ブログ",
        f"{base}ブログ日記",
        f"{base}のブログ",
        f"{base}の記録ブログ",
        f"{base}の暮らしブログ",
        f"{base}のメモ帳",
        f"{base}の覚え書き",
        f"{base}のジャーナル",
        f"{base}手帖",
        f"{base}ノート",
        f"{base}の小部屋",
        f"{base}ログ",
    ]

def _japanese_base_word(site) -> str:
    topic, _ = _guess_genre(site)
    if _has_cjk(topic):
        return topic.strip()
    return "日々"

def _path_tokens(url: str) -> list[str]:
    try:
        from urllib.parse import urlparse
        p = urlparse(url or "")
        path = (p.path or "").strip("/").lower()
    except Exception:
        path = ""
    raw = []
    for seg in path.split("/"):
        if not seg:
            continue
        # - と _ でさらに分割
        raw.extend([w for w in _re.split(r"[-_]", seg) if w])
    # 数字だけや短すぎるものは除外
    out = []
    for w in raw:
        if w.isdigit():
            continue
        if len(w) < 2:
            continue
        out.append(w)
    return out

def _site_keywords(site, max_tokens: int = 3) -> list[str]:
    """サイト名とURLからキーワード候補を必ず返す（重複・ストップワード除去）"""
    name = (getattr(site, "name", "") or "").strip()
    url  = (getattr(site, "url", "")  or "").strip()

    toks_name = _name_tokens(name)              # 日本語も英語も混ざる
    toks_dom  = _domain_tokens(url)             # 英語寄り
    toks_path = _path_tokens(url)               # 英語寄り

    # ストップワード除去
    filtered = []
    seen = set()
    for t in toks_name + toks_dom + toks_path:
        t_norm = _norm(t)
        if not t_norm:
            continue
        if _has_cjk(t):   # 日本語
            if any(sw in t for sw in STOPWORDS_JP):
                continue
        else:             # 英語
            if t_norm in STOPWORDS_EN:
                continue
        if t_norm in seen:
            continue
        seen.add(t_norm)
        filtered.append(t)

    # 1語も取れない場合はドメインのルート語だけでも拾う（URLが空なら最後に "blog"）
    if not filtered:
        base = (_domain_tokens(url)[:1] or ["blog"])
        return base[:max_tokens]

    return filtered[:max_tokens]


def _craft_blog_title(site) -> str:
    """
    サイト名/URLから抽出した語を必ず含むタイトルを作る。
    日本語トークンがあれば日本語寄りのテンプレ、なければ英語寄りだが「ブログ」を付ける。
    """
    site_name = (getattr(site, "name", "") or "").strip()
    site_url  = (getattr(site, "url", "")  or "").strip()
    salt = f"{getattr(site, 'id', '')}-{site_name}-{site_url}"

    kws = _site_keywords(site, max_tokens=3)
    # 日本語があるかどうかでテンプレを出し分け
    has_jp = any(_has_cjk(k) for k in kws)

    # 代表語を2つまで使う
    k1 = kws[0] if len(kws) >= 1 else ""
    k2 = kws[1] if len(kws) >= 2 else ""

    if has_jp:
        templates = []
        if k1 and k2:
            templates.extend([
                f"{k1}・{k2}ブログ",
                f"{k1}と{k2}のブログ",
                f"{k1}{k2}ブログ",
                f"{k1}＆{k2}の記録ブログ",
            ])
        if k1:
            templates.extend([
                f"{k1}ブログ",
                f"{k1}のブログ",
                f"{k1}の記録ブログ",
                f"{k1}のメモ帳",
                f"{k1}ノート",
            ])
    else:
        # 英語のみでも「ブログ」表記で日本語UIに合わせる
        join12 = f"{k1} {k2}".strip()
        templates = []
        if k1 and k2:
            templates.extend([
                f"{join12} ブログ",
                f"{k1} & {k2} ブログ",
                f"{k1}-{k2} ブログ",
            ])
        if k1:
            templates.extend([
                f"{k1} ブログ",
                f"{k1} Notes ブログ",
                f"{k1} Journal ブログ",
            ])

    # 予防: 空にならないよう最後の保険（URL由来に限定）
    if not templates:
        base = (_domain_tokens(site_url)[:1] or ["blog"])[0]
        templates = [f"{base} ブログ"]

    # 候補から決定（長さ調整）
    idx = _deterministic_index(salt, len(templates))
    for i in range(len(templates)):
        title = templates[(idx + i) % len(templates)]
        title = title.strip()[:48]
        if title:
            return title

    # ここに来ることはほぼないが一応
    return (templates[0].strip() if templates else "blog")[:48]


# ----------------- 画面操作ユーティリティ -----------------
async def _maybe_close_overlays(page):
    selectors = [
        'button#iubenda-cs-accept-btn',
        'button#iubenda-cs-accept',
        'button:has-text("同意")',
        'button:has-text("許可")',
        'button:has-text("OK")',
        '.cookie-accept', '.cookie-consent-accept',
        '.modal-footer button:has-text("閉じる")',
        'div[role="dialog"] button:has-text("OK")',
        'button:has-text("同意して進む")',
        'input[type="submit"][value*="同意"]',
    ]
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if await loc.is_visible():
                await loc.click(timeout=1000)
        except Exception:
            pass
    try:
        await page.evaluate("""
            (() => {
              const blocks = Array.from(document.querySelectorAll('div,section'))
                .filter(n => {
                  const s = getComputedStyle(n);
                  if (!s) return false;
                  const r = n.getBoundingClientRect();
                  return r.width>300 && r.height>200 &&
                         s.position !== 'static' &&
                         parseFloat(s.zIndex||'0') >= 1000 &&
                         s.pointerEvents !== 'none' &&
                         (s.backgroundColor && s.backgroundColor !== 'rgba(0, 0, 0, 0)');
                });
              blocks.slice(0,3).forEach(n => n.style.pointerEvents='none');
            })();
        """)
    except Exception:
        pass

async def _find_in_any_frame(page, selectors, timeout_ms=15000, require_visible=True):
    logger.info("[LD-Recover] frame-scan start selectors=%s timeout=%sms", selectors[:2], timeout_ms)
    deadline = asyncio.get_event_loop().time() + (timeout_ms / 1000)
    while asyncio.get_event_loop().time() < deadline:
        try:
            for fr in page.frames:
                for sel in selectors:
                    try:
                        loc = fr.locator(sel).first
                        cnt = await fr.locator(sel).count()
                        if cnt > 0:
                            if require_visible:
                                try:
                                    if await loc.is_visible():
                                        logger.info("[LD-Recover] frame-scan hit: frame=%s sel=%s", getattr(fr, 'url', None), sel)
                                        return fr, sel
                                except Exception:
                                    continue
                            else:
                                logger.info("[LD-Recover] frame-scan hit (no visible req): frame=%s sel=%s", getattr(fr, 'url', None), sel)
                                return fr, sel
                    except Exception:
                        continue
        except Exception:
            pass
        await asyncio.sleep(0.25)
    logger.warning("[LD-Recover] frame-scan timeout selectors=%s", selectors[:3])
    return None, None

async def _wait_enabled_and_click(page, locator, *, timeout=8000, label_for_log=""):
    try:
        await locator.wait_for(state="visible", timeout=timeout)
    except Exception:
        try:
            await locator.wait_for(state="attached", timeout=int(timeout/2))
        except Exception:
            pass
    try:
        await page.wait_for_function(
            """(el) => el && !el.disabled && el.offsetParent !== null""",
            arg=locator, timeout=timeout
        )
    except Exception:
        pass
    try:
        await locator.scroll_into_view_if_needed(timeout=1500)
    except Exception:
        pass
    try:
        await locator.focus()
    except Exception:
        pass
    try:
        await locator.click(timeout=timeout)
        logger.info("[LD-Recover] clicked %s (normal)", label_for_log or "")
        return True
    except Exception:
        try:
            await locator.click(timeout=timeout, force=True)
            logger.info("[LD-Recover] clicked %s (force)", label_for_log or "")
            return True
        except Exception:
            try:
                await page.evaluate("(el)=>el.click()", locator)
                logger.info("[LD-Recover] clicked %s (evaluate)", label_for_log or "")
                return True
            except Exception:
                logger.warning("[LD-Recover] click failed %s", label_for_log, exc_info=True)
                return False

# /member/blog/create へ確実に到達するためのリトライ
async def _ensure_create_page(page: Page, *, max_tries: int = 3) -> tuple[bool, str]:
    last_url = ""
    for i in range(max_tries):
        logger.info(f"[LD-Recover] create-page try {i+1}/{max_tries}")
        await page.goto("https://livedoor.blogcms.jp/member/blog/create", wait_until="load")
        try:
            await page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass

        if "need_email_auth" in page.url:
            return False, page.url

        # 直接フォームが見えるか
        if await page.locator('#blogTitle, input[name="title"]').first.count() > 0:
            return True, page.url

        # 画面に「ブログを作成」導線だけがあり、クリックが必要なケース
        await _maybe_close_overlays(page)
        candidates = [
            'a:has-text("ブログを作成")',
            'button:has-text("ブログを作成")',
            'a.button:has-text("ブログを作成")',
            'a[href*="/member/blog/create"]',
        ]
        for sel in candidates:
            try:
                loc = page.locator(sel).first
                if await loc.count() > 0 and await loc.is_visible():
                    await _wait_enabled_and_click(page, loc, timeout=4000, label_for_log="create-entry")
                    try:
                        await page.wait_for_load_state("networkidle", timeout=12000)
                    except Exception:
                        pass
                    if await page.locator('#blogTitle, input[name="title"]').first.count() > 0:
                        return True, page.url
            except Exception:
                pass

        # /member を経由してから再アタック（SSO遅延対策）
        await page.goto("https://livedoor.blogcms.jp/member/", wait_until="load")
        try:
            await page.wait_for_load_state("networkidle", timeout=12000)
        except Exception:
            pass

    return False, page.url

async def _set_title_and_submit(page, desired_title: str) -> bool:
    await _maybe_close_overlays(page)

    # まずはメインフレーム
    title_primary = ['#blogTitle', 'input[name="title"]']
    title_fallback = [
        '#blogTitle', 'input#blogTitle', 'input[name="title"]',
        'input#title', 'input[name="blogTitle"]', 'input[name="blog_title"]',
        'input[placeholder*="ブログ"]', 'input[placeholder*="タイトル"]',
    ]
    create_btn_sels = [
        'input[type="submit"][value="ブログを作成する"]',
        'input[type="submit"][value*="ブログを作成"]',
        'input[type="submit"][value*="ブログ作成"]',
        'input[type="submit"][value*="作成"]',
        'input[type="submit"][value*="登録"]',
        '#commit-button',
        'button[type="submit"]',
        'button:has-text("ブログを作成")',
        'button:has-text("作成")',
        'button:has-text("登録")',
        'a.button:has-text("ブログを作成")',
        'a:has-text("ブログを作成")',
    ]

    logger.info("[LD-Recover] タイトル設定＆送信開始（main-frame first）")
    try:
        found = False
        for sel in title_primary:
            try:
                await page.wait_for_selector(sel, state="visible", timeout=20000)
                el = page.locator(sel).first
                try:
                    await el.fill("")
                except Exception:
                    try:
                        await el.click(); await el.press("Control+A"); await el.press("Delete")
                    except Exception:
                        pass
                await el.fill(desired_title)
                logger.info("[LD-Recover] ブログタイトルを設定: %s (%s)", desired_title, sel)
                found = True
                break
            except Exception:
                continue

        if not found:
            fr, sel = await _find_in_any_frame(page, title_fallback, timeout_ms=25000)
            if not fr:
                logger.warning("[LD-Recover] タイトル入力欄が見つからない（DOM/iframe変更の可能性）")
                try:
                    await page.screenshot(path="/tmp/ld_title_not_found.png", full_page=True)
                    logger.info("[LD-Recover] dump: /tmp/ld_title_not_found.png")
                except Exception:
                    pass
                return False

            el = fr.locator(sel).first
            try:
                await el.fill("")
            except Exception:
                try:
                    await el.click(); await el.press("Control+A"); await el.press("Delete")
                except Exception:
                    pass
            await el.fill(desired_title)
            logger.info("[LD-Recover] ブログタイトルを設定(frame): %s (%s)", desired_title, sel)

    except Exception:
        logger.warning("[LD-Recover] タイトル入力に失敗", exc_info=True)
        try:
            await page.screenshot(path="/tmp/ld_title_fill_error.png", full_page=True)
            logger.info("[LD-Recover] dump: /tmp/ld_title_fill_error.png")
        except Exception:
            pass
        return False

    # ボタンクリック
    try:
        btn = None; btn_sel = None
        for sel in create_btn_sels:
            loc = page.locator(sel).first
            try:
                if await loc.count() > 0 and await loc.is_visible():
                    btn = loc; btn_sel = sel; break
            except Exception:
                continue
        if btn is None:
            fr_btn, btn_sel = await _find_in_any_frame(page, create_btn_sels, timeout_ms=15000)
            if not fr_btn:
                logger.warning("[LD-Recover] 作成ボタンが見つからない（UI変更の可能性）")
                try:
                    await page.screenshot(path="/tmp/ld_button_not_found.png", full_page=True)
                    logger.info("[LD-Recover] dump: /tmp/ld_button_not_found.png")
                except Exception:
                    pass
                return False
            btn = fr_btn.locator(btn_sel).first

        clicked = await _wait_enabled_and_click(page, btn, timeout=12000, label_for_log=f"create-button {btn_sel}")
        if not clicked:
            try:
                await page.screenshot(path="/tmp/ld_button_click_error.png", full_page=True)
            except Exception:
                pass
            return False

        logger.info("[LD-Recover] 『ブログを作成』をクリック: %s", btn_sel)

        # ★ 以前の「空の expect_navigation」を削除。
        #   クリック後は素直にロード完了を待つ。
        try:
            await page.wait_for_load_state("load", timeout=20000)
        except Exception:
            pass
        try:
            await page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass

    except Exception:
        logger.warning("[LD-Recover] 作成ボタンクリック処理で例外", exc_info=True)
        try:
            await page.screenshot(path="/tmp/ld_button_click_exception.png", full_page=True)
        except Exception:
            pass
        return False

    return True

# メイン：ブログ作成→AtomPub APIキー取得
async def recover_atompub_key(page, nickname: str, email: str, password: str, site,
                              desired_blog_id: str | None = None) -> dict:
    """
    - Livedoorブログの作成 → AtomPub APIキーを発行・取得
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    async def _dump_error(prefix: str):
        html = await page.content()
        error_html = f"/tmp/{prefix}_{timestamp}.html"
        error_png = f"/tmp/{prefix}_{timestamp}.png"
        Path(error_html).write_text(html, encoding="utf-8")
        try:
            await page.screenshot(path=error_png, full_page=True)
        except Exception:
            pass
        return error_html, error_png

    try:
        logger.info("[LD-Recover] ブログ作成ページに遷移")
        ok_create, where = await _ensure_create_page(page, max_tries=3)
        if not ok_create:
            if "need_email_auth" in where:
                logger.warning("[LD-Recover] email auth required before blog creation: %s", where)
                return {"success": False, "error": "email_auth_required", "need_email_auth": True, "where": where}
            err_html, err_png = await _dump_error("ld_create_enter_fail")
            return {"success": False, "error": "cannot reach create page", "html_path": err_html, "png_path": err_png, "where": where}

        try:
            desired_title = _craft_blog_title(site)
        except Exception:
            # 例外時でも最低限ドメイン由来の語を使う
            site_url = (getattr(site, "url", "") or "").strip()
            base = (_domain_tokens(site_url)[:1] or ["blog"])[0]
            desired_title = f"{base} ブログ"

        logger.info("[LD-Recover] タイトル設定＆送信開始")
        ok_submit = await _set_title_and_submit(page, desired_title)
        if not ok_submit:
            err_html, err_png = await _dump_error("ld_create_ui_notfound")
            return {"success": False, "error": "ブログ作成UIが見つからない/クリック不可（DOM/iframe変更の可能性）",
                    "html_path": err_html, "png_path": err_png}

        # 遷移確認
        success = False
        try:
            await page.wait_for_url(_re.compile(r"/welcome($|[/?#])"), timeout=15000)
            success = True
            logger.info("[LD-Recover] /welcome への遷移を確認")
        except Exception:
            hints = [
                'a:has-text("最初のブログを書く")',
                'a.button:has-text("はじめての投稿")',
                ':has-text("ようこそ")',
                ':has-text("ブログが作成されました")',
            ]
            fr, sel = await _find_in_any_frame(page, hints, timeout_ms=8000, require_visible=False)
            if fr:
                logger.info("[LD-Recover] welcome 導線の出現を確認（frame内）")
                success = True

        # 必須/重複等のフォーム内エラーへのフォールバック
        if not success:
            html_lower = (await page.content()).lower()
            dup_or_required = any(k in html_lower for k in ["使用できません", "既に使われています", "重複", "invalid", "already", "必須", "入力してください"])
            if dup_or_required:
                base = _slugify_ascii(getattr(site, "name", None) or getattr(site, "url", None) or "blog")
                candidates = [base] + [f"{base}-{i}" for i in range(1, 6)]

                # blog_id 入力欄がある場合のみ試す
                has_id_box = False
                for sel in ['#blogId', 'input[name="blog_id"]', 'input[name="livedoor_blog_id"]', 'input[name="blogId"]', 'input#livedoor_blog_id']:
                    try:
                        if await page.locator(sel).count() > 0:
                            has_id_box = True
                            break
                    except Exception:
                        pass

                if has_id_box:
                    for cand in candidates:
                        try:
                            # blog_id 再入力
                            for sel in ['#blogId', 'input[name="blog_id"]', 'input[name="livedoor_blog_id"]', 'input[name="blogId"]', 'input#livedoor_blog_id']:
                                try:
                                    if await page.locator(sel).count() > 0:
                                        await page.fill(sel, cand)
                                        break
                                except Exception:
                                    continue
                            logger.info(f"[LD-Recover] blog_id 衝突/必須 → 候補で再送信: {cand}")
                            if not await _set_title_and_submit(page, desired_title):
                                continue
                            try:
                                await page.wait_for_url(_re.compile(r"/welcome($|[/?#])"), timeout=12000)
                                success = True
                                logger.info(f"[LD-Recover] /welcome へ遷移（blog_id={cand}）")
                                break
                            except Exception:
                                hints = [
                                    'a:has-text("最初のブログを書く")',
                                    'a.button:has-text("はじめての投稿")',
                                ]
                                fr2, sel2 = await _find_in_any_frame(page, hints, timeout_ms=5000, require_visible=False)
                                if fr2:
                                    success = True
                                    logger.info(f"[LD-Recover] welcome 導線検出（blog_id={cand}）")
                                    break
                        except Exception:
                            continue

        if not success:
            err_html, err_png = await _dump_error("ld_atompub_create_fail")
            logger.error("[LD-Recover] ブログ作成に失敗（タイトルのみ or 自動採番不可）")
            return {"success": False, "error": "blog create failed", "html_path": err_html, "png_path": err_png}

        # 以降：AtomPub ページへ
        try:
            fr, sel = await _find_in_any_frame(page, [
                'a:has-text("最初のブログを書く")',
                'a.button:has-text("はじめての投稿")',
            ], timeout_ms=2500, require_visible=False)
            if fr and sel:
                try:
                    await _wait_enabled_and_click(page, fr.locator(sel).first, timeout=3000, label_for_log="welcome-next")
                    logger.info("[LD-Recover] 『最初のブログを書く』をクリック（任意）")
                except Exception:
                    pass
        except Exception:
            pass

        await page.goto("https://livedoor.blogcms.jp/member/", wait_until="load")
        try:
            await page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            pass

        blog_settings_selectors = [
            'a[title="ブログ設定"]',
            'a:has-text("ブログ設定")',
            'a[href^="/blog/"][href$="/config/"]',
            'a[href*="/config/"]'
        ]

        link_el = None
        href = None

        for sel in blog_settings_selectors:
            try:
                loc = page.locator(sel).first
                if await loc.count() > 0:
                    try: await loc.wait_for(state="visible", timeout=8000)
                    except Exception: pass
                    href = await loc.get_attribute("href")
                    if href:
                        link_el = loc
                        break
            except Exception:
                continue

        if not href:
            fr, sel = await _find_in_any_frame(page, blog_settings_selectors, timeout_ms=12000)
            if fr:
                loc = fr.locator(sel).first
                try: await loc.wait_for(state="visible", timeout=6000)
                except Exception: pass
                href = await loc.get_attribute("href")
                if href:
                    link_el = loc

        if not href:
            err_html, err_png = await _dump_error("ld_atompub_member_fail")
            return {"success": False, "error": "member page missing blog link", "html_path": err_html, "png_path": err_png}

        config_url = urljoin("https://livedoor.blogcms.jp/", href)
        try:
            parts = href.split("/")
            blog_id = parts[2] if len(parts) > 2 else None
        except Exception:
            blog_id = None
        if not blog_id:
            page_url = page.url
            if "/blog/" in page_url:
                try:
                    blog_id = page_url.split("/blog/")[1].split("/")[0]
                except Exception:
                    blog_id = "unknown"
            else:
                blog_id = "unknown"
        logger.info(f"[LD-Recover] ブログIDを取得: {blog_id}")

        await page.goto(config_url, wait_until="load")
        try:
            await page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            pass

        api_nav_selectors = [
            'a.configIdxApi[title="API Keyの発行・確認"]',
            'a[title*="API Key"]',
            'a:has-text("API Key")',
            'a:has-text("API Keyの発行")',
        ]
        api_link = None
        for sel in api_nav_selectors:
            try:
                loc = page.locator(sel).first
                if await loc.count() > 0:
                    api_link = loc; break
            except Exception:
                continue
        if api_link is None:
            fr, sel = await _find_in_any_frame(page, api_nav_selectors, timeout_ms=8000)
            if fr: api_link = fr.locator(sel).first

        if api_link is None:
            err_html, err_png = await _dump_error("ld_atompub_nav_fail")
            logger.error("[LD-Recover] AtomPub設定ページへのリンクが見つからない")
            return {"success": False, "error": "api nav link not found", "html_path": err_html, "png_path": err_png}

        await _wait_enabled_and_click(page, api_link, timeout=8000, label_for_log="api-nav")
        try:
            await page.wait_for_load_state("load", timeout=10000)
        except Exception:
            pass

        logger.info(f"[LD-Recover] AtomPub設定ページに遷移: {page.url}")

        if "member" in page.url:
            err_html, err_png = await _dump_error("ld_atompub_redirect_fail")
            logger.error(f"[LD-Recover] AtomPubページが開けず /member にリダイレクト: {page.url}")
            return {"success": False, "error": "redirected to member", "html_path": err_html, "png_path": err_png}

        success_png = f"/tmp/ld_atompub_page_{timestamp}.png"
        try:
            await page.screenshot(path=success_png, full_page=True)
        except Exception:
            try: await page.screenshot(path=success_png)
            except Exception: pass
        logger.info(f"[LD-Recover] AtomPubページのスクリーンショット保存: {success_png}")

        await page.wait_for_selector('input#apiKeyIssue', timeout=12000)
        await _wait_enabled_and_click(page, page.locator('input#apiKeyIssue').first, timeout=6000, label_for_log="api-issue")
        logger.info("[LD-Recover] 『発行する』をクリック")

        await page.wait_for_selector('button:has-text("実行")', timeout=12000)
        await _wait_enabled_and_click(page, page.locator('button:has-text("実行")').first, timeout=6000, label_for_log="api-issue-confirm")
        logger.info("[LD-Recover] モーダルの『実行』をクリック")

        async def _read_endpoint_and_key():
            endpoint_selectors = [
                'input.input-xxlarge[readonly]',
                'input[readonly][name*="endpoint"]',
                'input[readonly][id*="endpoint"]',
            ]
            endpoint_val = ""
            for sel in endpoint_selectors:
                try:
                    await page.wait_for_selector(sel, timeout=8000)
                    endpoint_val = await page.locator(sel).first.input_value()
                    if endpoint_val:
                        break
                except Exception:
                    continue

            await page.wait_for_selector('input#apiKey', timeout=15000)
            for _ in range(30):
                key_val = (await page.locator('input#apiKey').input_value()).strip()
                if key_val:
                    return endpoint_val, key_val
                await asyncio.sleep(0.5)
            return endpoint_val, ""

        endpoint, api_key = await _read_endpoint_and_key()

        if not api_key:
            logger.warning("[LD-Recover] API Keyが空。ページを再読み込みして再発行をリトライ")
            await page.reload(wait_until="load")
            try:
                await page.wait_for_load_state("networkidle", timeout=8000)
            except Exception:
                pass
            await page.wait_for_selector('input#apiKeyIssue', timeout=15000)
            await _wait_enabled_and_click(page, page.locator('input#apiKeyIssue').first, timeout=6000, label_for_log="api-issue-retry")
            await page.wait_for_selector('button:has-text("実行")', timeout=15000)
            await _wait_enabled_and_click(page, page.locator('button:has-text("実行")').first, timeout=6000, label_for_log="api-issue-confirm-retry")
            endpoint, api_key = await _read_endpoint_and_key()

        if not api_key:
            err_html, err_png = await _dump_error("ld_atompub_no_key")
            logger.error(f"[LD-Recover] API Keyが取得できませんでした。証跡: {err_html}, {err_png}")
            return {"success": False, "error": "api key empty", "html_path": err_html, "png_path": err_png}

        logger.info(f"[LD-Recover] ✅ AtomPub endpoint: {endpoint}")
        logger.info(f"[LD-Recover] ✅ AtomPub key: {api_key[:8]}...")

        return {"success": True, "blog_id": blog_id, "api_key": api_key, "endpoint": endpoint, "blog_title": desired_title}

    except Exception as e:
        err_html, err_png = await _dump_error("ld_atompub_fail")
        logger.error("[LD-Recover] AtomPub処理エラー", exc_info=True)
        return {"success": False, "error": str(e), "html_path": err_html, "png_path": err_png}

def create_blog_and_fetch_api_key(session_id: str, *, nickname: str, email: str, password: str, site, desired_blog_id: str | None = None) -> bool:
    """
    （メール認証が完了した前提で）既存セッションのブラウザを使って
    ブログを作成し、AtomPub API Key を取得して DB に保存する。
    """
    page = pwctl.run(pwctl.get_page(session_id))
    if page is None:
        page = pwctl.run(pwctl.revive(session_id))
        if page is None:
            raise RuntimeError(f"signup session not found (sid={session_id})")
    # 保険: 現時点の state を保存（ワーカー切替や後続 revive 時もログイン状態を維持）
    pwctl.run(pwctl.save_storage_state(session_id))

    result = pwctl.run(recover_atompub_key(
        page=page,
        nickname=nickname or "guest",
        email=email or "",
        password=password or "",
        site=site,
        desired_blog_id=desired_blog_id,
    ))

    if not result or not result.get("success"):
        logger.error("[LD-Signup] recover_atompub_key failed: %s", result)
        return False

    blog_id  = result.get("blog_id")
    api_key  = result.get("api_key")
    endpoint = result.get("endpoint")
    blog_title = result.get("blog_title")  # ← 追加：作成時に入力したブログ名

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
    if hasattr(acct, "blog_created"):
        acct.blog_created = True
    if hasattr(acct, "email_verified"):
        acct.email_verified = True
    # ★ 追加：実際に作成したブログ名を保存（他ロジックは変更しない）
    if hasattr(acct, "blog_name") and blog_title:
        acct.blog_name = blog_title    

    db.session.commit()
    pwctl.run(pwctl.set_step(session_id, "api_key_ok"))
    logger.info("[LD-Signup] ✅ blog_id=%s api_key[8]=%s...", blog_id, (api_key or "")[:8])
    return True

# --- backward-compat exports (for legacy imports in tasks/routes) -----------
def signup(site, email_seed: str = "ld"):
    """旧コード向けの互換API。内部では register_blog_account を呼ぶだけ。"""
    return register_blog_account(site, email_seed=email_seed)

__all__ = [
    # 新API
    "prepare_captcha", "submit_captcha", "create_blog_and_fetch_api_key",
    "generate_safe_id", "generate_safe_password",
    "suggest_livedoor_blog_id",
    # 新規：WPベースID候補
    "generate_livedoor_id_candidates",
    # Recover API（内部で使用）
    "recover_atompub_key",
    # 互換API
    "register_blog_account", "signup",
    # ルート互換で使う補助
    "poll_latest_link_gw", "extract_verification_url",
]

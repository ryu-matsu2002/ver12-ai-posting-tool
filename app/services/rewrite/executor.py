# app/services/rewrite/executor.py
# リライト実行の司令塔（安全設計：リンク完全保護 + 監査ログ + ドライラン既定）

import os
import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from flask import current_app
from sqlalchemy import select, text
from sqlalchemy.orm import joinedload, selectinload

from openai import OpenAI, BadRequestError

from app import db
from app.models import (
    Article,
    Site,
    ArticleRewritePlan,
    ArticleRewriteLog,
    GSCUrlStatus,
    GSCMetric,
    SerpOutlineCache,
    TokenUsageLog,
)
from app.wp_client import (
    fetch_single_post,
    update_post_content,
    resolve_wp_post_id,
    update_post_meta,
)

# === OpenAI 設定（article_generator.py と同じ流儀） ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

TOKENS = {
    "policy": 1200,     # 方針テキスト
    "rewrite": 3600,    # 本文リライト
    "summary": 400,     # diff 概要
}
TEMP = {
    "policy": 0.4,
    "rewrite": 0.5,
    "summary": 0.2,
}
TOP_P = 0.9
CTX_LIMIT = 12000
SHRINK = 0.85

META_MAX = 180  # メタ説明最大長（wp_clientのポリシーと整合）

# ========== ユーティリティ（article_generator.py と同系の振る舞い） ==========

def _tok(s: str) -> int:
    return int(len(s) / 1.8)

def _chat(msgs: List[Dict[str, str]], max_t: int, temp: float, user_id: Optional[int] = None) -> str:
    used = sum(_tok(m.get("content", "")) for m in msgs)
    available = CTX_LIMIT - used - 16
    max_t = min(max_t, max(1, available))
    if max_t < 1:
        raise ValueError("Calculated max_tokens is below minimum.")

    def _call(m: int) -> str:
        res = client.chat.completions.create(
            model=MODEL,
            messages=msgs,
            max_tokens=m,
            temperature=temp,
            top_p=TOP_P,
            timeout=120,
        )

        # TokenUsageLog（可能なら保存）
        try:
            if hasattr(res, "usage") and user_id:
                usage = res.usage
                log = TokenUsageLog(
                    user_id=user_id,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0),
                    completion_tokens=getattr(usage, "completion_tokens", 0),
                    total_tokens=getattr(usage, "total_tokens", 0),
                )
                db.session.add(log)
                db.session.commit()
        except Exception as e:
            logging.warning(f"[rewrite/_chat] トークンログ保存失敗: {e}")

        content = (res.choices[0].message.content or "").strip()
        finish = res.choices[0].finish_reason
        if finish == "length":
            logging.warning("⚠️ OpenAI response was cut off due to max_tokens.")
        return content

    try:
        return _call(max_t)
    except BadRequestError as e:
        if "max_tokens" in str(e):
            retry_t = max(1, int(max_t * SHRINK))
            return _call(retry_t)
        raise


# ========== 収集フェーズ：材料集め ==========

def _collect_wp_html(site: Site, article: Article) -> Tuple[Optional[int], Optional[str]]:
    """
    WP上の最新本文HTMLを取得。戻り値: (wp_post_id, content_html or None)
    """
    wp_id = article.wp_post_id
    if not wp_id:
        wp_id = resolve_wp_post_id(site, article, save=True)

    if not wp_id:
        return None, None

    post = fetch_single_post(site, wp_id)
    if post and post.content_html:
        return wp_id, post.content_html
    return wp_id, None


def _collect_gsc_snapshot(site_id: int, article: Article) -> Dict:
    """
    GSCのインデックス状況と最近のパフォーマンスを軽くスナップショット。
    無ければ空構造を返す（LLMに“無い”ことを伝える）。
    """
    snap: Dict = {"url_status": None, "metrics_recent": []}
    try:
        # URL Inspection キャッシュ（最新1件）
        if article.posted_url:
            s = (
                db.session.query(GSCUrlStatus)
                .filter(GSCUrlStatus.site_id == site_id, GSCUrlStatus.url == article.posted_url)
                .order_by(GSCUrlStatus.updated_at.desc())
                .first()
            )
            if s:
                snap["url_status"] = {
                    "indexed": s.indexed,
                    "coverage_state": s.coverage_state,
                    "verdict": s.verdict,
                    "last_crawl_time": s.last_crawl_time.isoformat() if s.last_crawl_time else None,
                    "robots_txt_state": s.robots_txt_state,
                    "page_fetch_state": s.page_fetch_state,
                    "last_inspected_at": s.last_inspected_at.isoformat() if s.last_inspected_at else None,
                }

        # 直近のGSCメトリクス（その記事のキーワード近傍で抽出…最低限は同キーワード）
        if article.keyword:
            rows = (
                db.session.query(GSCMetric)
                .filter(GSCMetric.site_id == site_id, GSCMetric.user_id == article.user_id, GSCMetric.query == article.keyword)
                .order_by(GSCMetric.date.desc())
                .limit(28)
                .all()
            )
            for r in rows:
                snap["metrics_recent"].append({
                    "date": r.date.isoformat(),
                    "impressions": r.impressions,
                    "clicks": r.clicks,
                    "ctr": r.ctr,
                    "position": r.position,
                })
    except Exception as e:
        logging.info(f"[rewrite/_collect_gsc_snapshot] skipped: {e}")
    return snap


def _collect_serp_outline(article: Article) -> List[Dict]:
    """
    競合見出しアウトライン（キャッシュ）を取り出す。無ければ空配列。
    """
    try:
        q = (
            db.session.query(SerpOutlineCache)
            .filter(SerpOutlineCache.article_id == article.id)
            .order_by(SerpOutlineCache.fetched_at.desc())
        )
        rec = q.first()
        if rec and rec.outlines:
            return rec.outlines
    except Exception as e:
        logging.info(f"[rewrite/_collect_serp_outline] skipped: {e}")
    return []


# ========== リンク完全保護（置換→復元） ==========

_LINK_RE = re.compile(r"<a\b[^>]*>.*?</a>", flags=re.IGNORECASE | re.DOTALL)

def _mask_links(html: str) -> Tuple[str, Dict[str, str]]:
    """
    本文内の <a ...>...</a> を [[LINK_i]] に置換し、マッピングを返す。
    後段のLLMには、このトークンを一切改変しないよう厳命する。
    """
    mapping: Dict[str, str] = {}
    def _repl(m):
        idx = len(mapping)
        key = f"[[LINK_{idx}]]"
        mapping[key] = m.group(0)
        return key
    masked = _LINK_RE.sub(_repl, html or "")
    return masked, mapping

def _unmask_links(html: str, mapping: Dict[str, str]) -> str:
    if not html or not mapping:
        return html or ""
    out = html
    for k, v in mapping.items():
        out = out.replace(k, v)
    return out


# ========== メタ生成（任意・安全トリム） ==========

def _strip_html_min(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", s, flags=re.I | re.S)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _smart_truncate(s: str, limit: int = META_MAX) -> str:
    if not s:
        return ""
    if len(s) <= limit:
        return s.strip()
    cut = s[:limit]
    for sep in ["。", "．", "！", "？", "、", "，", " ", "　"]:
        i = cut.rfind(sep)
        if i >= 60:
            cut = cut[:i]
            break
    return cut.strip()

def _gen_meta_from_body(title: str, body_html: str, user_id: Optional[int]) -> str:
    try:
        body_txt = _strip_html_min(body_html)[:1200]
        sys = "あなたは日本語のSEO編集者です。与えられた記事の要点を、自然でクリックを誘発しやすい1文にまとめてください。誇張や断定は避けます。"
        usr = (
            f"制約:\n- {META_MAX}文字以内\n- 文中で不自然に途切れない\n- 記号装飾を使わない\n\n"
            f"【タイトル】\n{title}\n\n【本文抜粋】\n{body_txt}\n"
        )
        meta = _chat(
            [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
            TOKENS["summary"], TEMP["summary"], user_id=user_id
        )
        return _smart_truncate(_strip_html_min(meta), META_MAX)
    except Exception as e:
        logging.info(f"[rewrite/_gen_meta_from_body] skipped: {e}")
        return ""


# ========== 方針生成 & 本文リライト ==========

def _build_policy_text(article: Article, gsc: Dict, outlines: List[Dict]) -> str:
    """
    LLMに「何を・どこを・どう直すか」の手順書を作らせる。
    ※ ここではHTMLを書かせない。あくまで“設計図”。
    """
    sys = (
        "あなたは日本語SEOの編集長です。与えられた材料（GSC指標、インデックス状況、競合見出し）"
        "から“なぜ伸びないのか”を仮説化し、どこをどう直すかの実行手順を作ってください。"
        "出力は箇条書きベースで、見出し構成・導入改善・E-E-A-T・FAQ・用語説明など具体策を含めます。"
        "内部リンクの追加・変更・削除は一切提案しないでください（既存リンクは厳禁で触らない）。"
    )
    usr = json.dumps({
        "article": {"id": article.id, "title": article.title, "keyword": article.keyword, "url": article.posted_url},
        "gsc_snapshot": gsc,
        "serp_outlines": outlines[:8],  # 冗長回避
    }, ensure_ascii=False, indent=2)
    return _chat(
        [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
        TOKENS["policy"], TEMP["policy"], user_id=article.user_id
    )

def _rewrite_html(original_html: str, policy_text: str, user_id: Optional[int]) -> str:
    """
    本文リライト（リンク完全保護）。<a> はすべて [[LINK_i]] に置換し、LLMへ。
    戻りで [[LINK_i]] を厳密復元する。
    """
    masked, mapping = _mask_links(original_html or "")

    sys = (
        "あなたは日本語SEOの編集者です。与えられた“修正方針”に従い、HTML本文を編集し直してください。"
        "重要: 以下を厳守してください。\n"
        "1) [[LINK_i]] というトークンは絶対に変更・削除・順序入替をしないこと（そのまま出力に残す）\n"
        "2) 元の本文に存在しない新しいハイパーリンクを追加しないこと\n"
        "3) 既存の見出し階層は概ね維持しつつ、導入・まとめ・FAQなどを改善してよい\n"
        "4) 事実に基づき、誇張・断定を避ける\n"
        "5) 出力はHTML断片のみ。<html>や<body>は含めない\n"
    )
    usr = (
        "=== 修正方針 ===\n"
        f"{policy_text}\n\n"
        "=== 編集対象（リンクは [[LINK_i]] に置換済み） ===\n"
        f"{masked}\n"
    )

    edited = _chat(
        [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
        TOKENS["rewrite"], TEMP["rewrite"], user_id=user_id
    )

    # 復元
    return _unmask_links(edited, mapping)


# ========== メイン：1件実行 ==========

def execute_one_plan(*, user_id: int, plan_id: Optional[int] = None, dry_run: bool = True) -> Dict:
    """
    1件のリライト計画を実行する。
    - dry_run=True: WP更新しない（方針と差分の生成・ログだけ）
    - dry_run=False: WPに更新反映まで行う
    戻り値は結果のサマリー辞書。
    """
    app = current_app._get_current_object()
    with app.app_context():
        # 1) まず ID だけを FOR UPDATE SKIP LOCKED で取得（JOINしない）
        id_q = db.session.query(ArticleRewritePlan.id).filter(
            ArticleRewritePlan.user_id == user_id,
            ArticleRewritePlan.is_active.is_(True),
            ArticleRewritePlan.status == "queued",
        ).order_by(
            ArticleRewritePlan.priority_score.desc(),
            ArticleRewritePlan.created_at.asc(),
        )
        if plan_id:
            id_q = id_q.filter(ArticleRewritePlan.id == plan_id)

        # Postgresに「どのテーブルをロックするか」を明示
        id_q = id_q.with_for_update(skip_locked=True, of=ArticleRewritePlan)

        target_id = id_q.limit(1).scalar()
        if not target_id:
            return {"status": "empty", "message": "実行可能な queued 計画が見つかりません"}

        # 2) 取得したIDで関連を別クエリでロード（このクエリにはロック不要）
        plan = db.session.query(ArticleRewritePlan).options(
            selectinload(ArticleRewritePlan.article),
            selectinload(ArticleRewritePlan.site),
        ).get(target_id)
        if not plan:
            return {"status": "empty", "message": f"ID={target_id} の計画が見つかりません"}

        plan.status = "running"
        plan.started_at = datetime.utcnow()
        plan.attempts = (plan.attempts or 0) + 1
        db.session.commit()

        article: Article = plan.article
        site: Site = plan.site

        # 2) 材料収集
        wp_post_id, wp_html = _collect_wp_html(site, article)
        original_html = wp_html or (article.body or "")
        if not original_html:
            plan.status = "error"
            plan.last_error = "本文が取得できませんでした（WP/DBとも空）"
            plan.finished_at = datetime.utcnow()
            db.session.commit()
            return {"status": "error", "message": plan.last_error, "plan_id": plan.id}

        gsc_snap = _collect_gsc_snapshot(site.id, article)
        outlines = _collect_serp_outline(article)

        # 3) 方針作成
        policy_text = _build_policy_text(article, gsc_snap, outlines)

        # 4) 本文リライト（リンク保護）
        edited_html = _rewrite_html(original_html, policy_text, user_id=article.user_id)

        # 5) 監査ログ（差分要約をLLMで要約）
        sys = "あなたは日本語の編集者です。修正前後の本文の違いを箇条書きで簡潔に要約してください。具体的に。"
        usr = f"【修正前】\n{_strip_html_min(original_html)[:3000]}\n\n【修正後】\n{_strip_html_min(edited_html)[:3000]}"
        diff_summary = _chat(
            [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
            TOKENS["summary"], TEMP["summary"], user_id=article.user_id
        )

        # 6) ログ保存（WP結果は後で上書き）
        log = ArticleRewriteLog(
            user_id=article.user_id,
            site_id=site.id,
            article_id=article.id,
            plan_id=plan.id,
            policy_text=policy_text,
            diff_summary=diff_summary,
            snapshot_before=original_html,
            snapshot_after=edited_html if dry_run else None,  # ドライラン時のみ“予定”として残す
            wp_status="unknown",
            wp_post_id=wp_post_id,
            executed_at=datetime.utcnow(),
        )
        db.session.add(log)
        db.session.commit()

        wp_ok = False
        wp_err = None

        # 7) WP更新（ドライランじゃなければ反映）
        if not dry_run:
            try:
                # 既存の ai-content ラッパがある場合は尊重（無ければそのまま）
                if '<div class="ai-content">' in edited_html:
                    new_html = edited_html
                else:
                    new_html = f'<div class="ai-content">{edited_html}</div>'

                wp_ok = update_post_content(site, wp_post_id, new_html) if wp_post_id else False

                # 任意：メタ説明を安全に生成・更新（内部リンクは一切触らない）
                if wp_ok:
                    meta = _gen_meta_from_body(article.title or "", edited_html, user_id=article.user_id)
                    if meta:
                        try:
                            update_post_meta(site, wp_post_id, meta)
                        except Exception as e:
                            logging.info(f"[rewrite/meta] meta push skipped: {e}")

                # 成功ならログを確定
                if wp_ok:
                    log.wp_status = "success"
                    log.snapshot_after = edited_html
                else:
                    log.wp_status = "error"
                    log.error_message = "WP更新に失敗しました"
                db.session.commit()
            except Exception as e:
                wp_err = str(e)
                log.wp_status = "error"
                log.error_message = wp_err
                db.session.commit()

        # 8) プランの終了処理
        plan.finished_at = datetime.utcnow()
        if dry_run:
            plan.status = "done"
            db.session.commit()
            return {
                "status": "done(dry)",
                "plan_id": plan.id,
                "article_id": article.id,
                "wp_post_id": wp_post_id,
            }
        else:
            plan.status = "done" if wp_ok else "error"
            if not wp_ok and not plan.last_error:
                plan.last_error = wp_err or "WP更新に失敗"
            db.session.commit()
            return {
                "status": "success" if wp_ok else "error",
                "plan_id": plan.id,
                "article_id": article.id,
                "wp_post_id": wp_post_id,
                "error": wp_err,
            }


# ========== CLI/ワンライナー補助（任意） ==========

def run_once(user_id: int, plan_id: Optional[int] = None, dry_run: bool = True) -> None:
    """
    python -c から呼びやすい薄いラッパ
    """
    res = execute_one_plan(user_id=user_id, plan_id=plan_id, dry_run=dry_run)
    print(json.dumps(res, ensure_ascii=False, indent=2))

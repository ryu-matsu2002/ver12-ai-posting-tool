# app/services/rewrite/executor.py
# リライト実行の司令塔（安全設計：リンク完全保護 + 監査ログ + ドライラン既定）

import os
import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import statistics

from flask import current_app
from sqlalchemy import select, text, func, or_
from sqlalchemy.orm import joinedload, selectinload
from urllib.parse import urlparse
from bs4 import BeautifulSoup, NavigableString, Comment
from openai import OpenAI, BadRequestError
import html  # ★ 追加：フォールバックのテキスト化に使用
from app import db
from app import redis_client  # ← 追加：サイト隔離フラグの持続化に利用
# フォールバック収集を ChatGPT検索API に切替
from app.services.rewrite.providers import openai_search as osearch
# 見出し抽出は自前HTTP＋BS4。serp_collectorの軽量ユーティリティだけを直接利用
from app.services.rewrite.serp_collector import (
    _fetch_page_outline as fetch_page_outline,           # 既存呼び出し維持
    fetch_page_outline_threadsafe,                       # フォールバック時のみ使用
)
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
    resolve_wp_post_id,
    update_post_content,
    update_post_meta,
)

def _rewrite_hard_stop() -> bool:
    """
    グローバル停止フラグ。
    環境変数 REWRITE_HARD_STOP が 1/true/on/yes のいずれかならデキューを即停止。
    既定は停止しない（= 従来どおり実行）。
    """
    v = os.getenv("REWRITE_HARD_STOP", "").strip().lower()
    return v in ("1", "true", "on", "yes")

def _max_rewrite_attempts() -> int:
    try:
        return max(1, int(os.getenv("MAX_REWRITE_ATTEMPTS", "3")))
    except Exception:
        return 3

def _ensure_wp_post_id(site, article) -> Optional[int]:
    """
    内部SEOと同様の手順で、更新前に必ず wp_post_id を解決する。
    - article.wp_post_id が空なら resolve_wp_post_id(site, article, save=True)
    - 解決できなければ None
    """
    wp_id = getattr(article, "wp_post_id", None)
    if not wp_id:
        try:
            wp_id = resolve_wp_post_id(site=site, art=article, save=True)
        except Exception as e:
            current_app.logger.warning("[rewrite] resolve_wp_post_id failed article_id=%s err=%s", getattr(article, "id", None), e)
            wp_id = None
    return wp_id

def _update_wp_with_retry(site, wp_post_id: int, new_html: str, *, max_retry: Optional[int] = None) -> tuple[bool, Optional[int]]:
    """
    内部SEO相当の“軽い”再試行。401/429/408/5xx 等の一時失敗を想定して最大 (1+max_retry) 回まで。
    戻り値: (ok, last_status_code)
    """
    attempt = 0
    if max_retry is None:
        max_retry = REWRITE_IMMEDIATE_RETRY
    last_status = None
    while True:
        ok = False
        try:
            ok = update_post_content(site, wp_post_id, new_html)
        except Exception as e:
            last_status = getattr(getattr(e, "response", None), "status_code", None)
            if last_status:
                current_app.logger.warning("[rewrite] WP update error status=%s post_id=%s err=%s", last_status, wp_post_id, e)
            else:
                current_app.logger.warning("[rewrite] WP update error post_id=%s err=%s", wp_post_id, e)
            # 401/403 は即時にサイト隔離し、これ以上のリトライを行わない（無限ループ防止）
            try:
                if last_status in (401, 403):
                    _set_site_auth_block(site.id, int(last_status), AUTH_BLOCK_TTL_SEC)
                    current_app.logger.error("[rewrite] auth_blocked site_id=%s status=%s (no further retries)", site.id, last_status)
                    return False, last_status
            except Exception:
                pass    
        else:
            # update_post_content が False を返すケースでは HTTP ステータスは取得できない
            # last_status は None のまま（＝分類は後段で推定）
            pass

        if ok:
            return True, last_status
        if attempt >= max_retry:
            return False, last_status
        attempt += 1
        # 短い間隔での軽い再試行（sleepは入れずに即時リトライ。ジョブのスループットを優先）

# === OpenAI 設定（article_generator.py と同じ流儀） ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

TOKENS = {
    "policy": 1600,     # 方針テキスト（微増してcut-offを緩和）
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
SERP_REQUIRED_FOR_REWRITE = os.getenv("SERP_REQUIRED_FOR_REWRITE", "0") == "1"

META_MAX = 180  # メタ説明最大長（wp_clientのポリシーと整合）

# === 運用しきい値を環境変数化（既定は現行値と同等〜やや緩め） ===
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

# 例）VALIDATION_TAG_DIFF_MAX_RATIO=0.20, VALIDATION_TEXTNODE_TOLERANCE=0.20
VALIDATION_TAG_DIFF_MAX_RATIO = _env_float("VALIDATION_TAG_DIFF_MAX_RATIO", 0.20)
VALIDATION_TEXTNODE_TOLERANCE = _env_float("VALIDATION_TEXTNODE_TOLERANCE", 0.20)
# WP即時リトライ回数（1+N回試す）。既定は従来の2（=最大3回）
REWRITE_IMMEDIATE_RETRY = _env_int("REWRITE_IMMEDIATE_RETRY", 2)
REWRITE_MAX_ATTEMPTS     = _env_int("REWRITE_MAX_ATTEMPTS", 6)
# 401/403 の自動隔離 TTL（秒）。既定 24h
AUTH_BLOCK_TTL_SEC       = _env_int("REWRITE_AUTH_BLOCK_TTL_SEC", 24 * 60 * 60)
# running貼り付きの判定しきい値（分）。既定 30分
STUCK_THRESHOLD_MIN      = _env_int("REWRITE_STUCK_THRESHOLD_MIN", 30)
STUCK_RECOVER_THROTTLE_S = _env_int("REWRITE_STUCK_RECOVER_THROTTLE_S", 60)

# ========== 401/403 サイト隔離（Redis） ==========
def _auth_block_key(site_id: int) -> str:
    return f"rewrite:auth_block:{int(site_id)}"

def _is_site_auth_blocked(site_id: int) -> bool:
    try:
        return bool(redis_client.get(_auth_block_key(site_id)))
    except Exception:
        # Redis不調時は“ブロックしない”で進め、可用性を優先
        return False

def _set_site_auth_block(site_id: int, reason_code: int, ttl_sec: int = AUTH_BLOCK_TTL_SEC) -> None:
    try:
        redis_client.setex(_auth_block_key(site_id), ttl_sec, str(reason_code))
    except Exception:
        # 監査に残せれば十分。Redisに依存しすぎない
        logging.warning(f"[rewrite/auth_block] setex failed site_id={site_id} reason={reason_code}")

def _get_blocked_site_ids_from_redis(limit: int = 512) -> List[int]:
    """
    取得時にクエリから除外するため、現在ブロック中の site_id をまとめて取り出す。
    keys/scan は重いので scan_iter を低頻度で使う想定（候補プラン取得の瞬間だけ）。
    """
    blocked: List[int] = []
    try:
        it = redis_client.scan_iter(match="rewrite:auth_block:*", count=limit)
        for k in it:
            # k は bytes になる環境がある
            ks = k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
            try:
                site_id = int(ks.rsplit(":", 1)[-1])
                blocked.append(site_id)
            except Exception:
                continue
    except Exception:
        # Redis不調時は“除外なし”で進める（停止させない）
        pass
    return blocked

# ========== running貼り付きの自動回復（軽量ウォッチドッグ） ==========
def _recover_stuck_plans_once() -> Dict[str, int]:
    """
    しきい値超の running を自動回復する。
    - attempts < REWRITE_MAX_ATTEMPTS: queued に戻して priority を微減点
    - attempts >= 上限: failed に確定
    Redisキーで60秒に1回だけ実行（多重実行抑制）。
    """
    stats = {"requeued": 0, "failed": 0}
    try:
        key = "rewrite:stuck_recover:throttle"
        if redis_client.get(key):
            return stats
        # 60秒スロットル
        redis_client.setex(key, STUCK_RECOVER_THROTTLE_S, "1")
    except Exception:
        # Redis不調なら実行自体は続ける（止めない）
        pass

    try:
        # 1) 再キュー対象（attempts < 上限）
        q1 = text(f"""
            UPDATE article_rewrite_plans
            SET status='queued',
                started_at=NULL,
                finished_at=NULL,
                -- 試行回数に応じて軽く優先度を減点（枯渇回避）
                priority_score = COALESCE(priority_score, 0) - LEAST(GREATEST(attempts,0)*0.5, 5)
            WHERE status='running'
              AND started_at < now() - interval '{STUCK_THRESHOLD_MIN} minutes'
              AND COALESCE(attempts,0) < :max_attempts
            RETURNING id
        """)
        rows1 = db.session.execute(q1, {"max_attempts": REWRITE_MAX_ATTEMPTS}).fetchall()
        stats["requeued"] = len(rows1)

        # 2) 上限到達は failed に確定
        q2 = text(f"""
            UPDATE article_rewrite_plans
            SET status='failed',
                finished_at=now(),
                attempts=COALESCE(attempts,0)+1,
                last_error = COALESCE(last_error,'') || CASE WHEN position('stuck_timeout' in COALESCE(last_error,''))=0 THEN ';stuck_timeout' ELSE '' END
            WHERE status='running'
              AND started_at < now() - interval '{STUCK_THRESHOLD_MIN} minutes'
              AND COALESCE(attempts,0) >= :max_attempts
            RETURNING id
        """)
        rows2 = db.session.execute(q2, {"max_attempts": REWRITE_MAX_ATTEMPTS}).fetchall()
        stats["failed"] = len(rows2)

        db.session.commit()
    except Exception as e:
        try:
            db.session.rollback()
        except Exception:
            pass
        logging.warning(f"[rewrite/stuck_recover] skipped: {e}")
    return stats

# ========== ユーティリティ（article_generator.py と同系の振る舞い） ==========

def _tok(s: str) -> int:
    return int(len(s) / 1.8)

def _chat(msgs: List[Dict[str, str]], max_t: int, temp: float, user_id: Optional[int] = None) -> str:
    def _recalc_max_tokens(messages: List[Dict[str, str]], desired: int) -> int:
        used = sum(_tok(m.get("content", "")) for m in messages)
        available = CTX_LIMIT - used - 16
        return min(desired, max(1, available))

    # 初回 max_tokens を計算
    max_t = _recalc_max_tokens(msgs, max_t)
    if max_t < 1:
        raise ValueError("Calculated max_tokens is below minimum.")

    def _call(messages: List[Dict[str, str]], m: int) -> str:
        # 一部SDKで timeout キーワード未対応 → フォールバック
        try:
            res = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=m,
                temperature=temp,
                top_p=TOP_P,
                timeout=300,
            )
        except TypeError:
            res = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=m,
                temperature=temp,
                top_p=TOP_P,
            )

        # TokenUsageLog（可能なら保存）— 失敗時は必ず rollback してセッションを健全化
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
            # ここが肝：例外時にロールバックしないとセッションが壊れっぱなしになる
            try:
                db.session.rollback()
            except Exception:
                pass
            logging.warning(f"[rewrite/_chat] トークンログ保存失敗: {e}")

        content = (res.choices[0].message.content or "").strip()
        finish = res.choices[0].finish_reason
        if finish == "length":
            logging.warning("⚠️ OpenAI response was cut off due to max_tokens.")
        return content

    # ★ 段階的縮約ロジック
    def _shrink_messages(messages: List[Dict[str, str]], target_chars: int) -> List[Dict[str, str]]:
        def clamp_text(text: str, limit: int) -> str:
            if text is None:
                return ""
            if len(text) <= limit:
                return text
            return text[:limit]
        shrunk: List[Dict[str, str]] = []
        for m in messages:
            c = m.get("content", "")
            # HTMLを含んでいそうならテキスト抽出して短縮（タグを除去）
            if "<" in c and ">" in c:
                try:
                    text_only = BeautifulSoup(c, "html.parser").get_text(" ", strip=True)
                except Exception:
                    text_only = html.unescape(c)
                shrunk.append({**m, "content": clamp_text(text_only, target_chars)})
            else:
                shrunk.append({**m, "content": clamp_text(c, target_chars)})
        return shrunk

    tries = 0
    curr_msgs = msgs
    while True:
        try:
            # 現在のメッセージ長に合わせて max_tokens を都度再計算
            curr_max_t = _recalc_max_tokens(curr_msgs, max_t)
            return _call(curr_msgs, curr_max_t)
        except BadRequestError as e:
            emsg = str(e)
            # 1) max_tokens 指摘 → そのまま縮小
            if "max_tokens" in emsg:
                max_t = max(1, int(max_t * SHRINK))
                continue
            # 2) 文脈長超過 → メッセージ自体を縮約して再試行（最大3回）
            if ("context_length_exceeded" in emsg) or ("maximum context length" in emsg) or ("This model's maximum context length" in emsg):
                tries += 1
                if tries >= 3:
                    raise
                # 試行ごとに強めに圧縮
                # 1回目: ~90k字, 2回目: ~60k字, 3回目: ~40k字相当（実トークンは概算）
                target = {1: 90000, 2: 60000}.get(tries, 40000)
                curr_msgs = _shrink_messages(curr_msgs, target_chars=target)
                # メッセージ縮約に伴い max_tokens も再調整
                max_t = _recalc_max_tokens(curr_msgs, max_t)
                continue
            # それ以外は従来通り上に投げる
            raise

def _safe_json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None

def _unique_urls(outlines: List[Dict], limit: int = 10) -> List[str]:
    seen = set()
    results: List[str] = []
    for o in outlines or []:
        u = (o or {}).get("url")
        if not u:
            continue
        if u in seen:
            continue
        seen.add(u)
        results.append(u)
        if len(results) >= limit:
            break
    return results    

def _take_sources_with_titles(outlines: List[Dict], limit: int = 8) -> List[Dict]:
    """
    参照元を人間可読に（url, title, snippet）で最大 limit 件。
    title/snippet が無いものは後方互換として url のみで出す。
    """
    out: List[Dict] = []
    seen = set()
    for o in outlines or []:
        u = (o or {}).get("url")
        if not u or u in seen:
            continue
        seen.add(u)
        out.append({
            "url": u,
            "title": (o or {}).get("title") or "",
            "snippet": (o or {}).get("snippet") or ""
        })
        if len(out) >= limit:
            break
    return out


# ========== 収集フェーズ：材料集め ==========

def _collect_wp_html(site: Site, article: Article) -> Tuple[Optional[int], Optional[str]]:
    """
    WP上の最新本文HTMLを取得。戻り値: (wp_post_id, content_html or None)
    ※ wp_post_id の解決は常に _ensure_wp_post_id を経由（例外ログを含め一元化）。
    """
    wp_id = _ensure_wp_post_id(site, article)
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


# ========== 競合構造シグナルの要約（FAQ/HowTo/表/語数など） ==========

def _summarize_serp_signals(outlines: List[Dict]) -> Dict:
    """
    serp_collector が保存した signals/schema/intro を集計して、
    - must_add_sections_suggested: ["FAQ","HowTo","Table"] のような“型”提案
    - word_count_stats: {"median": x, "p75": y}
    - estimated_length_range: "2500-3500" のような文字数レンジ仮説
    を返す。データが無ければ空ベース。
    """
    if not outlines:
        return {
            "must_add_sections_suggested": [],
            "word_count_stats": {},
            "estimated_length_range": None
        }
    has_faq = 0
    has_howto = 0
    has_table = 0
    wcounts: List[int] = []
    for o in outlines:
        schema = (o or {}).get("schema") or []
        sig = (o or {}).get("signals") or {}
        if "FAQ" in schema or sig.get("has_faq"):
            has_faq += 1
        if "HowTo" in schema or sig.get("has_howto"):
            has_howto += 1
        if sig.get("has_table"):
            has_table += 1
        wc = sig.get("word_count")
        if isinstance(wc, int) and wc > 0:
            wcounts.append(wc)
    n = max(1, len(outlines))
    suggest: List[str] = []
    # “半数以上が採用している型”は積極提案
    if has_faq >= (n // 2 + n % 2):
        suggest.append("FAQ")
    if has_howto >= (n // 2 + n % 2):
        suggest.append("HowTo")
    if has_table >= (n // 2 + n % 2):
        suggest.append("Table")
    stats = {}
    est = None
    try:
        if wcounts:
            med = int(statistics.median(wcounts))
            p75 = int(statistics.quantiles(wcounts, n=4)[2]) if len(wcounts) >= 4 else med
            stats = {"median": med, "p75": p75}
            # 日本語のだいたいの文字数=単語数×1.8（_tokの逆近似）を使い、範囲に丸める
            def _chars(words: int) -> int:
                return int(words * 1.8)
            low = max(1200, int(_chars(med) * 0.85))
            high = int(_chars(p75) * 1.15)
            # 500刻み程度に丸めて見やすく
            def _round_500(x: int) -> int:
                return int(round(x / 500.0) * 500)
            est = f"{_round_500(low)}-{_round_500(high)}"
    except Exception:
        pass
    return {
        "must_add_sections_suggested": suggest,
        "word_count_stats": stats,
        "estimated_length_range": est
    }

# ========== 競合とのギャップ分析 & ログ用データ整形 ==========

_H2_RE = re.compile(r"<h2[^>]*>(.*?)</h2>", flags=re.IGNORECASE | re.DOTALL)
_H3_RE = re.compile(r"<h3[^>]*>(.*?)</h3>", flags=re.IGNORECASE | re.DOTALL)

def _strip_tags(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _extract_my_headings(html: str) -> Dict[str, List[str]]:
    """
    自記事のH2/H3を抽出（テキスト化）。
    """
    h2s = [_strip_tags(m.group(1)) for m in _H2_RE.finditer(html or "")]
    h3s = [_strip_tags(m.group(1)) for m in _H3_RE.finditer(html or "")]
    # 空文字を除去
    h2s = [x for x in h2s if x]
    h3s = [x for x in h3s if x]
    return {"h2": h2s, "h3": h3s}

def _analyze_gaps(original_html: str, outlines: List[Dict]) -> Tuple[Dict, List[str], Dict]:
    """
    競合アウトライン（[{url, h:[...], notes?...}]）から
    - 参照URL一覧
    - 自記事に不足していそうな見出し候補
    - 補助統計（頻出上位テーマなど）
    を返す。outlines が空なら空の結果を返す。
    """
    referenced_urls = []
    comp_h2_counts = {}
    comp_h3_counts = {}

    for o in outlines or []:
        url = o.get("url")
        if url:
            referenced_urls.append(url)
        hs = o.get("h") or []
        for h in hs:
            t = _strip_tags(h or "")
            # H2/H3っぽい粒度だけをカウント（H1や雑多は除外ヒューリスティック）
            # 既に構造化済みなら "H2: xxx" 形式を想定、プレーンならそのまま扱う
            key = t
            if not key:
                continue
            # 簡易に“長めの見出し”を優先して学習（ノイズ除去）
            if len(key) < 2:
                continue
            comp_h2_counts[key] = comp_h2_counts.get(key, 0) + 1

    mine = _extract_my_headings(original_html or "")
    my_set = set(mine["h2"] + mine["h3"])

    # 頻度順で“自記事に無い見出し”を候補に
    sorted_h2 = sorted(comp_h2_counts.items(), key=lambda x: (-x[1], x[0]))[:30]
    missing = [h for h, c in sorted_h2 if h not in my_set][:15]

    stats = {
        "top_competitor_headings": [{"heading": h, "freq": c} for h, c in sorted_h2[:10]],
        "my_h2": mine["h2"][:20],
        "my_h3": mine["h3"][:20],
    }
    return stats, referenced_urls, {"missing_headings": missing}

#（重複定義が後ろにあるため、この版は削除）

# ========== ギャップ分析（SERP × 現本文 → 追加すべき項目の構造化） ==========

def _build_gap_analysis(article: Article, original_html: str, outlines: List[Dict], gsc_snapshot: Dict) -> Tuple[Dict, str]:
    """
    参照SERP（見出し骨子）と現行本文を比較して、“不足/改善”を構造化JSON + チェックリストで返す。
    - 戻り: (gap_summary_json, checklist_text)
      gap_summary_json 例:
        {
          "missing_topics": ["料金比較", "効果の目安(期間・回数)"],
          "must_add_sections": ["FAQ", "体験談/事例"],
          "quality_issues": ["導入が抽象的", "結論が曖昧"],
          "estimated_length_range": "2500-3500"
        }
    """
    # 本文をテキスト化（トークン節約）
    current_text = _strip_html_min(original_html)[:3500]
    # SERP要約（冗長回避のためH2/H3中心）＋ 競合の構造シグナル
    compact_outlines = []
    for o in (outlines or [])[:8]:
        compact_outlines.append({
            "url": o.get("url"),
            "h": (o.get("h") or [])[:30],
            "notes": o.get("notes", "")[:200]
        })
    serp_signals = _summarize_serp_signals(outlines)

    sys = (
        "あなたは日本語SEOの編集長です。以下の材料から“何が不足か/何を足すべきか”を構造化して返してください。"
        "返答は厳密なJSONのみ（前後に余計な文字を入れない）。"
        "キーは missing_topics(配列), must_add_sections(配列), quality_issues(配列), estimated_length_range(文字列)。"
        "内部リンクや新規リンク提案は一切禁止。"
    )
    usr = json.dumps({
        "article": {"id": article.id, "title": article.title, "keyword": article.keyword},
        "gsc": gsc_snapshot,
        "current_excerpt": current_text,
        "serp_outlines": compact_outlines,
        "serp_structure_signals": serp_signals
    }, ensure_ascii=False)
    raw = _chat(
        [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
        TOKENS["policy"], TEMP["policy"], user_id=article.user_id
    )
    gap = _safe_json_loads(raw) or {}
    # LLM出力を“データ駆動の仮説”で補強（不足していれば埋める／重複はユニーク化）
    try:
        ms = set((gap.get("must_add_sections") or []))
        for s in (serp_signals.get("must_add_sections_suggested") or []):
            ms.add(s)
        gap["must_add_sections"] = sorted(list(ms))
        if not gap.get("estimated_length_range") and serp_signals.get("estimated_length_range"):
            gap["estimated_length_range"] = serp_signals["estimated_length_range"]
    except Exception:
        pass
    # チェックリスト文字列も（UI表示向け）
    checklist_lines: List[str] = []
    for k in ("missing_topics", "must_add_sections", "quality_issues"):
        vals = gap.get(k) or []
        if isinstance(vals, list) and vals:
            title = {
                "missing_topics": "不足トピック",
                "must_add_sections": "追加必須セクション",
                "quality_issues": "品質課題",
            }[k]
            checklist_lines.append(f"■{title}")
            for v in vals:
                checklist_lines.append(f"- {v}")
    if gap.get("estimated_length_range"):
        checklist_lines.append(f"■推奨文字量: {gap['estimated_length_range']}")
    checklist = "\n".join(checklist_lines) if checklist_lines else ""
    return gap, checklist

def _derive_templates_from_gsc(gsc_snapshot: Dict) -> List[str]:
    """
    GSCの状態から、適用した施策テンプレートの“スラグ”を推定する軽い関数。
    UIで『どの方針を使ったか』を見せる目的。
    """
    slugs: List[str] = []
    st = ((gsc_snapshot or {}).get("url_status") or {})
    cov = (st.get("coverage_state") or "").lower()
    if "discovered" in cov and "not indexed" in cov:
        slugs.append("coverage_discovered_not_indexed")
    if "crawled" in cov and "not indexed" in cov:
        slugs.append("coverage_crawled_not_indexed")
    if "alternate page" in cov:
        slugs.append("coverage_alternate_canonical")
    # CTR/順位など簡易推定
    metrics = (gsc_snapshot or {}).get("metrics_recent") or []
    if metrics:
        # ざっくり最近5件平均
        last5 = metrics[:5]
        try:
            avg_pos = sum(m.get("position", 0) or 0 for m in last5) / max(1, len(last5))
            avg_ctr = sum(m.get("ctr", 0) or 0 for m in last5) / max(1, len(last5))
            if avg_pos <= 20 and avg_ctr < 0.01:
                slugs.append("low_ctr_ranked_20")
            if avg_pos > 30:
                slugs.append("low_visibility_ranked_30plus")
        except Exception:
            pass
    # 型を常に List[str] に統一（NULL混入での落ちを防ぐ）
    return slugs

def _gsc_based_rewrite_specs(gsc_snapshot: Dict) -> List[str]:
    """
    GSCの状態を “具体的な編集指示” に落とす。
    方針文 (_build_policy_text) の rewrite_specs に必ず混ぜるための関数。
    ※ WP設定や内部リンクを直接いじらない前提で、本文リライトで打てる手に限定。
    """
    specs: List[str] = []
    st = (gsc_snapshot or {}).get("url_status") or {}
    cov = (st.get("coverage_state") or "").lower()

    # --- 未インデックス系 ---
    # Discovered - not indexed → 重複/薄い/低価値の疑いが強いので独自性と冒頭訴求を強制
    if "discovered" in cov and "not indexed" in cov:
        specs += [
            "冒頭100〜150字で『誰の・どんな悩みを・どう解決する記事か』を明示（結論→具体ベネフィット→固有名詞の順）",
            "競合と被りやすい一般論は削る。各H2のはじめに“本記事固有の視点/事例”を1行要約で置く",
            "FAQを3問追加（検索者が迷いそうな比較/期間/金額/リスク）※リンク追加は禁止",
            "体験談/事例/データ/表のいずれかを最低1つ追加し、重複コンテンツの疑いを下げる",
        ]
    # Crawled - not indexed → 薄い/品質低の疑い。情報密度と整然性を上げる
    if "crawled" in cov and "not indexed" in cov:
        specs += [
            "各H2末尾に2〜3行の“要点サマリ”を追加（結論→根拠→次に読む場所）",
            "HowTo手順を3〜7ステップの番号リストで追加。各ステップは1〜2文で具体化",
            "比較表（<table>）を1つ追加：列は『項目/説明/目安』の3列",
        ]
    # Alternate page with proper canonical → 重複シグナル。独自差分を強制
    if "alternate page" in cov:
        specs += [
            "他ページと差分になる具体的な『対象者・目的・使用シーン』を導入で明記",
            "競合に無い切り口（体験談・地域性・よくある誤解の反証）を最低2つ追加",
        ]

    # --- CTR/順位系 ---
    metrics = (gsc_snapshot or {}).get("metrics_recent") or []
    if metrics:
        try:
            last5 = metrics[:5]
            avg_pos = sum((m.get("position") or 0) for m in last5) / max(1, len(last5))
            avg_ctr = sum((m.get("ctr") or 0) for m in last5) / max(1, len(last5))
            # 順位は悪くないのにCTRが低い → 冒頭訴求と見出しの具体化
            if avg_pos and avg_pos <= 20 and avg_ctr < 0.02:
                specs += [
                    "導入直後に『結論＋想定読者のベネフィット』を1文で置く（装飾なし・誇張なし）",
                    "H2/H3に“検索語を含む具体語”を追加（抽象見出しは避け、数値・期間・比較軸を明記）",
                ]
            # 順位も低い → 情報量と網羅性を増やす
            if avg_pos and avg_pos > 30:
                specs += [
                    "不足しがちな関連小見出しを2〜4個追加（用語定義/メリデメ/失敗例/よくある質問）",
                ]
        except Exception:
            pass
    return specs

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

_LINK_ANY_RE = re.compile(r"\[\[LINK_[^\]]*\]\]")

def _sanitize_link_tokens(html: str, allowed_keys: List[str]) -> str:
    """
    許可された [[LINK_n]] 以外の [[LINK_…]] を全て除去する安全弁。
    - allowed_keys: 例 ["[[LINK_0]]","[[LINK_1]]",...]
    """
    if not html:
        return html
    allowed = set(allowed_keys or [])
    def _replace(m):
        token = m.group(0)
        return token if token in allowed else ""
    return _LINK_ANY_RE.sub(_replace, html)

# --- 追加：LLMが紛れ込ませる“素の <a>…</a>”を物理的に除去するフェイルセーフ ----------------
_NEW_ANCHOR_RE = re.compile(r"<a\b[^>]*>.*?</a>", flags=re.I | re.S)

def _strip_new_anchors(html: str) -> str:
    """マスクされていない新規<a>…</a>は中身のテキストだけ残して除去"""
    if not html:
        return html
    # <a ...>テキスト</a> → “テキスト”のみ残す（テキスト内の他タグも除去）
    return _NEW_ANCHOR_RE.sub(lambda m: re.sub(r"<[^>]+>", "", m.group(0)), html)
# ----------------------------------------------------------------------------------

# --- 追加：復元後の“許可外href”を持つ<a>だけを除去（中身テキストは残す） ----------------------
_ANCHOR_WITH_HREF_RE = re.compile(r'<a\b[^>]*href=["\']([^"\']+)["\'][^>]*>.*?</a>', flags=re.I | re.S)

def _allowed_hrefs_from_mapping(mapping: Dict[str, str]) -> set:
    """マスク時に保存した元リンクHTMLから href の許可集合を作成"""
    allowed = set()
    if not mapping:
        return allowed
    for html in mapping.values():
        m = re.search(r'href=["\']([^"\']+)["\']', html, flags=re.I)
        if m:
            allowed.add(m.group(1).strip())
    return allowed

def _strip_anchors_not_in(html: str, allowed_hrefs: set) -> str:
    """許可hrefに含まれない <a href=...> はタグごと除去し、内側テキストのみ残す"""
    if not html or not allowed_hrefs:
        return html
    def repl(m):
        href = (m.group(1) or "").strip()
        if href in allowed_hrefs:
            return m.group(0)  # そのまま通す（クラス/スタイル含め完全保持）
        # 許可外 → タグ除去、内側テキストのみ残す（内側にタグがあればそれも除去）
        return re.sub(r"<[^>]+>", "", m.group(0))
    return _ANCHOR_WITH_HREF_RE.sub(repl, html)

# --- 追加：LLMが混入させる ```html や ``` のコードフェンスを除去 ----------------------------------
_CODEFENCE_OPEN_RE  = re.compile(r"```+\s*html\s*", flags=re.I)
_CODEFENCE_GENERIC_RE = re.compile(r"```+")
def _strip_codefences(s: str) -> str:
    if not s:
        return s
    s = _CODEFENCE_OPEN_RE.sub("", s)
    s = _CODEFENCE_GENERIC_RE.sub("", s)
    # 単独で残ったバッククォートのカケラを軽く掃除（文中の引用は温存）
    s = re.sub(r"\n?`html\s*", "", s, flags=re.I)
    return s
# --- 追加：タグ構造を保ったまま「テキストだけ」入れ替えるための属性同期 ------------------------
def _restore_attributes_preserve_text(original_html: str, edited_html: str) -> Tuple[str, bool]:
    """
    役割変更（2025-10-24）：
      - “文章だけ修正／リンクやデザインは一切触らない”ポリシーを厳守するため、
        以前の『属性を原文に強制同期』ロジックを撤回。
      - class/style/id/data-* などの属性を変更・同期・削除しない。
      - ここでは何もしない（No-Op）で edited_html をそのまま返す。

    戻り値:
      (edited_html, True) — 処理成功扱い（構造は別の検証で見る）
    """
    return edited_html, True

# --- 追加：テキストノード数の乖離チェック（文章限定リライトの逸脱を検知） ------------------------
def _textnode_divergence_too_large(original_html: str, edited_html: str, tolerance: Optional[float] = None) -> bool:
    """
    原文と編集後の『テキストノード個数』の乖離を見て、構造が大きく揺れていないかを判定。
    tolerance が None の場合は環境変数 VALIDATION_TEXTNODE_TOLERANCE を使用。
    """
    try:
        if tolerance is None:
            tolerance = VALIDATION_TEXTNODE_TOLERANCE
        def _count(h: str) -> int:
            s = BeautifulSoup(h or "", "html.parser")
            n = 0
            for d in s.descendants:
                if isinstance(d, NavigableString) and str(d).strip():
                    # a/PBLOCK 内の文字は既にマスク→復元済みなのでそのまま数える
                    n += 1
            return n
        o_n = _count(original_html)
        e_n = _count(edited_html)
        if o_n == 0:
            return False
        return abs(e_n - o_n) / o_n > tolerance
    except Exception:
        return False


# --- 追加：リンク周辺の“保護ブロック”を丸ごと凍結（位置/クラス/コメント含め不変化） ----------
# 例）<!-- ai-internal-link:... --> を含む段落や、<span class="topic">...</span> など見た目に影響するラッパ
_PBLOCK_RE = re.compile(
    r'(<!--\s*ai-internal-link:[^>]*-->.*?</p>)'
    r'|(<span\s+class=["\']topic["\'][^>]*>.*?</span>)',
    flags=re.I | re.S
)

def _mask_protected_blocks(html: str) -> Tuple[str, Dict[str, str]]:
    mapping: Dict[str, str] = {}
    def _repl(m):
        idx = len(mapping)
        key = f"[[PBLOCK_{idx}]]"
        mapping[key] = m.group(0)
        return key
    masked = _PBLOCK_RE.sub(_repl, html or "")
    return masked, mapping

def _unmask_protected_blocks(html: str, mapping: Dict[str, str]) -> str:
    if not html or not mapping:
        return html or ""
    out = html
    for k, v in mapping.items():
        out = out.replace(k, v)
    return out
# ----------------------------------------------------------------------------------
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

def _build_policy_text(article: Article, gsc: Dict, outlines: List[Dict], gap_summary: Optional[Dict] = None) -> str:
    """
    LLMに「何を・どこを・どう直すか」の手順書を作らせる。
    ※ ここではHTMLを書かせない。あくまで“設計図”。
    """
    sys = (
        "あなたは日本語SEOの編集長です。与えられた材料（GSC指標、インデックス状況、競合見出し）"
        "から“なぜ伸びないのか”を仮説化し、どこをどう直すかの実行手順を作ってください。"
        "出力は箇条書きベースで、見出し構成・導入改善・E-E-A-T・FAQ・用語説明など具体策を含めます。"
        "内部リンクの追加・変更・削除は一切提案しないでください（既存リンクは厳禁で触らない）。"
        "出力制約: 最大12行。各行は120字以内。冗長な説明や同義反復を避け、要点のみを簡潔に書く。"
    )
    # 競合の構造傾向を基に、出力仕様を条件付きで明示
    gap = gap_summary or {}
    must_sections = set((gap.get("must_add_sections") or []))
    length_hint = gap.get("estimated_length_range") or ""
    output_specs: List[str] = []
    if "FAQ" in must_sections:
        output_specs.append("FAQセクションを追加：H2配下でQを太字、Aは簡潔。最大5問。内部リンク・外部リンクは追加しない。")
    if "HowTo" in must_sections:
        output_specs.append("HowTo（手順）を追加：番号付きリストで3-7段階。各ステップ1-2文。")
    if "Table" in must_sections:
        output_specs.append("比較表を追加：<table>で列は『項目/説明/目安』の3列を基本。")
    if length_hint:
        output_specs.append(f"本文の総量は概ね {length_hint} 文字帯を目安（過度に盛らない）。")
    # GSCの状態を“必ず実行する編集指示”として差し込む
    gsc_specs = _gsc_based_rewrite_specs(gsc)
    if gsc_specs:
        output_specs.extend(gsc_specs)    

    usr = json.dumps({
        "article": {"id": article.id, "title": article.title, "keyword": article.keyword, "url": article.posted_url},
        "gsc_snapshot": gsc,
        "serp_outlines": outlines[:8],  # 冗長回避
        "rewrite_specs": output_specs
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
    # 保護ブロックも“丸ごと”凍結（リンクと同様に厳密復元）
    p_masked, p_mapping = _mask_protected_blocks(masked)
    masked_for_llm = p_masked
    allowed_tokens = list(mapping.keys())

    sys = (
        "あなたは日本語SEOの編集者です。与えられた“修正方針”に従い、HTML本文を編集し直してください。"
        "重要: 以下を厳守してください。\n"
        "1) **リンクトークンは厳密一致で保持**：今回許可されるのは次のトークンだけです→ {ALLOWED} 。\n"
        "   これらは削除・変更・順序入れ替えをしないでください。新しい [[LINK_…]] を作らないこと。\n"
        "2) 元の本文に存在しない新しいハイパーリンク（<a>）を追加しないこと\n"
        "3) **[[PBLOCK_*]] は保護ブロック**です。中身・位置・順序・タグ・クラス・属性・コメントを一切変更/移動/複製しないこと。\n"
        "   PBLOCKの前後のテキストのみ編集対象です。\n"
        "4) 既存の見出し階層は概ね維持しつつ、導入・まとめ・FAQなどを改善してよい\n"
        "5) 見出しルール：空の見出しを作らない。連続するH2/H3は禁止。新規H2は最大3つ、各H2直後に2–4文の本文を付与する\n"
        "6) 事実に基づき、誇張・断定を避ける\n"
        "7) 出力はHTML断片のみ。<html>や<body>は含めない\n"
    )
    usr = (
        "=== 修正方針 ===\n"
        f"{policy_text}\n\n"
        "=== 編集対象（リンクは [[LINK_i]] に置換済み） ===\n"
        f"{masked_for_llm}\n"
        "\n---\n"
        "注意: 上記本文に含まれるリンクトークンの正確な一覧は ALLOWED と同一です。ALLOWED に無い [[LINK_…]] を新規に出力しないでください。\n"
    )

    # プレースホルダを埋めてから呼ぶ
    sys = sys.replace("{ALLOWED}", ", ".join(allowed_tokens) if allowed_tokens else "(なし)")
    edited = _chat(
        [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
        TOKENS["rewrite"], TEMP["rewrite"], user_id=user_id
    )

    # 復元前に“不正なリンクトークン”を除去 + “素の<a>”を全面排除（リンクはトークン復元のみ許可）
    edited_clean = _sanitize_link_tokens(edited, allowed_tokens)
    edited_clean = _strip_new_anchors(edited_clean)
    # 保護ブロックを原文そのまま復元 → 既存リンクを復元
    edited_clean = _unmask_protected_blocks(edited_clean, p_mapping)
    restored = _unmask_links(edited_clean, mapping)
    # 復元後の最終ガード：許可href以外の<a>を除去（LLM混入や変換漏れ対策）
    try:
        allowed_hrefs = _allowed_hrefs_from_mapping(mapping)
        restored = _strip_anchors_not_in(restored, allowed_hrefs)
    except Exception as e:
        logging.info(f"[rewrite/_rewrite_html] skip allowlist strip: {e}")

    # コードフェンスのゴミ除去（```html など）
    restored = _strip_codefences(restored)
    # 属性同期はしない（文章だけ変更）— 互換のため関数は呼ぶが no-op
    restored, strict_ok = _restore_attributes_preserve_text(original_html, restored)
    if not strict_ok:
        logging.warning("[rewrite] tag sequence diverged; structure lock flagged (soft).")
    # フェイルセーフ：実質空なら元本文を返す（ログ WARNING を出す）
    try:
        if len(_strip_html_min(restored)) < 20:
            logging.warning("[rewrite] edited_html is empty or too short; fallback to original.")
            return original_html
    except Exception:
        pass
    return restored

def _same_domain(site_url: str, posted_url: str) -> bool:
    """
    ザックリ比較：ホスト名の末尾一致で同一ドメインとみなす。
    例: roof-pilates.com と www.roof-pilates.com は同一扱い。
    livedoor.blog など別ドメインは false。
    """
    if not site_url or not posted_url:
        return False
    try:
        s = urlparse(site_url).hostname or ""
        p = urlparse(posted_url).hostname or ""
        s = s.lower().strip(".").lstrip("www.")
        p = p.lower().strip(".").lstrip("www.")
        # 完全一致 or ドット境界つきのサフィックスのみ許可（spoof-roof-pilates.com を排除）
        return (p == s) or p.endswith("." + s)
    except Exception:
        return False
    
def _is_http_url(s: str) -> bool:
    try:
        return isinstance(s, str) and s.lower().startswith(("http://", "https://"))
    except Exception:
        return False


# ========== 追加：WP投稿前の安全バリデーション ==================================
def _validate_html_for_publish(before_html: str, after_html: str) -> Tuple[bool, str]:
    """
    リンクや保護ブロック、見出しの空要素、トークン残留など“壊し”を検知して投稿を止める。
    """
    import re
    from collections import Counter
    # 比較専用の軽い正規化（公開HTMLは触らない）
    # ・<div class="ai-meta-desc"> と <!-- ai-meta... --> コメントだけを除去
    def _normalize_for_diff_only(html: str) -> str:
        if not html:
            return html
        try:
            soup = BeautifulSoup(html, "html.parser")
            # 1) <div class="ai-meta-desc" ...> を比較から除外
            for node in soup.find_all("div", class_=lambda c: c and "ai-meta-desc" in (c if isinstance(c, list) else str(c))):
                node.decompose()
            # 2) <!-- ai-meta... --> コメントを比較から除外
            for c in soup.find_all(string=lambda t: isinstance(t, Comment) and "ai-meta" in t.lower()):
                c.extract()
            return str(soup)
        except Exception:
            return html

    cmp_before = _normalize_for_diff_only(before_html)
    cmp_after  = _normalize_for_diff_only(after_html)

    # href 集合・順序チェック（※比較は正規化したコピーで行う）
    def hrefs_with_order(h: str) -> List[str]:
        return [m.group(1).strip() for m in re.finditer(r'href=["\']([^"\']+)["\']', h or "", flags=re.I)]
    b_order = hrefs_with_order(cmp_before or "")
    a_order = hrefs_with_order(cmp_after  or "")
    # ✅ リンク集合は厳密一致（hrefが1つでも欠けたら停止）
    if set(b_order) != set(a_order):
        missing = sorted(set(b_order) - set(a_order))[:5]
        return False, f"links_changed_or_missing:{missing}"
    # タグ列が一致しているか（文章限定リライトの担保）
    # --- タグ差分の“ソフト許可”判定 ---
    # 見た目に影響しない軽量タグ（span/strong/em/b/i/u）は無視して構造比較。
    def _tag_multiset(html: str, ignore: set) -> Counter:
        soup = BeautifulSoup(html or "", "html.parser")
        tags = [t.name for t in soup.find_all(True) if t.name not in ignore]
        return Counter(tags)

    try:
        ignore_wrap = {"span", "strong", "em", "b", "i", "u"}
        ob_all = [t.name for t in BeautifulSoup(cmp_before or "", "html.parser").find_all(True)]
        ab_all = [t.name for t in BeautifulSoup(cmp_after  or "", "html.parser").find_all(True)]
        # 総タグ数の差（全タグベース）
        diff_ratio_all = abs(len(ob_all) - len(ab_all)) / max(len(ob_all), 1)

        # 無視対象を除いた“本質タグ”の差
        ob_core = _tag_multiset(cmp_before, ignore_wrap)
        ab_core = _tag_multiset(cmp_after,  ignore_wrap)
        # マルチセット一致なら OK（順序は問わない）
        core_equal = (ob_core == ab_core)

        # 許容差は環境変数 VALIDATION_TAG_DIFF_MAX_RATIO。core が一致しない場合のみ停止。
        if diff_ratio_all > VALIDATION_TAG_DIFF_MAX_RATIO and not core_equal:
            return False, f"tag_sequence_changed(Δ={diff_ratio_all:.2f})"
        # 20%以内、またはコア構造一致なら“ソフト許可”
    except Exception:
        # 解析に失敗した場合は従来通り通す（他のガードで守られる）
        pass
    # 保護ブロック改変/欠落（before から抽出 → after に原文そのまま存在するか）
    _, pmap_before = _mask_protected_blocks(before_html or "")
    for _k, raw in pmap_before.items():
        if raw not in (after_html or ""):
            return False, "pblock_missing_or_modified"
    
    # テキストノード乖離（過度な構造変化の間接指標）
    # テキストノード乖離が大きすぎる場合のみ停止（文章編集の範囲を超える変形を検知）
    if _textnode_divergence_too_large(before_html, after_html, tolerance=VALIDATION_TEXTNODE_TOLERANCE):
        return False, "textnode_divergence_too_large"
    # 空見出し検知
    if re.search(r'<h[23][^>]*>\s*</h[23]>', after_html or "", flags=re.I):
        return False, "empty_heading_detected"
    # [[LINK_…]] 残留（復元漏れ）検知
    if _LINK_ANY_RE.search(after_html or ""):
        return False, "link_token_residue"
    return True, "ok"
# ===============================================================================

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
        # 0) 実行前に running 貼り付きの軽量回復を一度だけ実行（60秒スロットル）
        try:
            _recover_stuck_plans_once()
        except Exception:
            pass

        # 1) まず ID だけを FOR UPDATE SKIP LOCKED で取得
        # ---- グローバル即時停止（キルスイッチ） ----
        if _rewrite_hard_stop():
            current_app.logger.warning("[rewrite] HARD_STOP enabled; skip dequeuing.")
            return {"status": "stopped", "message": "REWRITE_HARD_STOP is set"}

        id_q = db.session.query(ArticleRewritePlan.id).filter(
            ArticleRewritePlan.user_id == user_id,
            ArticleRewritePlan.is_active.is_(True),
            ArticleRewritePlan.status == "queued",
            # attempts 上限（None は 0 扱いで許容）— 既存のREWRITE_MAX_ATTEMPTSを採用
            or_(ArticleRewritePlan.attempts.is_(None),
                ArticleRewritePlan.attempts < REWRITE_MAX_ATTEMPTS),
            # 未来予約は除外（NULL は即実行可）
            or_(ArticleRewritePlan.scheduled_at.is_(None),
                ArticleRewritePlan.scheduled_at <= func.now()),
        ).order_by(
            ArticleRewritePlan.priority_score.desc(),
            ArticleRewritePlan.created_at.asc(),
        )
        # plan_id 指定はそのまま尊重
        if plan_id:
            id_q = id_q.filter(ArticleRewritePlan.id == plan_id)

        # Postgresに「どのテーブルをロックするか」を明示
        id_q = id_q.with_for_update(skip_locked=True, of=ArticleRewritePlan)

        # 1.1) ブロック中サイトをクエリ段階で除外（Redisで軽量に収集）
        blocked_site_ids = _get_blocked_site_ids_from_redis()
        if blocked_site_ids and not plan_id:
            try:
                # site_id 列があるため直接除外（追加JOIN不要）
                id_q = id_q.filter(~ArticleRewritePlan.site_id.in_(blocked_site_ids))
            except Exception:
                # IN 句で落ちたら“あとで弾く”方針にフォールバック
                pass

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

        # 2.1) サイトが“認証ブロック中”なら、このプランは選ばない（queuedのまま保持・スロット非占有）
        if _is_site_auth_blocked(plan.site_id):
            return {"status": "skipped(site_blocked)", "plan_id": plan.id, "site_id": plan.site_id}

        # 2.2) ここで初めて running 化（ブロック確認後に行うのが重要）
        plan.status = "running"
        plan.started_at = datetime.utcnow()
        # ✅ attempts は “本実行のみ” カウントする。dry_run では増やさない。
        if not dry_run:
            plan.attempts = (plan.attempts or 0) + 1
        db.session.commit()

        article: Article = plan.article
        site: Site = plan.site

        # 2) ドメイン不一致の安全ガード（posted_url がURLのときだけ判定）
        #    例: site.url=roof-pilates.com だが posted_url が livedoor.blog → WP更新対象外
        if article.posted_url and _is_http_url(article.posted_url) and not _same_domain(site.url, article.posted_url):
            reason = f"domain_mismatch: site={site.url} posted={article.posted_url}"
            current_app.logger.info(f"[rewrite] skip plan_id={plan.id} ({reason})")
            # dry_run でも 'running' のままにしないよう終端化
            plan.finished_at = datetime.utcnow()
            if dry_run:
                # ✅ ドライラン後は自動で再キュー（本番を続けて実行しやすくする）
                plan.status = "queued"
                plan.started_at = None
                plan.finished_at = None
                # （任意）ドライランではattemptsをカウントしない
                try:
                    if plan.attempts and plan.attempts > 0:
                        plan.attempts -= 1
                except Exception:
                    pass
                db.session.commit()
                return {
                    "status": "done(dry)",
                    "plan_id": plan.id,
                    "article_id": article.id,
                    "wp_post_id": None,
                    "note": "Dry-run finished and plan re-queued automatically."
                }
            else:
                # 本実行では計画をエラー終了（無効化はここでは行わない：手動判断の余地を残す）
                plan.status = "error"
                plan.last_error = f"permanent:{reason}"
                db.session.commit()
                return {
                    "status": "skipped",
                    "reason": reason,
                    "plan_id": plan.id,
                    "article_id": article.id,
                    "wp_post_id": None,
                }

        # 3) 材料収集
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
        if not outlines:
            logging.info("[rewrite] outlines empty for article_id=%s (SERP参考0件)", article.id)

            # === 修正版: OpenAI Search APIで直接URL収集しキャッシュ ===
            try:
                urls = osearch.search_top_urls(article.keyword or article.title or "", limit=6)
                if urls:
                    saved_outlines = []
                    for u in urls:
                        outline = fetch_page_outline_threadsafe(u, lang="ja", gl="jp")
                        # ★ 修正版：見出しが空でもURLだけキャッシュに残す
                        if outline:
                            # 通常のアウトライン
                            saved_outlines.append(outline)
                        else:
                            # fallback：最低限URL情報だけ記録してキャッシュ化
                            saved_outlines.append({
                                "url": u,
                                "h": [],
                                "title": "",
                                "snippet": "",
                                "fetched_at": datetime.utcnow().isoformat()
                            })

                    # URL単位でキャッシュを保存（空アウトラインも含める）
                    cache = SerpOutlineCache(
                        article_id=article.id,
                        outlines=saved_outlines,
                        fetched_at=datetime.utcnow()
                    )
                    db.session.add(cache)
                    db.session.commit()

                    if saved_outlines:
                        cache = SerpOutlineCache(
                            article_id=article.id,
                            outlines=saved_outlines,
                            fetched_at=datetime.utcnow()
                        )
                        db.session.add(cache)
                        db.session.commit()

                        outlines = saved_outlines
                        logging.info(
                            "[rewrite/fallback] ✅ OpenAI search success: article_id=%s keyword=%r n=%s",
                            article.id, (article.keyword or ""), len(saved_outlines),
                        )
                    else:
                        logging.warning(
                            "[rewrite/fallback] ⚠️ OpenAI search returned URLs but no outlines could be fetched (article_id=%s)",
                            article.id,
                        )
                else:
                    logging.warning(
                        "[rewrite/fallback] ⚠️ OpenAI search returned no URLs for article_id=%s keyword=%r",
                        article.id, (article.keyword or ""),
                    )
            except Exception as e:
                logging.exception("[rewrite/fallback] ❌ Exception during OpenAI search for article_id=%s: %r", article.id, e)

            # --- 0件継続時の通知強化（デバッグ・ハンドオフ用のヒントを添付） ---
            if not outlines:
                try:
                    debug_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "runtime", "serp_debug"))
                except Exception:
                    debug_dir = "runtime/serp_debug"
                logging.info(
                    "[rewrite/fallback] outlines STILL empty after attempt: article_id=%s keyword=%r title=%r debug_dir=%s",
                    article.id, (article.keyword or ""), (article.title or ""), debug_dir
                )

                # ここで“停止するか/続行するか”を環境変数で切り替え
                if SERP_REQUIRED_FOR_REWRITE:
                    reason = "serp_outlines_empty"
                    if dry_run:
                        # ドライラン時は再キューして早期終了（attemptsは加算しない）
                        plan.status = "queued"
                        plan.started_at = None
                        plan.finished_at = None
                        try:
                            if plan.attempts and plan.attempts > 0:
                                plan.attempts -= 1
                        except Exception:
                            pass
                        db.session.commit()
                        return {
                            "status": "done(dry)",
                            "plan_id": plan.id,
                            "article_id": article.id,
                            "wp_post_id": None,
                            "note": "SERP=0件のため停止（再キュー済み）"
                        }
                    else:
                        plan.status = "error"
                        plan.last_error = f"permanent:{reason}"
                        plan.finished_at = datetime.utcnow()
                        db.session.commit()
                        return {
                            "status": "error",
                            "plan_id": plan.id,
                            "article_id": article.id,
                            "wp_post_id": None,
                            "error": reason
                        } 
  

        # 3.4) URLだけキャッシュされて「h（見出し配列）」が空のときの軽量補完
        #     - OpenAI再ランキングや URL のみ保存モード(SERP_ONLY_URLS=1)運用時の安全弁
        #     - 上位 K 件のみヘッディング抽出を行い、outlines の各要素に h を埋める
        try:
            # すべてのアウトラインで h が空なら発動（部分的に埋まっている場合は何もしない）
            if outlines and all(not (o.get("h") or []) for o in outlines):
                K = int(os.getenv("SERP_HEADINGS_FILL_K", "3"))
                K = max(1, min(K, len(outlines)))
                filled_cnt = 0
                for o in outlines[:K]:
                    url_ = (o or {}).get("url")
                    if not url_:
                        continue
                    try:
                        # 軽量抽出器（本文をフル収集せず H2/H3 を中心に抜く）
                        filled = fetch_page_outline(url_, lang="ja", gl="jp")
                        if filled and (filled.get("h") or []):
                            o["h"] = filled["h"]
                            filled_cnt += 1
                    except Exception:
                        # 1件失敗しても続行
                        continue
                if filled_cnt:
                    logging.info("[rewrite] headings filled for %s/%s urls (K=%s)", filled_cnt, len(outlines), K)
        except Exception as e:
            logging.info("[rewrite] headings_fill skipped: %r", e)
        
        # 3.5) SERP × 現本文のギャップ分析（不足/追加セクション/品質課題 などを構造化）
        gap_summary_json, policy_checklist = _build_gap_analysis(article, original_html, outlines, gsc_snap)
        used_templates = _derive_templates_from_gsc(gsc_snap)
        referenced_urls = _unique_urls(outlines, limit=10)
        referenced_count = len(referenced_urls)
        referenced_sources = _take_sources_with_titles(outlines, limit=8)

        # 4) 方針作成
        policy_text = _build_policy_text(article, gsc_snap, outlines, gap_summary_json)
        # 人が一覧で判断しやすいよう、参照件数・URL・不足の要点を方針末尾に追記
        try:
            missing_topics = (gap_summary_json or {}).get("missing_topics") or []
            _mt_head = "、".join(missing_topics[:5])
            policy_text = (
                f"{policy_text}\n\n"
                f"---\n"
                f"【参照SERP件数】{referenced_count}\n"
                f"【不足トピック（要点）】{_mt_head if _mt_head else '—'}\n"
                f"※ 参照URLとタイトルの詳細はダッシュボードで確認できます。\n"
                f"{'(SERP参照0件: GSCのみで方針策定)' if referenced_count == 0 else ''}"
            )
        except Exception:
            pass

        # 4.5)（削除）— 以降は _build_gap_analysis の結果をそのまま使う

        # 5) 本文リライト（リンク保護）
        edited_html = _rewrite_html(original_html, policy_text, user_id=article.user_id)

        # 6) 監査ログ（差分要約をLLMで要約）
        sys = "あなたは日本語の編集者です。修正前後の本文の違いを箇条書きで簡潔に要約してください。具体的に。"
        usr = f"【修正前】\n{_strip_html_min(original_html)[:3000]}\n\n【修正後】\n{_strip_html_min(edited_html)[:3000]}"
        diff_summary = _chat(
            [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
            TOKENS["summary"], TEMP["summary"], user_id=article.user_id
        )

        # 7) ログ保存（WP結果は後で上書き）
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
            # 🔽 今回追加の監査情報
            referenced_count=referenced_count,
            referenced_urls=referenced_urls,
            gap_summary=gap_summary_json,
            policy_checklist=policy_checklist,
            used_templates=used_templates,
        )
        db.session.add(log)
        db.session.commit()

        wp_ok = False
        wp_err = None

        # 8) WP更新（ドライランじゃなければ反映）
        if not dry_run:
            try:
                # 念押し：直前でもう一度だけ ID を解決（WP側の状態変化に追随）
                # 例：直前の別処理で wp_post_id が付与された／変更された場合など。
                if not wp_post_id:
                    wp_post_id = _ensure_wp_post_id(site, article)
                # 見た目やクラスに触らない：LLM出力をそのまま使う
                ok, reason = _validate_html_for_publish(original_html, edited_html)
                if not ok:
                    log.wp_status = "error"
                    log.error_message = f"publish_aborted:{reason}"
                    # 監査性：停止時でも after を残す
                    if not log.snapshot_after:
                        log.snapshot_after = edited_html
                    db.session.commit()
                    wp_ok = False
                else:
                    # ✅ 本番反映は必ず“軽量リトライ”経由に統一
                    if wp_post_id:
                        wp_ok, last_status = _update_wp_with_retry(site, wp_post_id, edited_html)
                    else:
                        wp_ok, last_status = (False, None)
 

                # 任意：メタ説明を安全に生成・更新（内部リンクは一切触らない）
                if wp_ok:
                    meta = _gen_meta_from_body(article.title or "", edited_html, user_id=article.user_id)
                    if meta:
                        try:
                            update_post_meta(site, wp_post_id, meta)
                        except Exception as e:
                            logging.info(f"[rewrite/meta] meta push skipped: {e}")

                # 失敗時の“全自動”自己復旧（UI不要）
                if not wp_ok:
                    # 一時失敗（transient）を分類：429/408/5xx/ステータス不明(None)
                    status = last_status
                    is_transient = (
                        (status is None) or
                        (status == 429) or
                        (status == 408) or
                        (500 <= int(status or 0) <= 599)
                    )
                    # 401/403 はすでに _update_wp_with_retry 内で auth_block 済み。ここでは恒久扱い。
                    if status in (401, 403):
                        log.wp_status = "error"
                        if not log.error_message:
                            log.error_message = f"permanent_wp_error:{status}; site_auth_blocked"
                        plan.last_error = f"permanent:wp_status={status}"
                        db.session.commit()
                        return {
                            "status": "error",
                            "plan_id": plan.id,
                            "article_id": article.id,
                            "wp_post_id": wp_post_id,
                            "error": f"permanent_wp_error:{status}"
                        }
                    if is_transient:
                        # バックオフ：attempts に応じて優先度を軽く下げて再キュー（マイグレーション不要）
                        try:
                            # 既存 priority_score を軽く減点（例: 試行回数 × 0.5）
                            dec = float((plan.attempts or 1)) * 0.5
                            plan.priority_score = (plan.priority_score or 0.0) - dec
                        except Exception:
                            pass
                        # ステータスを queued に戻し、終了印を消す（完全自動復帰）
                        plan.status = "queued"
                        plan.started_at = None
                        plan.finished_at = None
                        # ログは “retry_queued” で確定（UIなしで追える）
                        log.wp_status = "retry_queued"
                        if not log.error_message:
                            log.error_message = f"transient_wp_error:{status}"
                        # 監査ログに分類を明示
                        if not log.error_message:
                            log.error_message = f"transient_wp_error:{status}"
                        db.session.commit()
                        return {
                            "status": "retry_queued",
                            "plan_id": plan.id,
                            "article_id": article.id,
                            "wp_post_id": wp_post_id,
                            "error": f"transient_wp_error:{status}",
                        }
                    else:
                        # 恒久的（permanent）と判断
                        # 401/403 は“サイト認証ブロック”を発火し、以降のプランは取得段階で除外する
                        log.wp_status = "error"
                        if not log.error_message:
                            log.error_message = f"permanent_wp_error:{status}"
                        if status in (401, 403):
                            _set_site_auth_block(site.id, status, AUTH_BLOCK_TTL_SEC)
                            # UI で分かるようにメッセージを明示
                            log.error_message = (log.error_message or "") + "; site_auth_blocked"
                        # plan 側にも分類を明示
                        plan.last_error = f"permanent:wp_status={status}"
                        db.session.commit()
                else:
                    # 成功ならログを確定
                    log.wp_status = "success"
                    log.snapshot_after = edited_html
                    db.session.commit()
            except Exception as e:
                wp_err = str(e)
                log.wp_status = "error"
                log.error_message = wp_err
                db.session.commit()

        # 9) プランの終了処理
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
            # 本番（dry_run=False）
            if wp_ok:
                plan.status = "done"
                db.session.commit()
                return {
                    "status": "success",
                    "plan_id": plan.id,
                    "article_id": article.id,
                    "wp_post_id": wp_post_id,
                    "error": None,
                }
            else:
                # ここに来るのは permanent 扱いで確定させたケース
                plan.status = "error"
                if not plan.last_error:
                    plan.last_error = f"permanent:{wp_err or getattr(log, 'error_message', 'WP更新に失敗')}"
                db.session.commit()
                return {
                    "status": "error",
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
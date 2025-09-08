# app/services/internal_seo/link_graph.py
from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from app import db
from app.models import ContentIndex, InternalLinkGraph

logger = logging.getLogger(__name__)

# indexer.py と同じ方針のゆるやかトークナイザ
JP_TOKEN = re.compile(r"[一-龥ぁ-んァ-ンーA-Za-z0-9]{2,}")

def _tokenize(title: str, text: str, keywords_csv: Optional[str]) -> List[str]:
    toks: List[str] = []
    if title:
        title_tokens = JP_TOKEN.findall(title.lower())
        # タイトルは重要なので重みブースト（2回入れる）
        toks.extend(title_tokens)
        toks.extend(title_tokens)
    if text:
        toks.extend(JP_TOKEN.findall(text.lower()))
    if keywords_csv:
        # keywords は更にブースト（例：3回入れる）
        kws = [k.strip().lower() for k in keywords_csv.split(",") if k.strip()]
        for _ in range(3):
            toks.extend(kws)
    # ごく軽いストップ語（日本語の機能語などは indexer 同様に適宜追加可）
    stop = {
        "こと","これ","それ","ため","よう","です","ます","する","いる","ある","ない","なる",
        "そして","また","ので","でも","について","まとめ","ポイント","こちら","今回","場合",
        "可能","原因","方法","対処","基本","入門","注意","解説","詳細","最新"
    }
    toks = [t for t in toks if t not in stop and not t.isdigit()]
    return toks

def _build_tfidf(
    docs: Dict[int, Tuple[str, str, Optional[str]]]
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, float], Dict[str, List[int]]]:
    """
    docs: post_id -> (title, raw_text, keywords_csv)
    返り値:
      - tfidf_vecs: post_id -> {token: weight}
      - norms: post_id -> L2ノルム
      - inverted: token -> [post_id, ...]
    """
    # TF
    tf: Dict[int, Counter] = {}
    df: Counter = Counter()
    for pid, (title, text, kws) in docs.items():
        toks = _tokenize(title or "", text or "", kws)
        c = Counter(toks)
        tf[pid] = c
        for t in c.keys():
            df[t] += 1

    N = max(1, len(docs))
    # IDF
    idf: Dict[str, float] = {}
    for t, dcnt in df.items():
        # スムージング付きIDF
        idf[t] = math.log((N + 1) / (dcnt + 1)) + 1.0

    # TF-IDF（log-tf）
    tfidf_vecs: Dict[int, Dict[str, float]] = {}
    norms: Dict[int, float] = {}
    inverted: Dict[str, List[int]] = defaultdict(list)

    for pid, c in tf.items():
        vec: Dict[str, float] = {}
        for t, freq in c.items():
            w = (1.0 + math.log(freq)) * idf.get(t, 0.0)
            if w > 0:
                vec[t] = w
                inverted[t].append(pid)
        # L2ノルム
        norm = math.sqrt(sum(w*w for w in vec.values())) or 1e-9
        tfidf_vecs[pid] = vec
        norms[pid] = norm

    return tfidf_vecs, norms, inverted

def _cosine_sim(
    pid_a: int,
    pid_b: int,
    vecs: Dict[int, Dict[str, float]],
    norms: Dict[int, float]
) -> float:
    if pid_a == pid_b:
        return 0.0
    va = vecs.get(pid_a, {})
    vb = vecs.get(pid_b, {})
    if not va or not vb:
        return 0.0
    # ドット積（共通トークンだけ）
    if len(va) > len(vb):
        va, vb = vb, va
    dot = 0.0
    for t, wa in va.items():
        wb = vb.get(t)
        if wb:
            dot += wa * wb
    denom = norms.get(pid_a, 1.0) * norms.get(pid_b, 1.0)
    if denom <= 0:
        return 0.0
    return dot / denom

def build_link_graph_for_site(
    site_id: int,
    max_targets_per_source: int = 20,
    min_score: float = 0.15,
    limit_posts: Optional[int] = None,
    batch_commit: int = 500,
) -> Dict[str, int]:
    """
    ContentIndex から同一サイトの記事を読み、各記事ごとに上位の“似てる記事”を
    InternalLinkGraph に UPSERT する。
    - max_targets_per_source: 1記事あたり保存するリンク候補の上限
    - min_score: 類似度の下限（0~1）
    - limit_posts: None なら全件、数値なら先頭N記事だけ（テスト用）
    """
    # 1) 対象記事の取得
    q = (
        ContentIndex.query
        .with_entities(
            ContentIndex.wp_post_id,
            ContentIndex.title,
            ContentIndex.raw_text,
            ContentIndex.keywords,
        )
        .filter(ContentIndex.site_id == site_id)
        .filter(ContentIndex.status == "publish")
        .filter(ContentIndex.wp_post_id.isnot(None))
    )
    rows = q.all()
    if not rows:
        logger.info("[LinkGraph] site=%s no content to process", site_id)
        return {"sources": 0, "edges_upserted": 0}

    if limit_posts:
        rows = rows[:limit_posts]

    docs: Dict[int, Tuple[str, str, Optional[str]]] = {}
    for pid, title, text, kws in rows:
        if not pid:  # 念のため
            continue
        docs[int(pid)] = (title or "", text or "", kws)

    # 2) TF-IDF 構築
    tfidf_vecs, norms, inverted = _build_tfidf(docs)

    # 3) 候補生成（倒立インデックス経由で近傍候補を集める）
    sources = list(docs.keys())
    edges_upserted = 0
    processed_sources = 0

    for src in sources:
        vec = tfidf_vecs.get(src)
        if not vec:
            continue

        # src のトークンが現れる doc を候補に
        candidate_ids = set()
        for t in vec.keys():
            for pid in inverted.get(t, []):
                if pid != src:
                    candidate_ids.add(pid)

        # 類似度スコア計算
        scored: List[Tuple[int, float]] = []
        for dst in candidate_ids:
            s = _cosine_sim(src, dst, tfidf_vecs, norms)
            if s >= min_score:
                scored.append((dst, s))

        # 上位K件だけ
        scored.sort(key=lambda x: x[1], reverse=True)
        topk = scored[:max_targets_per_source]

        # 4) UPSERT to InternalLinkGraph
        now = datetime.utcnow()
        for dst, sc in topk:
            # 既存があれば更新、無ければ作成（site_id + src + dst unique）
            row = (
                InternalLinkGraph.query
                .filter_by(site_id=site_id, source_post_id=src, target_post_id=dst)
                .one_or_none()
            )
            if row is None:
                row = InternalLinkGraph(
                    site_id=site_id,
                    source_post_id=src,
                    target_post_id=dst,
                    score=float(sc),
                    reason="keyword_match",
                    last_evaluated_at=now,
                )
                db.session.add(row)
            else:
                row.score = float(sc)
                row.reason = row.reason or "keyword_match"
                row.last_evaluated_at = now

            edges_upserted += 1
            if edges_upserted % batch_commit == 0:
                db.session.commit()
                logger.info("[LinkGraph] site=%s committed edges=%s", site_id, edges_upserted)

        processed_sources += 1

    db.session.commit()
    stats = {"sources": processed_sources, "edges_upserted": edges_upserted}
    logger.info("[LinkGraph] site=%s done: %s", site_id, stats)
    return stats

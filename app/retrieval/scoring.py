# app/retrieval/scoring.py
"""
Hybrid retrieval (vector + FTS) with a tiny, robust surface:
- Always return rows as dicts (via rows_to_dicts) to avoid tuple indexing errors
- Minimal SQL with proper ::vector cast to fix earlier type errors
- Simple score blending with alpha
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
from collections import defaultdict

from openai import OpenAI

from app.core.config import (
    EMBEDDING_MODEL,
    RETRIEVAL_TOP_K,
)
from app.core.utils import pg_cursor, rows_to_dicts, soft_normalize, clamp

# -----------------------------
# OpenAI client (reads OPENAI_API_KEY from env)
# -----------------------------
_client = OpenAI()

# -----------------------------
# Small helpers
# -----------------------------
def _to_vector_literal(embedding: Sequence[float]) -> str:
    # Postgres vector extension accepts e.g. '[0.1,0.2,0.3]'
    return "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"


def _embed_query(query: str) -> List[float]:
    resp = _client.embeddings.create(model=EMBEDDING_MODEL, input=query)
    return list(resp.data[0].embedding)


# -----------------------------
# SQL helpers (kept inline to avoid import drift)
# -----------------------------
_SQL_VECTOR = """
SELECT
  d.id     AS document_id,
  d.title  AS title,
  d.source_uri AS source_uri,
  c.id     AS chunk_id,
  c.chunk_no AS chunk_no,
  c.content AS content,
  /* cosine distance (pgvector) -> similarity */
  1 - (c.embedding <=> (%s)::vector) AS score
FROM corah_store.chunks c
JOIN corah_store.documents d ON d.id = c.document_id
ORDER BY c.embedding <=> (%s)::vector
LIMIT %s;
"""

_SQL_FTS = """
WITH q AS (
  SELECT websearch_to_tsquery('english', %s) AS query
)
SELECT
  d.id     AS document_id,
  d.title  AS title,
  d.source_uri AS source_uri,
  c.id     AS chunk_id,
  c.chunk_no AS chunk_no,
  c.content AS content,
  ts_rank(c.content_tsv, q.query) AS score
FROM corah_store.chunks c
JOIN corah_store.documents d ON d.id = c.document_id, q
WHERE c.content_tsv @@ q.query
ORDER BY score DESC
LIMIT %s;
"""


def _vector_search(db_url: str, embedding: Sequence[float], k: int) -> List[Dict[str, Any]]:
    """
    Return top-k chunks ordered by cosine distance (<=>).
    We cast the parameter to ::vector to avoid 'operator does not exist: vector <=> numeric' errors.
    """
    vec_txt = _to_vector_literal(embedding)
    with pg_cursor(db_url) as cur:
        cur.execute(_SQL_VECTOR, (vec_txt, vec_txt, k))
        return rows_to_dicts(cur)  # <-- ensures dicts, not tuples


def _fts_search(db_url: str, query: str, k: int) -> List[Dict[str, Any]]:
    with pg_cursor(db_url) as cur:
        cur.execute(_SQL_FTS, (query, k))
        return rows_to_dicts(cur)  # <-- ensures dicts, not tuples


# -----------------------------
# Public: hybrid retrieval
# -----------------------------
def hybrid_retrieve(
    db_url: str,
    query: str,
    k: int,
    alpha: float = 0.60,  # blend weight: vector (alpha) vs FTS (1 - alpha)
) -> List[Dict[str, Any]]:
    """
    Returns a unified list of hits with keys:
      document_id, title, source_uri, chunk_id, chunk_no, content, score
    """
    alpha = clamp(alpha, 0.0, 1.0)
    k = max(1, int(k))

    # 1) Embed the query
    embedding = _embed_query(query)

    # 2) Run searches (both return list[dict])
    vec_hits = _vector_search(db_url=db_url, embedding=embedding, k=k * 2)  # widen a bit before blend
    fts_hits = _fts_search(db_url=db_url, query=query, k=k * 2)

    # 3) Normalize scores to 0..1 (robust even if all equal)
    v_scores = [float(h.get("score", 0.0) or 0.0) for h in vec_hits]
    f_scores = [float(h.get("score", 0.0) or 0.0) for h in fts_hits]
    v_norm = soft_normalize(v_scores)
    f_norm = soft_normalize(f_scores)

    # index by chunk_id for blending
    by_chunk: Dict[int, Dict[str, Any]] = {}

    for h, s in zip(vec_hits, v_norm):
        cid = int(h["chunk_id"])
        by_chunk[cid] = {
            "document_id": int(h["document_id"]),
            "title": h.get("title"),
            "source_uri": h.get("source_uri"),
            "chunk_id": cid,
            "chunk_no": int(h.get("chunk_no") or 0),
            "content": h.get("content") or "",
            "vec": float(s),
            "fts": 0.0,
        }

    for h, s in zip(fts_hits, f_norm):
        cid = int(h["chunk_id"])
        if cid not in by_chunk:
            by_chunk[cid] = {
                "document_id": int(h["document_id"]),
                "title": h.get("title"),
                "source_uri": h.get("source_uri"),
                "chunk_id": cid,
                "chunk_no": int(h.get("chunk_no") or 0),
                "content": h.get("content") or "",
                "vec": 0.0,
                "fts": float(s),
            }
        else:
            by_chunk[cid]["fts"] = float(s)

    # 4) Blend and sort
    blended: List[Dict[str, Any]] = []
    for item in by_chunk.values():
        score = alpha * item["vec"] + (1.0 - alpha) * item["fts"]
        blended.append(
            {
                "document_id": item["document_id"],
                "title": item["title"],
                "source_uri": item["source_uri"],
                "chunk_id": item["chunk_id"],
                "chunk_no": item["chunk_no"],
                "content": item["content"],
                "score": float(score),
            }
        )

    blended.sort(key=lambda x: x["score"], reverse=True)
    return blended[:k]
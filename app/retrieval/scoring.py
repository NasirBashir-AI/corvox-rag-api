# app/retrieval/scoring.py
"""
Thin hybrid retrieval engine:
- vector search (pgvector) + Postgres FTS
- score blending with simple normalization
- returns unified hit shape for the retriever API
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from openai import OpenAI

from app.core.config import EMBEDDING_MODEL
from app.core.utils import pg_cursor, rows_to_dicts, soft_normalize
from app.retrieval.sql import SQL_VECTOR_SEARCH, SQL_FTS_SEARCH

_client = OpenAI()  # uses OPENAI_API_KEY from env


# -----------------------------
# Embedding helper
# -----------------------------
def _embed_query(text: str) -> List[float]:
    """Return the embedding vector for a query string."""
    resp = _client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return list(resp.data[0].embedding)


def _to_vector_literal(embedding: Sequence[float]) -> str:
    """Format a Python list[float] as a Postgres vector literal: '[0.1, -0.2, ...]'."""
    return "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"


# -----------------------------
# DB search helpers
# -----------------------------
def _vector_search(db_url: str, embedding: Sequence[float], k: int) -> List[Dict[str, Any]]:
    """
    Return top-k chunks ordered by cosine similarity (via distance `<=>` cast to ::vector).
    We convert distance to similarity in SQL: 1 - distance.
    """
    vec_txt = _to_vector_literal(embedding)
    sql = SQL_VECTOR_SEARCH  # see app/retrieval/sql.py

    with pg_cursor(db_url) as cur:
        cur.execute(sql, (vec_txt, vec_txt, k))
        return rows_to_dicts(cur)


def _fts_search(db_url: str, query: str, k: int) -> List[Dict[str, Any]]:
    """
    Return top-k chunks by Postgres full-text search with websearch_to_tsquery.
    """
    with pg_cursor(db_url) as cur:
        cur.execute(SQL_FTS_SEARCH, (query, k))
        return rows_to_dicts(cur)


# -----------------------------
# Hybrid combiner
# -----------------------------
def hybrid_retrieve(
    *,
    query: str,
    k: int,
    alpha: float = 0.6,  # blend weight: 0..1 (higher = more vector influence)
    db_url: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Run vector and FTS searches, normalize their scores to 0..1, then blend:
        blended = alpha * vec_norm + (1 - alpha) * fts_norm
    Return top-k combined results in a unified shape.

    Output keys per hit:
      - document_id: int
      - chunk_id: int
      - chunk_no: int
      - title: str|None
      - source_uri: str|None
      - content: str
      - score: float (0..1)
    """
    if db_url is None:
        # Let pg_cursor() read DB_URL from env
        from app.core.utils import getenv_str
        db_url = getenv_str("DB_URL")
        if not db_url:
            raise RuntimeError("DB_URL is not set")

    # 1) Embed query + get both result lists
    vec = _embed_query(query)
    vec_hits = _vector_search(db_url, vec, k)
    fts_hits = _fts_search(db_url, query, k)

    # 2) Normalize each list’s score to 0..1 independently
    vec_scores = [h.get("score", 0.0) for h in vec_hits]
    fts_scores = [h.get("score", 0.0) for h in fts_hits]
    vec_norm = soft_normalize(vec_scores)
    fts_norm = soft_normalize(fts_scores)

    # 3) Index by (document_id, chunk_id)
    def key_of(h: Dict[str, Any]) -> Tuple[int, int]:
        return int(h["document_id"]), int(h["chunk_id"])

    vec_map: Dict[Tuple[int, int], float] = {}
    for h, s in zip(vec_hits, vec_norm):
        vec_map[key_of(h)] = s

    fts_map: Dict[Tuple[int, int], float] = {}
    for h, s in zip(fts_hits, fts_norm):
        fts_map[key_of(h)] = s

    # 4) Merge keys and compute blended score
    all_keys = set(vec_map.keys()) | set(fts_map.keys())

    merged: Dict[Tuple[int, int], Dict[str, Any]] = {}
    # Prefer vector row shape if present, else FTS row shape
    base_rows: Dict[Tuple[int, int], Dict[str, Any]] = {
        key_of(h): h for h in vec_hits + fts_hits
    }

    for kkey in all_keys:
        base = base_rows[kkey]
        v = vec_map.get(kkey, 0.0)
        t = fts_map.get(kkey, 0.0)
        blended = alpha * v + (1.0 - alpha) * t

        merged[kkey] = {
            "document_id": base["document_id"],
            "chunk_id": base["chunk_id"],
            "chunk_no": base.get("chunk_no"),
            "title": base.get("title"),
            "source_uri": base.get("source_uri"),
            "content": base.get("content"),
            "score": float(blended),
        }

    # 5) Sort by blended score desc and return top-k
    results = sorted(merged.values(), key=lambda h: h["score"], reverse=True)
    return results[:k]
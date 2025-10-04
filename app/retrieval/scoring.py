"""
app/retrieval/scoring.py

Hybrid retrieval (vector + FTS) and score blending.
- Embeds the user query
- Runs pgvector similarity AND Postgres FTS
- Normalizes & blends scores
- Applies a per-document cap to avoid single-doc domination
- Returns top-k hits with a 0..1 blended score

No business logic about generation here—just retrieval scoring.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from app.core.config import EMBEDDING_MODEL, RETRIEVAL_TOP_K
from app.core.utils import pg_cursor, rows_to_dicts, soft_normalize, clamp
from app.retrieval.sql import SQL_VECTOR_SEARCH, SQL_FTS_SEARCH

# OpenAI client (reads OPENAI_API_KEY from environment)
_client = OpenAI()


# -----------------------------
# Embeddings
# -----------------------------

def embed_query(text: str) -> List[float]:
    """
    Create a single embedding vector for the input query.
    """
    rsp = _client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return rsp.data[0].embedding  # type: ignore[return-value]


# -----------------------------
# Low-level searches
# -----------------------------

def _vector_search(query_embedding: List[float], limit: int) -> List[Dict[str, Any]]:
    """
    Vector similarity search using pgvector cosine distance.
    Returns rows with 'distance' (lower is better).
    """
    with pg_cursor() as cur:
        cur.execute(
            SQL_VECTOR_SEARCH,
            {"query_embedding": query_embedding, "limit": limit},
        )
        return rows_to_dicts(cur)


def _fts_search(q: str, limit: int) -> List[Dict[str, Any]]:
    """
    Full-text search using websearch_to_tsquery.
    Returns rows with 'fts_rank' (higher is better).
    """
    with pg_cursor() as cur:
        cur.execute(
            SQL_FTS_SEARCH,
            {"q": q, "limit": limit},
        )
        return rows_to_dicts(cur)


# -----------------------------
# Blending & re-ranking
# -----------------------------

def _normalize_signals(
    vec_rows: List[Dict[str, Any]],
    fts_rows: List[Dict[str, Any]],
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Convert raw signals to 0..1 similarity for blending.
    - Vector: distance -> similarity (invert), then normalize across the set.
    - FTS: rank is already "the higher the better"; just normalize.
    Returns two dicts keyed by chunk_id.
    """
    # Vector distances -> preliminary similarity
    vec_ids: List[int] = []
    vec_sims_raw: List[float] = []
    for r in vec_rows:
        cid = int(r["chunk_id"])
        dist = float(r.get("distance", 1.0))
        sim = 1.0 - dist  # coarse inversion; normalization will follow
        vec_ids.append(cid)
        vec_sims_raw.append(sim)

    vec_norm_vals = soft_normalize(vec_sims_raw) if vec_sims_raw else []
    vec_sim_by_id: Dict[int, float] = {cid: clamp(s, 0.0, 1.0) for cid, s in zip(vec_ids, vec_norm_vals)}

    # FTS ranks -> normalize
    fts_ids: List[int] = []
    fts_raw: List[float] = []
    for r in fts_rows:
        cid = int(r["chunk_id"])
        rank = float(r.get("fts_rank", 0.0))
        fts_ids.append(cid)
        fts_raw.append(rank)

    fts_norm_vals = soft_normalize(fts_raw) if fts_raw else []
    fts_sim_by_id: Dict[int, float] = {cid: clamp(s, 0.0, 1.0) for cid, s in zip(fts_ids, fts_norm_vals)}

    return vec_sim_by_id, fts_sim_by_id


def _merge_and_blend(
    vec_rows: List[Dict[str, Any]],
    fts_rows: List[Dict[str, Any]],
    alpha: float,
    per_doc_cap: int,
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Merge vector & FTS hits, blend scores, cap per document, sort desc, take top_k.
    alpha: weight for vector similarity (0..1). (1-alpha) goes to FTS.
    """
    vec_sim_by_id, fts_sim_by_id = _normalize_signals(vec_rows, fts_rows)

    # Consolidate metadata by chunk_id (prefer vector row metadata; fallback to FTS if missing)
    by_id: Dict[int, Dict[str, Any]] = {}

    def _ingest_rows(rows: List[Dict[str, Any]], is_vec: bool) -> None:
        for r in rows:
            cid = int(r["chunk_id"])
            base = by_id.get(cid) or {
                "chunk_id": cid,
                "document_id": int(r["document_id"]),
                "chunk_no": int(r.get("chunk_no") or 0),
                "title": r.get("title"),
                "source_uri": r.get("source_uri"),
                "content": r.get("content"),
                # placeholders for scores
                "vec": 0.0,
                "fts": 0.0,
            }
            by_id[cid] = base

    _ingest_rows(vec_rows, is_vec=True)
    _ingest_rows(fts_rows, is_vec=False)

    # Attach normalized scores
    for cid, meta in by_id.items():
        meta["vec"] = vec_sim_by_id.get(cid, 0.0)
        meta["fts"] = fts_sim_by_id.get(cid, 0.0)
        meta["score"] = clamp(alpha * meta["vec"] + (1.0 - alpha) * meta["fts"], 0.0, 1.0)

    # Per-document cap
    per_doc: Dict[int, int] = defaultdict(int)
    capped: List[Dict[str, Any]] = []
    # sort by blended score desc
    for r in sorted(by_id.values(), key=lambda x: x["score"], reverse=True):
        doc_id = int(r["document_id"])
        if per_doc[doc_id] >= per_doc_cap:
            continue
        per_doc[doc_id] += 1
        capped.append(r)
        if len(capped) >= max(top_k * 3, top_k):  # keep a buffer before final trim
            break

    # Final trim to top_k
    return sorted(capped, key=lambda x: x["score"], reverse=True)[:top_k]


# -----------------------------
# Public API
# -----------------------------

def hybrid_retrieve(
    query: str,
    k: Optional[int] = None,
    alpha: float = 0.60,
    per_doc_cap: int = 2,
    vector_limit: Optional[int] = None,
    fts_limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run hybrid retrieval and return a list of hits:
      [{
        "document_id": int,
        "chunk_id": int,
        "chunk_no": int,
        "title": str|None,
        "source_uri": str|None,
        "content": str,
        "score": float (0..1),
      }, ...]
    """
    top_k = k or RETRIEVAL_TOP_K

    # Embed query once
    q_emb = embed_query(query)

    # Reasonable fan-out before re-rank
    v_lim = vector_limit or max(20, top_k * 5)
    f_lim = fts_limit or max(20, top_k * 5)

    vec_rows = _vector_search(q_emb, v_lim)
    fts_rows = _fts_search(query, f_lim)

    # Blend and re-rank
    hits = _merge_and_blend(vec_rows, fts_rows, alpha=alpha, per_doc_cap=per_doc_cap, top_k=top_k)

    return hits
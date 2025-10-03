# app/retrieval/retriever.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any
import psycopg2
import psycopg2.extras

from openai import OpenAI

from app.core.config import (
    DB_URL,
    EMBEDDING_MODEL,
    MIN_SIM,         # e.g. 0.25 (higher = stricter)
)
from app.api.schemas import SearchHit

client = OpenAI()

def _connect():
    return psycopg2.connect(DB_URL)

def _embed(query: str) -> List[float]:
    rsp = client.embeddings.create(model=EMBEDDING_MODEL, input=query)
    return rsp.data[0].embedding  # type: ignore

def _search_vec(cx, qvec: List[float], k: int) -> List[Dict[str, Any]]:
    """
    Cosine similarity search over pgvector.
    """
    with cx.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
              c.id            AS chunk_id,
              c.doc_id,
              d.title,
              d.uri,
              c.chunk_no,
              c.content,
              1 - (c.embedding <=> %s::vector) AS similarity,     -- cosine sim
              (c.embedding <=> %s::vector)     AS distance        -- cosine distance
            FROM corah_store.chunks c
            JOIN corah_store.documents d ON d.id = c.doc_id
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            (qvec, qvec, qvec, k)
        )
        return list(cur.fetchall())

def _to_hits(rows: List[Dict[str, Any]]) -> List[SearchHit]:
    hits: List[SearchHit] = []
    for r in rows:
        hits.append(
            SearchHit(
                doc=r.get("title") or "",
                uri=r.get("uri"),
                score=float(r.get("similarity") or 0.0),
                dist=float(r.get("distance") or 0.0),
                chunk_no=int(r.get("chunk_no") or 0),
                title=r.get("title") or "",
                text=r.get("content") or "",
            )
        )
    return hits

def search(query: str, k: int = 5) -> List[SearchHit]:
    """
    Returns top-k hits, filtered by MIN_SIMILARITY.
    """
    qvec = _embed(query)
    with _connect() as cx:
        rows = _search_vec(cx, qvec, k)
    hits = _to_hits(rows)

    # Similarity guard — filter obvious mismatches.
    if MIN_SIM is not None:
        hits = [h for h in hits if (h.score or 0.0) >= float(MIN_SIM)]

    return hits

def make_context(hits: List[SearchHit], max_chars: int = 3000) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Concatenate hit texts up to max_chars. Also return a lightweight 'used' list
    for generator debug/citations.
    """
    buf: List[str] = []
    used: List[Dict[str, Any]] = []
    total = 0
    for h in hits:
        t = (h.text or "").strip()
        if not t:
            continue
        if total + len(t) > max_chars:
            break
        buf.append(t)
        total += len(t)
        used.append({
            "title": h.title or h.doc,
            "uri": h.uri,
            "chunk_no": h.chunk_no,
            "similarity": h.score,
            "distance": h.dist,
        })
    return ("\n\n---\n\n".join(buf), used)
# app/retrieval/scoring.py

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from app.core.utils import pg_cursor, rows_to_dicts


def _to_vector_literal(vec: Sequence[float]) -> str:
    """
    Convert a Python list/sequence of floats into pgvector's text literal: [v1,v2,...].
    We then cast it with ::vector on the SQL side.
    """
    return "[" + ",".join(f"{float(x):.8f}" for x in vec) + "]"


def _vector_search(db_url: str, embedding: Sequence[float], k: int) -> List[Dict[str, Any]]:
    """
    Return top-k chunks ordered by cosine distance (<=>). We cast the parameter to ::vector
    to avoid 'operator does not exist: vector <=> numeric' errors.
    """
    vec_txt = _to_vector_literal(embedding)

    sql = """
        SELECT
            d.id   AS document_id,
            d.title,
            c.id   AS chunk_id,
            c.content,
            1 - (c.embedding <=> (%s)::vector) AS score
        FROM corah_store.chunks c
        JOIN corah_store.documents d ON d.id = c.document_id
        ORDER BY c.embedding <=> (%s)::vector
        LIMIT %s;
    """
    with pg_cursor(db_url) as cur:
        cur.execute(sql, (vec_txt, vec_txt, k))
        return rows_to_dicts(cur)
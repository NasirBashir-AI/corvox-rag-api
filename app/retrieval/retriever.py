# app/retrieval/retriever.py
"""
Thin retrieval facade used by the API and generator.

- Public search() -> returns top-k hybrid retrieval hits (vector + FTS)
- get_facts()     -> reads structured facts for contact/pricing answers
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from app.core.config import DB_URL, RETRIEVAL_TOP_K
from app.core.utils import pg_cursor, rows_to_dicts
from app.retrieval.scoring import hybrid_retrieve
from app.retrieval.sql import SQL_FACTS_SELECT_BY_NAMES


# -----------------------------
# Public search API
# -----------------------------
def search(query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Return top-k hybrid retrieval hits with a unified shape:
      {document_id, title, source_uri, chunk_id, chunk_no, content, score}
    """
    top_k = int(k or RETRIEVAL_TOP_K)
    # Pass DB_URL to the scoring layer (required)
    return hybrid_retrieve(db_url=DB_URL, query=query, k=top_k)


# -----------------------------
# Structured facts (contact/pricing)
# -----------------------------
def get_facts(names: Sequence[str]) -> Dict[str, str]:
    if not names:
        return {}
    placeholders = ",".join(["%s"] * len(names))
    sql = SQL_FACTS_SELECT_BY_NAMES.format(placeholders=placeholders)
    with pg_cursor(DB_URL) as cur:
        cur.execute(sql, tuple(names))
        rows = rows_to_dicts(cur)
    return {r["name"]: r["value"] for r in rows if r.get("value")}
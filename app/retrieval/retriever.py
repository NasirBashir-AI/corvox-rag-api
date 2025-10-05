# app/retrieval/retriever.py
"""
Thin retrieval facade used by the API and generator.

- search(query, k)      -> top-k hybrid retrieval hits (vector + FTS)
- get_facts(names)      -> structured facts for contact/pricing
- top_similarity(hits)  -> best score across hits (compat for generator)
- make_context(hits)    -> join hits into a single context string (compat)
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


# -----------------------------
# Compatibility helpers used by generator.py
# -----------------------------
def top_similarity(hits: Sequence[Dict[str, Any]]) -> float:
    """Return the highest score among hits (0.0 if empty)."""
    if not hits:
        return 0.0
    return max(float(h.get("score") or 0.0) for h in hits)


def make_context(hits: Sequence[Dict[str, Any]], max_chars: int = 2000) -> str:
    """
    Join hits into a single context string. Keeps things simple and compact.
    """
    parts: List[str] = []
    for i, h in enumerate(hits, 1):
        title = h.get("title") or f"Document {h.get('document_id', i)}"
        content = h.get("content") or ""
        source = h.get("source_uri") or ""
        block = f"## {title}\n{content}"
        if source:
            block += f"\n\n(SOURCE: {source})"
        parts.append(block.strip())
        if sum(len(p) for p in parts) >= max_chars:
            break
    ctx = "\n\n---\n\n".join(parts).strip()
    # Trim to hard cap if necessary
    if len(ctx) > max_chars:
        ctx = ctx[: max_chars - 1].rstrip() + "…"
    return ctx
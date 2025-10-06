from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple
from app.core.config import DB_URL, RETRIEVAL_TOP_K
from app.core.utils import pg_cursor, rows_to_dicts
from app.retrieval.scoring import hybrid_retrieve
from app.retrieval.sql import SQL_FACTS_SELECT_BY_NAMES

# app/retrieval/retriever.py

"""
Thin retrieval facade used by the API and generator.

- search(query, k)      -> top-k hybrid retrieval hits (vector + FTS)
- get_facts(names)      -> structured facts for contact/pricing
- top_similarity(hits)  -> best score across hits (compat for generator)
- make_context(hits)    -> join hits into a single context string (compat)
"""

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
def get_facts(names: Sequence[str], db_url: Optional[str] = None) -> Dict[str, str]:
    """
    Fetch named structured facts as a dict {name: value}.
    Uses rows_to_dicts to avoid tuple indexing errors.
    """
    if not names:
        return {}

    # De-duplicate & keep order stable
    uniq = list(dict.fromkeys([n for n in names if n]))

    if not uniq:
        return {}

    with pg_cursor(db_url) as cur:
        # SQL_FACTS_SELECT_BY_NAMES should be like:
        #   SELECT name, value, uri, updated_at
        #   FROM corah_store.facts
        #   WHERE name = ANY(%s)
        #   ORDER BY name;
        cur.execute(SQL_FACTS_SELECT_BY_NAMES, {"names": uniq})
        rows: List[Dict[str, Any]] = rows_to_dicts(cur)

    facts: Dict[str, str] = {}
    for r in rows:
        n = r.get("name")
        v = r.get("value")
        if n is not None and v is not None:
            facts[str(n)] = str(v)
    return facts


# -----------------------------
# Compatibility helpers used by generator.py
# -----------------------------
def top_similarity(hits: Sequence[Dict[str, Any]]) -> float:
    """Return the highest score among hits (0.0 if empty)."""
    if not hits:
        return 0.0
    return max(float(h.get("score") or 0.0) for h in hits)


def make_context(
    hits: Sequence[Dict[str, Any]],
    max_chars: int = 2000,
) -> Tuple[str, Optional[List[Dict[str, str]]]]:
    """
    Join hits into a single context string and also return lightweight citations.
    Returns: (context_text, citations or None)
    """
    parts: List[str] = []
    citations: List[Dict[str, str]] = []

    for i, h in enumerate(hits, 1):
        title = h.get("title") or f"Document {h.get('document_id', i)}"
        content = h.get("content") or ""
        source = (h.get("source_uri") or "").strip()

        block = f"## {title}\n{content}"
        if source:
            block += f"\n\n(SOURCE: {source})"
            citations.append({"title": title, "uri": source})

        parts.append(block.strip())

        # stop when we’re at/over the budget
        if sum(len(p) for p in parts) >= max_chars:
            break

    ctx = "\n\n---\n\n".join(parts).strip()
    if len(ctx) > max_chars:
        ctx = ctx[: max_chars - 1].rstrip() + "…"

    return ctx, (citations if citations else None)
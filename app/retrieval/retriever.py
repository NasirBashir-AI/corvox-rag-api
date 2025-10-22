# app/retrieval/retriever.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.config import DB_URL, RETRIEVAL_TOP_K
from app.core.utils import pg_cursor, rows_to_dicts
from app.retrieval.scoring import hybrid_retrieve
from app.retrieval.sql import SQL_FACTS_SELECT_BY_NAMES

"""
Thin retrieval facade used by the API and generator.

- search(query, k)      -> top-k hybrid retrieval hits (vector + FTS), with light canonical filtering
- get_facts(names)      -> structured facts for contact/pricing
- top_similarity(hits)  -> best score across hits
- make_context(hits)    -> join hits into a single context string (+ citations)
"""

# -----------------------------
# Public search API
# -----------------------------
def search(query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Return top-k hybrid retrieval hits with a unified shape:
      {document_id, title, source_uri, chunk_id, chunk_no, content, score}

    Adds a small, safe filter so “services” and “pricing/cost” queries prefer
    canonical docs by title. If filtering would empty results, we fall back
    to the original list.
    """
    top_k = int(k or RETRIEVAL_TOP_K)

    # Base retrieval (vector + FTS blend)
    raw_hits: List[Dict[str, Any]] = hybrid_retrieve(db_url=DB_URL, query=query, k=top_k)

    # Canonical filtering (non-destructive: only narrows when possible)
    qlow = (query or "").lower()
    titles = [(h.get("title") or "").lower() for h in raw_hits]

    filtered = raw_hits
    if "service" in qlow:
        filt = [
            h for h in raw_hits
            if "service" in (h.get("title") or "").lower()
               or "services" in (h.get("title") or "").lower()
        ]
        if filt:  # only narrow if we actually found better-scoped docs
            filtered = filt

    elif ("pricing" in qlow) or ("price" in qlow) or ("cost" in qlow) or ("budget" in qlow):
        filt = [
            h for h in raw_hits
            if any(token in (h.get("title") or "").lower() for token in ("pricing", "price", "cost"))
        ]
        if filt:
            filtered = filt

    return filtered[:top_k]

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

    # IMPORTANT: default to DB_URL if caller didn't pass one
    with pg_cursor(db_url or DB_URL) as cur:
        # SQL should be parameterized with %(names)s
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

    Each citation item uses keys aligned with the API schema:
      {"title": <str>, "source_uri": <str>}
    """
    parts: List[str] = []
    citations: List[Dict[str, str]] = []

    used = 0
    for i, h in enumerate(hits, 1):
        title = h.get("title") or f"Document {h.get('document_id', i)}"
        content = (h.get("content") or "").strip()
        source = (h.get("source_uri") or "").strip()

        if not content:
            continue

        block = f"## {title}\n{content}"
        if source:
            block += f"\n\n(SOURCE: {source})"
            citations.append({"title": title, "source_uri": source})

        if used + len(block) > max_chars:
            # take a clipped slice if nothing has been added yet
            if not parts:
                block = block[: max_chars - 1].rstrip() + "…"
                parts.append(block)
            break

        parts.append(block)
        used += len(block)

    ctx = "\n\n---\n\n".join(parts).strip()
    return ctx, (citations if citations else None)
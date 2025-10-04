"""
app/retrieval/retriever.py

Thin retrieval facade used by the API and generator:
- Hybrid search (vector + FTS) via scoring.hybrid_retrieve
- Shape results for API
- Build model context blocks from hits
- Read structured facts (contact/pricing) when requested
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.config import ENABLE_HYBRID, RETRIEVAL_TOP_K
from app.core.utils import pg_cursor, rows_to_dicts
from app.retrieval.scoring import hybrid_retrieve
from app.retrieval.sql import (
    SQL_FACTS_SELECT_BY_NAMES,
)

# -----------------------------
# Public search API
# -----------------------------

def search(query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Return top-k hybrid retrieval hits with a unified shape:
      {
        "document_id": int,
        "chunk_id": int,
        "chunk_no": int,
        "title": str|None,
        "source_uri": str|None,
        "content": str,
        "score": float (0..1)
      }
    """
    top_k = k or RETRIEVAL_TOP_K

    # For now, ENABLE_HYBRID simply guards hybrid_retrieve call (we can extend later if needed).
    if ENABLE_HYBRID:
        hits = hybrid_retrieve(query=query, k=top_k)
    else:
        # If hybrid disabled, still call hybrid with alpha=1.0 (vector-only path);
        # this keeps surface area stable and avoids code duplication.
        hits = hybrid_retrieve(query=query, k=top_k, alpha=1.0)

    return hits


# -----------------------------
# Context assembly for the generator
# -----------------------------

def make_context(
    hits: Sequence[Dict[str, Any]],
    max_chars: int = 3000,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build a single context string from hits, keeping order by score.
    Returns (context_text, used_metadata_list).
    """
    pieces: List[str] = []
    used: List[Dict[str, Any]] = []
    total = 0

    for h in hits:
        content = (h.get("content") or "").strip()
        if not content:
            continue
        # Compact header helps the model; we don't expose file names to the user output.
        title = (h.get("title") or "").strip()
        head = f"[{title or 'snippet'} · #{h.get('chunk_no', 0)}]\n"
        block = head + content + "\n"
        # Stop if adding this block would exceed cap
        if total + len(block) > max_chars and pieces:
            break
        pieces.append(block)
        used.append(
            {
                "title": title or None,
                "chunk_no": int(h.get("chunk_no") or 0),
                "score": float(h.get("score") or 0.0),
            }
        )
        total += len(block)

    return ("\n---\n".join(pieces)).strip(), used


def top_similarity(hits: Sequence[Dict[str, Any]]) -> float:
    """Return the highest blended score (0..1) or 0.0 if none."""
    if not hits:
        return 0.0
    try:
        return max(float(h.get("score", 0.0)) for h in hits)
    except Exception:
        return 0.0


# -----------------------------
# Structured facts (contact/pricing)
# -----------------------------

def get_facts(names: Sequence[str]) -> List[Dict[str, Any]]:
    """
    Fetch structured facts by canonical names from corah_store.facts.
    Returns list of dicts: {name, value, uri, updated_at}
    """
    if not names:
        return []
    with pg_cursor() as cur:
        cur.execute(SQL_FACTS_SELECT_BY_NAMES, {"names": list(names)})
        return rows_to_dicts(cur)
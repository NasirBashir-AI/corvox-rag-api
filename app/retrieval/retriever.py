# app/retrieval/retriever.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.config import DB_URL, RETRIEVAL_TOP_K
from app.core.utils import pg_cursor, rows_to_dicts
from app.retrieval.scoring import hybrid_retrieve
from app.retrieval.sql import SQL_FACTS_SELECT_BY_NAMES

"""
Thin retrieval facade used by the API and generator.
"""

def search(query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
    top_k = int(k or RETRIEVAL_TOP_K)
    hits = hybrid_retrieve(db_url=DB_URL, query=query, k=top_k)

    qlow = (query or "").lower()
    if any(t in qlow for t in ("email", "phone", "address", "contact", "where are you based", "website", "url")):
        scored = []
        for h in hits:
            title = (h.get("title") or "").lower()
            bonus = 0.15 if "contact" in title or "about" in title else 0.0
            scored.append({**h, "score": float(h.get("score") or 0.0) + bonus})
        scored.sort(key=lambda x: x["score"], reverse=True)
        hits = scored
    return hits

def get_facts(names: Sequence[str], db_url: Optional[str] = None) -> Dict[str, str]:
    if not names:
        return {}
    uniq = list(dict.fromkeys([n for n in names if n]))
    if not uniq:
        return {}
    with pg_cursor(db_url or DB_URL) as cur:
        cur.execute(SQL_FACTS_SELECT_BY_NAMES, {"names": uniq})
        rows: List[Dict[str, Any]] = rows_to_dicts(cur)

    facts: Dict[str, str] = {}
    for r in rows:
        n = r.get("name"); v = r.get("value")
        if n is not None and v is not None:
            facts[str(n)] = str(v)
    return facts

def top_similarity(hits: Sequence[Dict[str, Any]]) -> float:
    if not hits:
        return 0.0
    return max(float(h.get("score") or 0.0) for h in hits)

def make_context(hits: Sequence[Dict[str, Any]], max_chars: int = 2000) -> Tuple[str, Optional[List[Dict[str, str]]]]:
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
            if not parts:
                parts.append(block[: max_chars - 1].rstrip() + "…")
            break
        parts.append(block); used += len(block)
    ctx = "\n\n---\n\n".join(parts).strip()
    return ctx, (citations if citations else None)
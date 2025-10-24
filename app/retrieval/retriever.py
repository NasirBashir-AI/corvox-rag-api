from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.config import DB_URL, RETRIEVAL_TOP_K
from app.core.utils import pg_cursor, rows_to_dicts
from app.retrieval.scoring import hybrid_retrieve
from app.retrieval.sql import SQL_FACTS_SELECT_BY_NAMES

def search(query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
    top_k = int(k or RETRIEVAL_TOP_K)
    hits = hybrid_retrieve(db_url=DB_URL, query=query, k=top_k)
    # Light prioritization for common FAQs (keeps order stable otherwise)
    ql = query.lower()
    if "service" in ql:
        hits.sort(key=lambda h: 0 if "service" in (h.get("title","").lower()) else 1)
    if "pricing" in ql or "price" in ql or "cost" in ql:
        hits.sort(key=lambda h: 0 if "pricing" in (h.get("title","").lower()) else 1)
    if any(w in ql for w in ["email","address","where","based","website","url","contact"]):
        hits.sort(key=lambda h: 0 if "contact" in (h.get("title","").lower()) else 1)
    return hits

def get_facts(names: Sequence[str], db_url: Optional[str] = None) -> Dict[str, str]:
    if not names: return {}
    uniq = list(dict.fromkeys([n for n in names if n]))
    with pg_cursor(db_url or DB_URL) as cur:
        cur.execute(SQL_FACTS_SELECT_BY_NAMES, {"names": uniq})
        rows = rows_to_dicts(cur)
    out: Dict[str, str] = {}
    for r in rows:
        n, v = r.get("name"), r.get("value")
        if n and v: out[str(n)] = str(v)
    return out

def top_similarity(hits: Sequence[Dict[str, Any]]) -> float:
    return max([float(h.get("score") or 0.0) for h in hits], default=0.0)

def make_context(hits: Sequence[Dict[str, Any]], max_chars: int = 2000) -> Tuple[str, Optional[List[Dict[str,str]]]]:
    parts: List[str] = []; cits: List[Dict[str,str]] = []; used = 0
    for i,h in enumerate(hits,1):
        title = h.get("title") or f"Document {h.get('document_id', i)}"
        content = (h.get("content") or "").strip()
        if not content: continue
        source = (h.get("source_uri") or "").strip()
        block = f"## {title}\n{content}"
        if source:
            block += f"\n\n(SOURCE: {source})"
            cits.append({"title": title, "source_uri": source})
        if used + len(block) > max_chars:
            if not parts:
                parts.append(block[:max_chars-1].rstrip()+"…")
            break
        parts.append(block); used += len(block)
    ctx = "\n\n---\n\n".join(parts).strip()
    return ctx, (cits or None)
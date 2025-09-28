# app/retrieval/retriever.py
# -*- coding: utf-8 -*-

"""
Corah Retrieval Layer
- Embeds a user query with OpenAI
- Performs ANN search in Postgres (pgvector)
- Returns top-K chunks as context

Run:
  python -m app.retrieval.retriever "What's included in the service catalogue?" --k 5 --pretty
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Dict, Any, Optional, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor

try:
    # OpenAI >= 1.0 SDK
    from openai import OpenAI
except Exception:
    print("OpenAI SDK not found. Did you install in this venv?\n  pip install openai>=1.0.0")
    raise

# ---- Config -----------------------------------------------------------------

# Preferred: read DB URL from env (POSTGRES_URL). Fallback to app.core.config.DB_URL.
DB_URL = os.getenv("POSTGRES_URL")
if not DB_URL:
    try:
        from app.core.config import DB_URL as FALLBACK_DB_URL
        DB_URL = FALLBACK_DB_URL
    except Exception:
        raise RuntimeError(
            "No POSTGRES_URL in environment and app.core.config.DB_URL not available."
        )

# Embedding model (must match your 1536-dim schema)
try:
    from app.core.config import EMBEDDING_MODEL
except Exception:
    EMBEDDING_MODEL = "text-embedding-3-small"  # safe default

# Optional: tiktoken for later trimming/packing
try:
    import tiktoken  # noqa
    HAVE_TIKTOKEN = True
except Exception:
    HAVE_TIKTOKEN = False

# ---- OpenAI client -----------------------------------------------------------

_client: Optional[OpenAI] = None


def _client_or_die() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()  # reads OPENAI_API_KEY from env
    return _client


def embed_text(text: str) -> List[float]:
    """
    Create a 1536-dim embedding vector using OpenAI.
    """
    client = _client_or_die()
    text = text.strip()
    if not text:
        raise ValueError("Cannot embed empty text")
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


# ---- DB helpers --------------------------------------------------------------

def get_connection():
    """
    Simple connection helper. For services, you might add pooling later.
    """
    return psycopg2.connect(DB_URL)


def _to_pgvector(vec: List[float]) -> str:
    """
    pgvector textual format: '[v1,v2,...,vn]'
    Psycopg2 doesn't adapt python lists to pgvector by default, so we pass text.
    """
    # keep 6 decimal places to reduce payload
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


# ---- Retrieval ---------------------------------------------------------------

def search(
    query: str,
    k: int = 5,
    min_chars: int = 10,
) -> List[Dict[str, Any]]:
    """
    Embed the query and retrieve top-K chunks ordered by distance (L2).
    Returns a list of dicts: title, uri, chunk_id, chunk_no, content, distance.

    NOTE:
      - We use L2 distance (<->). Smaller is better.
      - If you prefer a similarity score, you can convert to 1/(1+distance) on return.
    """
    emb = embed_text(query)
    qv = _to_pgvector(emb)

    sql = """
    SELECT
        d.id       AS doc_id,
        d.title    AS title,
        d.uri      AS uri,
        c.id       AS chunk_id,
        c.chunk_no AS chunk_no,
        c.content  AS content,
        (c.embedding <-> %s::vector) AS distance
    FROM corah_store.chunks c
    JOIN corah_store.documents d ON d.id = c.doc_id
    ORDER BY c.embedding <-> %s::vector
    LIMIT %s;
    """

    rows: List[Dict[str, Any]] = []
    with get_connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        # pass same vector twice (for ORDER BY and SELECT)
        cur.execute(sql, (qv, qv, k))
        fetched = cur.fetchall()

    # Basic hygiene: drop accidental empty chunks, compute a simple similarity display
    for r in fetched:
        content = (r.get("content") or "").strip()
        if len(content) < min_chars:
            continue
        distance = float(r["distance"])
        similarity = 1.0 / (1.0 + distance)  # simple monotonic transform
        rows.append(
            {
                "doc_id": r["doc_id"],
                "title": r.get("title"),
                "uri": r.get("uri"),
                "chunk_id": r["chunk_id"],
                "chunk_no": r["chunk_no"],
                "content": content,
                "distance": distance,
                "similarity": similarity,
            }
        )

    return rows


def make_context(
    hits: List[Dict[str, Any]],
    max_chars: int = 3000,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Concatenate retrieved chunks into a single context string, up to max_chars.
    Returns (context, used_hits).
    """
    used: List[Dict[str, Any]] = []
    buf: List[str] = []
    total = 0

    for h in hits:
        snippet = h["content"].strip()
        if not snippet:
            continue
        if total + len(snippet) + 2 > max_chars:
            break
        buf.append(snippet)
        used.append(h)
        total += len(snippet) + 2

    return ("\n\n".join(buf), used)


# ---- CLI ---------------------------------------------------------------------

def _pretty_print(hits: List[Dict[str, Any]], show: int = 5):
    show = max(1, min(show, len(hits)))
    for i, h in enumerate(hits[:show], start=1):
        print(f"\n[{i}] score≈{h['similarity']:.4f}  dist={h['distance']:.4f}  "
              f"doc='{h.get('title') or ''}'  uri='{h.get('uri') or ''}'")
        print("-" * 80)
        print(h["content"][:1200])
        if len(h["content"]) > 1200:
            print("... [truncated]")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Corah: pgvector retriever")
    parser.add_argument("query", type=str, help="User query text")
    parser.add_argument("--k", type=int, default=5, help="How many chunks to fetch")
    parser.add_argument("--max-chars", type=int, default=3000, help="Context size")
    parser.add_argument("--pretty", action="store_true", help="Pretty print results")
    args = parser.parse_args(argv)

    hits = search(args.query, k=args.k)
    ctx, used = make_context(hits, max_chars=args.max_chars)

    if args.pretty:
        print(f"\nTop {len(used)} of {len(hits)} hits")
        _pretty_print(hits, show=args.k)
        print("\n" + "=" * 80)
        print("CONTEXT (truncated to max-chars)")
        print("=" * 80 + "\n")
        print(ctx[:3000])
    else:
        # machine-readable
        import json
        print(json.dumps({"context": ctx, "hits": used}, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
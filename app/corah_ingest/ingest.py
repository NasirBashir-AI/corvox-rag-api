"""
app/corah_ingest/ingest.py

Knowledge ingestion for Corah.

Pipeline (local files → Postgres):
1) Walk RAW_DOCS_PATH (or --root) for .md / .txt files
2) For each file:
   - Read text, infer title
   - Chunk text with overlap
   - Embed chunks (OpenAI embeddings)
   - Upsert document row; insert chunks with embeddings
   - Extract structured facts (contact/pricing) and upsert into facts table
3) Optional --rebuild wipes documents+chunks before reloading (facts preserved)

Notes:
- DB_URL is read from env; passing --db will set it for this process.
- Chunk size/overlap and embedding model are read from app.core.config.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from openai import OpenAI

from app.core.config import (
    RAW_DOCS_PATH,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from app.corah_ingest.db_utils import (
    ensure_schema,
    begin_rebuild,
    upsert_document,
    insert_chunk,
    upsert_fact,
)
from app.corah_ingest.extract_facts import extract_facts_from_markdown


# -----------------------------
# File helpers
# -----------------------------

def _read_text_file(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        # last-ditch: binary-safe decode
        return p.read_bytes().decode("utf-8", errors="ignore")


def _infer_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            # take heading content after hashes
            return s.lstrip("#").strip()[:200] or fallback
        if s:
            # first non-empty line if no markdown heading
            return s[:200]
    return fallback


def _rel_uri(root: Path, file_path: Path) -> str:
    """
    Stable logical URI for the document (used for upserts).
    Example: "kb/about-us.md"
    """
    try:
        rel = file_path.relative_to(root)
    except Exception:
        rel = file_path.name
    # Use forward slashes for consistency
    return str(rel).replace("\\", "/")


# -----------------------------
# Chunking
# -----------------------------

def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """
    Simple fixed-window chunking on characters with overlap.
    Keeps boundaries simple; upstream retrieval is robust to this.
    """
    t = text.strip()
    if not t:
        return []
    if size <= 0:
        return [t]

    step = max(1, size - max(0, overlap))
    chunks: List[str] = []
    i = 0
    n = len(t)
    while i < n:
        chunk = t[i : i + size].strip()
        if chunk:
            chunks.append(chunk)
        i += step
    return chunks


# -----------------------------
# Embeddings
# -----------------------------

_client = OpenAI()  # reads OPENAI_API_KEY

def embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of texts in one call.
    """
    rsp = _client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [row.embedding for row in rsp.data]  # type: ignore[return-value]


def embed_all(chunks: List[str], batch_size: int = 64) -> List[List[float]]:
    out: List[List[float]] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        out.extend(embed_batch(batch))
    return out


# -----------------------------
# Ingest one file
# -----------------------------

def ingest_file(root: Path, file_path: Path) -> Tuple[int, int]:
    """
    Ingest a single file. Returns (document_id, chunk_count).
    """
    text = _read_text_file(file_path)
    if not text.strip():
        return (-1, 0)

    title = _infer_title(text, fallback=file_path.stem.replace("_", " ").replace("-", " ").title())
    uri = _rel_uri(root, file_path)

    # Upsert document
    doc_id = upsert_document(title=title, uri=uri, source_uri=str(file_path))

    # Chunk
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        return (doc_id, 0)

    # Embed
    embeddings = embed_all(chunks)

    # Insert chunks
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings), start=1):
        token_count = max(1, int(len(chunk) / 4))  # rough heuristic; OK for analytics
        insert_chunk(document_id=doc_id, chunk_no=i, content=chunk, embedding=emb, token_count=token_count)

    # Extract & store structured facts (append-most-recent)
    facts = extract_facts_from_markdown(text, uri=uri)
    for f in facts:
        upsert_fact(name=f["name"], value=f["value"], uri=f.get("uri") or uri)

    return (doc_id, len(chunks))


# -----------------------------
# Driver
# -----------------------------

def ingest_root(root: Path) -> Tuple[int, int]:
    """
    Walk the root and ingest all supported files.
    Returns (file_count, total_chunks).
    """
    files = sorted(
        [*root.rglob("*.md"), *root.rglob("*.txt")],
        key=lambda p: str(p).lower(),
    )
    if not files:
        print(f"[ingest] No .md or .txt files found under: {root}")
        return (0, 0)

    print(f"[ingest] Found {len(files)} files under {root}")
    total_files = 0
    total_chunks = 0
    for fp in files:
        doc_id, n_chunks = ingest_file(root, fp)
        if doc_id != -1:
            total_files += 1
            total_chunks += n_chunks
            print(f"[ingest] {fp.name}: doc_id={doc_id}, chunks={n_chunks}", flush=True)
    print(f"[ingest] DONE. files={total_files}, chunks={total_chunks}")
    return (total_files, total_chunks)


def main() -> None:
    ap = argparse.ArgumentParser(description="Corah Ingestion")
    ap.add_argument("--root", type=str, default=RAW_DOCS_PATH, help="Root folder with .md/.txt files")
    ap.add_argument(
        "--db",
        type=str,
        default=os.getenv("DB_URL", ""),
        help="Postgres URL (overrides env for this run only)",
    )
    ap.add_argument("--rebuild", action="store_true", help="Wipe documents+chunks before ingest (facts preserved)")
    args = ap.parse_args()

    # If --db provided, set env so lower layers pick it up
    if args.db:
        os.environ["DB_URL"] = args.db

    root = Path(args.root).expanduser().resolve()

    # Prepare schema; optionally wipe
    ensure_schema()
    if args.rebuild:
        print("[ingest] Rebuild requested: wiping documents+chunks…")
        begin_rebuild()

    # Run
    ingest_root(root)


if __name__ == "__main__":
    main()
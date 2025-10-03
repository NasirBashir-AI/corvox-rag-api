# app/corah_ingest/ingest.py
from __future__ import annotations
import os
import glob
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
import psycopg2
import psycopg2.extras

# --- Postgres / pgvector setup ---
from pgvector.psycopg2 import register_vector
from app.core.config import DB_URL

def get_pg_conn():
    conn = psycopg2.connect(DB_URL)
    register_vector(conn)  # so we can insert Python lists into VECTOR(1536)
    return conn


from openai import OpenAI

from app.core.config import (
    DB_URL,
    RAW_DOCS_PATH,          # e.g. "/home/ec2-user/corah/data/raw"
    EMBEDDING_MODEL,        # e.g. "text-embedding-3-small"
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

client = OpenAI()

# ---------- utils ----------

def _connect():
    return psycopg2.connect(DB_URL)

def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _clean(s: str) -> str:
    # light cleanup; keep markdown headings etc.
    s = s.replace("\r\n", "\n").strip()
    return s

@dataclass
class Chunk:
    chunk_no: int
    text: str
    embedding: Optional[List[float]] = None

# ---------- chunking ----------

def chunk_text(txt: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    txt = _clean(txt)
    if not txt:
        return []
    chunks: List[Chunk] = []
    start = 0
    cno = 1
    while start < len(txt):
        end = min(len(txt), start + size)
        piece = txt[start:end]
        chunks.append(Chunk(chunk_no=cno, text=piece))
        cno += 1
        if end == len(txt):
            break
        start = max(0, end - overlap)
    return chunks

# ---------- embeddings ----------

def embed_batch(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]

# ---------- DB I/O ----------

def upsert_document(cx, title: str, uri: str) -> int:
    """
    Insert or update a document record in Postgres.
    Returns the document id.
    """
    conn = get_pg_conn()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO documents (uri, title)
        VALUES (%s, %s)
        ON CONFLICT (uri) DO UPDATE SET title = EXCLUDED.title
        RETURNING id;
    """, (uri, title))

    doc_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return doc_id


def insert_chunks(cx, doc_id: int, chunks: List[Chunk]):
    """
    Bulk-insert chunks into public.chunks with pgvector embeddings.
    Expects each Chunk to have: chunk_no, text, embedding (list[float]).
    """
    with cx.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO chunks (document_id, chunk_index, content, embedding)
            VALUES %s
            ON CONFLICT (document_id, chunk_index) DO NOTHING
            """,
            [
                (doc_id, ch.chunk_no, ch.text, ch.embedding)
                for ch in chunks
            ],
        )


def clear_all(cx):
    with cx.cursor() as cur:
        cur.execute("TRUNCATE corah_store.chunks RESTART IDENTITY CASCADE;")
        cur.execute("TRUNCATE corah_store.documents RESTART IDENTITY CASCADE;")

# ---------- main pipeline ----------

def ingest_folder(root: str, rebuild: bool = False) -> Tuple[int, int]:
    """
    Walks root, ingests .md/.txt files.
    Returns (#documents, #chunks).
    """
    patterns = [os.path.join(root, "**", "*.md"), os.path.join(root, "**", "*.txt")]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    files = sorted(files)

    if not files:
        print(f"[ingest] No files found under: {root}")
        return (0, 0)

    doc_count = 0
    chunk_count = 0

    with _connect() as cx:
        if rebuild:
            print("[ingest] Rebuild requested → clearing existing documents/chunks")
            clear_all(cx)
            cx.commit()

        for path in files:
            title = os.path.basename(path)
            uri = os.path.abspath(path)
            text = _read_text(path)
            parts = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            if not parts:
                continue

            # embed chunk texts in batches (keep one batch for simplicity)
            embeddings = embed_batch([c.text for c in parts])
            for c, emb in zip(parts, embeddings):
                c.embedding = emb

            doc_id = upsert_document(cx, title=title, uri=uri)
            insert_chunks(cx, doc_id, parts)
            doc_count += 1
            chunk_count += len(parts)
            cx.commit()
            print(f"[ingest] OK: {title} ({len(parts)} chunks)")

    return (doc_count, chunk_count)

# ---------- CLI ----------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Corah: ingest markdown/txt into pgvector")
    ap.add_argument("--root", type=str, default=RAW_DOCS_PATH, help="root folder of raw docs")
    ap.add_argument("--rebuild", action="store_true", help="truncate and rebuild store")
    args = ap.parse_args()

    docs, chunks = ingest_folder(args.root, rebuild=args.rebuild)
    print(f"[ingest] Done → documents={docs}, chunks={chunks}")
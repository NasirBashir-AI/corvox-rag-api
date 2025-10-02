# app/corah_ingest/db_utils.py
from __future__ import annotations
from typing import Iterable, List, Sequence, Tuple

import psycopg2
import psycopg2.extras
from psycopg2.extras import execute_values

from app.core.config import DB_URL


# -----------------------------
# Connections
# -----------------------------

def get_connection():
    """
    Return a fresh psycopg2 connection. Callers are responsible for closing it.
    We keep this simple (no pool) to remain compatible with existing call sites
    that do conn.close() in finally blocks.
    """
    # If your RDS forces SSL, you can add: sslmode="require"
    return psycopg2.connect(DB_URL)


# -----------------------------
# Helpers
# -----------------------------

def _vector_literal(vec: Sequence[float]) -> str:
    """
    Build a pgvector literal string from a Python list, e.g. "[0.1,-0.2,...]".
    We will cast this to ::vector inside SQL.
    """
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


# -----------------------------
# Schema management (optional)
# -----------------------------

def ensure_schema(cur) -> None:
    """
    Create schema/tables if they do not exist. This is optional; most of the time
    your schema will already be provisioned by migrations or a one-time SQL.
    """
    cur.execute(
        """
        CREATE SCHEMA IF NOT EXISTS corah_store;

        CREATE TABLE IF NOT EXISTS corah_store.documents (
            id          BIGSERIAL PRIMARY KEY,
            source      TEXT NOT NULL DEFAULT 'raw',
            uri         TEXT NOT NULL,
            title       TEXT,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        -- Adjust vector dimension to match your embeddings model
        CREATE TABLE IF NOT EXISTS corah_store.chunks (
            id          BIGSERIAL PRIMARY KEY,
            doc_id      BIGINT NOT NULL REFERENCES corah_store.documents(id) ON DELETE CASCADE,
            chunk_no    INT NOT NULL,
            title       TEXT,
            content     TEXT,
            embedding   VECTOR(1536),
            token_count INT DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_doc
            ON corah_store.chunks (doc_id, chunk_no);

        -- cosine distance ops
        CREATE INDEX IF NOT EXISTS idx_chunks_embed
            ON corah_store.chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

        CREATE INDEX IF NOT EXISTS idx_chunks_content_trgm
            ON corah_store.chunks USING gin (content gin_trgm_ops);
        """
    )


# -----------------------------
# Insert operations
# -----------------------------

def insert_document(cur, source: str, uri: str, title: str) -> int:
    """
    Insert a single document row and return its id.
    """
    cur.execute(
        """
        INSERT INTO corah_store.documents (source, uri, title)
        VALUES (%s, %s, %s)
        RETURNING id;
        """,
        (source, uri, title),
    )
    return int(cur.fetchone()[0])


def insert_chunks(
    cur,
    doc_id: int,
    rows: Iterable[Tuple[int, str, Sequence[float], int]],
) -> None:
    """
    Bulk insert chunk rows for a document.

    Args:
        cur: psycopg cursor
        doc_id: target document id
        rows: iterable of tuples
              (chunk_no, content, embedding_vector, token_count)

    We pass embeddings as a pgvector literal and cast to ::vector inside SQL.
    """
    # Prepare transformed rows for execute_values
    # Convert each embedding list into a vector literal string
    prepared = [
        (doc_id, chunk_no, content, _vector_literal(embedding), token_count)
        for (chunk_no, content, embedding, token_count) in rows
    ]

    # Use a VALUES template that casts the embedding param to ::vector
    template = "(%s, %s, %s, %s::vector, %s)"

    execute_values(
        cur,
        """
        INSERT INTO corah_store.chunks
            (doc_id, chunk_no, content, embedding, token_count)
        VALUES %s
        """,
        prepared,
        template=template,
        page_size=1000,
    )
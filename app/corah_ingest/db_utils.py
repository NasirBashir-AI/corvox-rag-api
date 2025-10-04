"""
app/corah_ingest/db_utils.py

Postgres helpers for ingestion:
- insert/upsert documents and chunks
- optional wipe/rebuild helpers
- lightweight facts helpers for contact/pricing

Design notes
- Uses app.core.utils.pg_cursor / rows_to_dicts to avoid duplicating DB glue.
- Keeps DDL to a minimum; safe-guards are wrapped in try/except so ingestion never
  dies on already-existing objects.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from app.core.utils import pg_cursor, rows_to_dicts


# -----------------------------
# Schema helpers (safe / optional)
# -----------------------------

def ensure_schema() -> None:
    """
    Ensure expected schema objects exist. This is defensive and safe to call;
    it won't error if objects already exist.
    """
    with pg_cursor() as cur:
        # Schema
        try:
            cur.execute("CREATE SCHEMA IF NOT EXISTS corah_store;")
        except Exception:
            pass

        # Documents
        try:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS corah_store.documents (
                    id           BIGSERIAL PRIMARY KEY,
                    title        TEXT,
                    uri          TEXT,          -- logical identifier (s3://... or slug)
                    source_uri   TEXT,          -- optional original path
                    created_at   TIMESTAMPTZ DEFAULT NOW()
                );
                """
            )
        except Exception:
            pass

        # Chunks
        try:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS corah_store.chunks (
                    id           BIGSERIAL PRIMARY KEY,
                    document_id  BIGINT REFERENCES corah_store.documents(id) ON DELETE CASCADE,
                    chunk_no     INT NOT NULL,
                    content      TEXT NOT NULL,
                    embedding    vector,        -- pgvector; dimension set at extension level
                    token_count  INT DEFAULT 0
                );
                """
            )
        except Exception:
            pass

        # Facts (structured values extracted at ingest)
        try:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS corah_store.facts (
                    id         BIGSERIAL PRIMARY KEY,
                    name       TEXT NOT NULL,     -- e.g., contact_email, contact_phone, pricing_bullet
                    value      TEXT NOT NULL,
                    uri        TEXT,              -- which doc this fact came from (best-effort)
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
                """
            )
        except Exception:
            pass

        # Helpful index for FTS (optional; safe if it already exists)
        try:
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON corah_store.chunks
                USING GIN (to_tsvector('english', content));
                """
            )
        except Exception:
            pass


# -----------------------------
# Documents
# -----------------------------

def insert_document(title: str, uri: Optional[str] = None, source_uri: Optional[str] = None) -> int:
    """
    Insert a document row and return its id.
    If a row with the same uri already exists, return that id (idempotent).
    """
    with pg_cursor() as cur:
        if uri:
            cur.execute(
                "SELECT id FROM corah_store.documents WHERE uri = %s LIMIT 1;",
                (uri,),
            )
            row = cur.fetchone()
            if row:
                return int(row[0])

        cur.execute(
            """
            INSERT INTO corah_store.documents (title, uri, source_uri)
            VALUES (%s, %s, %s)
            RETURNING id;
            """,
            (title, uri, source_uri),
        )
        return int(cur.fetchone()[0])


def upsert_document(title: str, uri: Optional[str] = None, source_uri: Optional[str] = None) -> int:
    """
    Upsert variant for convenience.
    - If uri exists, update title/source_uri if provided and return id.
    - Otherwise insert new.
    """
    with pg_cursor() as cur:
        if uri:
            cur.execute(
                "SELECT id, title, source_uri FROM corah_store.documents WHERE uri = %s LIMIT 1;",
                (uri,),
            )
            r = cur.fetchone()
            if r:
                doc_id = int(r[0])
                # Update if new values are provided
                if (title or source_uri):
                    cur.execute(
                        """
                        UPDATE corah_store.documents
                           SET title = COALESCE(%s, title),
                               source_uri = COALESCE(%s, source_uri)
                         WHERE id = %s;
                        """,
                        (title, source_uri, doc_id),
                    )
                return doc_id

        # Fallback to insert
        cur.execute(
            """
            INSERT INTO corah_store.documents (title, uri, source_uri)
            VALUES (%s, %s, %s)
            RETURNING id;
            """,
            (title, uri, source_uri),
        )
        return int(cur.fetchone()[0])


# -----------------------------
# Chunks
# -----------------------------

def insert_chunk(
    document_id: int,
    chunk_no: int,
    content: str,
    embedding: Sequence[float],
    token_count: int = 0,
) -> int:
    """
    Insert a single chunk with its embedding. Returns new chunk id.
    """
    with pg_cursor() as cur:
        cur.execute(
            """
            INSERT INTO corah_store.chunks (document_id, chunk_no, content, embedding, token_count)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id;
            """,
            (document_id, chunk_no, content, embedding, token_count),
        )
        return int(cur.fetchone()[0])


def delete_all_documents_and_chunks() -> None:
    """Dangerous but useful for a full rebuild."""
    with pg_cursor() as cur:
        # Chunks depend on documents; delete chunks first for clarity (CASCADE also handles it)
        cur.execute("DELETE FROM corah_store.chunks;")
        cur.execute("DELETE FROM corah_store.documents;")


# -----------------------------
# Facts
# -----------------------------

def insert_fact(name: str, value: str, uri: Optional[str] = None) -> int:
    """
    Simple append-only insert for a fact. Retrieval sorts by updated_at DESC,
    so the latest value naturally wins.
    """
    with pg_cursor() as cur:
        cur.execute(
            """
            INSERT INTO corah_store.facts (name, value, uri)
            VALUES (%s, %s, %s)
            RETURNING id;
            """,
            (name, value, uri),
        )
        return int(cur.fetchone()[0])


def upsert_fact(name: str, value: str, uri: Optional[str] = None) -> int:
    """
    Upsert-style: delete older identical name(s) then insert fresh.
    (We avoid assuming a unique constraint; this keeps it simple and deterministic.)
    """
    with pg_cursor() as cur:
        try:
            cur.execute("DELETE FROM corah_store.facts WHERE name = %s;", (name,))
        except Exception:
            # If delete fails for any reason, we still try to insert the new value
            pass
    return insert_fact(name=name, value=value, uri=uri)


def delete_facts_by_names(names: Sequence[str]) -> None:
    if not names:
        return
    with pg_cursor() as cur:
        cur.execute("DELETE FROM corah_store.facts WHERE name = ANY(%s);", (list(names),))


# -----------------------------
# Utilities for ingestion flows
# -----------------------------

def begin_rebuild() -> None:
    """
    Optional convenience for ingest --rebuild flows:
    ensure schema and wipe existing docs/chunks (facts left intact).
    """
    ensure_schema()
    delete_all_documents_and_chunks()
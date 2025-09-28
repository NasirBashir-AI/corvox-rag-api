# app/core/schema.py
"""
Safe, idempotent DDL helpers. You can call ensure_schema(conn)
anytime; it will create the schema/tables/indexes if they don't exist.
"""

DDL = """
CREATE SCHEMA IF NOT EXISTS corah_store;

CREATE TABLE IF NOT EXISTS corah_store.documents (
  id         BIGSERIAL PRIMARY KEY,
  source     TEXT NOT NULL,
  uri        TEXT NOT NULL,
  title      TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS corah_store.chunks (
  id          BIGSERIAL PRIMARY KEY,
  doc_id      BIGINT REFERENCES corah_store.documents(id) ON DELETE CASCADE,
  chunk_no    INT NOT NULL,
  content     TEXT NOT NULL,
  embedding   vector(1536),
  token_count INT,
  created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON corah_store.chunks(doc_id);
-- If you prefer HNSW for pgvector (supported versions):
-- CREATE INDEX IF NOT EXISTS idx_chunks_emb_hnsw ON corah_store.chunks
--   USING hnsw (embedding vector_l2_ops) WITH (m = 16, ef_construction = 128);
"""

def ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()
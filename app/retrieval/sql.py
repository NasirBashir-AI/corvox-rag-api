"""
app/retrieval/sql.py

All SQL strings for retrieval in one place.
- Vector search (pgvector, cosine distance)
- Full-Text Search (FTS) using websearch_to_tsquery
- Facts lookup for contact/pricing (optional)
Keep these queries minimal and ANSI+Postgres; do not embed business logic here.
"""

# -----------------------------
# Vector similarity (pgvector)
# -----------------------------
# Notes:
# - Uses cosine distance operator `<=>` (lower is better).
# - We return raw `distance` and normalize/convert to similarity in scoring.py.

SQL_VECTOR_SEARCH = """
SELECT
    c.id            AS chunk_id,
    d.id            AS document_id,
    c.chunk_no      AS chunk_no,
    d.title         AS title,
    COALESCE(d.source_uri, d.uri) AS source_uri,
    c.content       AS content,
    (c.embedding <=> %(query_embedding)s) AS distance
FROM corah_store.chunks c
JOIN corah_store.documents d
  ON d.id = c.document_id
ORDER BY c.embedding <=> %(query_embedding)s
LIMIT %(limit)s;
"""

# -----------------------------
# Full-Text Search (FTS)
# -----------------------------
# Notes:
# - `websearch_to_tsquery` parses natural language terms (handles quotes, -negation).
# - We compute ts_rank and sort descending (higher is better).
# - No GIN requirement at query-time, but it is strongly recommended in the DB.

SQL_FTS_SEARCH = """
WITH q AS (
  SELECT websearch_to_tsquery('english', %(q)s) AS tsq
)
SELECT
    c.id            AS chunk_id,
    d.id            AS document_id,
    c.chunk_no      AS chunk_no,
    d.title         AS title,
    COALESCE(d.source_uri, d.uri) AS source_uri,
    c.content       AS content,
    ts_rank(
      to_tsvector('english', c.content),
      (SELECT tsq FROM q)
    ) AS fts_rank
FROM corah_store.chunks c
JOIN corah_store.documents d
  ON d.id = c.document_id
WHERE to_tsvector('english', c.content) @@ (SELECT tsq FROM q)
ORDER BY fts_rank DESC
LIMIT %(limit)s;
"""

# -----------------------------
# Facts lookup (optional, recommended)
# -----------------------------
# Expected table: corah_store.facts(name text, value text, uri text, updated_at timestamptz)
# Populated during ingestion by corah_ingest/extract_facts.py

SQL_FACTS_SELECT_BY_NAMES = """
SELECT
  name,
  value,
  uri,
  updated_at
FROM corah_store.facts
WHERE name = ANY (%(names)s::text[])
ORDER BY updated_at DESC, name ASC;
"""

# Optional: simple wildcard pull (used sparingly; prefer named list)
SQL_FACTS_SELECT_LIKE = """
SELECT
    name,
    value,
    uri,
    updated_at
FROM corah_store.facts
WHERE name ILIKE %(name_like)s
ORDER BY updated_at DESC, name ASC
LIMIT %(limit)s;
"""
from psycopg2.extras import execute_values
import psycopg2
from app.core.config import DB_URL  # ← single source of truth

def get_connection():
    return psycopg2.connect(DB_URL)

def insert_document(cur, source, uri, title=None):
    cur.execute(
    "INSERT INTO corah_store.documents (source, uri, title) VALUES (%s, %s, %s) RETURNING id",
    (source, uri, title),
    )
    return cur.fetchone()[0]

def insert_chunks(cur, rows):
    execute_values(
        cur,
        """
        INSERT INTO corah_store.chunks
            (doc_id, chunk_no, content, embedding, token_count)
        VALUES %s
        """,
        rows,
    )
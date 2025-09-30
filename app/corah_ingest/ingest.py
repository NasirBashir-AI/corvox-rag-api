# ingest.py
import os
from openai import OpenAI

from app.core.config import (
    RAW_DOCS_PATH,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from app.corah_ingest.db_utils import get_connection, insert_document, insert_chunks

client = OpenAI()  # uses OPENAI_API_KEY from env

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + size)
        chunks.append(text[start:end])
        start += max(1, size - overlap)
    # drop tiny chunks
    return [c.strip() for c in chunks if len(c.strip()) >= 200]

def embed_texts(texts):
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def ingest_file(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    if not chunks:
        print(f"⚠️  Skipped (no usable chunks): {path}")
        return

    embeddings = embed_texts(chunks)
    conn = get_connection()
    cur = conn.cursor()
    doc_id = insert_document(cur, uri=path, title=os.path.basename(path))
    rows = [(doc_id, i, c, e, len(c.split())) for i, (c, e) in enumerate(zip(chunks, embeddings), start=1)]
    insert_chunks(cur, doc_id, rows)
    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Ingested {path} ({len(chunks)} chunks)")

if __name__ == "__main__":
    files = [os.path.join(RAW_DOCS_PATH, f) for f in os.listdir(RAW_DOCS_PATH) if f.endswith(".md")]
    files.sort()
    if not files:
        print(f"❌ No .md files found in {RAW_DOCS_PATH}")
    for fp in files:
        ingest_file(fp)
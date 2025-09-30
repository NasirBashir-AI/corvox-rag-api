# app/core/config.py
import os

# --- DB URL (read from env; optional local fallback) ---
DB_URL = (
    os.getenv("POSTGRES_URL")
    or os.getenv("DATABASE_URL")
)
# Optional local fallback (only used if neither env var is set)
FALLBACK_DB_URL = "postgresql://corah_user:YOUR_PASSWORD@localhost:5432/corah"

# --- Files / paths (adjust as you like for local dev) ---
# For local Mac dev you can point this to your Knowledge/ folder if you want
RAW_DOCS_PATH = os.getenv("RAW_DOCS_PATH", "/home/ec2-user/corah/data/raw")

# --- Models (central place so all code imports from here) ---
EMBEDDING_MODEL = "text-embedding-3-small"   # 1536-dim, matches pgvector(1536)
CHAT_MODEL      = "gpt-4o-mini"              # used by generator

# --- Chunking defaults (shared by ingest/retrieval) ---
CHUNK_SIZE     = 500
CHUNK_OVERLAP  = 50

# existing imports & settings...

# Feature flags
SHOW_CITATIONS = False      # <- production default: no citations in responses
DEBUG_RAG = False           # <- production default: no retrieval dump in responses
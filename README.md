# Corah – Conversational AI Assistant (Backend)

Corah is a FastAPI backend that uses OpenAI + Postgres to answer questions from your knowledge base and capture leads with a natural, sales-aware flow.

---

## Quick Start (EC2 or local)

```bash
# 1) Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Set environment (see next section), then:
# Ingest your KB (rebuild = wipe & re-embed)
python -m app.corah_ingest.ingest --db "$DB_URL" --root "$RAW_DOCS_PATH" --rebuild

# 4) Run API
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
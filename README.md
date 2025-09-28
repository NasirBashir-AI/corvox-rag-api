# Corah RAG API

Corah is a **Retrieval-Augmented Generation (RAG) chatbot framework** built by **Corvox**.  
It combines document ingestion, semantic search, and grounded answer generation into a clean API.

---

## 🚀 Features
- **Ingestion**: Load and chunk Markdown/text docs, store in PostgreSQL with pgvector.  
- **Retrieval**: Semantic search with embeddings and top-k chunk selection.  
- **Generation**: GPT-powered answers, grounded only in retrieved context.  
- **Self-query Rewriting**: LLM rewrites user queries for better recall & precision.  
- **API Layer**: Expose endpoints for chatbots or web UIs.  

---

## 📂 Project Structure

app/
api/           # FastAPI app (main.py, schemas.py)
corah_ingest/  # Data ingestion (db_utils.py, ingest.py)
generation/    # Answer generation (generator.py, prompt.py)
retrieval/     # Semantic search & retrieval (retriever.py)
requirements.txt # Python dependencies
README.md        # Project documentation

---

## 🛠️ Requirements

- Python **3.10+**
- PostgreSQL (with **pgvector** extension enabled)
- OpenAI API key (set in environment variable)

Install dependencies:

```bash
pip install -r requirements.txt

## ⚙️ Environment Setup

export POSTGRES_URL="postgresql://corah_user:YOUR_PASSWORD@localhost:5432/corah"
export OPENAI_API_KEY="sk-xxxxx"

---

## ▶️ Usage

### 1. Ingest documents
Load and store Markdown files into PostgreSQL:

```bash
python -m app.corah_ingest.ingest
# server.py — Corvox RAG API (clean)

import os
import re
from pathlib import Path
from typing import List, Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, Body
from pydantic import BaseModel
from qdrant_client import QdrantClient
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware


# ----------------- TUNING -----------------
COLLECTION   = "corvox_kb"
TOP_K        = 5     # final snippets sent to LLM
CANDIDATES   = 20    # fetched from Qdrant before rerank
THRESHOLD    = 0.0   # set >0.0 to drop weak hits early
EMB_MODEL    = "text-embedding-3-small"
CHAT_MODEL   = "gpt-4o-mini"
# ------------------------------------------

# env & clients
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
)
client = OpenAI()  # reads OPENAI_API_KEY from env

app = FastAPI(title="Corvox RAG API")

# ---------- helpers (accept ScoredPoint OR legacy tuples) ----------
def _score(p: Any) -> float:
    return (getattr(p, "score", None)
            if hasattr(p, "score") else (p[1] if isinstance(p, tuple) else 0.0)) or 0.0

def _payload(p: Any) -> Dict[str, Any]:
    return (getattr(p, "payload", None)
            if hasattr(p, "payload") else (p[2] if isinstance(p, tuple) else {})) or {}

def _embed(text: str) -> List[float]:
    return client.embeddings.create(model=EMB_MODEL, input=text).data[0].embedding

def _retrieve(vec: List[float]):
    """Fetch candidates; fall back to legacy API if needed."""
    try:
        res = qdrant.query_points(
            collection_name=COLLECTION,
            query=vec,
            limit=CANDIDATES,
            with_payload=True,
        )
        pts = list(res.points or [])
    except Exception:
        res = qdrant.search(
            collection_name=COLLECTION,
            query_vector=vec,
            limit=CANDIDATES,
            with_payload=True,
        )
        pts = list(res or [])

    if THRESHOLD > 0.0:
        pts = [p for p in pts if _score(p) >= THRESHOLD]
    return pts

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # narrow to your domain later
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)
# ---------- rerank: boost services, penalize contact/hours ----------
_SERVICE_HINTS = ("service", "services", "offer", "offers", "provide", "provides")
_CONTACT_NOISE = ("contact", "email", "phone", "mon–fri", "mon-fri", "call", "support")

def _rerank(question: str, hits):
    q_terms = set(re.findall(r"\w+", question.lower()))

    def lexical_overlap(text: str) -> float:
        t = set(re.findall(r"\w+", (text or "").lower()))
        return 0.0 if not q_terms else len(q_terms & t) / max(1, len(q_terms))

    def service_bonus(text: str) -> float:
        tl = (text or "").lower()
        return 0.35 if any(w in tl for w in _SERVICE_HINTS) else 0.0

    def contact_penalty(text: str) -> float:
        tl = (text or "").lower()
        return 0.25 if any(w in tl for w in _CONTACT_NOISE) else 0.0

    def fused_score(p):
        vec = _score(p)
        txt = _payload(p).get("text", "")
        lex = lexical_overlap(txt)
        return 0.7 * vec + 0.25 * lex + service_bonus(txt) - contact_penalty(txt)

    hits = sorted(hits, key=fused_score, reverse=True)
    return hits[:TOP_K]

# ---------- extract service bullets from text ----------
def _extract_services_from_hits(hits):
    lines = []
    for p in hits:
        text = (_payload(p).get("text") or "")
        # 1) bullet-like lines
        for line in text.splitlines():
            s = line.strip()
            if s.startswith(("-", "•", "*")):
                lines.append(s.lstrip("-•* ").strip())

        # 2) sentence candidates that look like services
        for sent in re.split(r"[.;]\s+|\n+", text):
            s = sent.strip()
            if not s:
                continue
            if re.search(r"\b(create|creates|provide|provides|include|includes|offer|offers|focus(?:es)?|services?)\b", s, re.I):
                if not re.search(r"\b(email|contact|phone|mon–fri|mon-fri|9am|5pm|support)\b", s, re.I):
                    lines.append(s)

    # de-dup keep order
    seen, out = set(), []
    for l in lines:
        k = l.lower()
        if k and k not in seen:
            seen.add(k)
            out.append(l.lstrip("-•* ").strip())
    return out[:10]

# ---------- routes ----------
class Ask(BaseModel):
    message: str

@app.get("/health")
def health():
    return {"ok": True}

_SERVICE_Q = re.compile(r"\b(what\s+services|services|offer|offers|provide|provides)\b", re.I)

@app.post("/chat")
def chat(body: Ask = Body(...)):
    vec = _embed(body.message)
    hits = _retrieve(vec)
    if not hits:
        return {"answer": "I don't know."}

    hits = _rerank(body.message, hits)

    # If asking about services: try direct extraction
    if _SERVICE_Q.search(body.message):
        items = _extract_services_from_hits(hits)
        if len(items) >= 3:
            return {"answer": "\n".join(f"- {it}" for it in items)}

        # LLM fallback: list only services
        context = "\n".join([_payload(p).get("text", "") for p in hits])
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    "From the context below, list ONLY the services Corvox provides "
                    "as 3–8 bullet points. Do not include contact info, hours, or pricing. "
                    "If services are not present, say 'I don't know.'\n\n"
                    f"Context:\n{context}\n\nAnswer:"
                )
            }],
            temperature=0
        )
        return {"answer": resp.choices[0].message.content.strip()}

    # otherwise general Q&A with reranked context
    context = "\n".join([_payload(p).get("text", "") for p in hits])
    prompt = (
        "You are a helpful business assistant. Answer using ONLY the context. "
        "If the context lacks the answer, say 'I don't know.'\n\n"
        f"Context:\n{context}\n\nQuestion: {body.message}\nAnswer:"
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return {"answer": resp.choices[0].message.content.strip()}

@app.post("/chat_debug")
def chat_debug(body: Ask = Body(...)):
    vec = _embed(body.message)
    hits = _retrieve(vec)
    hits = _rerank(body.message, hits)
    return {
        "hits": [
            {"score": float(_score(p)), "text": (_payload(p).get("text") or "")[:250]}
            for p in hits
        ]
    }
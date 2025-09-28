from __future__ import annotations
from fastapi import FastAPI, Query
from app.retrieval.retriever import search
from app.generation.generator import generate_answer
from app.api.schemas import SearchResponse, ChatRequest, ChatResponse, SearchHit
from app.api.schemas import AnswerOut

from app.api.schemas import (
    SearchResponse,
    ChatResponse,
    SearchHit,
    AnswerOut,
    ChatRequest,
    QuestionIn,   # ✅ add this
)

@app.post("/answer", response_model=AnswerOut)
def answer(req: QuestionIn):
    out = generate_answer(req.question, k=req.k)
    return AnswerOut(answer=out["answer"], citations=out.get("citations"))

app = FastAPI(title="Corah API", version="0.1.0")

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/search", response_model=SearchResponse)
def search_endpoint(q: str = Query(..., min_length=2), k: int = 5):
    hits = search(q, k=k)
    # FastAPI will coerce dicts to SearchHit automatically
    return {"query": q, "k": k, "hits": hits}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest):
    out = generate_answer(
        question=payload.question,
        k=payload.k,
        max_context_chars=payload.max_context,
    )
    # Shape matches ChatResponse (answer, rewritten_query, used)
    return out
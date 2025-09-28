# app/api/main.py
from __future__ import annotations
from fastapi import FastAPI, Query
from app.retrieval.retriever import search
from app.generation.generator import generate_answer
from app.api.schemas import QuestionIn, AnswerOut, SearchResponse, ChatRequest

app = FastAPI(title="Corah API", version="1.0.0")


@app.post("/answer", response_model=AnswerOut)
def answer(req: QuestionIn):
    out = generate_answer(req.question, k=req.k)
    return AnswerOut(answer=out["answer"])   # return ONLY answer


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/search", response_model=SearchResponse)
def search_endpoint(q: str = Query(..., min_length=2), k: int = 5):
    hits = search(q, k=k)
    return {"query": q, "k": k, "hits": hits}


@app.post("/chat", response_model=AnswerOut)
def chat_endpoint(payload: ChatRequest):
    out = generate_answer(
        question=payload.question,
        k=payload.k,
        max_context_chars=payload.max_context,
    )
    return AnswerOut(answer=out["answer"])   # return ONLY answer
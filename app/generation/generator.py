from __future__ import annotations
# app/generation/generator.py
from app.core.config import CHAT_MODEL as CHAT_MODEL
import os
from typing import Dict, Any, List, Tuple
from openai import OpenAI

from app.retrieval.retriever import search, make_context
from app.core.config import EMBEDDING_MODEL  # just to keep models centralized

client = OpenAI()  # uses OPENAI_API_KEY from env

# ---------- Self-query rewrite ----------

SELF_QUERY_SYSTEM = """You are a retrieval query optimizer.
Rewrite the user's question to maximize retrieval recall and precision.
- Expand important terms with synonyms.
- Keep it concise (<= 40 tokens).
- No answers, just a better search query.
- Never include quotes or formatting, plain text only.
"""

def self_query_rewrite(user_question: str) -> str:
    msg = [
        {"role": "system", "content": SELF_QUERY_SYSTEM},
        {"role": "user", "content": user_question.strip()},
    ]
    rsp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=msg,
        temperature=0.2,
        max_tokens=80,
    )
    return rsp.choices[0].message.content.strip()

# ---------- Answer synthesis (grounded RAG) ----------

ANSWER_SYSTEM = """You are Corah, an AI-first (marketing-second) assistant for Corvox.
Answer using ONLY the provided context. If the context is insufficient, say "I don’t have that information yet."
Style: concise, professional, friendly. No hallucinations. Cite source titles inline like (Source: <title>) when helpful.
"""

def build_prompt(context: str, question: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question.strip()}"},
    ]

def generate_answer(question: str, k: int = 5, max_context_chars: int = 3000) -> Dict[str, Any]:
    # 1) Self-query rewrite
    rewritten = self_query_rewrite(question) if question.strip() else question

    # 2) Retrieve top-k chunks using rewritten query
    hits = search(rewritten, k=k)

    # 3) Build context block
    context, used = make_context(hits, max_chars=max_context_chars)

    # 4) If nothing found, fail gracefully
    if not context.strip():
        return {
            "answer": "I don’t have that information yet.",
            "used": [],
            "rewritten_query": rewritten,
        }

    # 5) Generate grounded answer
    messages = build_prompt(context, question)
    rsp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
        max_tokens=600,
    )
    answer = rsp.choices[0].message.content.strip()

    return {
        "answer": answer,
        "used": used,               # includes title/uri/chunk_no so you can show sources
        "rewritten_query": rewritten,
    }

# ---------- CLI helper ----------

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Corah: Self-query RAG generator")
    ap.add_argument("question", type=str)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-context", type=int, default=3000)
    args = ap.parse_args()

    out = generate_answer(args.question, k=args.k, max_context_chars=args.max_context)
    print(json.dumps(out, ensure_ascii=False, indent=2))
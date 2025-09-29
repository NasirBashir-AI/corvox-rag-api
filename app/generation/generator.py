# app/generation/generator.py
from __future__ import annotations
from typing import Dict, Any, List
from openai import OpenAI

from app.retrieval.retriever import search, make_context
from app.core.config import SHOW_CITATIONS, DEBUG_RAG

client = OpenAI()  # uses OPENAI_API_KEY from env

# ---------- Self-query rewrite ----------
SELF_QUERY_SYSTEM = """You are a retrieval query optimizer.
Rewrite the user's question to maximize retrieval recall and precision.
- Expand important terms with synonyms.
- Keep it concise (<= 40 tokens).
- No answers, just a better search query.
- Plain text only.
"""

def self_query_rewrite(user_question: str) -> str:
    q = user_question.strip()
    if not q:
        return q
    rsp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SELF_QUERY_SYSTEM},
            {"role": "user", "content": q},
        ],
        temperature=0.2,
        max_tokens=80,
    )
    return rsp.choices[0].message.content.strip()

# ---------- Answer synthesis (grounded RAG) ----------

# Build system instruction dynamically (with/without citation guidance)
CITE_LINE = "" if not SHOW_CITATIONS else " Cite source titles inline like (Source: <title>) when helpful."
ANSWER_SYSTEM = (
    "You are Corah, an AI-first (marketing-second) assistant for Corvox. "
    "Answer using ONLY the provided context. If the context is insufficient, say "
    "\"I don’t have that information yet.\" "
    "Style: concise, professional, friendly. No hallucinations." + CITE_LINE
)

def build_prompt(context: str, question: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question.strip()}"},
    ]

def generate_answer(
    question: str,
    k: int = 5,
    max_context_chars: int = 3000,
    debug: bool | None = None,           # optional one-off override
    show_citations: bool | None = None,  # optional one-off override
) -> Dict[str, Any]:

    # Resolve flags (param beats config)
    debug = DEBUG_RAG if debug is None else debug
    show_citations = SHOW_CITATIONS if show_citations is None else show_citations

    # 1) self-query rewrite
    rewritten = self_query_rewrite(question)

    # 2) retrieve using the rewritten query
    hits = search(rewritten, k=k)

    # 3) context
    context, used = make_context(hits, max_chars=max_context_chars)
    if not context.strip():
        base = {
            "answer": "I don’t have that information yet.",
            "rewritten_query": rewritten,
            "used": [],
        }
        if debug:
            base["debug"] = {"rewritten_query": rewritten, "used": []}
        return base

    # 4) call LLM
    messages = build_prompt(context, question)
    rsp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
        max_tokens=600,
    )
    answer = rsp.choices[0].message.content.strip()

    # 5) construct response
    out: Dict[str, Any] = {
        "answer": answer,
        "rewritten_query": rewritten,  # <-- top-level for main.py
        "used": used,                  # <-- top-level for main.py
    }

    # optional lightweight citations list
    if show_citations:
        out["citations"] = [{"title": u["title"], "chunk_no": u["chunk_no"]} for u in used]

    # debug bundle
    if debug:
        out.setdefault("debug", {})
        out["debug"]["rewritten_query"] = rewritten
        out["debug"]["used"] = used

    return out

# ---------- CLI helper ----------
if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Corah: RAG generator")
    ap.add_argument("question", type=str)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-context", type=int, default=3000)
    ap.add_argument("--debug", action="store_true", help="include retrieval details")
    ap.add_argument("--citations", action="store_true", help="include citations list")
    args = ap.parse_args()

    out = generate_answer(
        args.question,
        k=args.k,
        max_context_chars=args.max_context,
        debug=args.debug,
        show_citations=args.citations,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
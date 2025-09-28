from __future__ import annotations
from typing import Dict, Any, List
from openai import OpenAI

from app.retrieval.retriever import search, make_context
from app.core.config import SHOW_CITATIONS, DEBUG_RAG

client = OpenAI()  # uses OPENAI_API_KEY from env

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

    # 1) self-query rewrite (keep as you already have it, or remove if not used)
    # rewritten = self_query_rewrite(question) if question.strip() else question
    rewritten = question.strip()

    # 2) retrieve
    hits = search(rewritten, k=k)

    # 3) context
    context, used = make_context(hits, max_chars=max_context_chars)
    if not context.strip():
        base = {"answer": "I don’t have that information yet."}
        if debug:
            base["debug"] = {"rewritten_query": rewritten, "used": used}
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
    out: Dict[str, Any] = {"answer": answer}

    # if you *do* want lightweight citations sometimes
    if show_citations:
        out["citations"] = [{"title": u["title"], "chunk_no": u["chunk_no"]} for u in used]

    # debug only when explicitly requested or feature-flagged
    if debug:
        out.setdefault("debug", {})
        out["debug"]["rewritten_query"] = rewritten
        out["debug"]["used"] = used  # full objects with uri/similarity, etc.

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
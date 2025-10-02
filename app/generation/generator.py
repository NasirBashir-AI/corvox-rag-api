# app/generation/generator.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
from openai import OpenAI

from app.retrieval.retriever import search, make_context
from app.core.config import SHOW_CITATIONS, DEBUG_RAG, TEMPERATURE, MIN_SIM, ENABLE_SELF_QUERY

client = OpenAI()  # uses OPENAI_API_KEY from env

# ---------- Self-query rewrite ----------

SELF_QUERY_SYSTEM = (
    "You are a retrieval query optimizer. Rewrite the user's question to improve "
    "search recall and precision.\n"
    "- Expand important terms with synonyms where useful.\n"
    "- Remove filler words.\n"
    "- Keep it concise (<= 40 tokens).\n"
    "- Output only the rewritten query, no quotes, no extra words."
)

def self_query_rewrite(user_question: str) -> str:
    if not user_question.strip():
        return user_question
    if not ENABLE_SELF_QUERY:
        return user_question.strip()

    rsp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SELF_QUERY_SYSTEM},
            {"role": "user", "content": user_question.strip()},
        ],
        temperature=0.2,
        max_tokens=80,
    )
    return rsp.choices[0].message.content.strip()

# ---------- Answer synthesis (grounded RAG) ----------

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

def _top_similarity(hits: List[Dict[str, Any]]) -> float:
    """
    Try to read a normalized similarity from retrieval hits.
    Fallbacks:
      - 'similarity' if present (0..1, higher is better)
      - derive from 'dist' if it's 0..2 cosine distance: sim = 1 - dist/2
      - last resort: use 'score' if it looks like a similarity
    """
    if not hits:
        return 0.0
    h0 = hits[0]
    if "similarity" in h0 and isinstance(h0["similarity"], (int, float)):
        return float(h0["similarity"])
    if "dist" in h0 and isinstance(h0["dist"], (int, float)):
        # assume cosine distance in [0,2]; convert to similarity-ish
        d = float(h0["dist"])
        return max(0.0, min(1.0, 1.0 - d / 2.0))
    if "score" in h0 and isinstance(h0["score"], (int, float)):
        s = float(h0["score"])
        # if score already looks like a 0..1 similarity, clamp
        if 0.0 <= s <= 1.0:
            return s
    return 0.0

def _clarifying_question(question: str, hits: List[Dict[str, Any]]) -> str:
    """
    Build a short, polite clarifying question using titles from the top few hits.
    Keeps everything inside generator.py so we don't change API shapes.
    """
    options: List[str] = []
    for h in hits[:3]:
        title = h.get("title") or h.get("doc") or ""
        title = str(title).strip()
        if title and title not in options:
            options.append(title)
    # De-duplicate and trim
    options = [o[:60] for o in options if o]
    if options:
        # Suggest up to 2–3 options
        sample = options[:3]
        bullets = " / ".join(f"“{o}”" for o in sample)
        return (
            "I want to make sure I answer precisely. "
            f"Are you asking about {bullets} — or something else?"
        )
    # Fallback generic clarifier
    return (
        "I can help with pricing, services, industries, or delivery. "
        "Could you clarify which area you mean?"
    )

def generate_answer(
    question: str,
    k: int = 5,
    max_context_chars: int = 3000,
    debug: Optional[bool] = None,
    show_citations: Optional[bool] = None,
) -> Dict[str, Any]:
    # Resolve flags (param beats config)
    debug = DEBUG_RAG if debug is None else debug
    show_citations = SHOW_CITATIONS if show_citations is None else show_citations

    # 1) Self-query rewrite (enabled by config)
    rewritten = self_query_rewrite(question)

    # 2) Retrieve
    hits = search(rewritten, k=k)

    # 3) Similarity gate — ask for clarification if match looks weak
    top_sim = _top_similarity(hits)
    if top_sim < MIN_SIM:
        clarifier = _clarifying_question(question, hits)
        out: Dict[str, Any] = {"answer": clarifier}
        if debug:
            out["debug"] = {
                "reason": "low_similarity",
                "top_similarity": top_sim,
                "threshold": MIN_SIM,
                "rewritten_query": rewritten,
                "used": hits[:3],
            }
        return out

    # 4) Build context
    context, used = make_context(hits, max_chars=max_context_chars)
    if not context.strip():
        base: Dict[str, Any] = {"answer": "I don’t have that information yet."}
        if debug:
            base["debug"] = {"rewritten_query": rewritten, "used": used}
        return base

    # 5) Call LLM for grounded answer
    messages = build_prompt(context, question)
    rsp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=TEMPERATURE,   # now controlled centrally
        max_tokens=600,
    )
    answer = rsp.choices[0].message.content.strip()

    # 6) Construct response
    out: Dict[str, Any] = {"answer": answer}
    if show_citations:
        out["citations"] = [{"title": u["title"], "chunk_no": u["chunk_no"]} for u in used if "title" in u]

    if debug:
        out.setdefault("debug", {})
        out["debug"]["rewritten_query"] = rewritten
        out["debug"]["used"] = used
        out["debug"]["top_similarity"] = top_sim

    return out

# ---------- CLI helper ----------
if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Corah: RAG generator with self-query + similarity gate")
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
"""
app/generation/generator.py

Answer generation for Corah.
- Optional self-query rewrite
- Hybrid retrieval via retriever.search
- "Never deflect": answer even on low similarity (but steer the model gently)
- Concise, grounded style (1–2 sentences or up to 3 bullets)
- Optional citations & debug

Lead summary helpers (if used in your flows) remain here as well.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple, Optional

from openai import OpenAI

from app.core.config import (
    TEMPERATURE,
    MAX_TOKENS,
    MIN_SIM,
    ENABLE_SELF_QUERY,
    DEBUG_RAG,
    SHOW_CITATIONS,
)
from app.core.utils import strip_source_tokens
from app.retrieval.retriever import search, make_context, top_similarity


_client = OpenAI()  # reads OPENAI_API_KEY from env


# -----------------------------
# Query rewrite (optional)
# -----------------------------

def self_query_rewrite(question: str) -> str:
    """
    Rewrite a conversational question into a crisp search query.
    If disabled or anything fails, return the original question.
    """
    if not ENABLE_SELF_QUERY:
        return question
    try:
        msg = [
            {
                "role": "system",
                "content": (
                    "Rewrite the user message into a concise search query for a company knowledge base. "
                    "Keep key nouns; remove chit-chat; no quotes; max ~12 words."
                ),
            },
            {"role": "user", "content": question},
        ]
        rsp = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=msg,
            temperature=0.2,
            max_tokens=64,
        )
        out = (rsp.choices[0].message.content or "").strip()
        return out or question
    except Exception:
        return question


# -----------------------------
# Prompt building
# -----------------------------

def _system_style() -> str:
    return (
        "You are Corah, an AI assistant for Corvox. "
        "Answer ONLY using the provided context. "
        "Be concise and helpful: prefer 1–2 sentences or up to 3 short bullets. "
        "Do not mention file names, paths, or where the info came from. "
        "If context seems only loosely related, still give the best helpful answer—be clear and neutral."
    )


def build_prompt(context: str, question: str) -> List[Dict[str, str]]:
    user = (
        "Context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Answer succinctly and professionally."
    )
    return [
        {"role": "system", "content": _system_style()},
        {"role": "user", "content": user},
    ]


# -----------------------------
# Main API
# -----------------------------

def generate_answer(
    question: str,
    k: int = 5,
    max_context_chars: int = 3000,
    debug: Optional[bool] = None,
    show_citations: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    End-to-end answer:
      - optional self-query
      - retrieve (hybrid)
      - build context
      - call LLM with concise style
      - never deflect; provide an answer even if similarity is low
    """
    # Resolve flags (param beats config)
    debug = DEBUG_RAG if debug is None else debug
    show_citations = SHOW_CITATIONS if show_citations is None else show_citations

    # 1) Self-query rewrite (optional)
    rewritten = self_query_rewrite(question)

    # 2) Retrieve
    hits = search(rewritten, k=k)

    # 3) Similarity check (diagnostic only — we still answer)
    top_sim = top_similarity(hits)
    low_confidence = top_sim < MIN_SIM

    # 4) Build context (if empty, still provide a graceful generic answer)
    context, used = make_context(hits, max_chars=max_context_chars)
    if not context.strip():
        base: Dict[str, Any] = {
            "answer": "Here’s a quick overview: Corvox provides AI agents and assistants to help businesses handle enquiries, capture leads, and automate routine tasks.",
        }
        if debug:
            base["debug"] = {"rewritten_query": rewritten, "used": used, "top_similarity": top_sim}
        return base

    # If similarity was low, prepend gentle instruction for synthesis
    if low_confidence:
        context = (
            "The following materials may be loosely related; synthesize the best clear answer.\n\n"
            + context
        )

    # 5) Call LLM for grounded answer
    messages = build_prompt(context, question)
    rsp = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    answer_raw = (rsp.choices[0].message.content or "").strip()
    answer = strip_source_tokens(answer_raw)

    # 6) Construct response
    out: Dict[str, Any] = {"answer": answer}

    if show_citations:
        out["citations"] = [
            {"title": u.get("title"), "chunk_no": u.get("chunk_no")} for u in used if u.get("title")
        ]

    if debug:
        out.setdefault("debug", {})
        out["debug"]["rewritten_query"] = rewritten
        out["debug"]["used"] = used
        out["debug"]["top_similarity"] = top_sim
        out["debug"]["low_confidence"] = low_confidence

    return out
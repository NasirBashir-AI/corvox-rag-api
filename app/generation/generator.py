# app/generation/generator.py
from __future__ import annotations

import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from app.core.utils import normalize_ws
from app.retrieval.retriever import search

# ----------------------------
# Models & Temperatures
# ----------------------------
_PLANNER_MODEL = os.getenv("OPENAI_PLANNER_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
_FINAL_MODEL   = os.getenv("OPENAI_FINAL_MODEL",   os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

_LEGACY_TEMP   = float(os.getenv("TEMPERATURE", "0.5"))
_PLANNER_TEMP  = float(os.getenv("PLANNER_TEMPERATURE", os.getenv("PLANNER_TEMP", "0.3")))
_FINAL_TEMP    = float(os.getenv("FINAL_TEMPERATURE",   os.getenv("FINAL_TEMP",   str(_LEGACY_TEMP))))

_client = OpenAI()
client = _client  # exported compatibility

# ----------------------------
# Context markers & regex
# ----------------------------
_CTX_START = "[Context]"
_CTX_END   = "[End Context]"

# Terms that should force retrieval
_FACT_QUERIES = [
    "address", "where are you based", "where are you located", "location",
    "services", "what services", "pricing", "price", "cost", "how much",
    "contact", "email", "phone", "number", "website", "url", "multi-agent", "multi agents"
]

# ----------------------------
# Helpers: context parsing
# ----------------------------
def _split_user_and_ctx(q: str) -> Tuple[str, str]:
    if _CTX_START in q and _CTX_END in q:
        head, rest = q.split(_CTX_START, 1)
        ctx, _ = rest.split(_CTX_END, 1)
        return head.strip(), ctx.strip()
    return q.strip(), ""

def _extract_section(ctx_block: str, header: str) -> str:
    if not ctx_block:
        return ""
    start_pat = re.compile(rf"^\s*-\s*{re.escape(header)}\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE)
    m = start_pat.search(ctx_block)
    if not m:
        return ""
    start_idx = m.end(0)
    tail = ctx_block[start_idx:]
    nxt = re.search(r"^\s*-\s*[A-Za-z].*?:", tail, re.MULTILINE)
    chunk = tail[:nxt.start()] if nxt else tail
    text = (m.group(1) + "\n" + chunk).strip()
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()

def _extract_intent(ctx_block: str) -> Tuple[str, Optional[str]]:
    if not ctx_block:
        return "other", None
    m_kind = re.search(r"^\s*kind\s*:\s*([A-Za-z_]+)\s*$", ctx_block, re.IGNORECASE | re.MULTILINE)
    m_topic = re.search(r"^\s*topic\s*:\s*([A-Za-z_]+)\s*$", ctx_block, re.IGNORECASE | re.MULTILINE)
    kind = (m_kind.group(1).strip().lower() if m_kind else "other")
    topic = (m_topic.group(1).strip().lower() if m_topic else None)
    if topic in ("none", ""):
        topic = None
    return kind, topic

def _chat(model: str, system: str, user: str, temperature: float) -> str:
    resp = _client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return (resp.choices[0].message.content or "").strip()

# ----------------------------
# Planner (tiny / defensive)
# ----------------------------
def _planner(user_text: str, intent_kind: str, intent_topic: Optional[str]) -> Dict[str, Any]:
    needs = intent_kind in ("info", "contact") or (intent_kind == "info" and intent_topic in ("pricing", "services"))
    system = (
        "Return ONLY JSON with keys: {needs_retrieval:boolean, search_query:string|null}\n"
        "- If the user is asking for company facts (address/location/services/pricing/contact), needs_retrieval=true.\n"
        "- Otherwise false. search_query should usually be the user text."
    )
    user = f"User text:\n{user_text}\n\nHeuristic needs_retrieval (pre): {str(needs).lower()}"
    raw = _chat(_PLANNER_MODEL, system, user, temperature=_PLANNER_TEMP)
    out = {"needs_retrieval": needs, "search_query": user_text}
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            out.update({
                "needs_retrieval": bool(obj.get("needs_retrieval", needs)),
                "search_query": (obj.get("search_query") or user_text)
            })
    except Exception:
        pass
    return out

# ----------------------------
# Final answer composer
# ----------------------------
def _final_answer(
    model: str,
    user_text: str,
    retrieved_snippets: str,
    summary: str,
    current_topic: str,
    recent_turns: str,
    user_details: str,
    contact_ctx: str,
    pricing_ctx: str,
    intent_kind: str,
    intent_topic: Optional[str],
    phase: str,                 # 'early' | 'engaged'
) -> str:
    # Phase control (answer-only until intent/engagement)
    ask_policy = "Do not ask follow-up questions in EARLY phase unless the user explicitly asks to book/schedule/quote."
    if phase != "early":
        ask_policy = "You may ask at most one short, helpful follow-up if it clearly advances the user’s goal."

    system = (
        "You are Corah, Corvox’s professional front-desk assistant—confident, concise, factual, warm.\n"
        "Rules:\n"
        f"- {ask_policy}\n"
        "- Always answer the user's question first using [Retrieved] when available.\n"
        "- Never invent company facts (address/location/services/pricing/contact). "
        "  If [Retrieved]/[Company contact]/[Pricing] lacks it, say you don’t have it in view and offer a next step.\n"
        "- If a public contact detail (email, office address, website URL) appears in [Retrieved] or [Company contact], you may share it verbatim.\n"
        "- Use the user’s name once if present in [User details]. Avoid repeating email/office details in consecutive turns unless asked again.\n"
        "- For capabilities (multi-agent systems etc.), answer confidently and concretely: we can build multi-agent systems tailored to operations.\n"
        "- Tone: professional and friendly; no filler. 1–3 sentences.\n"
        "- If the question is unrelated to Corvox/AI, redirect politely to services/pricing.\n"
    )

    user = (
        f"[Intent] kind={intent_kind or 'other'}; topic={intent_topic or 'None'}; phase={phase}\n\n"
        f"User: {user_text}\n\n"
        f"[Summary]\n{summary or 'None'}\n\n"
        f"[Current topic]\n{current_topic or 'None'}\n\n"
        f"[Recent turns]\n{recent_turns or 'None'}\n\n"
        f"[User details]\n{user_details or 'None'}\n\n"
        f"[Company contact]\n{contact_ctx or 'None'}\n\n"
        f"[Pricing]\n{pricing_ctx or 'None'}\n\n"
        f"[Retrieved]\n{retrieved_snippets or 'None'}\n\n"
        "Now reply as Corah following the rules."
    )
    return _chat(model, system, user, temperature=_FINAL_TEMP)

# ----------------------------
# Public: generate_answer
# ----------------------------
def generate_answer(
    question: str,
    k: int = 5,
    max_context_chars: int = 3000,
    debug: Optional[bool] = False,
    show_citations: Optional[bool] = False,
) -> Dict[str, Any]:
    """
    Sandwich:
      1) Use [Intent] + phase from context to guide retrieval and follow-up policy.
      2) Retrieval (bounded).
      3) Final compose (help-first, KB-first).
    """
    user_text, ctx_block = _split_user_and_ctx(question)

    # Sections provided by orchestrator
    summary       = _extract_section(ctx_block, "Summary")
    current_topic = _extract_section(ctx_block, "Current topic")
    recent_turns  = _extract_section(ctx_block, "Recent turns")
    user_details  = _extract_section(ctx_block, "User details")
    contact_ctx   = _extract_section(ctx_block, "Company contact")
    pricing_ctx   = _extract_section(ctx_block, "Pricing")
    intent_kind, intent_topic = _extract_intent(ctx_block)

    # Phase hint
    phase_m = re.search(r"^\s*phase\s*:\s*(early|engaged)\s*$", ctx_block, re.IGNORECASE | re.MULTILINE)
    phase = (phase_m.group(1).lower() if phase_m else "early")

    # Planner (light) + heuristics
    plan = _planner(user_text, intent_kind=intent_kind, intent_topic=intent_topic)
    needs_retrieval = bool(plan.get("needs_retrieval", False))
    search_query = (plan.get("search_query") or user_text).strip()

    # Force retrieval if user_text obviously asks for a fact
    low = user_text.lower()
    if any(term in low for term in _FACT_QUERIES):
        needs_retrieval = True

    # Retrieval
    hits: List[Dict[str, Any]] = []
    retrieved_snippets = ""
    if needs_retrieval:
        try:
            raw_hits = search(search_query, k=k)
            pieces: List[str] = []
            total = 0
            for h in raw_hits:
                snippet = normalize_ws(h.get("content", "")).strip()
                title = (h.get("title") or "")[:120]
                if not snippet:
                    continue
                one = f"[{title}] {snippet}"
                if total + len(one) > max_context_chars:
                    break
                pieces.append(one); total += len(one)
            hits = raw_hits
            retrieved_snippets = "\n\n".join(pieces)
        except Exception:
            hits, retrieved_snippets = [], ""

    # Final answer
    answer = _final_answer(
        model=_FINAL_MODEL,
        user_text=user_text,
        retrieved_snippets=retrieved_snippets,
        summary=summary,
        current_topic=current_topic,
        recent_turns=recent_turns,
        user_details=user_details,
        contact_ctx=contact_ctx,
        pricing_ctx=pricing_ctx,
        intent_kind=intent_kind,
        intent_topic=intent_topic,
        phase=phase,
    ).strip()

    # Citations
    citations: List[Dict[str, Any]] = []
    if show_citations and hits:
        seen = set()
        for h in hits:
            key = (h.get("title"), h.get("source_uri"))
            if key in seen:
                continue
            seen.add(key)
            citations.append({
                "title": h.get("title"),
                "source_uri": h.get("source_uri"),
                "document_id": h.get("document_id"),
                "chunk_no": h.get("chunk_no"),
            })

    dbg = {"intent": {"kind": intent_kind, "topic": intent_topic, "phase": phase},
           "needs_retrieval": needs_retrieval,
           "num_hits": len(hits)} if debug else None

    return {"answer": answer, "citations": citations or None, "debug": dbg}
# app/generation/generator.py
from __future__ import annotations

import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from app.core.utils import normalize_ws
from app.retrieval.retriever import search

# -----------------------------
# Config
# -----------------------------
_PLANNER_MODEL = os.getenv("OPENAI_PLANNER_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
_FINAL_MODEL   = os.getenv("OPENAI_FINAL_MODEL",   os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
_TEMPERATURE   = float(os.getenv("TEMPERATURE", "0.5"))  # friendlier, still tidy

_client = OpenAI()

# -----------------------------
# Context parsing helpers
# -----------------------------
_CTX_START = "[Context]"
_CTX_END   = "[End Context]"

_LEAD_HINT_RE   = re.compile(r"^Lead hint:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_LAST_ASKED_RE  = re.compile(r"^last_asked:\s*([A-Za-z_]+)\s*$", re.IGNORECASE | re.MULTILINE)
_CUR_TOPIC_RE   = re.compile(r"^current_topic:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_EARLIER_RE     = re.compile(r"^Earlier mention:\s*(.+)$", re.IGNORECASE | re.MULTILINE)

def _split_user_and_ctx(q: str) -> Tuple[str, str]:
    """Split original user text and the injected [Context] block (if present)."""
    if _CTX_START in q and _CTX_END in q:
        head, rest = q.split(_CTX_START, 1)
        ctx, _ = rest.split(_CTX_END, 1)
        return head.strip(), ctx.strip()
    return q.strip(), ""

def _extract_section(ctx_block: str, header: str) -> str:
    """
    Pull the lines after “- <header>:” until the next “- <other>:” or end.
    """
    if not ctx_block:
        return ""
    pat = re.compile(rf"-\s*{re.escape(header)}\s*:\s*(.*)", re.IGNORECASE)
    m = pat.search(ctx_block)
    if not m:
        return ""
    start = m.end(0)
    tail = ctx_block[start:]
    nxt = re.search(r"^\s*-\s*[A-Za-z].*?:", tail, re.MULTILINE)
    chunk = tail[: nxt.start()] if nxt else tail
    text = (m.group(1) + "\n" + chunk).strip()
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()

def _extract_ctx_fields(ctx: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not ctx:
        return out
    m = _LEAD_HINT_RE.search(ctx)
    if m: out["lead_hint"] = m.group(1).strip()
    m = _LAST_ASKED_RE.search(ctx)
    if m: out["last_asked"] = m.group(1).strip().lower()
    m = _CUR_TOPIC_RE.search(ctx)
    if m: out["current_topic"] = m.group(1).strip()
    m = _EARLIER_RE.search(ctx)
    if m: out["earlier_mention"] = m.group(1).strip()
    return out

# -----------------------------
# LLM calls
# -----------------------------
def _chat(model: str, system: str, user: str, temperature: float = _TEMPERATURE) -> str:
    resp = _client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return (resp.choices[0].message.content or "").strip()

def _planner(user_text: str, lead_hint: str = "") -> Dict[str, Any]:
    """
    First pass: lightweight route/plan.
    We DO NOT answer here; return compact JSON only.
    """
    system = (
        "You are a classifier/planner for Corah. "
        "Return ONLY JSON on one line with fields:\n"
        "{kind: 'smalltalk'|'lead'|'contact'|'pricing'|'qa'|'other', "
        " needs_retrieval: boolean, "
        " search_query: string|null, "
        " lead_prompt: string|null}\n"
        "Heuristics:\n"
        "- Use 'lead' if user asks to start/arrange a callback or provides contact.\n"
        "- Use 'contact' for explicit company contact requests.\n"
        "- Use 'pricing' for cost/plans.\n"
        "- Use 'qa' for knowledge that likely needs the KB.\n"
        "- If in doubt, set needs_retrieval=true and pass a trimmed search_query."
    )
    user = f"User text:\n{user_text}\n\nLead hint (optional): {lead_hint or 'none'}"
    raw = _chat(_PLANNER_MODEL, system, user, temperature=0.0)

    # Robust JSON parse
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # Safe default
    return {"kind": "qa", "needs_retrieval": True, "search_query": user_text, "lead_prompt": None}

# -----------------------------
# Final answer composition
# -----------------------------
def _final_answer(
    model: str,
    user_text: str,
    retrieved_snippets: str,
    summary_ctx: str,
    user_details: str,
    contact_ctx: str,
    pricing_ctx: str,
    lead_hint: str,
    last_asked: str,
    current_topic: str,
    earlier_mention: str,
) -> str:
    """
    Second pass: the actual user-visible reply.
    The controller (capture.py) owns flow; we only phrase the next step.
    """
    system = (
        "You are **Corah**, Corvox’s warm front-desk assistant — polite, patient, and genuinely helpful.\n"
        "Speak naturally like a professional receptionist who enjoys helping people.\n\n"
        "Core style (follow strictly):\n"
        "• Friendly and concise (1–3 sentences by default). Vary openers; avoid repeating the user’s words.\n"
        "• Ask at most ONE short question in a turn. No stacked CTAs.\n"
        "• Respect refusals: if the user declines to share contact, keep helping and (at most once) offer the email from [Company contact] as an alternative.\n"
        "• Use [Summary] and any [Earlier mention] to maintain continuity and resolve references.\n"
        "• Use current_topic to stay on the same thread when the user says “tell me more / yes / go on”.\n"
        "• Use ONLY [Company contact] for Corvox’s contact details; NEVER treat [User details] as company contact.\n"
        "• When the user asks about their own saved details, read from [User details] plainly.\n"
        "• Never restart or advance the lead flow yourself; obey the hint from the controller.\n"
        "• Do not start every message with a greeting or the user’s name; use name only once when first captured and when natural.\n\n"
        "Lead hint policy:\n"
        "• If hint=ask_name/contact/time/notes — ask exactly one short, friendly question to gather that field.\n"
        "• If hint=bridge_back_to_<field> — first answer the user’s message briefly, then gently steer back to that single field without repeating the exact prior wording.\n"
        "• If hint=confirm_done — give a concise confirmation summarizing Name/Phone/Email/Preferred time and a warm sign-off (no further asks).\n"
        "• If no hint is present — just answer helpfully using [Summary]/current_topic.\n\n"
        "Few-shot style mini-examples (follow tone, not content):\n"
        "1) (bridge_back_to_time) User: “What tech do you use?” → “We build on WhatsApp Business + a secure AI backend. When’s a good time for a quick call?”\n"
        "2) (ask_contact) After name captured → “Thanks, Ayesha — what’s the best number or email to reach you?”\n"
        "3) (confirm_done) → “All set — I’ve logged your callback for Tue 3pm. We’ll call +44 7123… Anything else before I close this chat?”\n"
    )

    # Compose the user message with all sections
    user = (
        f"User: {user_text}\n\n"
        f"[Summary]\n{summary_ctx or 'None'}\n\n"
        f"[User details]\n{user_details or 'None'}\n\n"
        f"[Company contact]\n{contact_ctx or 'None'}\n\n"
        f"[Pricing]\n{pricing_ctx or 'None'}\n\n"
        f"[Retrieved]\n{retrieved_snippets or 'None'}\n\n"
        f"[Lead hint]\n{lead_hint or 'None'}\n"
        f"[last_asked]\n{last_asked or 'None'}\n"
        f"[current_topic]\n{current_topic or 'None'}\n"
        f"[Earlier mention]\n{earlier_mention or 'None'}\n\n"
        "Now reply as Corah, following the rules above."
    )
    return _chat(model, system, user, temperature=_TEMPERATURE)

# -----------------------------
# Public API (called by app/api/main.py)
# -----------------------------
def generate_answer(
    question: str,
    k: int = 5,
    max_context_chars: int = 3000,
    debug: Optional[bool] = False,
    show_citations: Optional[bool] = False,
) -> Dict[str, Any]:
    """
    Two-pass LLM pipeline:
      1) Planner LLM: classify + (optional) search query rewrite
      2) Retrieval (if needed)
      3) Final LLM: compose the answer with facts + evidence + controller hints
    """
    # 0) Split augmented question: user text + our [Context] block
    user_text, ctx_block = _split_user_and_ctx(question)

    # Structured sections
    summary_ctx   = _extract_section(ctx_block, "Summary")
    user_details  = _extract_section(ctx_block, "User details")
    contact_ctx   = _extract_section(ctx_block, "Company contact")
    pricing_ctx   = _extract_section(ctx_block, "Pricing")

    # Inline fields (hints & helpers)
    fields = _extract_ctx_fields(ctx_block)
    lead_hint       = fields.get("lead_hint", "")
    last_asked      = fields.get("last_asked", "")
    current_topic   = fields.get("current_topic", "")
    earlier_mention = fields.get("earlier_mention", "")

    # 1) Planner
    plan = _planner(user_text, lead_hint=lead_hint)
    needs_retrieval = bool(plan.get("needs_retrieval", True))
    search_query = (plan.get("search_query") or user_text).strip()

    # 2) Retrieval (compact snippets)
    hits: List[Dict[str, Any]] = []
    retrieved_snippets = ""
    if needs_retrieval:
        try:
            hits = search(search_query, k=k)
            pieces: List[str] = []
            total = 0
            for h in hits:
                snippet = normalize_ws(h.get("content", "")).strip()
                title = (h.get("title") or "")[:120]
                if not snippet:
                    continue
                one = f"[{title}] {snippet}"
                if total + len(one) > max_context_chars:
                    break
                pieces.append(one)
                total += len(one)
            retrieved_snippets = "\n\n".join(pieces)
        except Exception:
            hits, retrieved_snippets = [], ""

    # 3) Final LLM
    answer = _final_answer(
        model=_FINAL_MODEL,
        user_text=user_text,
        retrieved_snippets=retrieved_snippets,
        summary_ctx=summary_ctx,
        user_details=user_details,
        contact_ctx=contact_ctx,
        pricing_ctx=pricing_ctx,
        lead_hint=lead_hint,
        last_asked=last_asked,
        current_topic=current_topic,
        earlier_mention=earlier_mention,
    ).strip()

    # Citations: surface doc titles/URIs when asked
    citations: List[Dict[str, Any]] = []
    if show_citations and hits:
        seen = set()
        for h in hits:
            key = (h.get("title"), h.get("source_uri"))
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                {
                    "title": h.get("title"),
                    "source_uri": h.get("source_uri"),
                    "document_id": h.get("document_id"),
                    "chunk_no": h.get("chunk_no"),
                }
            )

    dbg = None
    if debug:
        dbg = {
            "planner": plan,
            "used_search_query": search_query if needs_retrieval else None,
            "num_hits": len(hits),
            "lead_hint": lead_hint,
            "last_asked": last_asked,
            "current_topic": current_topic,
        }

    return {"answer": answer, "citations": citations or None, "debug": dbg}
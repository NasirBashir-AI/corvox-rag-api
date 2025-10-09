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
_TEMPERATURE   = float(os.getenv("TEMPERATURE", "0.2"))  # concise/controlled answers

_client = OpenAI()

# -----------------------------
# Context parsing helpers
# -----------------------------

_CTX_START = "[Context]"
_CTX_END   = "[End Context]"
_LEAD_HINT_RE = re.compile(r"^Lead hint:\s*(.+)$", re.IGNORECASE | re.MULTILINE)

def _split_user_and_ctx(q: str) -> Tuple[str, str]:
    """Split original user text and the injected [Context] block (if present)."""
    if _CTX_START in q and _CTX_END in q:
        head, rest = q.split(_CTX_START, 1)
        ctx, _ = rest.split(_CTX_END, 1)
        return head.strip(), ctx.strip()
    return q.strip(), ""

def _extract_line(ctx_block: str, key: str) -> str:
    if not ctx_block:
        return ""
    m = re.search(rf"{re.escape(key)}\s*:\s*(.+)", ctx_block, re.IGNORECASE)
    return m.group(1).strip() if m else ""

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

def _extract_ctx_sections(ctx: str) -> Dict[str, str]:
    """
    Extract lead hint (ask=name / bridge_back_to_time / etc.) and last_asked.
    """
    out: Dict[str, str] = {}
    if not ctx:
        out["lead_hint"] = ""
        out["last_asked"] = ""
        return out
    m = _LEAD_HINT_RE.search(ctx)
    out["lead_hint"] = m.group(1).strip() if m else ""
    out["last_asked"] = _extract_line(ctx, "Lead last asked")
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
    First pass: classify & optionally rewrite search query.
    """
    system = (
        "You are a classifier/planner for a business assistant. "
        "Return ONLY compact JSON with fields:\n"
        "{kind: 'smalltalk'|'lead'|'contact'|'pricing'|'qa'|'other', "
        " needs_retrieval: boolean, "
        " search_query: string|null, "
        " lead_prompt: string|null}\n"
        "Rules:\n"
        "- 'smalltalk' for greetings/thanks.\n"
        "- 'lead' if the user wants to start, requests a callback, gives phone/email, or asks how to start.\n"
        "- 'contact' if explicitly asking for email/phone/address/website.\n"
        "- 'pricing' for cost/fees/plans.\n"
        "- 'qa' for questions that likely require the knowledge base.\n"
        "- If kind in {'qa','pricing','contact'} and the KB would help, set needs_retrieval=true and produce a short search_query.\n"
        "- Keep JSON a single line, no commentary."
    )
    user = f"User text:\n{user_text}\n\nLead hint (optional): {lead_hint or 'none'}"
    raw = _chat(_PLANNER_MODEL, system, user, temperature=0.0)

    # Parse tolerant JSON
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
    return {"kind": "qa", "needs_retrieval": True, "search_query": user_text, "lead_prompt": None}

# -----------------------------
# Second pass: final wording (LLM = words only)
# -----------------------------

def _final_answer(
    model: str,
    user_text: str,
    retrieved_snippets: str,
    user_details: str,
    contact_ctx: str,
    pricing_ctx: str,
    lead_hint: str,
    last_asked: str,
) -> str:
    """
    Compose the reply in a natural tone using the hint and last_asked to
    avoid repetition and gently steer the flow.
    """
    system = (
        "You are Corah, a professional, friendly assistant for Corvox.\n"
        "Follow these rules strictly:\n"
        "1) Ask at most ONE short question (<= 20 words) per turn when a lead step is needed.\n"
        "2) If [Lead hint] is 'ask=<field>' and [Lead last asked] equals <field>, do NOT repeat the same question.\n"
        "   Instead: briefly address what the user just said (1 sentence), then add ONE gentle bridge back to that field.\n"
        "3) Use [Company contact] ONLY for Corvox contact details. "
        "   NEVER treat values from [User details] as company contact. If company info is missing, say so briefly.\n"
        "4) If the user asks about their own name/phone/email, read it from [User details] and answer plainly.\n"
        "5) Match the user's tone; be concise; no stacked CTAs; never restart or change the lead stage yourself.\n"
        "6) Vary phrasing naturally (avoid boilerplate). Do not start every turn with 'Hi/Hello' or repeat the user's name every turn.\n"
    )

    # Give the model clear, structured control inputs
    user = (
        f"User: {user_text}\n\n"
        f"[User details]\n{user_details or 'None'}\n\n"
        f"[Company contact]\n{contact_ctx or 'None'}\n\n"
        f"[Pricing]\n{pricing_ctx or 'None'}\n\n"
        f"[Retrieved]\n{retrieved_snippets or 'None'}\n\n"
        f"[Lead control]\n"
        f"hint: {lead_hint or 'none'}\n"
        f"last_asked: {last_asked or '-'}\n\n"
        "Now respond as Corah:\n"
        "- If hint starts with 'ask=', ask exactly ONE short question that advances that field.\n"
        "- If hint starts with 'bridge_back_to_', first answer the user helpfully in one sentence, then add a single gentle bridge to that field.\n"
        "- Otherwise, answer the user normally, using Retrieved and Company contact when relevant."
    )
    return _chat(model, system, user, temperature=_TEMPERATURE)

# -----------------------------
# Public API (planner -> retrieval -> final)
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
      3) Final LLM: compose the answer with facts + evidence
    """
    # 0) Split question into user text + [Context]
    user_text, ctx_block = _split_user_and_ctx(question)
    ctx = _extract_ctx_sections(ctx_block)

    # 1) Pull structured sections for the final pass
    user_details  = _extract_section(ctx_block, "User details")
    contact_ctx   = _extract_section(ctx_block, "Company contact")
    pricing_ctx   = _extract_section(ctx_block, "Pricing")
    lead_hint     = ctx.get("lead_hint", "")
    last_asked    = ctx.get("last_asked", "") or _extract_line(ctx_block, "Lead last asked")

    # 2) Planner
    plan = _planner(user_text, lead_hint=lead_hint)
    needs_retrieval = bool(plan.get("needs_retrieval", True))
    search_query = (plan.get("search_query") or user_text).strip()

    # 3) Retrieval (if needed)
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

    # 4) Final wording
    answer = _final_answer(
        model=_FINAL_MODEL,
        user_text=user_text,
        retrieved_snippets=retrieved_snippets,
        user_details=user_details,
        contact_ctx=contact_ctx,
        pricing_ctx=pricing_ctx,
        lead_hint=lead_hint,
        last_asked=last_asked or "",
    ).strip()

    # 5) Optional citations payload
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
            "last_asked": last_asked or "",
        }

    return {"answer": answer, "citations": citations or None, "debug": dbg}
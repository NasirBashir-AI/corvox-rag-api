# app/generation/generator.py
from __future__ import annotations

import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from app.core.utils import truncate, normalize_ws
from app.retrieval.retriever import search

# -----------------------------
# Config
# -----------------------------

_PLANNER_MODEL = os.getenv("OPENAI_PLANNER_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
_FINAL_MODEL   = os.getenv("OPENAI_FINAL_MODEL",   os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
_TEMPERATURE   = float(os.getenv("TEMPERATURE", "0.2"))  # keep answers crisp/controlled

_client = OpenAI()


# -----------------------------
# Helpers: parse the augmented prompt we get from main.py
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

def _extract_ctx_sections(ctx: str) -> Dict[str, str]:
    """
    Very light parser: we expect lines like:
      - Company contact:\n...
      - Pricing:\n...
      - Lead state: name=..., phone=..., email=..., preferred_time=...
      Lead hint: ...
    We’ll just pass these strings forward; the LLM can read them.
    """
    out: Dict[str, str] = {}
    out["raw"] = ctx or ""
    # Lead hint (single line) if present
    m = _LEAD_HINT_RE.search(ctx or "")
    out["lead_hint"] = m.group(1).strip() if m else ""
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
    First-pass LLM “planner”: classify and (optionally) rewrite a search query.
    Return JSON with keys:
      - kind: 'smalltalk' | 'lead' | 'contact' | 'pricing' | 'qa' | 'other'
      - needs_retrieval: bool
      - search_query: string or null
      - lead_prompt: string or null   (e.g., the next question to ask)
    """
    system = (
        "You are a classifier/planner for a business assistant. "
        "Do NOT answer the question. Return ONLY compact JSON with fields:\n"
        "{kind: 'smalltalk'|'lead'|'contact'|'pricing'|'qa'|'other', "
        " needs_retrieval: boolean, "
        " search_query: string|null, "
        " lead_prompt: string|null}\n"
        "Rules:\n"
        "- 'smalltalk' for greetings/thanks.\n"
        "- 'lead' if the user shows intent to start, asks for a callback, gives phone/email, or asks 'how to start'.\n"
        "- 'contact' if explicitly requesting email/phone/address/website.\n"
        "- 'pricing' if asking cost/fees/plans.\n"
        "- 'qa' for knowledge questions that require the KB.\n"
        "- If lead_hint is given, prefer kind='lead' and include a brief lead_prompt.\n"
        "- If kind in {'qa','pricing','contact'} and you expect KB help, set needs_retrieval=true and craft search_query.\n"
        "- Keep JSON a single line, no extra commentary."
    )
    user = f"User text:\n{user_text}\n\nLead hint (optional): {lead_hint or 'none'}"
    raw = _chat(_PLANNER_MODEL, system, user, temperature=0.0)

    # Tolerant JSON extraction
    try:
        # Try direct parse
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Fallback: pull {...} substring
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


def _final_answer(
    model: str,
    user_text: str,
    retrieved_snippets: str,
    contact_ctx: str,
    pricing_ctx: str,
    lead_state_line: str,
    lead_hint: str,
) -> str:
    """
    Second LLM pass: compose the actual reply using the retrieved evidence + facts.
    Keep it short, friendly, specific, and ask only ONE thing if a lead_hint is present.
    """
    system = (
        "You are Corah, a professional, friendly assistant for Corvox.\n"
        "Goals:\n"
        "1) Answer helpfully and concisely in a natural tone (no bullet-dumps unless asked).\n"
        "2) Use retrieved evidence and company facts when relevant; never contradict them.\n"
        "3) If a lead_hint is present, ask exactly one short follow-up question to move the lead forward.\n"
        "4) If the user asked for contact details, provide only the requested items (email/phone/address/website), not all.\n"
        "5) Avoid repeating the user's sentence verbatim. Paraphrase naturally.\n"
        "6) If unsure, say so briefly and propose a next helpful step.\n"
    )

    # We provide sections the model can scan; keep them short.
    user = (
        f"User: {user_text}\n\n"
        f"[Company Contact]\n{contact_ctx or 'None'}\n\n"
        f"[Pricing]\n{pricing_ctx or 'None'}\n\n"
        f"[Lead State]\n{lead_state_line}\n\n"
        f"[Retrieved Evidence]\n{retrieved_snippets or 'None'}\n\n"
        f"[Lead Hint]\n{lead_hint or 'None'}\n\n"
        "Now respond as Corah. Keep it to 1–3 sentences unless the user asked for a list or detailed steps."
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
      3) Final LLM: compose the answer with facts + evidence
    """
    # 0) Split the augmented question: user text + our [Context] block
    user_text, ctx_block = _split_user_and_ctx(question)
    ctx = _extract_ctx_sections(ctx_block)

    # For the final pass, we need these strings available (we don’t parse deeply)
    contact_ctx = _extract_section(ctx_block, "Company contact")
    pricing_ctx = _extract_section(ctx_block, "Pricing")
    lead_state_line = _extract_line(ctx_block, "Lead state")
    lead_hint = ctx.get("lead_hint", "")

    # 1) Planner
    plan = _planner(user_text, lead_hint=lead_hint)
    kind = (plan.get("kind") or "qa").lower().strip()
    needs_retrieval = bool(plan.get("needs_retrieval", True))
    search_query = (plan.get("search_query") or user_text).strip()

    # 2) Retrieval (only if needed)
    hits: List[Dict[str, Any]] = []
    retrieved_snippets = ""
    if needs_retrieval:
        try:
            hits = search(search_query, k=k)
            # flatten snippets into a compact text block for the final LLM
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
        except Exception as e:
            # If retrieval fails, proceed without it; LLM will answer generically
            retrieved_snippets = ""
            hits = []

    # 3) Final LLM
    answer = _final_answer(
        model=_FINAL_MODEL,
        user_text=user_text,
        retrieved_snippets=retrieved_snippets,
        contact_ctx=contact_ctx,
        pricing_ctx=pricing_ctx,
        lead_state_line=lead_state_line,
        lead_hint=lead_hint,
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
            "kind": kind,
        }

    return {"answer": answer, "citations": citations or None, "debug": dbg}


# -----------------------------
# Tiny helpers to pull sections/lines from the context block
# -----------------------------

_SECTION_RE = re.compile(r"^- ([^\n:]+):\s*(.*)$", re.MULTILINE)

def _extract_section(ctx_block: str, header: str) -> str:
    """
    Given the raw [Context] block, pull the lines after “- <header>:”
    until the next “- <other>:” or end. We keep it simple and robust.
    """
    if not ctx_block:
        return ""
    # Find the header start
    pat = re.compile(rf"-\s*{re.escape(header)}\s*:\s*(.*)", re.IGNORECASE)
    m = pat.search(ctx_block)
    if not m:
        return ""
    start = m.end(0)
    # Grab until next header marker "- Something:"
    tail = ctx_block[start:]
    nxt = re.search(r"^\s*-\s*[A-Za-z].*?:", tail, re.MULTILINE)
    chunk = tail[: nxt.start()] if nxt else tail
    text = (m.group(1) + "\n" + chunk).strip()
    # Squash excessive whitespace
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()

def _extract_line(ctx_block: str, key: str) -> str:
    if not ctx_block:
        return ""
    pat = re.compile(rf"{re.escape(key)}\s*:\s*(.+)", re.IGNORECASE)
    m = pat.search(ctx_block)
    return m.group(1).strip() if m else ""
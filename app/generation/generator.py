from __future__ import annotations

import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI
from app.core.utils import normalize_ws
from app.retrieval.retriever import search

# Models
_PLANNER_MODEL = os.getenv("OPENAI_PLANNER_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
_FINAL_MODEL   = os.getenv("OPENAI_FINAL_MODEL",   os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

# Temperatures (Phase 2: separate planner/final temps; fall back to legacy TEMPERATURE)
_LEGACY_TEMP   = float(os.getenv("TEMPERATURE", "0.5"))
_PLANNER_TEMP  = float(os.getenv("PLANNER_TEMPERATURE", os.getenv("PLANNER_TEMP", "0.3")))
_FINAL_TEMP    = float(os.getenv("FINAL_TEMPERATURE",   os.getenv("FINAL_TEMP",   str(_LEGACY_TEMP))))

_client = OpenAI()
# Export for lead_intent.py compatibility
client = _client

_CTX_START = "[Context]"
_CTX_END   = "[End Context]"
_LEAD_HINT_RE = re.compile(r"^Lead hint:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_LAST_ASKED_RE = re.compile(r"^last_asked\s*:\s*([A-Za-z_]+)\s*$", re.IGNORECASE | re.MULTILINE)

# NEW: lightweight pricing intent detector (keeps us from inventing “tiers”)
_PRICING_RE = re.compile(
    r"\b(price|pricing|cost|how\s*much|budget|quote|estimate|charges?|fees?)\b",
    re.IGNORECASE,
)

def _split_user_and_ctx(q: str) -> Tuple[str, str]:
    if _CTX_START in q and _CTX_END in q:
        head, rest = q.split(_CTX_START, 1)
        ctx, _ = rest.split(_CTX_END, 1)
        return head.strip(), ctx.strip()
    return q.strip(), ""

def _extract_section(ctx_block: str, header: str) -> str:
    if not ctx_block: return ""
    pat = re.compile(rf"-\s*{re.escape(header)}\s*:\s*(.*)", re.IGNORECASE)
    m = pat.search(ctx_block)
    if not m: return ""
    start = m.end(0)
    tail = ctx_block[start:]
    nxt = re.search(r"^\s*-\s*[A-Za-z].*?:", tail, re.MULTILINE)
    chunk = tail[: nxt.start()] if nxt else tail
    text = (m.group(1) + "\n" + chunk).strip()
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()

def _extract_ctx_bits(ctx: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    out["lead_hint"]   = (_LEAD_HINT_RE.search(ctx or "") or (lambda:None)()) and _LEAD_HINT_RE.search(ctx or "").group(1).strip() if ctx else ""
    m = _LAST_ASKED_RE.search(ctx or "")
    out["last_asked"]  = m.group(1).strip() if m else ""
    return out

def _chat(model: str, system: str, user: str, temperature: float) -> str:
    resp = _client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
    )
    return (resp.choices[0].message.content or "").strip()

def _planner(user_text: str, lead_hint: str = "", current_topic: str = "") -> Dict[str, Any]:
    """
    Classifier/planner. Keeps us on the same thread when the user back-channels
    with 'yes/sure/tell me more/go ahead/sounds good'. Prioritises lead flow.
    """
    system = (
        "You are a tiny planner. Return ONLY JSON:\n"
        "{kind:'qa'|'lead'|'contact'|'pricing'|'follow_up_on_current_topic'|'out_of_scope',"
        " needs_retrieval:boolean, search_query:string|null, lead_prompt:string|null}\n"
        "- If the user expresses intent like 'can you make/build', 'I want', 'interested', 'how much', "
        "'pricing', 'cost', or provides phone/email/time, classify as 'lead'.\n"
        "- Map 'yes'/'sure'/'okay'/'tell me more'/'go ahead'/'sounds good' to follow_up_on_current_topic "
        "  (do NOT ask which topic; continue the same thread).\n"
        "- Use 'out_of_scope' for general trivia not related to the user's project or Corvox.\n"
        "- needs_retrieval true only when the user asks for factual details from KB (products/policies/pricing text)."
    )
    user = (
        f"User text:\n{user_text}\n\n"
        f"Lead hint (optional): {lead_hint or 'none'}\n"
        f"Current topic (if any): {current_topic or 'none'}\n"
        "Back-channel words to treat as follow-up: yes, sure, ok, okay, go ahead, tell me more, sounds good."
    )

    raw = _chat(_PLANNER_MODEL, system, user, temperature=_PLANNER_TEMP)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    # Default fallback: basic QA with retrieval
    return {"kind": "qa", "needs_retrieval": True, "search_query": user_text, "lead_prompt": None}

# inside app/generation/generator.py

def _final_answer(
    model: str,
    user_text: str,
    retrieved_snippets: str,
    user_details: str,
    contact_ctx: str,
    pricing_ctx: str,
    summary: str,
    current_topic: str,
    recent_turns: str,
    lead_hint: str,
    last_asked: str,
) -> str:
    system = (
        "You are Corah, Corvox’s warm front-desk assistant — helpful first, concise, factual.\n"
        "Rules:\n"
        "• Always answer the user’s question before any lead capture.\n"
        "• Never ask for a field that is already known in [Lead slots].\n"
        "• At most one short question when it clearly advances the conversation.\n"
        "• If a name exists in [Lead slots], use it naturally once (e.g., “Thanks, Nasir”).\n"
        "• Do not invent pricing; you may offer to arrange a call if the user asks for price specifics.\n"
        "• If the user declines sharing contact, keep helping without nagging.\n"
    )

    user = (
        f"User: {user_text}\n\n"
        f"[Summary]\n{summary or 'None'}\n\n"
        f"[Current topic]\n{current_topic or 'None'}\n\n"
        f"[Recent turns]\n{recent_turns or 'None'}\n\n"
        f"[User details]\n{user_details or 'None'}\n\n"
        f"[Company contact]\n{contact_ctx or 'None'}\n\n"
        f"[Pricing]\n{pricing_ctx or 'None'}\n\n"
        f"[Retrieved]\n{retrieved_snippets or 'None'}\n\n"
        f"[Lead slots]  # dict of known lead fields; do NOT ask again if present\n{getattr(__import__('json'), 'dumps')({'placeholder':'moved to main context'}, indent=2)}\n\n"
        "Reply now as Corah following the rules."
    )
    return _chat(model, system, user, temperature=_FINAL_TEMP)

def generate_answer(
    question: str,
    k: int = 5,
    max_context_chars: int = 3000,
    debug: Optional[bool] = False,
    show_citations: Optional[bool] = False,
) -> Dict[str, Any]:
    user_text, ctx_block = _split_user_and_ctx(question)

    # Pull structured sections the orchestrator sends
    summary       = _extract_section(ctx_block, "Summary")
    current_topic = _extract_section(ctx_block, "Current topic")
    recent_turns  = _extract_section(ctx_block, "Recent turns")
    user_details  = _extract_section(ctx_block, "User details")
    contact_ctx   = _extract_section(ctx_block, "Company contact")
    pricing_ctx   = _extract_section(ctx_block, "Pricing")

    bits          = _extract_ctx_bits(ctx_block)
    lead_hint     = bits.get("lead_hint","")
    last_asked    = bits.get("last_asked","")

    # Planner
    plan = _planner(user_text, lead_hint=lead_hint, current_topic=current_topic)
    kind = plan.get("kind","qa")
    needs_retrieval = bool(plan.get("needs_retrieval", True))

    # Pricing sanity & pivot
    from app.lead.pricing import maybe_add_pricing_context
    pricing_reply = maybe_add_pricing_context(session_id="web", text=user_text)
    if pricing_reply:
        return {"answer": pricing_reply, "citations": None, "debug": {"planner": plan, "num_hits": 0}}

    # NEW: nudge the generator when user directly asked about pricing
    # (Don’t change DB state here — this is UI/LLM-level behaviour only.)
    if not lead_hint and _PRICING_RE.search(user_text or ""):
        lead_hint = "ask_contact"   # one gentle ask after a safe pricing statement

    # Retrieval (bounded)
    hits: List[Dict[str, Any]] = []
    retrieved_snippets = ""
    if needs_retrieval and kind != "out_of_scope":
        try:
            search_query = (plan.get("search_query") or user_text).strip()
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
        user_details=user_details,
        contact_ctx=contact_ctx,
        pricing_ctx=pricing_ctx,
        summary=summary,
        current_topic=current_topic,
        recent_turns=recent_turns,
        lead_hint=lead_hint,
        last_asked=last_asked,
    ).strip()

    # Citations (optional)
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

    dbg = {"planner": plan, "num_hits": len(hits)} if debug else None
    return {"answer": answer, "citations": citations or None, "debug": dbg}
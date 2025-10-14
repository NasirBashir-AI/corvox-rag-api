from __future__ import annotations

import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI
from app.core.utils import normalize_ws
from app.retrieval.retriever import search

_PLANNER_MODEL = os.getenv("OPENAI_PLANNER_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
_FINAL_MODEL   = os.getenv("OPENAI_FINAL_MODEL",   os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
_TEMPERATURE   = float(os.getenv("TEMPERATURE", "0.5"))

_client = OpenAI()

_CTX_START = "[Context]"
_CTX_END   = "[End Context]"
_LEAD_HINT_RE = re.compile(r"^Lead hint:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_LAST_ASKED_RE = re.compile(r"^last_asked\s*:\s*([A-Za-z_]+)\s*$", re.IGNORECASE | re.MULTILINE)

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

def _chat(model: str, system: str, user: str, temperature: float = _TEMPERATURE) -> str:
    resp = _client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
    )
    return (resp.choices[0].message.content or "").strip()

def _planner(user_text: str, lead_hint: str = "", current_topic: str = "") -> Dict[str, Any]:
    """
    Classifier/planner. Keeps us on the same thread when the user back-channels
    with 'yes/sure/tell me more/go ahead/sounds good'.
    """
    system = (
        "You are a tiny planner. Return ONLY JSON:\n"
        "{kind:'qa'|'lead'|'contact'|'pricing'|'follow_up_on_current_topic'|'out_of_scope',"
        " needs_retrieval:boolean, search_query:string|null, lead_prompt:string|null}\n"
        "- Map 'yes'/'sure'/'okay'/'tell me more'/'go ahead'/'sounds good' to follow_up_on_current_topic "
        "  (do NOT ask which topic; continue the same thread).\n"
        "- Use 'lead' when user asks to start/arrange a callback or gives phone/email/time.\n"
        "- Use 'out_of_scope' for general trivia not related to the user's project or Corvox.\n"
        "- needs_retrieval true only when the user asks for factual details about Corvox/products/pricing we might have in KB."
    )
    user = (
        f"User text:\n{user_text}\n\n"
        f"Lead hint (optional): {lead_hint or 'none'}\n"
        f"Current topic (if any): {current_topic or 'none'}\n"
        "Back-channel words to treat as follow-up: yes, sure, ok, okay, go ahead, tell me more, sounds good."
    )

    raw = _chat(_PLANNER_MODEL, system, user, temperature=0.0)
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
    # ---- System rules (concise; no tone overhaul) ----
    system = (
        "You are Corah, Corvox’s warm front-desk assistant—polite, concise, genuinely helpful.\n"
        "Core behavior (obey strictly):\n"
        "1) Keep replies short (1–3 sentences). No filler. Vary openers (avoid repeating 'Hi there!').\n"
        "2) Ask at most ONE short, purposeful question when it advances the goal. No stacked CTAs.\n"
        "3) Use [Summary], [Current topic], and [Recent turns] to continue the SAME thread; "
        "   do NOT re-ask what the user already told you.\n"
        "4) If you have a Lead hint (ask_<field> or bridge_back_to_<field> or confirm_done), phrase it naturally. "
        "   If last_asked equals the current ask target, do NOT repeat; briefly acknowledge and gently bridge back.\n"
        "5) Use [Company contact] ONLY for Corvox’s contact; never treat [User details] as company info.\n"
        "6) Stay in scope: Corvox services, the user’s project, lead capture, pricing, and our policies. "
        "   If the user asks general trivia (unrelated), decline briefly and steer back.\n"
        "7) If the user declines to share contact, respect it and continue helping without nagging.\n"
        "8) End neutrally when appropriate: 'I’m here if you want to dive deeper.' (no hard CTA).\n"
    )

    # ---- Few-shot (tiny) to anchor follow-ups on current topic ----
    shots = (
        "Example A\n"
        "[Summary] user exploring WhatsApp chatbot for a jewellery store; wants features.\n"
        "User: yes, tell me more\n"
        "Assistant: We can start with quick replies for FAQs and stock checks, then add order-tracking. "
        "If you use Shopify, we can read inventory directly. Would you want stock checks live or from a daily export?\n\n"
        "Example B\n"
        "[Summary] user just asked about pricing for a callback service.\n"
        "User: sure, go ahead\n"
        "Assistant: For a simple callback agent, most teams start on our lower tier and add call logging later. "
        "Ballpark is a small setup + modest monthly. Want me to sketch the two options?\n\n"
    )

    user = (
        f"{shots}"
        f"User: {user_text}\n\n"
        f"[Summary]\n{summary or 'None'}\n\n"
        f"[Current topic]\n{current_topic or 'None'}\n\n"
        f"[Recent turns]\n{recent_turns or 'None'}\n\n"
        f"[User details]\n{user_details or 'None'}\n\n"
        f"[Company contact]\n{contact_ctx or 'None'}\n\n"
        f"[Pricing]\n{pricing_ctx or 'None'}\n\n"
        f"[Retrieved]\n{retrieved_snippets or 'None'}\n\n"
        f"[Lead hint]\n{lead_hint or 'None'}\n\n"
        f"[last_asked]\n{last_asked or 'None'}\n\n"
        "Now reply as Corah following the rules above."
    )
    return _chat(model, system, user, temperature=_TEMPERATURE)

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
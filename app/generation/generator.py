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
    # ---- System rules (Phase 2.2 behavioural guard) ----
    system = (
        "You are Corah, Corvox’s warm front-desk assistant—polite, concise, genuinely helpful.\n"
        "Conversation priorities (strict):\n"
        "A) If lead capture is incomplete (missing name OR missing contact (email/phone) OR missing company), "
        "   prefer asking for exactly ONE missing item that most advances scheduling a discovery call.\n"
        "B) If user asked for pricing/cost, give a short, safe statement (no hard numbers) and pivot to lead capture.\n"
        "C) If lead is complete or user says 'no/that’s all/bye', summarise once and close politely.\n"
        "\n"
        "Behavioural rules:\n"
        "1) Keep replies short (1–3 sentences). Avoid filler.\n"
        "2) Ask at most ONE question only when it advances A/B above. Otherwise use a statement.\n"
        "3) Use [Summary], [Current topic], and [Recent turns] to stay on the SAME thread; do NOT re-ask known details.\n"
        "4) Hallucination guard: do NOT invent prices/timelines/features; if info is missing, say so and propose a short discovery call.\n"
        "5) Brand: Corvox BUILDS custom chat + voice agents (Corah). Do NOT imply we only integrate third-party chatbots.\n"
        "6) Third-party tools: only mention external plugins if the user explicitly asks for third-party options; otherwise prefer Corah.\n"
        "7) If last_asked equals the current ask target, do NOT repeat; briefly acknowledge and move to the next most useful step.\n"
        "8) If user declines to share contact, respect it; continue helping without nagging.\n"
    )

    # Minimal few-shot to anchor “continue same topic” without re-asking
    shots = (
        "Example A\n"
        "[Summary] user exploring chatbot for a toy store; wants stock updates.\n"
        "User: yes, tell me more\n"
        "Assistant: We can connect inventory to answer “in stock?” in real time and suggest alternatives. "
        "If you like, I can set a quick discovery call—what's your name and the best email?\n\n"
        "Example B\n"
        "[Summary] user just asked about pricing.\n"
        "User: how much\n"
        "Assistant: Pricing depends on scope; we start with a short discovery call, then share a clear quote. "
        "Shall I note your name and email to arrange it?\n\n"
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
# app/lead/pricing.py
from __future__ import annotations
from typing import Optional
from app.core.session_mem import get_state, set_state

def handle_pricing_query(session_id: str) -> str:
    """
    Return a polite, flexible pricing message and mark pricing context.
    Keeps it human and routes smoothly toward discovery call if appropriate.
    """
    session = get_state(session_id) or {}
    session["context"]["pricing_discussed"] = True
    # If the code was persisting the whole session dict:
    set_state(session_id, **session)

    # (If it was setting a single value, e.g. pricing_quip=quip,
    # you can also do: set_state(session_id, pricing_quip=quip))

    return (
        "Pricing depends on the project scope — smaller chatbots start from flexible monthly plans, "
        "while more advanced multi-agent or voice setups are quoted individually. "
        "The best next step is a short discovery call where we can understand your goals and share a clear proposal."
    )


def maybe_add_pricing_context(session_id: str, text: str) -> Optional[str]:
    """
    Detect pricing-related questions in user text and trigger a contextual reply.
    """
    if any(x in text.lower() for x in ["price", "pricing", "cost", "how much", "quote"]):
        return handle_pricing_query(session_id)
    return None
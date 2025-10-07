# app/generation/lead_intent.py
from __future__ import annotations
import json
from typing import Any, Dict, List

from app.generation.generator import client  # reuse the existing OpenAI client

_SYSTEM = (
    "You are a classifier that labels a short chat transcript for lead intent. "
    "Return strict JSON with keys: interest_level (one of: none, curious, buying, explicit_cta), "
    "confidence (0..1), and signals (array of short lowercase phrases). "
    "Signals should capture behaviors like: asked_to_start, asked_for_call, asked_price, "
    "gave_contact, positive_sentiment, business_described, timeline_soon, budget_mentioned."
)

def classify_lead_intent(turns: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    turns: list like [{"role":"user"|"assistant", "content":"..."}]
    returns: {"interest_level": "...", "confidence": 0.0-1.0, "signals": [...]}
    """
    if not turns:
        return {"interest_level": "none", "confidence": 0.0, "signals": []}

    try:
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": json.dumps({"turns": turns})},
            ],
        )
        txt = rsp.choices[0].message.content or "{}"
        data = json.loads(txt)
        # minimal validation
        lvl = str(data.get("interest_level", "none")).lower()
        if lvl not in {"none", "curious", "buying", "explicit_cta"}:
            lvl = "none"
        conf = float(data.get("confidence", 0.0))
        sigs = data.get("signals", [])
        if not isinstance(sigs, list):
            sigs = []
        return {"interest_level": lvl, "confidence": max(0.0, min(1.0, conf)), "signals": sigs}
    except Exception:
        return {"interest_level": "none", "confidence": 0.0, "signals": []}
# app/generation/lead_intent.py
from __future__ import annotations
import json
from typing import Any, Dict, List

from app.generation.generator import client  # reuse the existing OpenAI client

_SYSTEM = (
    "You are a compact classifier. Read a short chat transcript and return STRICT JSON:\n"
    "{\n"
    '  "intent_level": "none|curious|warm|hot|explicit_cta",\n'
    '  "sentiment": "negative|neutral|positive",\n'
    "  \"confidence\": 0..1,\n"
    "  \"signals\": [array of short lowercase cues like 'asked_price','gave_contact','asked_for_call','business_described']\n"
    "}\n"
    "- Map phrases like: 'can you make/build', 'i want', 'i'm interested', 'how much', 'pricing', 'cost', "
    "'book/schedule/set up a call', 'call me', or when the user gives phone/email/time → intent >= warm.\n"
    "- If the user explicitly asks for a callback or provides contact for follow-up → intent = explicit_cta.\n"
    "- Keep it conservative; prefer 'neutral' sentiment unless clear positivity/negativity.\n"
)

def classify_lead_intent(turns: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    turns: list like [{"role":"user"|"assistant", "content":"..."}]
    returns:
      {
        "intent_level": "none|curious|warm|hot|explicit_cta",
        "sentiment": "negative|neutral|positive",
        "confidence": float,
        "signals": [str, ...]
      }
    """
    if not turns:
        return {"intent_level": "none", "sentiment": "neutral", "confidence": 0.0, "signals": []}

    try:
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": json.dumps({"turns": turns[-8:]}, ensure_ascii=False)},
            ],
        )
        txt = (rsp.choices[0].message.content or "{}").strip()
        data = json.loads(txt)
    except Exception:
        return {"intent_level": "none", "sentiment": "neutral", "confidence": 0.0, "signals": []}

    # Minimal validation/sanitise
    lvl = str(data.get("intent_level", "none")).lower()
    if lvl not in {"none", "curious", "warm", "hot", "explicit_cta"}:
        lvl = "none"

    sent = str(data.get("sentiment", "neutral")).lower()
    if sent not in {"negative", "neutral", "positive"}:
        sent = "neutral"

    try:
        conf = float(data.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    sigs = data.get("signals", [])
    if not isinstance(sigs, list):
        sigs = []

    return {"intent_level": lvl, "sentiment": sent, "confidence": conf, "signals": sigs}
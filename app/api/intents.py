# app/api/intents.py
"""
Lightweight intent routing for Corah – v2 conversational edition.
Expanded detection for callbacks, yes-followups, and polite continuity.
"""

from __future__ import annotations
import re
from typing import Optional, Tuple
from app.core.utils import normalize_ws

# -----------------------------
# Smalltalk patterns
# -----------------------------
_SMALLTALK_PATTERNS = [
    r"^\s*(hi|hello|hey|hiya|yo)\s*[!.]?\s*$",
    r"^\s*(good\s*(morning|afternoon|evening))\s*[!.]?\s*$",
    r"^\s*(how\s+are\s+you|how's\s+it\s+going)\s*\??\s*$",
    r"^\s*(thanks|thank\s*you)\s*[!.]?\s*$",
    r"^\s*(ok|okay|cool|great|nice)\s*[!.]?\s*$",
]

# -----------------------------
# Information-related keywords
# -----------------------------
_PRICING_KEYWORDS = [
    r"\b(price|pricing|cost|rate|fee|plan|quote|budget|estimate|subscription)\b",
    r"\bhow\s+much\b", r"\bper\s*month\b",
]

_SERVICES_KEYWORDS = [
    r"\bservice(s)?\b", r"\bwhat\s+do\s+you\s+do\b",
    r"\bwhat\s+does\s+corvox\b", r"\boffer(s|ings)?\b",
    r"\bcapabilit(y|ies)\b", r"\bsolution(s)?\b", r"\bproduct(s)?\b",
]

# -----------------------------
# Lead / conversion triggers
# -----------------------------
_LEAD_KEYWORDS = [
    r"\b(book|schedule|arrange)\s+(a\s*)?(call|meeting|demo)\b",
    r"\b(call\s*back|callback)\b",
    r"\b(contact|reach|get\s*in\s*touch|someone\s*call)\b",
    r"\b(sign\s*up|get\s*started|free\s*consultation)\b",
    r"\b(interested|want\s+to\s+talk|talk\s+to\s+(someone|team))\b",
]

# -----------------------------
# Contact details requests
# -----------------------------
def _contact_focus(q: str) -> Optional[str]:
    if re.search(r"\b(e[-\s]?mail|email)\b", q, re.IGNORECASE):
        return "email"
    if re.search(r"\b(phone|number|call)\b", q, re.IGNORECASE):
        return "phone"
    if re.search(r"\b(address|office|location|where\s+are\s+you|based)\b", q, re.IGNORECASE):
        return "address"
    if re.search(r"\b(website|url|site)\b", q, re.IGNORECASE):
        return "url"
    return None

# -----------------------------
# Follow-up / affirmation words
# -----------------------------
_FOLLOWUP_KEYWORDS = [
    r"^\s*(yes|yeah|yep|sure|ok|okay|please\s+do|go\s+ahead|alright)\s*[.!]?\s*$",
]

# -----------------------------
# Helpers
# -----------------------------
def _rx_any(patterns):
    rxs = [re.compile(p, re.IGNORECASE) for p in patterns]
    def _match(q: str) -> bool:
        return any(rx.search(q) for rx in rxs)
    return _match

_is_smalltalk = lambda q: any(re.search(p, q, re.IGNORECASE) for p in _SMALLTALK_PATTERNS)
_is_pricing   = _rx_any(_PRICING_KEYWORDS)
_is_services  = _rx_any(_SERVICES_KEYWORDS)
_is_lead      = _rx_any(_LEAD_KEYWORDS)
_is_followup  = _rx_any(_FOLLOWUP_KEYWORDS)

# -----------------------------
# Public detection
# -----------------------------
def detect_intent(text: str) -> Tuple[str, Optional[str]]:
    q = normalize_ws(text or "")

    # Follow-up intent
    if _is_followup(q):
        return "followup", None

    # Smalltalk
    if _is_smalltalk(q):
        return "smalltalk", None

    # Contact info requests
    cf = _contact_focus(q)
    if cf:
        return "contact", cf

    # Info-type queries
    if _is_services(q):
        return "info", "services"
    if _is_pricing(q):
        return "info", "pricing"

    # Lead-type queries
    if _is_lead(q):
        return "lead", None

    # Otherwise
    return "other", None

# -----------------------------
# Smalltalk replies
# -----------------------------
def smalltalk_reply(text: str) -> str:
    t = (text or "").lower().strip()
    if "how are" in t:
        return "I’m doing well—thanks for asking! How can I help today?"
    if "good morning" in t:
        return "Good morning! How can I help today?"
    if "good afternoon" in t:
        return "Good afternoon! What can I do for you?"
    if "good evening" in t:
        return "Good evening! How can I help?"
    if any(w in t for w in ["hi", "hello", "hey", "hiya"]):
        return "Hi—how can I help today?"
    if "thank" in t:
        return "You’re welcome! Is there anything else I can help with?"
    return "Hello! How can I help today?"
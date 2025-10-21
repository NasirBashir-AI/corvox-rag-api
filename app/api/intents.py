# app/api/intents.py
"""
Lightweight intent routing for Corah.
- Never calls the retriever or LLM here; this module only classifies.
- Categories: smalltalk | contact | pricing | services | lead | other
- If smalltalk, we also return a ready-made polite reply (no RAG needed).
"""

from __future__ import annotations

import re
from typing import Optional, Tuple
from app.core.utils import normalize_ws

# -----------------------------
# Minimal inline detectors (replace old harvest_* usage)
# -----------------------------
EMAIL_RE  = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
PHONE_RE  = re.compile(r"\b(?:\+?\d[\d\-\s]{7,}\d)\b")  # loose but safe enough for intent
NAME_HINT = re.compile(
    r"\b(my\s+name\s+is|i\s*am|i'm|this\s+is)\s+[A-Z][A-Za-z'’\-]{1,}\b",
    re.IGNORECASE,
)

# -----------------------------
# Keyword lexicons (compact)
# -----------------------------

_SMALLTALK_PATTERNS = [
    r"^\s*(hi|hello|hey|yo|hiya)\s*[!.]?\s*$",
    r"^\s*(good\s*(morning|afternoon|evening))\s*[!.]?\s*$",
    r"^\s*(how\s+are\s+you|how's\s+it\s+going|sup|what'?s\s+up)\s*\??\s*$",
    r"^\s*(thanks|thank\s+you)\s*[!.]?\s*$",
    r"^\s*(ok|okay|cool|great|nice)\s*[!.]?\s*$",
]

_CONTACT_PATTERNS = [
    re.compile(r"\bphone\s*(?:no\.?|number)\b", re.IGNORECASE),
    re.compile(r"\bemail\s*(?:id|address)\b", re.IGNORECASE),
    re.compile(r"\bcall\s*-?\s*back\b", re.IGNORECASE),
    re.compile(r"\bwhere\s+(?:are\s+you\s+)?based\b", re.IGNORECASE),
    re.compile(r"\bwhats?app\b", re.IGNORECASE),
]

_CONTACT_KEYWORDS = [
    "contact", "get in touch", "reach you", "reach out",
    "email", "email address", "email id", "e-mail",
    "phone", "phone number", "contact number",
    "call", "call back", "callback", "whatsapp",
    "address", "where are you based", "where based",
    "location", "located", "office", "website", "url",
]

_PRICING_KEYWORDS = [
    "price", "pricing", "cost", "fees", "rates", "plans", "how much",
    "subscription", "quote", "estimate", "price list", "how much is it",
    "per month", "budget",
]

_SERVICES_KEYWORDS = [
    "services", "what do you do", "what does corvox do", "capabilities",
    "solutions", "offerings", "products", "what you offer", "use cases",
    "areas you cover",
]

_LEAD_KEYWORDS = [
    "callback", "call back", "book a call", "schedule a call", "arrange a call",
    "speak to someone", "talk to a human", "demo", "book a demo",
    "next step", "get started", "i want to take your services", "sign up",
    "quote", "consultation",
]

def _contact_focus(q: str) -> str:
    """Return which contact detail the user is asking for."""
    if re.search(r"\b(e-?mail|email)\b", q, re.IGNORECASE):
        return "email"
    if re.search(r"\b(phone|number|call)\b", q, re.IGNORECASE):
        return "phone"
    if re.search(r"\b(address|location|office|where\s+are\s+you\s+based)\b", q, re.IGNORECASE):
        return "address"
    if re.search(r"\b(website|site|url)\b", q, re.IGNORECASE):
        return "url"
    return "generic"

def _compile_keywords(patterns):
    """
    Treat entries as regex fragments. For plain single words, add \b…\b.
    Leave phrases/regex (those containing spaces or regex tokens) as-is.
    """
    parts = []
    for p in patterns:
        if any(tok in p for tok in ["\\", "(", "[", "?", "|", " "]):
            parts.append(p)
        else:
            parts.append(rf"\b{p}\b")
    return re.compile("|".join(parts), re.IGNORECASE)

_RE_CONTACT  = _compile_keywords(_CONTACT_KEYWORDS)
_RE_PRICING  = _compile_keywords(_PRICING_KEYWORDS)
_RE_SERVICES = _compile_keywords(_SERVICES_KEYWORDS)
_RE_SMALLTALK = [re.compile(p, re.IGNORECASE) for p in _SMALLTALK_PATTERNS]
_RE_LEAD     = _compile_keywords(_LEAD_KEYWORDS)

# -----------------------------
# Public API
# -----------------------------

def detect_intent(text: str) -> Tuple[str, Optional[str]]:
    """
    Return (category, detail).

    Categories:
      - "smalltalk": include a canned reply as detail
      - "contact":  detail ∈ {"email","phone","address","url","generic"} or None
      - "pricing":  None
      - "services": None
      - "lead":     None (capture flow can proceed)
      - "other":    None
    """
    q = normalize_ws(text or "")

    # If the user actually provided email/phone, treat as contact so we don't lose it.
    if EMAIL_RE.search(q) or PHONE_RE.search(q):
        return "contact", None

    # If they announced a name (e.g., "my name is …"), treat as lead continuation.
    if NAME_HINT.search(q):
        return "lead", None

    # Smalltalk first
    for rx in _RE_SMALLTALK:
        if rx.search(q):
            return "smalltalk", smalltalk_reply(q)

    # Conversion-ish intents before generic services
    if _RE_LEAD.search(q):
        return "lead", None

    # Contact queries
    if _RE_CONTACT.search(q):
        return "contact", _contact_focus(q)
    for rx in _CONTACT_PATTERNS:
        if rx.search(q):
            return "contact", None

    # Pricing / services
    if _RE_PRICING.search(q):
        return "pricing", None
    if _RE_SERVICES.search(q):
        return "services", None

    return "other", None

def smalltalk_reply(text: str) -> str:
    """
    Provide a short, friendly reply for greetings/acknowledgements.
    Keep it generic and professional.
    """
    t = text.lower().strip()

    if re.search(r"\bhow\s+are\s+you\b", t):
        return "I’m doing well—thanks for asking! How can I help today?"
    if re.search(r"\bgood\s*morning\b", t):
        return "Good morning! How can I help today?"
    if re.search(r"\bgood\s*afternoon\b", t):
        return "Good afternoon! What can I do for you?"
    if re.search(r"\bgood\s*evening\b", t):
        return "Good evening! How can I help?"

    if re.search(r"\b(hi|hello|hey|hiya|yo)\b", t):
        return "Hi—how can I help today?"
    if re.search(r"\b(thanks|thank\s*you)\b", t):
        return "You’re welcome! What can I help you with?"

    return "Hello! How can I help today?"
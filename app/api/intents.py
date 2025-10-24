# app/api/intents.py
"""
Lightweight intent routing for Corah.
- No retrieval here; only classifies the user message.
- Returns (kind, topic):
    kind  ∈ {'smalltalk','info','contact','lead','goodbye','other'}
    topic ∈ {'services','pricing','email','phone','address','url', None}
- Also exposes helpers for interest detection and goodbye.
"""

from __future__ import annotations
import re
from typing import Optional, Tuple

from app.core.utils import normalize_ws

# -----------------------------
# Smalltalk / goodbye patterns
# -----------------------------
_SMALLTALK_PATTERNS = [
    r"^\s*(hi|hello|hey|hiya|yo)\s*[!.]?\s*$",
    r"^\s*(good\s*(morning|afternoon|evening))\s*[!.]?\s*$",
    r"^\s*(how\s+are\s+you|how's\s+it\s+going)\s*\??\s*$",
    r"^\s*(thanks|thank\s*you)\s*[!.]?\s*$",
    r"^\s*(ok|okay|cool|great|nice)\s*[!.]?\s*$",
]
_GOODBYE_RE = re.compile(r"\b(bye|goodbye|that'?s\s+all|i'?m\s+done|close\s+(the\s+)?(chat|session))\b", re.IGNORECASE)

# -----------------------------
# Info topics (services / pricing)
# -----------------------------
_PRICING_KEYWORDS = [
    r"\bprice(s|ing)?\b", r"\bcost(s)?\b", r"\brates?\b", r"\bfee(s)?\b",
    r"\bplan(s)?\b", r"\bhow\s+much\b", r"\bquote\b", r"\bestimate\b",
    r"\bper\s*month\b", r"\bsubscription\b", r"\bbudget\b",
]

_SERVICES_KEYWORDS = [
    r"\bservice(s)?\b", r"\bwhat\s+do\s+you\s+do\b", r"\bwhat\s+does\s+corvox\b",
    r"\bcapabilit(y|ies)\b", r"\bsolution(s)?\b", r"\boffer(s|ings)?\b",
    r"\bproduct(s)?\b", r"\buse\s*case(s)?\b", r"\bwhat\s+you\s+offer\b",
    r"\bmulti-?agent(s)?\b", r"\bai\s+(agent|team|department|voice)\b",
]

# -----------------------------
# Lead-ish words (conversion)
# -----------------------------
_LEAD_KEYWORDS = [
    r"\b(call\s*back|book\s*(a\s*)?call|schedule\s*(a\s*)?call|arrange\s*(a\s*)?call|demo|consultation|quote|proposal)\b",
    r"\b(personali[sz]ed|custom)\s+quote\b",
    r"\b(get|next)\s+started\b",
    r"\b(contact\s*me|reach\s*me|get\s*in\s*touch)\b",
]

# -----------------------------
# Contact detection
# -----------------------------
# If the user asks *for* contact details (address/phone/email/url)
def _contact_focus(q: str) -> Optional[str]:
    if re.search(r"\b(e[-\s]?mail|email)\b", q, re.IGNORECASE):
        return "email"
    if re.search(r"\b(phone|number|call)\b", q, re.IGNORECASE):
        return "phone"
    if re.search(r"\b(address|hq|head\s*office|location|office|where\s+(are\s+you\s+)?(based|located))\b",
                 q, re.IGNORECASE):
        return "address"
    if re.search(r"\b(website|site|url)\b", q, re.IGNORECASE):
        return "url"
    return None

# If the user *provides* contact (PII) — opportunistic capture
_EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b(?:\+?\d[\s\-()]*){7,}\d\b")

def looks_like_pii(q: str) -> bool:
    return bool(_EMAIL_RE.search(q) or _PHONE_RE.search(q))

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

# -----------------------------
# Public API
# -----------------------------
def detect_intent(text: str) -> Tuple[str, Optional[str]]:
    # Normalize whitespace for stability
    q = normalize_ws(text or "")

    if _GOODBYE_RE.search(q):
        return "goodbye", None

    # 0) If the user *gave* phone/email, route to 'lead' (opportunistic capture)
    if looks_like_pii(q):
        return "lead", None

    # 1) Smalltalk first (no RAG needed)
    if _is_smalltalk(q):
        return "smalltalk", None

    # 2) Clear info requests (KB-first)
    if _is_services(q):
        return "info", "services"
    if _is_pricing(q):
        return "info", "pricing"

    # 3) Direct contact requests
    cf = _contact_focus(q)
    if cf:
        return "contact", cf

    # 4) Conversion / next-step intent
    if _is_lead(q):
        return "lead", None

    return "other", None


def smalltalk_reply(text: str) -> str:
    t = (text or "").lower().strip()
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


def is_interest_intent(kind: str, topic: Optional[str], text: str) -> bool:
    """Return True if the message signals purchase interest (demo/quote/call/pricing follow-up)."""
    if kind == "lead":
        return True
    if kind == "info" and (topic in {"pricing", "services"}):
        return True
    # fallback: lexical triggers
    q = (text or "").lower()
    return bool(re.search(r"\b(book|schedule|arrange)\b.*\bcall\b", q) or "quote" in q or "demo" in q)
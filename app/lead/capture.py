# app/lead/capture.py
"""
Lead detail harvesting and payload assembly for Corah.

Includes:
- harvest_name / harvest_email / harvest_phone  (safe, conservative extractors)
- build_lead_payload  (single source of truth for saved lead structure)
- save_lead          (delegates to app.retrieval.leads if available)

Design goals
- Never force DB changes here; just shape a clean payload and hand off.
- Be conservative when extracting details from free text.
- Keep corrections/audit info outside of PII fields (in metadata).
"""

from __future__ import annotations

from typing import Optional, Dict, Any
from datetime import datetime, timezone
import re

# Optional downstream store
try:
    from app.retrieval import leads as _leads_store
except Exception:  # pragma: no cover
    _leads_store = None


# ---------------------------
# Harvesters (conservative)
# ---------------------------

_EMAIL_RE = re.compile(r"([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})")
# UK-leaning, but accepts international “+” formats; keeps only digits and leading +
_PHONE_RE = re.compile(r"(\+?\d[\d\-\s]{7,}\d)")
# Name phrases: "my name is X", "I'm X", "I am X" (avoid grabbing long sentences)
_NAME_PATTERNS = (
    re.compile(r"\bmy\s+name\s+is\s+([A-Z][a-zA-Z\-\.'\s]{1,40})\b", re.I),
    re.compile(r"\bi\s*am\s+([A-Z][a-zA-Z\-\.'\s]{1,40})\b", re.I),
    re.compile(r"\bi'm\s+([A-Z][a-zA-Z\-\.'\s]{1,40})\b", re.I),
)

def _clean(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = s.strip()
    return s or None

def harvest_email(text: str) -> Optional[str]:
    if not text:
        return None
    m = _EMAIL_RE.search(text)
    if not m:
        return None
    email = m.group(1).strip()
    # Basic sanity check (avoid trailing punctuation)
    email = email.rstrip(".,);:")
    return email

def harvest_phone(text: str) -> Optional[str]:
    if not text:
        return None
    m = _PHONE_RE.search(text)
    if not m:
        return None
    raw = m.group(1)
    # Normalise: keep leading + and digits
    digits = "".join(ch for ch in raw if (ch.isdigit() or ch == "+"))
    # Very short numbers are noise
    if len(digits.replace("+", "")) < 8:
        return None
    return digits

def harvest_name(text: str) -> Optional[str]:
    if not text:
        return None
    for pat in _NAME_PATTERNS:
        m = pat.search(text)
        if m:
            # Trim to first two tokens max (avoid grabbing sentence tails)
            candidate = " ".join(m.group(1).split()[:3]).strip()
            # Avoid obviously generic answers
            if candidate and len(candidate) <= 40:
                return candidate
    return None


# ---------------------------
# Lead payload builder
# ---------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def build_lead_payload(
    *,
    session_id: str,
    session_summary: str,
    contact: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a normalized lead dict ready for storage.
    metadata may include: sentiment, intent_score/level, priority, audit_note, closure_type, corrections, etc.
    """
    md = dict(metadata or {})
    # Flatten a few well-known fields for easier querying later
    payload = {
        "created_at": _now_iso(),
        "session_id": session_id,
        "name": _clean(contact.get("name")),
        "email": _clean(contact.get("email")),
        "phone": _clean(contact.get("phone")),
        "company": _clean(contact.get("company")),
        "preferred_time": _clean(contact.get("preferred_time")),
        "summary": session_summary or "",
        # Phase-2 analytics (optional but recommended)
        "sentiment": md.get("sentiment") or md.get("last_sentiment"),
        "intent_level": md.get("intent_level") or md.get("intent"),
        "priority": md.get("priority"),
        # Audit fields
        "audit_note": md.get("audit_note"),
        "closure_type": md.get("closure_type"),
        "corrections": md.get("corrections"),
        # Keep the raw meta as well for future-proofing
        "meta": md,
    }
    return payload


# ---------------------------
# Storage delegation
# ---------------------------

def save_lead(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Persist the lead using the project’s storage module if available.
    Fallback: return the payload (so the caller can still proceed gracefully).
    Expected downstream functions (first match wins):
      - app.retrieval.leads.save_lead(payload)
      - app.retrieval.leads.store_lead(payload)
      - app.retrieval.leads.add_lead(payload)
    """
    if _leads_store:
        for fname in ("save_lead", "store_lead", "add_lead"):
            fn = getattr(_leads_store, fname, None)
            if callable(fn):
                try:
                    return fn(payload)
                except Exception:
                    # Fall through to next option or fallback
                    pass
    # Fallback – return unchanged; upstream may log or handle separately
    return payload
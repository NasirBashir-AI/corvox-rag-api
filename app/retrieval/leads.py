# app/retrieval/leads.py
"""
Lead storage & helpers (file-based, zero-dependency).

Functions
- normalize_priority(priority) -> "hot"|"warm"|"cold"
- save_lead(payload) -> dict (adds id, writes JSONL + CSV)
- store_lead(payload) / add_lead(payload) -> aliases

Storage
- data/leads.jsonl  (one JSON per line; full fidelity)
- data/leads.csv    (selected columns for quick review/export)

Notes
- Safe to swap later for Postgres/CRM; keep function contracts identical.
"""

from __future__ import annotations

from typing import Dict, Any, List
import os
import uuid
import json
import csv
from datetime import datetime, timezone

# ---------------------------
# Paths
# ---------------------------

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
_JSONL = os.path.join(_DATA_DIR, "leads.jsonl")
_CSV = os.path.join(_DATA_DIR, "leads.csv")

os.makedirs(_DATA_DIR, exist_ok=True)

# ---------------------------
# Helpers
# ---------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def normalize_priority(priority: str) -> str:
    p = (priority or "").strip().lower()
    if p in ("hot", "high", "p1", "urgent"):
        return "hot"
    if p in ("warm", "medium", "p2"):
        return "warm"
    if p in ("cold", "low", "p3"):
        return "cold"
    return "cold"

# ---------------------------
# Save
# ---------------------------

_CSV_FIELDS: List[str] = [
    "id", "created_at", "session_id",
    "name", "email", "phone", "company", "preferred_time",
    "sentiment", "intent_level", "priority",
    "closure_type",
    "summary",
]

def _write_jsonl(obj: Dict[str, Any]) -> None:
    with open(_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _write_csv(obj: Dict[str, Any]) -> None:
    # Create file with header if missing
    file_exists = os.path.exists(_CSV)
    with open(_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        row = {k: obj.get(k) for k in _CSV_FIELDS}
        writer.writerow(row)

def save_lead(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Persist the lead to local files. Returns the saved payload with an id.
    Expected keys (best-effort): created_at, session_id, name, email, phone,
    company, preferred_time, sentiment, intent_level, priority, summary,
    closure_type, audit_note, corrections, meta.
    """
    lead = dict(payload or {})
    lead.setdefault("id", str(uuid.uuid4()))
    lead.setdefault("created_at", _now_iso())
    # Normalize priority for consistency
    lead["priority"] = normalize_priority(lead.get("priority"))

    # Write
    try:
        _write_jsonl(lead)
    except Exception:
        # If JSONL write fails, continue to CSV to avoid total loss
        pass
    try:
        _write_csv(lead)
    except Exception:
        # CSV failure should not block API flow
        pass

    return lead

# Aliases for compatibility with other modules
def store_lead(payload: Dict[str, Any]) -> Dict[str, Any]:
    return save_lead(payload)

def add_lead(payload: Dict[str, Any]) -> Dict[str, Any]:
    return save_lead(payload)
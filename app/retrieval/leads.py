# app/retrieval/leads.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from app.core.utils import pg_cursor
from app.core.config import DB_URL

# We keep all lead data in structured columns (no JSON payload paths).
# Table is created if missing; we also add a UNIQUE index on session_id so
# we can use ON CONFLICT for clean upserts.

DDL = """
CREATE SCHEMA IF NOT EXISTS corah_store;

CREATE TABLE IF NOT EXISTS corah_store.leads (
  id             BIGSERIAL PRIMARY KEY,
  session_id     TEXT NOT NULL,
  name           TEXT,
  phone          TEXT,
  email          TEXT,
  preferred_time TEXT,
  notes          TEXT,
  source         TEXT DEFAULT 'chat',
  stage          TEXT,
  done           BOOLEAN DEFAULT FALSE,
  done_at        TIMESTAMPTZ,
  created_at     TIMESTAMPTZ DEFAULT now(),
  updated_at     TIMESTAMPTZ DEFAULT now()
);

-- unique index allows ON CONFLICT(session_id)
CREATE UNIQUE INDEX IF NOT EXISTS corah_leads_session_id_uq
  ON corah_store.leads(session_id);
"""

UPSERT = """
INSERT INTO corah_store.leads (
  session_id, name, phone, email, preferred_time, notes, source, stage, done, done_at, updated_at
) VALUES (
  %(session_id)s, %(name)s, %(phone)s, %(email)s, %(preferred_time)s, %(notes)s, %(source)s, %(stage)s, %(done)s, %(done_at)s, now()
)
ON CONFLICT (session_id) DO UPDATE
SET
  name           = COALESCE(EXCLUDED.name,           corah_store.leads.name),
  phone          = COALESCE(EXCLUDED.phone,          corah_store.leads.phone),
  email          = COALESCE(EXCLUDED.email,          corah_store.leads.email),
  preferred_time = COALESCE(EXCLUDED.preferred_time, corah_store.leads.preferred_time),
  notes          = COALESCE(EXCLUDED.notes,          corah_store.leads.notes),
  source         = COALESCE(EXCLUDED.source,         corah_store.leads.source),
  stage          = COALESCE(EXCLUDED.stage,          corah_store.leads.stage),
  done           = COALESCE(EXCLUDED.done,           corah_store.leads.done),
  done_at        = COALESCE(EXCLUDED.done_at,        corah_store.leads.done_at),
  updated_at     = now()
RETURNING id;
"""

SELECT_ONE = """
SELECT
  id, session_id, name, phone, email, preferred_time, notes,
  source, stage, done, done_at, created_at, updated_at
FROM corah_store.leads
WHERE session_id = %(session_id)s
LIMIT 1;
"""

def _ensure_schema() -> None:
    with pg_cursor(DB_URL) as cur:
        cur.execute(DDL)

def save_lead(
    session_id: str,
    name: Optional[str] = None,
    phone: Optional[str] = None,
    email: Optional[str] = None,
    preferred_time: Optional[str] = None,
    notes: Optional[str] = None,
    source: Optional[str] = "chat",
    stage: Optional[str] = None,
    done: Optional[bool] = None,
    done_at: Optional[str] = None,  # ISO string is fine; DB casts to timestamptz if valid
) -> int:
    """
    Upsert a lead row for this session_id.
    - Only non-null fields overwrite existing values (via COALESCE in the UPSERT).
    - Returns the lead id.
    """
    _ensure_schema()
    params = {
        "session_id": session_id,
        "name": name,
        "phone": phone,
        "email": email,
        "preferred_time": preferred_time,
        "notes": notes,
        "source": source,
        "stage": stage,
        "done": done,
        "done_at": done_at,
    }
    with pg_cursor(DB_URL) as cur:
        cur.execute(UPSERT, params)
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else 0

def get_lead(session_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single lead by session_id."""
    _ensure_schema()
    with pg_cursor(DB_URL) as cur:
        cur.execute(SELECT_ONE, {"session_id": session_id})
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

def mark_stage(session_id: str, stage: str, **kwargs: Any) -> int:
    """
    Convenience: set the current stage and persist any provided fields.
    Usage: mark_stage(sid, 'contact', phone=phone)  # persists safely
    """
    return save_lead(session_id, stage=stage, **kwargs)

def mark_done(session_id: str, **kwargs: Any) -> int:
    """
    Mark lead as done; optionally pass final fields (notes, preferred_time, etc).
    """
    # If caller didn't supply done_at, stamp now.
    if "done_at" not in kwargs or not kwargs.get("done_at"):
        kwargs["done_at"] = datetime.now(timezone.utc).isoformat()
    return save_lead(session_id, done=True, **kwargs)

# -------- Optional helper: append a compact JSON "report" into notes (no schema changes) --------

def _append_to_notes(existing: Optional[str], addition: str) -> str:
    existing = (existing or "").strip()
    stamp = datetime.now(timezone.utc).isoformat()
    block = f"\n\n---\nREPORT {stamp}\n{addition}\n"
    return (existing + block) if existing else (f"REPORT {stamp}\n{addition}\n")

def save_lead_report(session_id: str, report: Dict[str, Any]) -> int:
    """
    Append a compact JSON report into `notes` for this lead.
    - No schema change: we just append a stamped 'REPORT <ISO>\n<json>' block.
    - Returns the lead id (creates the row if missing).
    """
    _ensure_schema()
    existing = get_lead(session_id)
    prev_notes = existing.get("notes") if existing else None
    addition = json.dumps(report, separators=(",", ":"), ensure_ascii=False)
    new_notes = _append_to_notes(prev_notes, addition)
    return save_lead(session_id, notes=new_notes)
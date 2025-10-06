# app/retrieval/leads.py
from __future__ import annotations
from typing import Optional
from app.core.config import DB_URL
from app.core.utils import pg_cursor

_SQL_LEAD_INSERT = """
INSERT INTO corah_store.leads
  (session_id, name, phone, email, preferred_time, notes, source)
VALUES
  (%(session_id)s, %(name)s, %(phone)s, %(email)s, %(preferred_time)s, %(notes)s, %(source)s)
RETURNING id;
"""

def save_lead(
    session_id: str,
    *,
    name: Optional[str] = None,
    phone: Optional[str] = None,
    email: Optional[str] = None,
    preferred_time: Optional[str] = None,
    notes: Optional[str] = None,
    source: str = "chat",
) -> int:
    with pg_cursor(DB_URL) as cur:
        cur.execute(_SQL_LEAD_INSERT, {
            "session_id": session_id,
            "name": name,
            "phone": phone,
            "email": email,
            "preferred_time": preferred_time,
            "notes": notes,
            "source": source,
        })
        new_id = cur.fetchone()[0]
        return int(new_id)

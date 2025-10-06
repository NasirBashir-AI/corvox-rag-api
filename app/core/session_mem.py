# app/core/session_mem.py
from typing import Dict, Any
from collections import defaultdict

# naive in-memory store (per process). Good enough for demo.
_SESS: Dict[str, Dict[str, Any]] = defaultdict(dict)

def get_state(session_id: str) -> Dict[str, Any]:
    return _SESS[session_id]

def set_state(session_id: str, **updates):
    _SESS[session_id].update({k: v for k, v in updates.items() if v})
# app/api/schemas.py
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field

# -----------------------------
# Search / Retrieval
# -----------------------------

class SearchHit(BaseModel):
    doc: str
    uri: Optional[str] = None
    score: Optional[float] = None
    dist: Optional[float] = None
    chunk_no: Optional[int] = None
    title: Optional[str] = None
    text: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    k: int
    hits: List[SearchHit]

# -----------------------------
# Q&A / Chat
# -----------------------------

class QuestionIn(BaseModel):
    question: str
    k: int = 5

class ChatRequest(BaseModel):
    question: str
    k: int = 5
    max_context: int = 3000
    session_id: Optional[str] = None  # <-- add this

class AnswerOut(BaseModel):
    answer: str

class ChatResponse(BaseModel):
    answer: str
    rewritten_query: Optional[str] = None
    used: Optional[List[SearchHit]] = None

# -----------------------------
# Lead Capture – session flow
# -----------------------------

class LeadStart(BaseModel):
    session_id: str

class LeadMessageIn(BaseModel):
    session_id: str
    message: str

class LeadOut(BaseModel):
    reply: str
    done: bool = False
    missing: List[str] = []
    lead_id: Optional[int] = None

# -----------------------------
# Lead Capture – validated payload
# (useful for saving a complete/partial lead)
# -----------------------------

class Lead(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[constr(pattern=r'^[\d+\-\s]{7,20}$')] = None # flexible phone validation
    company_name: Optional[str] = None
    company_size: Optional[str] = None
    industry: Optional[str] = None
    requirements: Optional[str] = None
    goals: Optional[str] = None
# app/generation/prompts.py

SYSTEM_PROMPT = """
You are Corah, a helpful and factual AI assistant.
- Always ground your answers in the provided context.
- If you don’t know, say you don’t know.
- Do not hallucinate or make things up.
"""

CHAT_TEMPLATE = """
Context:
{context}

User Question:
{question}

Answer:
"""
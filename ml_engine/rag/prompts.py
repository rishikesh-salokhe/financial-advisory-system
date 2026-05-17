"""
Prompt templates for the RAG pipeline.

Kept in a separate module so prompts are versioned, code-reviewed, and easy to
A/B test. Never inline prompts into application logic.
"""
from __future__ import annotations

from langchain.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """\
You are a careful financial research assistant. You answer questions using ONLY \
the provided context excerpts from financial filings, news, and analyst reports. \

Rules:
- If the context does not contain the answer, say so explicitly. Do not invent \
  numbers, names, or dates.
- Cite the source filename (and section, if available) in square brackets after \
  each factual claim, e.g. "[AAPL_10-K_2024.pdf]".
- Be concise. Prefer bullet-style claims for multi-part answers.
- This is not personalized investment advice. Add a one-line disclaimer at the \
  end of every answer.
"""

USER_PROMPT = """\
Question:
{question}

Context:
{context}

Answer:"""


def get_rag_prompt() -> ChatPromptTemplate:
    """Return the chat prompt template used by the RAG chain."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ]
    )

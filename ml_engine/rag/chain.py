"""
RAG chain assembly using the LCEL (LangChain Expression Language) style.

The chain is built once and reused across requests. It exposes a uniform
``invoke({"question": ..., "top_k": ...})`` interface that returns:

    {
        "answer": str,
        "source_documents": list[Document],
        "tokens_used": int | None,   # populated when the LLM reports usage
    }
"""
from __future__ import annotations

from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from loguru import logger

from backend.core.config import settings
from ml_engine.rag.prompts import get_rag_prompt
from ml_engine.rag.retriever import build_retriever


# ─── LLM factory ──────────────────────────────────────────────────────────


def get_chat_llm() -> BaseChatModel:
    """Return the chat LLM configured for the app. Swap providers here."""
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Either configure it or replace get_chat_llm() "
            "with a local model (e.g. Ollama, llama.cpp) before calling the chain."
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.openai_chat_model,
        api_key=settings.openai_api_key,
        temperature=0.1,
        timeout=60,
    )


# ─── Helpers ──────────────────────────────────────────────────────────────


def _format_docs(docs: list[Document]) -> str:
    """Render retrieved chunks for the prompt with source markers."""
    if not docs:
        return "(no relevant context found)"
    blocks: list[str] = []
    for i, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        header = f"[{i}] source={src}" + (f" page={page}" if page is not None else "")
        blocks.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(blocks)


# ─── Chain factory ────────────────────────────────────────────────────────


class RAGChain:
    """Callable wrapper that mirrors the legacy `RetrievalQA` shape.

    Using a class (rather than a raw LCEL chain) lets us return the retrieved
    documents alongside the answer in a single ``invoke`` call.
    """

    def __init__(self, store: FAISS, llm: BaseChatModel) -> None:
        self._store = store
        self._llm = llm
        self._prompt = get_rag_prompt()
        self._parser = StrOutputParser()

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        question: str = payload["question"]
        top_k: int | None = payload.get("top_k")
        # ``use_mmr=True`` switches retrieval to Maximum Marginal Relevance,
        # which actively penalizes redundancy across retrieved chunks. Useful
        # for comparison questions ("compare AAPL and MSFT…") that otherwise
        # tend to get K chunks all from the same document.
        search_type = "mmr" if payload.get("use_mmr") else "similarity"

        retriever = build_retriever(self._store, top_k=top_k, search_type=search_type)
        docs: list[Document] = retriever.invoke(question)
        logger.debug(f"Retrieved {len(docs)} chunk(s) for question (search={search_type})")

        context = _format_docs(docs)
        formatted = self._prompt.format_messages(question=question, context=context)
        response = self._llm.invoke(formatted)
        answer = self._parser.invoke(response)

        tokens_used: int | None = None
        usage = getattr(response, "response_metadata", {}).get("token_usage")
        if isinstance(usage, dict):
            tokens_used = usage.get("total_tokens")

        return {
            "answer": answer,
            "source_documents": docs,
            "tokens_used": tokens_used,
        }


def build_rag_chain(store: FAISS, llm: BaseChatModel | None = None) -> RAGChain:
    """Public factory used by the service layer."""
    return RAGChain(store=store, llm=llm or get_chat_llm())


# ─── LCEL pipe (alternative, for streaming use cases) ─────────────────────


def build_lcel_pipeline(store: FAISS, llm: BaseChatModel | None = None) -> Runnable:
    """A streaming-friendly LCEL chain.

    Use this when you want to ``.stream()`` tokens back to a websocket client.
    It returns only the answer string — wrap with a retriever side-channel if
    you also need source documents.
    """
    llm = llm or get_chat_llm()
    retriever = build_retriever(store)
    prompt = get_rag_prompt()

    return (
        {
            "context": retriever | RunnableLambda(_format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

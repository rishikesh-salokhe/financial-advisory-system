"""
Retriever utilities.

Right now we expose a plain similarity-search retriever. The interface is
intentionally narrow so we can later swap in MMR, contextual compression, or
hybrid (BM25 + dense) without touching the chain.
"""
from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever

from backend.core.config import settings


def build_retriever(
    store: FAISS,
    *,
    top_k: int | None = None,
    search_type: str = "similarity",
    score_threshold: float | None = None,
) -> BaseRetriever:
    """Construct a retriever over the given FAISS store.

    Parameters
    ----------
    store:
        A loaded FAISS vector store.
    top_k:
        Number of chunks to return. Defaults to ``settings.rag_top_k``.
    search_type:
        One of ``"similarity"``, ``"mmr"``, ``"similarity_score_threshold"``.
    score_threshold:
        Used only with ``"similarity_score_threshold"``.
    """
    k = top_k or settings.rag_top_k
    kwargs: dict[str, int | float] = {"k": k}
    if search_type == "similarity_score_threshold" and score_threshold is not None:
        kwargs["score_threshold"] = score_threshold
    if search_type == "mmr":
        kwargs["fetch_k"] = max(k * 4, 20)
        kwargs["lambda_mult"] = 0.5

    return store.as_retriever(search_type=search_type, search_kwargs=kwargs)

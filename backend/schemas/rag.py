"""Request/response models for the RAG-based query module."""
from __future__ import annotations

from pydantic import BaseModel, Field


class RAGQueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int | None = Field(None, ge=1, le=20, description="Override default retrieval depth")
    session_id: str | None = Field(None, description="Conversation id for multi-turn context")
    use_mmr: bool = Field(
        False,
        description=(
            "Use Maximum Marginal Relevance retrieval, which diversifies the "
            "retrieved chunks. Recommended for comparison questions that span "
            "multiple source documents."
        ),
    )


class RetrievedChunk(BaseModel):
    content: str
    source: str
    score: float | None = None
    metadata: dict[str, str | int | float] = Field(default_factory=dict)


class RAGQueryResponse(BaseModel):
    answer: str
    sources: list[RetrievedChunk]
    model: str
    tokens_used: int | None = None


class IngestRequest(BaseModel):
    paths: list[str] = Field(..., min_length=1, description="File or directory paths to ingest")
    recursive: bool = True


class IngestResponse(BaseModel):
    documents_added: int
    chunks_added: int
    vectorstore_size: int

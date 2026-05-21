"""
RAG service — thin wrapper around ``ml_engine.rag`` that adds persistence
(query/answer logs) and dependency injection.
"""
from __future__ import annotations

from datetime import datetime

from loguru import logger
from motor.motor_asyncio import AsyncIOMotorDatabase

from backend.core.config import settings
from backend.schemas.rag import (
    IngestRequest,
    IngestResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    RetrievedChunk,
)
from ml_engine.rag.chain import build_rag_chain
from ml_engine.rag.ingestion import ingest_paths
from ml_engine.rag.vectorstore import VectorStoreManager


class RAGService:
    COLLECTION = "rag_queries"

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db
        self._collection = db[self.COLLECTION]
        # The chain is cheap to build but expensive to recreate per request.
        # Cache lazily on first use; tests can patch this attribute.
        self._chain = None

    # ─── Querying ─────────────────────────────────────────────────────────

    async def query(self, req: RAGQueryRequest) -> RAGQueryResponse:
        logger.info(f"RAG query: '{req.question[:80]}...'")
        chain = self._get_chain()
        top_k = req.top_k or settings.rag_top_k
        result = chain.invoke({
            "question": req.question,
            "top_k": top_k,
            "use_mmr": req.use_mmr,
        })

        response = RAGQueryResponse(
            answer=result["answer"],
            sources=[
                RetrievedChunk(
                    content=doc.page_content,
                    source=str(doc.metadata.get("source", "unknown")),
                    score=doc.metadata.get("score"),
                    metadata={k: v for k, v in doc.metadata.items() if k != "score"},
                )
                for doc in result.get("source_documents", [])
            ],
            model=settings.openai_chat_model,
            tokens_used=result.get("tokens_used"),
        )

        # Fire-and-forget log of the interaction
        await self._collection.insert_one(
            {
                "question": req.question,
                "answer": response.answer,
                "session_id": req.session_id,
                "model": response.model,
                "created_at": datetime.utcnow(),
            }
        )
        return response

    # ─── Ingestion ────────────────────────────────────────────────────────

    async def ingest(self, req: IngestRequest) -> IngestResponse:
        logger.info(f"Ingesting {len(req.paths)} path(s) into vectorstore")
        stats = ingest_paths(req.paths, recursive=req.recursive)
        # Reset cached chain so the next query sees the fresh index.
        self._chain = None
        return IngestResponse(**stats)

    # ─── Internals ────────────────────────────────────────────────────────

    def _get_chain(self):
        if self._chain is None:
            store = VectorStoreManager().load_or_init()
            self._chain = build_rag_chain(store)
        return self._chain

"""RAG-based query endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from backend.api.deps import RAGSvc
from backend.schemas.common import APIResponse
from backend.schemas.rag import (
    IngestRequest,
    IngestResponse,
    RAGQueryRequest,
    RAGQueryResponse,
)

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post(
    "/query",
    response_model=APIResponse[RAGQueryResponse],
    summary="Ask a question against the indexed knowledge base",
)
async def query(req: RAGQueryRequest, service: RAGSvc) -> APIResponse[RAGQueryResponse]:
    result = await service.query(req)
    return APIResponse(data=result)


@router.post(
    "/ingest",
    response_model=APIResponse[IngestResponse],
    summary="Ingest one or more files / directories into the vectorstore",
)
async def ingest(req: IngestRequest, service: RAGSvc) -> APIResponse[IngestResponse]:
    result = await service.ingest(req)
    return APIResponse(data=result)

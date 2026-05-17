"""Per-request access log with timing and a request-id."""
from __future__ import annotations

import time
from uuid import uuid4

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("x-request-id") or uuid4().hex[:12]
        start = time.perf_counter()
        with logger.contextualize(request_id=request_id):
            try:
                response = await call_next(request)
            except Exception:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.exception(
                    f"{request.method} {request.url.path} → 500 in {elapsed_ms:.1f}ms"
                )
                raise
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                f"{request.method} {request.url.path} → {response.status_code} in {elapsed_ms:.1f}ms"
            )
            response.headers["x-request-id"] = request_id
            response.headers["x-response-time-ms"] = f"{elapsed_ms:.1f}"
            return response

"""
Domain exceptions and a single point for registering global handlers.

Pattern: route handlers raise *domain* exceptions; FastAPI converts them to
HTTP responses via the handlers registered in :func:`register_exception_handlers`.
This keeps service-layer code free of HTTP concerns.
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger


# ─── Domain exceptions ────────────────────────────────────────────────────


class AppException(Exception):
    """Base class for all expected application-level errors."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(AppException):
    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"


class ValidationError(AppException):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    code = "validation_error"


class UnauthorizedError(AppException):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "unauthorized"


class ExternalServiceError(AppException):
    """Raised when a downstream service (Yahoo, OpenAI, etc.) misbehaves."""

    status_code = status.HTTP_502_BAD_GATEWAY
    code = "external_service_error"


class ModelNotReadyError(AppException):
    """Forecaster / sentiment model has not been trained or loaded yet."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "model_not_ready"


# ─── Handler registration ─────────────────────────────────────────────────


def _error_payload(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": details or {}}}


def register_exception_handlers(app: FastAPI) -> None:
    """Attach JSON handlers for our domain exceptions + Pydantic validation."""

    @app.exception_handler(AppException)
    async def _handle_app_exception(_: Request, exc: AppException) -> JSONResponse:
        logger.warning(f"AppException [{exc.code}] {exc.message} | details={exc.details}")
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(exc.code, exc.message, exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        logger.info(f"RequestValidationError: {exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_error_payload(
                "validation_error",
                "Request validation failed",
                {"errors": exc.errors()},
            ),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_payload("internal_error", "An unexpected error occurred"),
        )

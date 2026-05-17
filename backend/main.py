"""
FastAPI application entry point.

Run locally:
    uvicorn backend.main:app --reload --port 8000

The application uses the modern ``lifespan`` context manager (over the
deprecated ``startup`` / ``shutdown`` events) to manage long-lived resources
like the MongoDB connection pool.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from loguru import logger

from backend.api.v1.router import api_router
from backend.core.config import settings
from backend.core.exceptions import register_exception_handlers
from backend.core.logging import configure_logging
from backend.db import mongodb
from backend.middleware.request_logging import RequestLoggingMiddleware


# ─── Lifespan ─────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001 — FastAPI requires the param
    """Acquire resources on startup, release them on shutdown."""
    configure_logging()
    logger.info(
        f"Starting {settings.app_name} v{settings.app_version} ({settings.app_env})"
    )

    await mongodb.connect()
    # Place to warm ML caches once they exist:
    # await rag_warmup()
    # await forecaster_registry.load_disk_artifacts()

    yield

    logger.info("Shutting down")
    await mongodb.close()


# ─── App factory ──────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """Construct the FastAPI app. Useful for tests that need a fresh instance."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.app_debug,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS — Streamlit dashboard and any web clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["x-request-id", "x-response-time-ms"],
    )

    # Custom middleware (added after CORS so it runs *inside* the CORS layer)
    app.add_middleware(RequestLoggingMiddleware)

    # Global exception → JSON converters
    register_exception_handlers(app)

    # API routes mounted under /api/v1
    app.include_router(api_router, prefix="/api/v1")

    # Friendly root redirect to the OpenAPI docs
    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    return app


# ASGI entry-point used by uvicorn / gunicorn.
app = create_app()

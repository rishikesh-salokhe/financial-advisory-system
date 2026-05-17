"""Liveness / readiness probes."""
from __future__ import annotations

from fastapi import APIRouter

from backend.core.config import settings
from backend.db.mongodb import mongo
from backend.schemas.common import APIResponse, HealthStatus

router = APIRouter(tags=["health"])


@router.get("/health", response_model=APIResponse[HealthStatus])
async def health() -> APIResponse[HealthStatus]:
    mongo_ok = False
    if mongo.client is not None:
        try:
            await mongo.client.admin.command("ping")
            mongo_ok = True
        except Exception:  # noqa: BLE001 — surfaced via the response, not raised
            mongo_ok = False

    return APIResponse(
        data=HealthStatus(
            status="ok" if mongo_ok else "degraded",
            version=settings.app_version,
            environment=settings.app_env,
            mongo_ok=mongo_ok,
        )
    )

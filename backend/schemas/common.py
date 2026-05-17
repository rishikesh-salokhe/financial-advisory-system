"""Shared response envelopes and primitive types."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    """Uniform success envelope used across all endpoints."""

    success: bool = True
    data: T
    meta: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthStatus(BaseModel):
    status: str
    version: str
    environment: str
    mongo_ok: bool

"""
Async MongoDB client managed by FastAPI's lifespan.

Why a module-level container rather than ``app.state``? Because services and
repositories are constructed at import time (or lazily via dependencies) and
shouldn't need a ``Request`` to reach the DB. The container's attributes are
populated in ``connect()`` and cleared in ``close()``.
"""
from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from backend.core.config import settings


@dataclass
class _MongoState:
    client: AsyncIOMotorClient | None = None
    db: AsyncIOMotorDatabase | None = None


mongo: _MongoState = _MongoState()


async def connect() -> None:
    """Open the connection pool and verify the server is reachable."""
    logger.info(f"Connecting to MongoDB at {settings.mongodb_uri}")
    mongo.client = AsyncIOMotorClient(
        settings.mongodb_uri,
        minPoolSize=settings.mongodb_min_pool_size,
        maxPoolSize=settings.mongodb_max_pool_size,
        serverSelectionTimeoutMS=5_000,
        uuidRepresentation="standard",
    )
    # Force a round-trip so we fail fast if Mongo isn't reachable.
    await mongo.client.admin.command("ping")
    mongo.db = mongo.client[settings.mongodb_db_name]
    logger.info(f"Connected to MongoDB database '{settings.mongodb_db_name}'")

    await _ensure_indexes()


async def close() -> None:
    """Close the connection pool cleanly on shutdown."""
    if mongo.client is not None:
        logger.info("Closing MongoDB connection")
        mongo.client.close()
        mongo.client = None
        mongo.db = None


def get_db() -> AsyncIOMotorDatabase:
    """Return the active database handle. Raises if called pre-startup."""
    if mongo.db is None:
        raise RuntimeError("MongoDB is not initialized — was connect() called?")
    return mongo.db


async def _ensure_indexes() -> None:
    """Create indexes that the app relies on. Idempotent."""
    db = get_db()
    await db["users"].create_index("email", unique=True)
    await db["portfolios"].create_index([("user_id", 1), ("created_at", -1)])
    await db["forecasts"].create_index([("ticker", 1), ("created_at", -1)])
    await db["sentiment_runs"].create_index([("ticker", 1), ("created_at", -1)])
    logger.debug("MongoDB indexes ensured")

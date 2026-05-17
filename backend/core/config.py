"""
Centralized application settings.

Uses ``pydantic-settings`` so values can be overridden by environment variables
(or a ``.env`` file) without touching code. Import the singleton ``settings``
elsewhere — never read ``os.environ`` directly in app code.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Strongly-typed configuration loaded from environment / .env."""

    # ─── App ──────────────────────────────────────────────────────────────
    app_name: str = "Financial Advisory & Risk Intelligence"
    app_env: Literal["development", "staging", "production"] = "development"
    app_debug: bool = True
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_version: str = "0.1.0"
    log_level: str = "INFO"

    # ─── Security ─────────────────────────────────────────────────────────
    secret_key: str = "change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # ─── CORS ─────────────────────────────────────────────────────────────
    # Stored as a comma-separated string to dodge pydantic-settings' JSON
    # decoding of list fields. Use ``settings.cors_origins_list`` to read
    # the parsed list.
    cors_origins: str = "http://localhost:8501"

    # ─── MongoDB ──────────────────────────────────────────────────────────
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "financial_advisory"
    mongodb_min_pool_size: int = 5
    mongodb_max_pool_size: int = 50

    # ─── LLM / Embeddings ─────────────────────────────────────────────────
    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    use_local_embeddings: bool = False
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ─── RAG ──────────────────────────────────────────────────────────────
    vectorstore_dir: Path = PROJECT_ROOT / "data" / "vectorstore"
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 150
    rag_top_k: int = 4

    # ─── External data ────────────────────────────────────────────────────
    news_api_key: str | None = None
    finnhub_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Parsed list view of the comma-separated ``cors_origins`` string."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor — instantiate exactly once per process."""
    return Settings()


settings = get_settings()

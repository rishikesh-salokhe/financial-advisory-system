"""Dashboard configuration loaded from environment."""
from __future__ import annotations

import os

BACKEND_BASE_URL: str = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")
API_PREFIX: str = "/api/v1"
REQUEST_TIMEOUT_SECONDS: int = int(os.getenv("DASHBOARD_TIMEOUT", "120"))

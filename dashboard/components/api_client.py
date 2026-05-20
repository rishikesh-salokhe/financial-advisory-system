"""
HTTP client for the FastAPI backend.

Wraps httpx with a typed surface so Streamlit pages stay readable. All errors
are normalized to a single ``APIError`` so pages can render a friendly message.
"""
from __future__ import annotations

from typing import Any

import httpx

from dashboard.config import API_PREFIX, BACKEND_BASE_URL, REQUEST_TIMEOUT_SECONDS


class APIError(Exception):
    """Raised when the backend returns a non-2xx response."""

    def __init__(self, status_code: int, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.details = details or {}


class APIClient:
    def __init__(self, base_url: str = BACKEND_BASE_URL, timeout: int = REQUEST_TIMEOUT_SECONDS):
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    # ─── Generic ─────────────────────────────────────────────────────────

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        url = f"{API_PREFIX}{path}"
        try:
            r = self._client.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            raise APIError(0, f"Network error contacting backend: {exc}") from exc

        if r.status_code >= 400:
            try:
                payload = r.json().get("error", {})
            except ValueError:
                payload = {"message": r.text}
            raise APIError(
                r.status_code,
                payload.get("message", f"HTTP {r.status_code}"),
                payload.get("details"),
            )
        return r.json()["data"]

    # ─── Domain helpers ───────────────────────────────────────────────────

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def forecast(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/forecasting/predict", json=payload)

    def rag_query(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/rag/query", json=payload)

    def rag_ingest(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/rag/ingest", json=payload)

    def sentiment(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/sentiment/analyze", json=payload)

    def risk_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/risk/profile", json=payload)

    def asset_risk(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/risk/asset", json=payload)

    def portfolio_optimize(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/risk/optimize", json=payload)


# Streamlit pages share a single client per session.
_client: APIClient | None = None


def get_client() -> APIClient:
    global _client
    if _client is None:
        _client = APIClient()
    return _client

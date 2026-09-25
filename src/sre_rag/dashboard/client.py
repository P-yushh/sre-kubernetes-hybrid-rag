"""Typed HTTP client used by the Streamlit dashboard."""

from types import TracebackType
from typing import Any

import httpx
from pydantic import ValidationError

from sre_rag.api.health import HealthResponse
from sre_rag.api.query import QueryResult


class DashboardAPIError(RuntimeError):
    """User-safe error raised when the dashboard cannot obtain a valid API response."""


class DashboardClient:
    """Call FastAPI without coupling UI rendering to transport details."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("API base URL must use HTTP or HTTPS")
        if timeout_seconds <= 0:
            raise ValueError("API timeout must be positive")
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._owns_client = client is None
        self._client = client or httpx.Client()

    def __enter__(self) -> "DashboardClient":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        if self._owns_client:
            self._client.close()

    def health(self) -> HealthResponse:
        payload = self._request("GET", "/healthz")
        try:
            return HealthResponse.model_validate(payload)
        except ValidationError as error:
            raise DashboardAPIError("The API returned an invalid health response.") from error

    def query(self, question: str) -> QueryResult:
        payload = self._request("POST", "/v1/query", json_body={"question": question})
        try:
            return QueryResult.model_validate(payload)
        except ValidationError as error:
            raise DashboardAPIError("The API returned an invalid query response.") from error

    def _request(self, method: str, path: str, *, json_body: object | None = None) -> Any:
        try:
            response = self._client.request(
                method,
                f"{self._base_url}{path}",
                timeout=self._timeout_seconds,
                json=json_body,
            )
        except httpx.TimeoutException as error:
            raise DashboardAPIError("The API request timed out.") from error
        except httpx.RequestError as error:
            raise DashboardAPIError(
                "The API is unreachable. Start FastAPI and try again."
            ) from error

        if response.is_error:
            raise DashboardAPIError(_error_message(response))
        try:
            return response.json()
        except ValueError as error:
            raise DashboardAPIError("The API returned invalid JSON.") from error


def _error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, dict):
            message = detail.get("message")
            if isinstance(message, str):
                return message
        if isinstance(detail, str):
            return detail
    return f"The API request failed with status {response.status_code}."

"""Tests for typed dashboard-to-API communication."""

import httpx
import pytest

from sre_rag.dashboard.client import DashboardAPIError, DashboardClient


def _client(handler: httpx.MockTransport) -> tuple[DashboardClient, httpx.Client]:
    http_client = httpx.Client(transport=handler)
    return DashboardClient("http://api.test/", client=http_client), http_client


def test_client_parses_health_response() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            request=request,
            json={
                "status": "ok",
                "service": "SRE RAG",
                "version": "0.1.0",
                "environment": "test",
            },
        )
    )
    client, http_client = _client(transport)

    with http_client:
        health = client.health()

    assert health.status == "ok"
    assert health.environment == "test"


def test_client_parses_guarded_query_response() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            request=request,
            json={
                "query_id": "41c24871-2907-4c4b-9779-1760c10a977f",
                "latency_ms": 42.5,
                "response": {
                    "outcome": "refusal",
                    "code": "INSUFFICIENT_CONTEXT",
                    "message": "Not enough evidence.",
                },
                "contexts": [],
                "generation": None,
            },
        )
    )
    client, http_client = _client(transport)

    with http_client:
        result = client.query("unknown issue")

    assert result.response.outcome == "refusal"
    assert result.latency_ms == 42.5


def test_client_surfaces_structured_api_error() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            503,
            request=request,
            json={
                "detail": {
                    "code": "WORKFLOW_UNAVAILABLE",
                    "message": "The RAG workflow is unavailable.",
                }
            },
        )
    )
    client, http_client = _client(transport)

    with http_client, pytest.raises(DashboardAPIError, match="workflow is unavailable"):
        client.query("Why OOMKilled?")


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(500), "status 500"),
        (httpx.Response(200, content=b"not-json"), "invalid JSON"),
        (httpx.Response(200, json={"status": "wrong"}), "invalid health response"),
    ],
)
def test_client_rejects_invalid_api_responses(response: httpx.Response, message: str) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            response.status_code,
            request=request,
            content=response.content,
            headers=response.headers,
        )
    )
    client, http_client = _client(transport)

    with http_client, pytest.raises(DashboardAPIError, match=message):
        client.health()


def test_client_reports_network_timeout() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    client, http_client = _client(httpx.MockTransport(timeout))

    with http_client, pytest.raises(DashboardAPIError, match="timed out"):
        client.health()


@pytest.mark.parametrize(("base_url", "timeout"), [("localhost:8000", 1), ("http://x", 0)])
def test_client_rejects_invalid_configuration(base_url: str, timeout: float) -> None:
    with pytest.raises(ValueError):
        DashboardClient(base_url, timeout_seconds=timeout)

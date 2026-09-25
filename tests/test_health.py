"""Tests for the service health endpoint."""

import pytest
from httpx import ASGITransport, AsyncClient

from sre_rag.config import Settings
from sre_rag.main import create_app


@pytest.mark.anyio
async def test_health_endpoint_reports_service_metadata() -> None:
    settings = Settings(
        app_name="Test SRE RAG",
        app_version="test-version",
        environment="test",
    )
    transport = ASGITransport(app=create_app(settings))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "Test SRE RAG",
        "version": "test-version",
        "environment": "test",
    }


@pytest.mark.anyio
async def test_openapi_schema_exposes_health_endpoint() -> None:
    transport = ASGITransport(app=create_app(Settings(environment="test")))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/openapi.json")

    assert response.status_code == 200
    assert "/healthz" in response.json()["paths"]
    assert "/v1/query" in response.json()["paths"]

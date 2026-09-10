"""Service health endpoint."""

from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel

from sre_rag.config import Settings

router = APIRouter(tags=["system"])


class HealthResponse(BaseModel):
    """Public health status returned by the service."""

    status: Literal["ok"]
    service: str
    version: str
    environment: str


@router.get("/healthz", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Confirm that the API process is alive and its configuration is loaded."""

    settings: Settings = request.app.state.settings
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )

"""Service health endpoint."""

from typing import Literal

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
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


@router.get("/readyz", response_model=None)
async def readiness(request: Request) -> dict[str, str] | JSONResponse:
    """Report whether the immutable index and query workflow were loaded."""

    if request.app.state.query_workflow is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unavailable", "reason": "workflow_not_loaded"},
        )
    return {"status": "ready"}

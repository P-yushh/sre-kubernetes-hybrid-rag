"""Prometheus scrape endpoint."""

from fastapi import APIRouter, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest

router = APIRouter(tags=["system"])


@router.get("/metrics", include_in_schema=False)
async def metrics(request: Request) -> Response:
    """Expose the application registry in Prometheus text format."""

    registry: CollectorRegistry = request.app.state.metrics_registry
    return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

"""FastAPI application factory."""

from fastapi import FastAPI
from prometheus_client import REGISTRY, CollectorRegistry

from sre_rag.api.health import router as health_router
from sre_rag.api.metrics import router as metrics_router
from sre_rag.api.query import QueryWorkflow
from sre_rag.api.query import router as query_router
from sre_rag.config import Settings, get_settings


def create_app(
    settings: Settings | None = None,
    metrics_registry: CollectorRegistry | None = None,
    query_workflow: QueryWorkflow | None = None,
) -> FastAPI:
    """Build an application instance with validated runtime settings."""

    resolved_settings = settings or get_settings()
    application = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
        description="Observable hybrid search and grounded generation for SRE documentation.",
    )
    application.state.settings = resolved_settings
    application.state.metrics_registry = metrics_registry or REGISTRY
    application.state.query_workflow = query_workflow
    application.include_router(health_router)
    application.include_router(metrics_router)
    application.include_router(query_router)
    return application


app = create_app()

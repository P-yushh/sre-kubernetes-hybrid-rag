"""FastAPI application factory."""

from fastapi import FastAPI

from sre_rag.api.health import router as health_router
from sre_rag.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build an application instance with validated runtime settings."""

    resolved_settings = settings or get_settings()
    application = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
        description="Observable hybrid search and grounded generation for SRE documentation.",
    )
    application.state.settings = resolved_settings
    application.include_router(health_router)
    return application


app = create_app()

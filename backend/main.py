"""
VEXA FastAPI Application

Entry point for the backend service.
Registers all routers and configures middleware.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.agent_runs import router as agent_runs_router
from backend.api.health import router as health_router
from backend.api.workspaces import router as workspaces_router
from backend.core.config import get_settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Handle startup and shutdown events."""
    settings = get_settings()
    logger.info(
        "VEXA backend starting | version=%s env=%s",
        settings.app_version,
        settings.environment.value,
    )
    yield
    logger.info("VEXA backend shutting down")


def create_app() -> FastAPI:
    """Factory function that creates and configures the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "VEXA — Architecture-Aware Autonomous Multi-Agent "
            "Software Engineering System"
        ),
        lifespan=_lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — restrict in production via environment config
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(health_router)
    app.include_router(workspaces_router)
    app.include_router(agent_runs_router)

    return app


app = create_app()


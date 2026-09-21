"""
Pytest configuration and shared fixtures for VEXA test suite.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Force testing environment before any app module is imported
os.environ.setdefault("ENVIRONMENT", "testing")
os.environ.setdefault("LOG_LEVEL", "WARNING")

# ---------------------------------------------------------------------------
# M1 fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def app():
    """Return the FastAPI application instance."""
    from backend.main import create_app
    return create_app()


@pytest.fixture(scope="session")
def client(app):
    """Return a synchronous TestClient wrapping the app."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def settings():
    """Return the application settings (uses testing environment)."""
    # Clear cache so it re-reads with the env vars set above
    from backend.core.config import get_settings
    get_settings.cache_clear()
    return get_settings()


# ---------------------------------------------------------------------------
# M2 fixtures
# ---------------------------------------------------------------------------

FIXTURE_PROJECT = Path(__file__).parent / "fixtures" / "sample_project"


@pytest.fixture(scope="session")
def tmp_workspace_root(tmp_path_factory) -> Path:
    """Session-scoped temporary directory used as the workspace root."""
    return tmp_path_factory.mktemp("vexa_workspaces")


@pytest.fixture(scope="session")
def tool_service(tmp_workspace_root: Path):
    """Return a ToolService backed by the session temp workspace root."""
    from backend.tools.service import ToolService
    return ToolService(workspace_root=str(tmp_workspace_root))


@pytest.fixture(scope="session")
def workspace_id(tool_service) -> str:
    """Create a blank workspace and return its ID."""
    return tool_service.create_workspace()


@pytest.fixture(scope="session")
def sample_project_workspace(tool_service) -> str:
    """
    Create a workspace pre-populated with the sample_project fixture files.

    Returns the workspace_id string.
    """
    ws_id = tool_service.create_workspace()
    from backend.tools.workspace import WorkspaceManager
    mgr = WorkspaceManager(workspace_root=tool_service._manager.root)
    repo_dir = mgr.require(ws_id)
    # Copy fixture files into the workspace repository directory
    shutil.copytree(str(FIXTURE_PROJECT), str(repo_dir), dirs_exist_ok=True)
    return ws_id

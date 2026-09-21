"""
VEXA Backend Configuration

Reads all settings from environment variables.
No secrets are hardcoded.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    development = "development"
    testing = "testing"
    production = "production"


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Application
    # -------------------------------------------------------------------------
    app_name: str = "VEXA"
    app_version: str = "0.1.0"
    environment: Environment = Environment.development
    debug: bool = False

    # -------------------------------------------------------------------------
    # API server
    # -------------------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # -------------------------------------------------------------------------
    # Database (optional — app starts without it)
    # -------------------------------------------------------------------------
    database_url: str | None = Field(
        default=None,
        description=(
            "PostgreSQL connection URL, e.g. "
            "postgresql+asyncpg://user:password@localhost:5432/vexa"
        ),
    )

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = False

    # -------------------------------------------------------------------------
    # VEXA research settings
    # -------------------------------------------------------------------------
    max_iterations: int = Field(
        default=5,
        description="Maximum debug/repair iterations per run before giving up",
    )

    # -------------------------------------------------------------------------
    # M2 — Workspace & Tool Layer
    # -------------------------------------------------------------------------
    workspace_root: str = Field(
        default="workspaces",
        description=(
            "Root directory for isolated project workspaces. "
            "Relative paths are resolved from the current working directory. "
            "Set an absolute path in production."
        ),
    )

    sandbox_backend: str = Field(
        default="local",
        description="Execution sandbox backend. 'local' = subprocess on host. "
                    "'docker' = reserved for future Docker isolation.",
    )

    # Execution limits
    tool_exec_timeout_seconds: int = Field(
        default=120,
        description="Maximum seconds a sandboxed command may run before it is killed.",
    )
    tool_max_output_bytes: int = Field(
        default=1_048_576,  # 1 MiB
        description="Maximum captured stdout+stderr bytes per execution.",
    )
    tool_max_file_bytes: int = Field(
        default=524_288,  # 512 KiB
        description="Maximum file size that read_file will load.",
    )
    tool_max_list_files: int = Field(
        default=500,
        description="Maximum number of files returned by list_files.",
    )
    tool_max_search_results: int = Field(
        default=200,
        description="Maximum number of matches returned by search_code.",
    )

    # -------------------------------------------------------------------------
    # M3 — Agent system (see also backend/agents/config.py for full agent settings)
    # -------------------------------------------------------------------------
    # These mirror AgentSettings for documentation purposes.
    # AgentSettings is loaded independently with VEXA_ prefix.
    vexa_llm_model: str = Field(
        default="gpt-4o-mini",
        description="LiteLLM model string for all agents",
    )
    vexa_debug_max_iter: int = Field(
        default=3,
        description="Maximum repair iterations in the Debugger loop",
    )

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return upper

    @field_validator("sandbox_backend")
    @classmethod
    def _validate_sandbox_backend(cls, v: str) -> str:
        allowed = {"local", "docker"}
        lower = v.lower()
        if lower not in allowed:
            raise ValueError(f"sandbox_backend must be one of {allowed}")
        return lower


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()


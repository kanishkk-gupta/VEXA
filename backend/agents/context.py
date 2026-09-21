"""
VEXA M3 — RunContext

Lightweight execution context threaded through the agent pipeline.
Designed to be expanded by M8 observability without schema replacement.

The RunContext:
- carries shared state across agents
- records timing and file-change events
- provides the workspace_id reference for tool calls
- will be persisted to the database in M8
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from backend.core.logging import get_logger

logger = get_logger(__name__)


class AgentEvent(BaseModel):
    """A single structured log event within a run."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    agent: str
    event: str
    detail: str = ""
    duration_ms: float | None = None


class RunContext:
    """
    Mutable execution context threaded through the agent pipeline.

    Not a Pydantic model (needs mutable methods). Serialised to dict for
    API responses.

    Architecture is set at creation time (defaults to 'sequential').
    M4/M5/M6 will pass their architecture names.
    """

    def __init__(
        self,
        workspace_id: str,
        requirement: str,
        architecture: str = "sequential",
        run_id: str | None = None,
    ) -> None:
        self.run_id: str = run_id or str(uuid.uuid4())
        self.workspace_id: str = workspace_id
        self.requirement: str = requirement
        self.architecture: str = architecture
        self.execution_mode: str = "mock"
        self.provider: str | None = None
        self.model: str | None = None
        self.current_agent: str = "none"
        self.started_at: datetime = datetime.now(UTC)
        self.finished_at: datetime | None = None

        # Mutable state accumulated during the run
        self.files_changed: list[str] = []
        self.test_attempts: int = 0
        self.debug_iterations: int = 0
        self.errors: list[str] = []
        self.events: list[AgentEvent] = []

        # Usage and Cost Tracking
        self.usage_available: bool = False
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.total_tokens: int = 0
        self.estimated_cost_usd: float | None = None

    # ------------------------------------------------------------------
    # Event recording
    # ------------------------------------------------------------------

    def record(self, agent: str, event: str, detail: str = "", duration_ms: float | None = None) -> None:
        """Record a structured event. This is the foundation for M8 tracing."""
        ev = AgentEvent(agent=agent, event=event, detail=detail, duration_ms=duration_ms)
        self.events.append(ev)
        if duration_ms is not None:
            logger.info("run=%s agent=%s event=%s duration_ms=%.1f | %s", self.run_id, agent, event, duration_ms, detail)
        else:
            logger.info("run=%s agent=%s event=%s | %s", self.run_id, agent, event, detail)

    def agent_started(self, agent_name: str) -> None:
        self.current_agent = agent_name
        self.record(agent_name, "started")

    def agent_completed(self, agent_name: str, duration_ms: float = 0.0) -> None:
        self.record(agent_name, "completed", duration_ms=duration_ms)

    def tool_invoked(self, agent_name: str, tool: str, detail: str = "") -> None:
        self.record(agent_name, "tool_invoked", f"{tool}: {detail}")

    def tool_failed(self, agent_name: str, tool: str, error: str) -> None:
        self.record(agent_name, "tool_failed", f"{tool}: {error}")
        self.errors.append(f"{agent_name}:{tool} — {error}")

    def file_changed(self, path: str) -> None:
        if path not in self.files_changed:
            self.files_changed.append(path)

    def test_attempted(self) -> None:
        self.test_attempts += 1
        self.record("Tester", "test_started", f"attempt #{self.test_attempts}")

    def debug_iterated(self) -> None:
        self.debug_iterations += 1
        self.record("Debugger", "debug_iteration", f"iteration #{self.debug_iterations}")

    def finish(self) -> None:
        self.finished_at = datetime.now(UTC)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "workspace_id": self.workspace_id,
            "requirement": self.requirement,
            "architecture": self.architecture,
            "execution_mode": self.execution_mode,
            "provider": self.provider,
            "model": self.model,
            "current_agent": self.current_agent,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "files_changed": self.files_changed,
            "test_attempts": self.test_attempts,
            "debug_iterations": self.debug_iterations,
            "errors": self.errors,
            "usage_available": self.usage_available,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "events": [e.model_dump(mode="json") for e in self.events],
        }

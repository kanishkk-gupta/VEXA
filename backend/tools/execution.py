"""
VEXA Tool Layer — Execution Engine

SandboxExecutor hierarchy:
  SandboxExecutor (abstract base)
  └── LocalSandboxExecutor   — subprocess on the host, workspace-bounded
  └── DockerSandboxExecutor  — interface stub; Docker isolation for future use

The concrete executor is selected by the ``sandbox_backend`` configuration key.
Agents must NEVER execute arbitrary commands. The executor enforces:

  - cwd = workspace repository directory
  - configurable hard timeout (SIGKILL after timeout)
  - stdout / stderr capture with byte limit
  - structured ExecResult return

Allowlisted command prefixes guard against obvious dangerous invocations.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path

from backend.core.config import get_settings
from backend.core.logging import get_logger
from backend.tools.models import ExecResult
from backend.tools.workspace import WorkspaceManager, WorkspaceNotFoundError

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Command allowlist
# ---------------------------------------------------------------------------
# Only commands whose executable resolves to one of these names are permitted.
# This is a defence-in-depth layer, not a security perimeter on its own —
# the Docker sandbox provides stronger isolation when available.

_ALLOWED_EXECUTABLES: frozenset[str] = frozenset({
    "python", "python3", "python.exe",
    "pytest", "pytest.exe",
    "git", "git.exe",
    "pip", "pip3", "pip.exe",
    "ruff", "ruff.exe",
    "mypy", "mypy.exe",
})


def _check_allowlist(argv: list[str]) -> str | None:
    """
    Return an error message if the command is not allowed, else None.

    Only the executable (argv[0]) is checked. The intention is to block
    obviously dangerous invocations (rm, format, shutdown, etc.) while
    permitting the development toolchain.
    """
    if not argv:
        return "Empty command"
    exe = Path(argv[0]).name.lower()
    if exe not in _ALLOWED_EXECUTABLES:
        return (
            f"Command '{exe}' is not in the permitted executable list. "
            f"Allowed: {sorted(_ALLOWED_EXECUTABLES)}"
        )
    return None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class SandboxExecutor(ABC):
    """Abstract executor interface. Subclasses differ in isolation level."""

    @abstractmethod
    def run(
        self,
        argv: list[str],
        workspace_id: str,
        *,
        timeout: int | None = None,
        env_extra: dict[str, str] | None = None,
    ) -> ExecResult:
        """
        Execute *argv* in the workspace's repository directory.

        Parameters
        ----------
        argv:
            Command + arguments, e.g. ["pytest", "tests/", "-v"]
        workspace_id:
            Workspace identifier. The cwd will be set to its repository dir.
        timeout:
            Seconds before the process is killed. None uses the config default.
        env_extra:
            Additional environment variables (merged with a sanitised host env).

        Returns
        -------
        ExecResult with stdout, stderr, exit_code, duration_ms, timed_out.
        """


# ---------------------------------------------------------------------------
# Local implementation
# ---------------------------------------------------------------------------


class LocalSandboxExecutor(SandboxExecutor):
    """
    Runs commands directly on the host using subprocess.

    **Isolation level:** Limited. The process runs as the current user inside
    the workspace directory. The PATH and most environment variables are
    inherited, minus known sensitive variables.

    For production use, replace this with DockerSandboxExecutor which provides
    kernel-level namespace isolation.
    """

    _SCRUB_ENV_KEYS: frozenset[str] = frozenset({
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
        "DATABASE_URL", "SECRET_KEY", "AWS_SECRET_ACCESS_KEY",
        "AWS_ACCESS_KEY_ID",
    })

    def __init__(self, manager: WorkspaceManager) -> None:
        self._mgr = manager
        self._settings = get_settings()

    def run(
        self,
        argv: list[str],
        workspace_id: str,
        *,
        timeout: int | None = None,
        env_extra: dict[str, str] | None = None,
    ) -> ExecResult:
        # Guard: allowlist check
        deny_reason = _check_allowlist(argv)
        if deny_reason:
            return ExecResult(
                success=False,
                operation="exec",
                message=f"Command rejected: {deny_reason}",
                exit_code=None,
                error=deny_reason,
            )

        # Guard: workspace must exist
        try:
            cwd = self._mgr.require(workspace_id)
        except WorkspaceNotFoundError as exc:
            return ExecResult(
                success=False,
                operation="exec",
                message=str(exc),
                exit_code=None,
                error=str(exc),
            )

        effective_timeout = timeout if timeout is not None else self._settings.tool_exec_timeout_seconds
        max_out = self._settings.tool_max_output_bytes

        # Build a sanitised environment
        import os
        env = {k: v for k, v in os.environ.items() if k not in self._SCRUB_ENV_KEYS}
        if env_extra:
            env.update(env_extra)

        logger.debug("Executing %s | cwd=%s | timeout=%ss", argv, cwd, effective_timeout)

        t0 = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(
                argv,
                cwd=str(cwd),
                capture_output=True,
                timeout=effective_timeout,
                env=env,
            )
            stdout = proc.stdout.decode("utf-8", errors="replace")
            stderr = proc.stderr.decode("utf-8", errors="replace")
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            stdout = ""
            stderr = f"Process killed after {effective_timeout}s timeout"
            exit_code = -1
        except FileNotFoundError as exc:
            return ExecResult(
                success=False,
                operation="exec",
                message=f"Executable not found: {argv[0]}",
                exit_code=None,
                error=str(exc),
                duration_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception as exc:
            return ExecResult(
                success=False,
                operation="exec",
                message=f"Execution failed: {exc}",
                exit_code=None,
                error=str(exc),
                duration_ms=(time.monotonic() - t0) * 1000,
            )

        duration_ms = (time.monotonic() - t0) * 1000

        # Truncate oversized output
        def _truncate(text: str, limit: int) -> str:
            encoded = text.encode("utf-8")
            if len(encoded) > limit:
                truncated = encoded[:limit].decode("utf-8", errors="replace")
                return truncated + f"\n[...output truncated at {limit} bytes]"
            return text

        half = max_out // 2
        stdout = _truncate(stdout, half)
        stderr = _truncate(stderr, half)

        success = (exit_code == 0) and not timed_out
        return ExecResult(
            success=success,
            operation="exec",
            message="OK" if success else ("Timed out" if timed_out else f"Exit code {exit_code}"),
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            duration_ms=duration_ms,
        )


# ---------------------------------------------------------------------------
# Docker stub — future isolation backend
# ---------------------------------------------------------------------------


class DockerSandboxExecutor(SandboxExecutor):
    """
    Future Docker-based isolation backend.

    When implemented, this will:
    - Mount the workspace repository as a read-write volume
    - Use a locked-down base image (python:3.11-slim or equivalent)
    - Drop all Linux capabilities except those needed for test execution
    - Apply resource limits (CPU, memory, no network by default)
    - Remove the container after each execution

    This is the recommended production sandbox for VEXA.
    Implement in Milestone 2b or a dedicated sandbox milestone.
    """

    def run(self, argv: list[str], workspace_id: str, **kwargs) -> ExecResult:  # type: ignore[override]
        raise NotImplementedError(
            "DockerSandboxExecutor is not yet implemented. "
            "Set SANDBOX_BACKEND=local to use LocalSandboxExecutor."
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_executor(manager: WorkspaceManager) -> SandboxExecutor:
    """
    Return the configured SandboxExecutor based on SANDBOX_BACKEND env var.
    """
    backend = get_settings().sandbox_backend
    if backend == "docker":
        logger.warning("Docker sandbox selected but not yet implemented; falling back to local")
        return LocalSandboxExecutor(manager)
    return LocalSandboxExecutor(manager)

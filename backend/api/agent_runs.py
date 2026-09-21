"""
VEXA M3 — Agent Run API Endpoints

POST /agent-runs       Submit a new baseline sequential agent run
GET  /agent-runs/{id}  (stub — in-memory only for M3; DB persistence in M4+)

Note: M3 runs are synchronous (blocking until completion).
      Async/streaming is deferred to a later milestone.
"""

from __future__ import annotations

import uuid
from typing import Any
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field

from backend.agents.contracts import AgentStatus, RunResult
from backend.agents.context import RunContext
from backend.agents.crew import SoftwareEngineeringCrew
from backend.core.logging import get_logger
from backend.tools.service import ToolService

logger = get_logger(__name__)
router = APIRouter(prefix="/agent-runs", tags=["agent-runs"])

# Single ToolService for the lifetime of the process (same as workspace router)
_svc = ToolService()

# In-memory run store (M4+ will replace with DB persistence)
# We store either a RunResult (for completed synchronous runs) or a RunContext (for active/async runs)
_run_store: dict[str, RunResult | RunContext] = {}


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class AgentRunRequest(BaseModel):
    workspace_id: str = Field(description="ID of the workspace to operate on")
    requirement: str = Field(
        min_length=10,
        max_length=4096,
        description="Natural-language software task requirement",
    )
    execution_mode: str = Field(
        default="mock",
        description="Mode of execution: 'mock' (deterministic logic tests) or 'real' (live LLM).",
    )
    llm_model: str | None = Field(default=None, description="Optional LiteLLM model string override")
    llm_api_key: str | None = Field(default=None, description="Optional primary API key override")
    llm_fallback_api_key: str | None = Field(default=None, description="Optional fallback API key override")


class AgentRunResponse(BaseModel):
    run_id: str
    workspace_id: str
    execution_mode: str
    architecture: str
    status: str
    total_duration_ms: float
    files_changed: list[str]
    debug_iterations: int
    errors: list[str]
    result: dict  # Full RunResult or RunContext serialised — typed access via SDK later


class AgentRunListResponse(BaseModel):
    runs: list[AgentRunResponse]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("", status_code=status.HTTP_201_CREATED, response_model=AgentRunResponse)
def submit_agent_run(body: AgentRunRequest) -> AgentRunResponse:
    """
    Submit a software engineering task to the baseline sequential agent crew.

    The call is synchronous — it blocks until all agents have completed.
    Returns the full structured result of the run.
    """
    # Validate workspace
    if not _svc.workspace_exists(body.workspace_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{body.workspace_id}' not found",
        )

    logger.info("agent_run_requested | workspace=%s mode=%s req=%s", body.workspace_id, body.execution_mode, body.requirement[:80])

    try:
        from backend.agents.config import get_agent_settings
        settings = get_agent_settings()
        if body.llm_model:
            settings.llm_model = body.llm_model
        if body.llm_api_key:
            settings.llm_api_key = body.llm_api_key
        if body.llm_fallback_api_key:
            settings.llm_fallback_api_key = body.llm_fallback_api_key

        crew = SoftwareEngineeringCrew(tool_service=_svc, settings=settings)
        run_result = crew.run_task(
            workspace_id=body.workspace_id,
            requirement=body.requirement,
            execution_mode=body.execution_mode,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    # Store in memory for GET endpoint
    _run_store[run_result.run_id] = run_result

    return AgentRunResponse(
        run_id=run_result.run_id,
        workspace_id=run_result.workspace_id,
        execution_mode=run_result.execution_mode,
        architecture=run_result.architecture,
        status=run_result.status.value,
        total_duration_ms=run_result.total_duration_ms,
        files_changed=run_result.files_changed,
        debug_iterations=run_result.debug_iterations,
        errors=run_result.errors,
        result=run_result.model_dump(mode="json"),
    )

def _execute_async_run(ctx: RunContext, crew: SoftwareEngineeringCrew, execution_mode: str) -> None:
    try:
        run_result = crew.run_task(
            workspace_id=ctx.workspace_id,
            requirement=ctx.requirement,
            execution_mode=execution_mode,
            ctx=ctx
        )
        _run_store[ctx.run_id] = run_result
    except Exception as e:
        logger.error(f"Async run {ctx.run_id} failed: {e}")
        ctx.errors.append(str(e))
        ctx.finish()

@router.post("/async", status_code=status.HTTP_202_ACCEPTED, response_model=AgentRunResponse)
def submit_agent_run_async(body: AgentRunRequest, bg_tasks: BackgroundTasks) -> AgentRunResponse:
    """
    Submit a task asynchronously for frontend polling.
    """
    if not _svc.workspace_exists(body.workspace_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{body.workspace_id}' not found",
        )

    run_id = str(uuid.uuid4())
    ctx = RunContext(workspace_id=body.workspace_id, requirement=body.requirement, run_id=run_id)
    ctx.execution_mode = body.execution_mode
    _run_store[run_id] = ctx
    
    from backend.agents.config import get_agent_settings
    settings = get_agent_settings()
    if body.llm_model:
        settings.llm_model = body.llm_model
    if body.llm_api_key:
        settings.llm_api_key = body.llm_api_key
    if body.llm_fallback_api_key:
        settings.llm_fallback_api_key = body.llm_fallback_api_key

    crew = SoftwareEngineeringCrew(tool_service=_svc, settings=settings)
    bg_tasks.add_task(_execute_async_run, ctx, crew, body.execution_mode)

    return _format_run_response(run_id, ctx)

@router.get("", response_model=AgentRunListResponse)
def list_agent_runs() -> AgentRunListResponse:
    """List all recent runs in memory."""
    runs = []
    for run_id, record in _run_store.items():
        runs.append(_format_run_response(run_id, record))
    # Return newest first
    runs.reverse()
    return AgentRunListResponse(runs=runs)


@router.get("/{run_id}", response_model=AgentRunResponse)
def get_agent_run(run_id: str) -> AgentRunResponse:
    """Retrieve a past run result from the in-memory store."""
    if run_id not in _run_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found",
        )
    record = _run_store[run_id]
    return _format_run_response(run_id, record)

def _format_run_response(run_id: str, record: RunResult | RunContext) -> AgentRunResponse:
    if isinstance(record, RunResult):
        return AgentRunResponse(
            run_id=record.run_id,
            workspace_id=record.workspace_id,
            execution_mode=record.execution_mode,
            architecture=record.architecture,
            status=record.status.value,
            total_duration_ms=record.total_duration_ms,
            files_changed=record.files_changed,
            debug_iterations=record.debug_iterations,
            errors=record.errors,
            result=record.model_dump(mode="json"),
        )
    else:
        # It's an active RunContext
        current_status = "running"
        if record.finished_at:
            current_status = "failed" if record.errors else "completed"
            
        import time
        from datetime import UTC, datetime
        now = datetime.now(UTC)
        duration_ms = (now - record.started_at).total_seconds() * 1000
        if record.finished_at:
            duration_ms = (record.finished_at - record.started_at).total_seconds() * 1000
            
        return AgentRunResponse(
            run_id=record.run_id,
            workspace_id=record.workspace_id,
            execution_mode=record.execution_mode,
            architecture=record.architecture,
            status=current_status,
            total_duration_ms=duration_ms,
            files_changed=record.files_changed,
            debug_iterations=record.debug_iterations,
            errors=record.errors,
            result=record.to_dict(),
        )

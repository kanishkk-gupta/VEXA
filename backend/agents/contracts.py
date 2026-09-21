"""
VEXA M3 — Typed Agent Contracts

Every agent in the VEXA pipeline communicates via typed Pydantic models.
These contracts carry enough information for observability (M8) to build on
without needing to restructure the schemas.

Design principles:
- Every result has: agent_name, status, summary, timestamp
- Evidence fields carry structured data, not arbitrary strings
- Errors are explicit Optional fields, never buried in summary text
- All models are serialisable (JSON-compatible)
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared enumerations
# ---------------------------------------------------------------------------


class AgentStatus(str, Enum):
    success = "success"
    partial = "partial"   # completed with caveats
    failed = "failed"
    skipped = "skipped"


# ---------------------------------------------------------------------------
# Base result
# ---------------------------------------------------------------------------


class AgentResult(BaseModel):
    """Base class for all agent result types."""
    agent_name: str
    status: AgentStatus
    summary: str
    duration_ms: float = 0.0
    errors: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# 1. Requirement Analyst output
# ---------------------------------------------------------------------------


class RequirementAnalysis(AgentResult):
    """
    Structured output from the Requirement Analyst.

    Downstream agents use this to understand the task without re-parsing
    the original natural-language requirement.
    """
    agent_name: str = "RequirementAnalyst"
    original_requirement: str = ""
    task_summary: str = ""
    functional_requirements: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    affected_areas: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 2. Project Analyst output
# ---------------------------------------------------------------------------


class ProjectAnalysis(AgentResult):
    """
    Structured output from the Project Analyst.

    Produced by actual deterministic inspection of the workspace (M2 tools).
    NOT semantic/RAG retrieval — that is M7.
    """
    agent_name: str = "ProjectAnalyst"
    workspace_id: str = ""
    total_files: int = 0
    languages: dict[str, int] = Field(default_factory=dict)
    important_files: list[str] = Field(default_factory=list)
    relevant_files: list[str] = Field(
        default_factory=list,
        description="Files identified as relevant to this task"
    )
    relevant_file_contents: dict[str, str] = Field(
        default_factory=dict,
        description="Content of key files (paths → content snippets)"
    )
    existing_tests: list[str] = Field(default_factory=list)
    git_status: str = ""
    project_context: str = Field(
        default="",
        description="LLM-synthesised narrative summary of the project for Planner"
    )


# ---------------------------------------------------------------------------
# 3. Planner output
# ---------------------------------------------------------------------------


class ImplementationStep(BaseModel):
    """A single step in an implementation plan."""
    step_number: int
    description: str
    files_affected: list[str] = Field(default_factory=list)
    action: str = Field(description="write | patch | delete | test | verify")


class ImplementationPlan(AgentResult):
    """
    Structured output from the Planner.
    """
    agent_name: str = "Planner"
    objective: str = ""
    steps: list[ImplementationStep] = Field(default_factory=list)
    files_to_modify: list[str] = Field(default_factory=list)
    files_to_create: list[str] = Field(default_factory=list)
    test_strategy: str = ""
    risks: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 4. Coder output
# ---------------------------------------------------------------------------


class FileChange(BaseModel):
    """Record of a single file modification made by the Coder."""
    path: str
    action: str = Field(description="created | patched | overwritten | deleted")
    description: str = ""


class CodingResult(AgentResult):
    """
    Structured output from the Coder.
    """
    agent_name: str = "Coder"
    files_changed: list[FileChange] = Field(default_factory=list)
    implementation_notes: str = ""


# ---------------------------------------------------------------------------
# 5. Tester output
# ---------------------------------------------------------------------------


class AgentTestResult(AgentResult):
    """
    Structured output from the Tester.

    Wraps the raw ExecResult from M2 sandbox execution.
    """
    agent_name: str = "Tester"
    passed: bool = False
    exit_code: int | None = None
    tests_found: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    failures: list[str] = Field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = 0.0
    timed_out: bool = False


# ---------------------------------------------------------------------------
# 6. Debugger output
# ---------------------------------------------------------------------------


class DebugIteration(BaseModel):
    """Record of one debugging iteration."""
    iteration: int
    root_cause: str
    fix_applied: str
    files_changed: list[FileChange] = Field(default_factory=list)
    test_result_after: AgentTestResult | None = None


class DebugResult(AgentResult):
    """
    Structured output from the Debugger.
    """
    agent_name: str = "Debugger"
    iterations: list[DebugIteration] = Field(default_factory=list)
    final_test_result: AgentTestResult | None = None
    max_iterations_reached: bool = False
    root_cause_category: str = Field(
        default="unknown",
        description="implementation_bug | test_bug | environment | pre_existing"
    )


# ---------------------------------------------------------------------------
# 7. Verifier output
# ---------------------------------------------------------------------------


class VerificationResult(AgentResult):
    """
    Structured output from the Verifier.

    Tests passing alone does NOT equal verified — the Verifier must evaluate
    requirement satisfaction independently.
    """
    agent_name: str = "Verifier"
    verified: bool = False
    requirements_satisfied: list[str] = Field(default_factory=list)
    requirements_failed: list[str] = Field(default_factory=list)
    tests_passed: bool = False
    files_changed: list[str] = Field(default_factory=list)
    remaining_risks: list[str] = Field(default_factory=list)
    evidence: str = ""


# ---------------------------------------------------------------------------
# 8. Final Run Result (top-level)
# ---------------------------------------------------------------------------


class RunResult(BaseModel):
    """
    The complete result of one VEXA baseline sequential run.

    Returned by SoftwareEngineeringCrew.run_task().
    """
    run_id: str
    workspace_id: str
    requirement: str
    architecture: str = "sequential"
    execution_mode: str = "mock"
    provider: str | None = None
    model: str | None = None
    status: AgentStatus = AgentStatus.failed
    total_duration_ms: float = 0.0

    # Per-agent results (None means agent was not reached)
    requirement_analysis: RequirementAnalysis | None = None
    project_analysis: ProjectAnalysis | None = None
    implementation_plan: ImplementationPlan | None = None
    coding_result: CodingResult | None = None
    initial_test_result: AgentTestResult | None = None
    debug_result: DebugResult | None = None
    final_test_result: AgentTestResult | None = None
    verification_result: VerificationResult | None = None

    # Convenience fields
    files_changed: list[str] = Field(default_factory=list)
    debug_iterations: int = 0
    errors: list[str] = Field(default_factory=list)

    # Usage and Cost
    usage_available: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = None

    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None

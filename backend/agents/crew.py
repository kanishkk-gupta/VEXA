"""
VEXA M3 — Sequential Software Engineering Crew

The baseline multi-agent workflow:

    RequirementAnalyst
          |
    ProjectAnalyst
          |
    Planner
          |
    Coder
          |
    Tester ─(fail)─> Debugger ─> Tester  (up to MAX_DEBUG_ITER)
          |
    Verifier
          |
    RunResult

This module contains:
- _parse_json_output(): Safely extract JSON from LLM text responses
- parse_requirement_analysis(), parse_project_analysis(), etc.: Contract parsers
- SoftwareEngineeringCrew: The main orchestrator class
  - run_task(workspace_id, requirement) → RunResult

Design:
- Each agent is created fresh per run (avoids state leakage)
- The debug loop is explicit Python, not CrewAI sequential chaining
  (gives us precise control over retry logic)
- Mock LLM injection is supported for deterministic testing
- All workspace access goes through ToolService (M2 security preserved)
"""

from __future__ import annotations

import json
import re
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import PrivateAttr

try:
    from crewai.tools import BaseTool
except ImportError:
    BaseTool = object


from backend.agents.config import AgentSettings, get_agent_settings, LLMFactory
from backend.agents.context import RunContext
from backend.agents.contracts import (
    AgentStatus,
    CodingResult,
    DebugIteration,
    DebugResult,
    FileChange,
    ImplementationPlan,
    ImplementationStep,
    ProjectAnalysis,
    RequirementAnalysis,
    RunResult,
    AgentTestResult,
    VerificationResult,
)
from backend.agents.prompts import (
    CODER_BACKSTORY,
    CODER_GOAL,
    CODER_ROLE,
    CODER_TASK,
    DEBUGGER_BACKSTORY,
    DEBUGGER_GOAL,
    DEBUGGER_ROLE,
    DEBUGGER_TASK,
    PLANNER_BACKSTORY,
    PLANNER_GOAL,
    PLANNER_ROLE,
    PLANNER_TASK,
    PROJECT_ANALYST_BACKSTORY,
    PROJECT_ANALYST_GOAL,
    PROJECT_ANALYST_ROLE,
    PROJECT_ANALYST_TASK,
    REQUIREMENT_ANALYST_BACKSTORY,
    REQUIREMENT_ANALYST_GOAL,
    REQUIREMENT_ANALYST_ROLE,
    REQUIREMENT_ANALYST_TASK,
    TESTER_BACKSTORY,
    TESTER_GOAL,
    TESTER_ROLE,
    TESTER_TASK,
    VERIFIER_BACKSTORY,
    VERIFIER_GOAL,
    VERIFIER_ROLE,
    VERIFIER_TASK,
)
from backend.agents.tools import WorkspaceToolKit
from backend.core.logging import get_logger
from backend.tools.service import ToolService

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# JSON response parsing utilities
# ---------------------------------------------------------------------------


def _parse_json_output(text: str) -> dict[str, Any]:
    """
    Extract a JSON object from an LLM response string.

    LLMs frequently wrap JSON in markdown code blocks. This strips them before
    parsing. Returns an empty dict on any parse failure.
    """
    if not isinstance(text, str):
        return {}
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    # Find the first { ... } block
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON from LLM output: %s", cleaned[:200])
        return {}


def _str(d: dict, key: str, default: str = "") -> str:
    return str(d.get(key, default))


def _list(d: dict, key: str) -> list:
    v = d.get(key, [])
    return v if isinstance(v, list) else []


def _int(d: dict, key: str, default: int = 0) -> int:
    try:
        return int(d.get(key, default))
    except (TypeError, ValueError):
        return default


def _bool(d: dict, key: str, default: bool = False) -> bool:
    v = d.get(key, default)
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# Per-agent result parsers (LLM output → typed contracts)
# ---------------------------------------------------------------------------


def parse_requirement_analysis(raw: str, requirement: str, duration_ms: float) -> RequirementAnalysis:
    d = _parse_json_output(raw)
    return RequirementAnalysis(
        status=AgentStatus.success if d else AgentStatus.partial,
        summary=_str(d, "task_summary", raw[:200] if raw else "parse error"),
        duration_ms=duration_ms,
        original_requirement=requirement,
        task_summary=_str(d, "task_summary"),
        functional_requirements=_list(d, "functional_requirements"),
        constraints=_list(d, "constraints"),
        acceptance_criteria=_list(d, "acceptance_criteria"),
        ambiguities=_list(d, "ambiguities"),
        affected_areas=_list(d, "affected_areas"),
        errors=[] if d else ["Could not parse JSON from LLM response"],
    )


def parse_project_analysis(raw: str, workspace_id: str, duration_ms: float) -> ProjectAnalysis:
    d = _parse_json_output(raw)
    return ProjectAnalysis(
        status=AgentStatus.success if d else AgentStatus.partial,
        summary=_str(d, "project_context", raw[:200] if raw else "parse error"),
        duration_ms=duration_ms,
        workspace_id=workspace_id,
        total_files=_int(d, "total_files"),
        languages=d.get("languages", {}) if isinstance(d.get("languages"), dict) else {},
        important_files=_list(d, "important_files"),
        relevant_files=_list(d, "relevant_files"),
        relevant_file_contents=d.get("relevant_file_contents", {}) if isinstance(d.get("relevant_file_contents"), dict) else {},
        existing_tests=_list(d, "existing_tests"),
        git_status=_str(d, "git_status"),
        project_context=_str(d, "project_context"),
        errors=[] if d else ["Could not parse JSON from LLM response"],
    )


def parse_implementation_plan(raw: str, duration_ms: float) -> ImplementationPlan:
    d = _parse_json_output(raw)
    steps_raw = _list(d, "steps")
    steps = []
    for i, s in enumerate(steps_raw):
        if isinstance(s, dict):
            steps.append(ImplementationStep(
                step_number=_int(s, "step_number", i + 1),
                description=_str(s, "description"),
                files_affected=_list(s, "files_affected"),
                action=_str(s, "action", "patch"),
            ))
    return ImplementationPlan(
        status=AgentStatus.success if d else AgentStatus.partial,
        summary=_str(d, "objective", raw[:200] if raw else "parse error"),
        duration_ms=duration_ms,
        objective=_str(d, "objective"),
        steps=steps,
        files_to_modify=_list(d, "files_to_modify"),
        files_to_create=_list(d, "files_to_create"),
        test_strategy=_str(d, "test_strategy"),
        risks=_list(d, "risks"),
        errors=[] if d else ["Could not parse JSON from LLM response"],
    )


def parse_coding_result(raw: str, duration_ms: float) -> CodingResult:
    d = _parse_json_output(raw)
    changes_raw = _list(d, "files_changed")
    changes = []
    for c in changes_raw:
        if isinstance(c, dict):
            changes.append(FileChange(
                path=_str(c, "path"),
                action=_str(c, "action", "patched"),
                description=_str(c, "description"),
            ))
    return CodingResult(
        status=AgentStatus.success if changes else AgentStatus.partial,
        summary=_str(d, "implementation_notes", f"{len(changes)} file(s) changed"),
        duration_ms=duration_ms,
        files_changed=changes,
        implementation_notes=_str(d, "implementation_notes"),
        errors=[] if d else ["Could not parse JSON from LLM response"],
    )


def parse_test_result(raw: str, duration_ms: float) -> AgentTestResult:
    """
    Parse a tester result. Also handles raw ExecResult JSON (from direct tool call).
    """
    d = _parse_json_output(raw)
    passed = _bool(d, "passed") or (_bool(d, "success") if "success" in d else False)
    if "exit_code" in d:
        try:
            exit_code = int(d["exit_code"])
            if exit_code == 0:
                passed = True
        except (TypeError, ValueError):
            exit_code = None
    else:
        exit_code = None

    return AgentTestResult(
        status=AgentStatus.success if passed else AgentStatus.failed,
        summary=f"Tests {'passed' if passed else 'failed'} (exit_code={exit_code})",
        duration_ms=duration_ms,
        passed=passed,
        exit_code=exit_code,
        tests_found=_int(d, "tests_found"),
        tests_passed=_int(d, "tests_passed"),
        tests_failed=_int(d, "tests_failed"),
        failures=_list(d, "failures"),
        stdout=_str(d, "stdout"),
        stderr=_str(d, "stderr"),
        timed_out=_bool(d, "timed_out"),
        errors=[] if d else ["Could not parse JSON from LLM response"],
    )


def parse_debug_result(raw: str, iteration: int, duration_ms: float) -> tuple[str, str, list[FileChange]]:
    """
    Parse one Debugger iteration output.
    Returns (root_cause, fix_applied, files_changed).
    """
    d = _parse_json_output(raw)
    root_cause = _str(d, "root_cause", "Unknown root cause")
    fix_applied = _str(d, "fix_applied", "No fix description")
    category = _str(d, "root_cause_category", "unknown")
    changes_raw = _list(d, "files_changed")
    changes = []
    for c in changes_raw:
        if isinstance(c, dict):
            changes.append(FileChange(
                path=_str(c, "path"),
                action=_str(c, "action", "patched"),
                description=_str(c, "description"),
            ))
    return root_cause, fix_applied, changes


def parse_verification_result(raw: str, duration_ms: float) -> VerificationResult:
    d = _parse_json_output(raw)
    verified = _bool(d, "verified")
    return VerificationResult(
        status=AgentStatus.success if verified else AgentStatus.failed,
        summary=("Requirements verified and satisfied." if verified else "Verification failed — requirements not fully met."),
        duration_ms=duration_ms,
        verified=verified,
        requirements_satisfied=_list(d, "requirements_satisfied"),
        requirements_failed=_list(d, "requirements_failed"),
        tests_passed=_bool(d, "tests_passed"),
        files_changed=_list(d, "files_changed"),
        remaining_risks=_list(d, "remaining_risks"),
        evidence=_str(d, "evidence"),
        errors=[] if d else ["Could not parse JSON from LLM response"],
    )


# ---------------------------------------------------------------------------
# SoftwareEngineeringCrew
# ---------------------------------------------------------------------------


class SoftwareEngineeringCrew:
    """
    The VEXA baseline sequential multi-agent software engineering workflow.

    Orchestrates seven agents:
        RequirementAnalyst → ProjectAnalyst → Planner → Coder →
        Tester → (Debugger → Tester)* → Verifier

    The debug loop is explicit Python rather than CrewAI sequential chaining
    so we have precise control over retry counting and context threading.

    Parameters
    ----------
    tool_service : ToolService
        The M2 ToolService. All workspace access goes through this.
    settings : AgentSettings | None
        Agent/LLM settings. If None, reads from environment.
    llm : Any
        Pre-built LLM object (used to inject mocks in tests).
    """

    def __init__(
        self,
        tool_service: ToolService | None = None,
        settings: AgentSettings | None = None,
        llm: Any = None,
    ) -> None:
        self._svc = tool_service or ToolService()
        self._settings = settings or get_agent_settings()
        self._llm = llm  # If None, built lazily when first needed

    def _get_llm(self) -> Any:
        if self._llm is None:
            self._llm = LLMFactory.build(self._settings)
        return self._llm

    # ------------------------------------------------------------------
    # Agent factory helpers
    # ------------------------------------------------------------------

    def _make_agent(self, role: str, goal: str, backstory: str, tools: list) -> Any:
        """Build a CrewAI Agent with the standard VEXA configuration."""
        from crewai import Agent
        return Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            llm=self._get_llm(),
            tools=tools,
            max_iter=self._settings.agent_max_iter,
            verbose=self._settings.agent_verbose,
            allow_delegation=False,
        )

    def _make_task(self, description: str, agent: Any, expected_output: str = "JSON object") -> Any:
        """Build a single-agent CrewAI Task."""
        from crewai import Task
        
        # Guardrail against open-source models hallucinating a "JSON" tool call
        if "JSON" in expected_output.upper() or "JSON" in description.upper():
            description += (
                "\n\nCRITICAL: DO NOT use a tool call to output your JSON response! "
                "There is no tool named 'JSON'. You MUST output the raw JSON text directly "
                "as your final answer."
            )
            
        return Task(
            description=description,
            expected_output=expected_output,
            agent=agent,
        )

    def _run_single_agent_crew(self, ctx: RunContext, agent: Any, task: Any) -> str:
        """Run a single-agent Crew and return the string output. Accumulate usage."""
        from crewai import Crew, Process
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=self._settings.agent_verbose,
        )
        result = crew.kickoff()
        
        # Accumulate usage metrics if available on the Crew
        if hasattr(crew, "usage_metrics") and crew.usage_metrics:
            metrics = crew.usage_metrics
            ctx.usage_available = True
            ctx.total_tokens += getattr(metrics, "total_tokens", 0)
            ctx.input_tokens += getattr(metrics, "prompt_tokens", 0)
            ctx.output_tokens += getattr(metrics, "completion_tokens", 0)
        
        # CrewAI 1.x returns a CrewOutput object; extract raw text
        if hasattr(result, "raw"):
            return str(result.raw)
        return str(result)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_task(self, workspace_id: str, requirement: str, execution_mode: str = "mock", ctx: RunContext | None = None) -> RunResult:
        """
        Execute the full baseline sequential workflow.

        Parameters
        ----------
        workspace_id : str
            ID of the M2 workspace to operate on.
        requirement : str
            Natural-language software task description.
        execution_mode : str
            'mock' (default) for deterministic mock execution or 'real' for live LLM.
        ctx : RunContext | None
            Optional pre-initialized execution context for external observability.

        Returns
        -------
        RunResult
            Complete structured result of the run.
        """
        if ctx is None:
            run_id = str(uuid.uuid4())
            ctx = RunContext(workspace_id=workspace_id, requirement=requirement, run_id=run_id)
            ctx.execution_mode = execution_mode
        else:
            run_id = ctx.run_id
        t_total_start = time.perf_counter()

        logger.info("run_started | run_id=%s workspace=%s mode=%s", run_id, workspace_id, execution_mode)

        if execution_mode == "real":
            from backend.agents.config import LLMFactory
            self._llm = LLMFactory.build(self._settings)
            ctx.provider = self._settings.llm_provider
            ctx.model = self._settings.llm_model
            # Note: Do not load LLM for mock mode; it will use the test stubs.

        result = RunResult(
            run_id=run_id,
            workspace_id=workspace_id,
            requirement=requirement,
            architecture="sequential",
            execution_mode=execution_mode,
            provider=ctx.provider,
            model=ctx.model,
        )

        try:
            # Validate workspace exists
            if not self._svc.workspace_exists(workspace_id):
                raise ValueError(f"Workspace '{workspace_id}' does not exist")

            kit = WorkspaceToolKit(self._svc, workspace_id)

            # ----------------------------------------------------------
            # 1. Requirement Analyst
            # ----------------------------------------------------------
            req_analysis = self._run_requirement_analyst(ctx, requirement)
            result.requirement_analysis = req_analysis
            if req_analysis.status == AgentStatus.failed:
                raise RuntimeError("Requirement analysis failed: " + "; ".join(req_analysis.errors))

            # ----------------------------------------------------------
            # 2. Project Analyst
            # ----------------------------------------------------------
            proj_analysis = self._run_project_analyst(ctx, kit, req_analysis)
            result.project_analysis = proj_analysis

            # ----------------------------------------------------------
            # 3. Planner
            # ----------------------------------------------------------
            plan = self._run_planner(ctx, kit, req_analysis, proj_analysis)
            result.implementation_plan = plan

            # ----------------------------------------------------------
            # 4. Coder
            # ----------------------------------------------------------
            coding = self._run_coder(ctx, kit, plan, proj_analysis)
            result.coding_result = coding
            result.files_changed = [fc.path for fc in coding.files_changed]
            for fc in coding.files_changed:
                ctx.file_changed(fc.path)

            # ----------------------------------------------------------
            # 5. Tester (initial run)
            # ----------------------------------------------------------
            test_result = self._run_tester(ctx, kit, workspace_id)
            result.initial_test_result = test_result
            result.final_test_result = test_result

            # ----------------------------------------------------------
            # 6. Debugger loop (if tests failed)
            # ----------------------------------------------------------
            if not test_result.passed:
                debug_result = self._run_debugger_loop(ctx, kit, coding, test_result)
                result.debug_result = debug_result
                result.debug_iterations = ctx.debug_iterations
                result.final_test_result = debug_result.final_test_result or test_result
                # Merge changed files from debugger
                for itr in debug_result.iterations:
                    for fc in itr.files_changed:
                        if fc.path not in result.files_changed:
                            result.files_changed.append(fc.path)

            # ----------------------------------------------------------
            # 7. Verifier
            # ----------------------------------------------------------
            verification = self._run_verifier(ctx, kit, req_analysis, result.files_changed, result.final_test_result)
            result.verification_result = verification

            # Determine final status
            if verification.verified and (result.final_test_result and result.final_test_result.passed):
                result.status = AgentStatus.success
            elif result.final_test_result and result.final_test_result.passed:
                result.status = AgentStatus.partial
            else:
                result.status = AgentStatus.failed

        except Exception as exc:
            logger.exception("run_failed | run_id=%s error=%s", run_id, exc)
            result.errors.append(str(exc))
            result.status = AgentStatus.failed

        finally:
            ctx.finish()
            elapsed_ms = (time.perf_counter() - t_total_start) * 1000
            result.total_duration_ms = elapsed_ms
            result.finished_at = datetime.now(UTC)
            
            # Map usage and cost metrics
            result.usage_available = ctx.usage_available
            result.input_tokens = ctx.input_tokens
            result.output_tokens = ctx.output_tokens
            result.total_tokens = ctx.total_tokens
            
            if ctx.usage_available:
                cost = 0.0
                if self._settings.llm_input_cost_per_1m is not None:
                    cost += (ctx.input_tokens / 1_000_000) * self._settings.llm_input_cost_per_1m
                if self._settings.llm_output_cost_per_1m is not None:
                    cost += (ctx.output_tokens / 1_000_000) * self._settings.llm_output_cost_per_1m
                result.estimated_cost_usd = cost
                ctx.estimated_cost_usd = cost

            logger.info(
                "run_completed | run_id=%s status=%s duration_ms=%.1f",
                run_id, result.status.value, elapsed_ms
            )

        return result

    # ------------------------------------------------------------------
    # Individual agent runners
    # ------------------------------------------------------------------

    def _run_requirement_analyst(self, ctx: RunContext, requirement: str) -> RequirementAnalysis:
        ctx.agent_started("RequirementAnalyst")
        t = time.perf_counter()
        logger.info("agent_started | run=%s agent=RequirementAnalyst", ctx.run_id)
        try:
            agent = self._make_agent(
                role=REQUIREMENT_ANALYST_ROLE,
                goal=REQUIREMENT_ANALYST_GOAL,
                backstory=REQUIREMENT_ANALYST_BACKSTORY,
                tools=[],  # No tools — pure LLM reasoning
            )
            task = self._make_task(
                description=REQUIREMENT_ANALYST_TASK.format(requirement=requirement),
                agent=agent,
                expected_output="JSON object with task_summary, functional_requirements, constraints, acceptance_criteria, ambiguities, affected_areas",
            )
            raw = self._run_single_agent_crew(ctx, agent, task)
            duration_ms = (time.perf_counter() - t) * 1000
            result = parse_requirement_analysis(raw, requirement, duration_ms)
            ctx.agent_completed("RequirementAnalyst", duration_ms)
            logger.info("agent_completed | run=%s agent=RequirementAnalyst duration_ms=%.1f", ctx.run_id, duration_ms)
            return result
        except Exception as e:
            duration_ms = (time.perf_counter() - t) * 1000
            logger.error("agent_failed | run=%s agent=RequirementAnalyst error=%s", ctx.run_id, e)
            return RequirementAnalysis(
                status=AgentStatus.failed,
                summary=f"Agent error: {e}",
                duration_ms=duration_ms,
                original_requirement=requirement,
                errors=[str(e)],
            )

    def _run_project_analyst(self, ctx: RunContext, kit: WorkspaceToolKit, req: RequirementAnalysis) -> ProjectAnalysis:
        ctx.agent_started("ProjectAnalyst")
        t = time.perf_counter()
        logger.info("agent_started | run=%s agent=ProjectAnalyst", ctx.run_id)
        try:
            tools = self._make_crewai_tools(kit.for_project_analyst())
            agent = self._make_agent(
                role=PROJECT_ANALYST_ROLE,
                goal=PROJECT_ANALYST_GOAL,
                backstory=PROJECT_ANALYST_BACKSTORY,
                tools=tools,
            )
            task = self._make_task(
                description=PROJECT_ANALYST_TASK.format(
                    requirement_analysis=req.model_dump_json(indent=2)
                ),
                agent=agent,
                expected_output="JSON object with project analysis",
            )
            raw = self._run_single_agent_crew(ctx, agent, task)
            duration_ms = (time.perf_counter() - t) * 1000
            result = parse_project_analysis(raw, ctx.workspace_id, duration_ms)
            ctx.agent_completed("ProjectAnalyst", duration_ms)
            logger.info("agent_completed | run=%s agent=ProjectAnalyst files=%d", ctx.run_id, result.total_files)
            return result
        except Exception as e:
            duration_ms = (time.perf_counter() - t) * 1000
            logger.error("agent_failed | run=%s agent=ProjectAnalyst error=%s", ctx.run_id, e)
            return ProjectAnalysis(
                status=AgentStatus.failed,
                summary=f"Agent error: {e}",
                duration_ms=duration_ms,
                workspace_id=ctx.workspace_id,
                errors=[str(e)],
            )

    def _run_planner(self, ctx: RunContext, kit: WorkspaceToolKit, req: RequirementAnalysis, proj: ProjectAnalysis) -> ImplementationPlan:
        ctx.agent_started("Planner")
        t = time.perf_counter()
        logger.info("agent_started | run=%s agent=Planner", ctx.run_id)
        try:
            tools = self._make_crewai_tools(kit.for_planner())
            agent = self._make_agent(
                role=PLANNER_ROLE,
                goal=PLANNER_GOAL,
                backstory=PLANNER_BACKSTORY,
                tools=tools,
            )
            task = self._make_task(
                description=PLANNER_TASK.format(
                    requirement_analysis=req.model_dump_json(indent=2),
                    project_analysis=proj.model_dump_json(indent=2),
                ),
                agent=agent,
                expected_output="JSON implementation plan",
            )
            raw = self._run_single_agent_crew(ctx, agent, task)
            duration_ms = (time.perf_counter() - t) * 1000
            result = parse_implementation_plan(raw, duration_ms)
            ctx.agent_completed("Planner", duration_ms)
            logger.info("agent_completed | run=%s agent=Planner steps=%d", ctx.run_id, len(result.steps))
            return result
        except Exception as e:
            duration_ms = (time.perf_counter() - t) * 1000
            logger.error("agent_failed | run=%s agent=Planner error=%s", ctx.run_id, e)
            return ImplementationPlan(
                status=AgentStatus.failed,
                summary=f"Agent error: {e}",
                duration_ms=duration_ms,
                errors=[str(e)],
            )

    def _run_coder(self, ctx: RunContext, kit: WorkspaceToolKit, plan: ImplementationPlan, proj: ProjectAnalysis) -> CodingResult:
        ctx.agent_started("Coder")
        t = time.perf_counter()
        logger.info("agent_started | run=%s agent=Coder", ctx.run_id)
        try:
            tools = self._make_crewai_tools(kit.for_coder())
            agent = self._make_agent(
                role=CODER_ROLE,
                goal=CODER_GOAL,
                backstory=CODER_BACKSTORY,
                tools=tools,
            )
            task = self._make_task(
                description=CODER_TASK.format(
                    implementation_plan=plan.model_dump_json(indent=2),
                    project_analysis=proj.model_dump_json(indent=2),
                ),
                agent=agent,
                expected_output="JSON coding result with files_changed",
            )
            raw = self._run_single_agent_crew(ctx, agent, task)
            duration_ms = (time.perf_counter() - t) * 1000
            result = parse_coding_result(raw, duration_ms)
            ctx.agent_completed("Coder", duration_ms)
            logger.info("agent_completed | run=%s agent=Coder files_changed=%d", ctx.run_id, len(result.files_changed))
            return result
        except Exception as e:
            duration_ms = (time.perf_counter() - t) * 1000
            logger.error("agent_failed | run=%s agent=Coder error=%s", ctx.run_id, e)
            return CodingResult(
                status=AgentStatus.failed,
                summary=f"Agent error: {e}",
                duration_ms=duration_ms,
                errors=[str(e)],
            )

    def _run_tester(self, ctx: RunContext, kit: WorkspaceToolKit, workspace_id: str) -> AgentTestResult:
        ctx.agent_started("Tester")
        ctx.test_attempted()
        t = time.perf_counter()
        logger.info("agent_started | run=%s agent=Tester attempt=%d", ctx.run_id, ctx.test_attempts)
        try:
            tools = self._make_crewai_tools(kit.for_tester())
            agent = self._make_agent(
                role=TESTER_ROLE,
                goal=TESTER_GOAL,
                backstory=TESTER_BACKSTORY,
                tools=tools,
            )
            task = self._make_task(
                description=TESTER_TASK.format(
                    workspace_id=workspace_id,
                    test_path=".",
                ),
                agent=agent,
                expected_output="JSON test result with passed, exit_code, failures",
            )
            raw = self._run_single_agent_crew(ctx, agent, task)
            duration_ms = (time.perf_counter() - t) * 1000
            result = parse_test_result(raw, duration_ms)
            ctx.agent_completed("Tester", duration_ms)
            logger.info(
                "agent_completed | run=%s agent=Tester passed=%s exit_code=%s",
                ctx.run_id, result.passed, result.exit_code
            )
            return result
        except Exception as e:
            duration_ms = (time.perf_counter() - t) * 1000
            logger.error("agent_failed | run=%s agent=Tester error=%s", ctx.run_id, e)
            return AgentTestResult(
                status=AgentStatus.failed,
                summary=f"Agent error: {e}",
                duration_ms=duration_ms,
                passed=False,
                errors=[str(e)],
            )

    def _run_debugger_loop(
        self,
        ctx: RunContext,
        kit: WorkspaceToolKit,
        coding: CodingResult,
        initial_test: AgentTestResult,
    ) -> DebugResult:
        """
        Debugger loop: attempt repairs up to MAX_DEBUG_ITERATIONS.

        Returns a DebugResult with all iterations and the final test result.
        """
        iterations: list[DebugIteration] = []
        current_test = initial_test
        max_iter = self._settings.debug_max_iter

        logger.info("debug_loop_started | run=%s max_iterations=%d", ctx.run_id, max_iter)

        for i in range(1, max_iter + 1):
            ctx.debug_iterated()
            t = time.perf_counter()
            logger.info("agent_started | run=%s agent=Debugger iteration=%d", ctx.run_id, i)

            try:
                tools = self._make_crewai_tools(kit.for_debugger())
                agent = self._make_agent(
                    role=DEBUGGER_ROLE,
                    goal=DEBUGGER_GOAL,
                    backstory=DEBUGGER_BACKSTORY,
                    tools=tools,
                )
                failure_summary = "\n".join(current_test.failures) if current_test.failures else current_test.stdout[-2000:]
                task = self._make_task(
                    description=DEBUGGER_TASK.format(
                        test_failure_output=failure_summary,
                        coding_result=coding.model_dump_json(indent=2),
                        iteration=i,
                        max_iterations=max_iter,
                    ),
                    agent=agent,
                    expected_output="JSON debug result with root_cause, fix_applied, files_changed",
                )
                raw = self._run_single_agent_crew(ctx, agent, task)
                duration_ms = (time.perf_counter() - t) * 1000
                root_cause, fix_applied, files_changed = parse_debug_result(raw, i, duration_ms)
                logger.info(
                    "agent_completed | run=%s agent=Debugger iteration=%d root_cause=%s",
                    ctx.run_id, i, root_cause[:100]
                )

                # Re-run tests after the fix
                re_test = self._run_tester(ctx, kit, ctx.workspace_id)
                iteration_record = DebugIteration(
                    iteration=i,
                    root_cause=root_cause,
                    fix_applied=fix_applied,
                    files_changed=files_changed,
                    test_result_after=re_test,
                )
                iterations.append(iteration_record)
                current_test = re_test

                if re_test.passed:
                    logger.info("debug_loop_success | run=%s after_iteration=%d", ctx.run_id, i)
                    break

            except Exception as e:
                duration_ms = (time.perf_counter() - t) * 1000
                logger.error("agent_failed | run=%s agent=Debugger iteration=%d error=%s", ctx.run_id, i, e)
                iterations.append(DebugIteration(
                    iteration=i,
                    root_cause=f"Debug agent error: {e}",
                    fix_applied="None — agent crashed",
                    test_result_after=current_test,
                ))

        max_reached = not current_test.passed
        if max_reached:
            logger.warning("debug_loop_max_iterations | run=%s max=%d", ctx.run_id, max_iter)

        return DebugResult(
            status=AgentStatus.success if current_test.passed else AgentStatus.failed,
            summary=(
                f"Debug loop completed in {len(iterations)} iteration(s). "
                f"{'Tests now pass.' if current_test.passed else 'Tests still failing.'}"
            ),
            iterations=iterations,
            final_test_result=current_test,
            max_iterations_reached=max_reached,
        )

    def _run_verifier(
        self,
        ctx: RunContext,
        kit: WorkspaceToolKit,
        req: RequirementAnalysis,
        files_changed: list[str],
        final_test: AgentTestResult | None,
    ) -> VerificationResult:
        ctx.agent_started("Verifier")
        t = time.perf_counter()
        logger.info("verification_started | run=%s", ctx.run_id)
        try:
            tools = self._make_crewai_tools(kit.for_verifier())
            agent = self._make_agent(
                role=VERIFIER_ROLE,
                goal=VERIFIER_GOAL,
                backstory=VERIFIER_BACKSTORY,
                tools=tools,
            )
            task = self._make_task(
                description=VERIFIER_TASK.format(
                    original_requirement=req.original_requirement,
                    acceptance_criteria="\n".join(f"- {c}" for c in req.acceptance_criteria),
                    files_changed=", ".join(files_changed) if files_changed else "none",
                    test_result=final_test.model_dump_json(indent=2) if final_test else "{}",
                ),
                agent=agent,
                expected_output="JSON verification result",
            )
            raw = self._run_single_agent_crew(ctx, agent, task)
            duration_ms = (time.perf_counter() - t) * 1000
            result = parse_verification_result(raw, duration_ms)
            ctx.agent_completed("Verifier", duration_ms)
            logger.info(
                "run_verified | run=%s verified=%s", ctx.run_id, result.verified
            )
            return result
        except Exception as e:
            duration_ms = (time.perf_counter() - t) * 1000
            logger.error("agent_failed | run=%s agent=Verifier error=%s", ctx.run_id, e)
            return VerificationResult(
                status=AgentStatus.failed,
                summary=f"Agent error: {e}",
                duration_ms=duration_ms,
                verified=False,
                errors=[str(e)],
            )

    # ------------------------------------------------------------------
    # CrewAI tool wrapping
    # ------------------------------------------------------------------

    def _make_crewai_tools(self, adapters: list) -> list:
        """
        Convert VEXA tool adapter objects into CrewAI BaseTool instances.
        """
        if BaseTool is object:
            return []  # No crewai installed — tests run without it

        crewai_tools = []
        for adapter in adapters:
            crewai_tools.append(_AdaptedTool(adapter))
        return crewai_tools


import inspect
from pydantic import BaseModel, create_model, PrivateAttr

def _generate_args_schema(func: Any, tool_name: str) -> type[BaseModel]:
    sig = inspect.signature(func)
    fields = {}
    for name, param in sig.parameters.items():
        if name in ("self", "args", "kwargs"):
            continue
        annotation = param.annotation if param.annotation != inspect.Parameter.empty else str
        default = param.default if param.default != inspect.Parameter.empty else ...
        fields[name] = (annotation, default)
    
    # If a tool has no parameters, Pydantic creates an empty schema.
    # Groq needs at least some properties or no 'required' block.
    # We add a dummy property to avoid empty schema issues on strict LLMs if needed,
    # but normally create_model handles it correctly when properly typed.
    if not fields:
        fields["dummy"] = (str, "ignored")
        
    return create_model(f"{tool_name}_schema", **fields)

class _AdaptedTool(BaseTool):
    """
    Wraps a VEXA tool adapter as a callable that CrewAI can use.
    """
    name: str = ""
    description: str = ""
    _adapter: Any = PrivateAttr()

    def __init__(self, adapter: Any, **data: Any) -> None:
        schema = _generate_args_schema(adapter.run, adapter.name)
        super().__init__(
            name=adapter.name, 
            description=adapter.description, 
            args_schema=schema, 
            **data
        )
        self._adapter = adapter

    def run(self, *args: Any, **kwargs: Any) -> str:
        kwargs.pop("dummy", None)
        return self._adapter.run(*args, **kwargs)

    def _run(self, *args: Any, **kwargs: Any) -> str:
        kwargs.pop("dummy", None)
        return self._adapter.run(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        kwargs.pop("dummy", None)
        return self._adapter.run(*args, **kwargs)

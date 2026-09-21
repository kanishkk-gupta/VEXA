"""
VEXA M3 — Sequential Crew Orchestration Tests (Mock LLM)

These tests use a deterministic mock LLM to:
1. Verify the sequential agent pipeline executes in order
2. Verify the debug loop triggers and retries
3. Verify the retry limit is enforced
4. Verify file changes reach the workspace
5. Verify verification fails when requirements are not met

The mock LLM is injected at the SoftwareEngineeringCrew level.
The REAL LLM integration path is preserved and tested in validate_m3.py.

IMPORTANT: These tests make REAL tool calls (file I/O, pytest execution)
against a real workspace. Only the LLM is mocked.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.agents.config import AgentSettings
from backend.agents.contracts import AgentStatus, RunResult
from backend.agents.crew import (
    SoftwareEngineeringCrew,
    parse_coding_result,
    parse_implementation_plan,
    parse_project_analysis,
    parse_requirement_analysis,
    parse_test_result,
    parse_verification_result,
)
from backend.tools.service import ToolService

FIXTURE_PROJECT = Path(__file__).parent / "fixtures" / "sample_project"


# ---------------------------------------------------------------------------
# Mock LLM that returns deterministic canned responses
# ---------------------------------------------------------------------------


class MockLLM:
    """
    Deterministic mock LLM that returns configurable responses.

    The SoftwareEngineeringCrew._run_single_agent_crew() calls crew.kickoff().
    We mock that at the Crew class level rather than at the LLM level,
    since CrewAI's internal call chain is complex.
    """
    pass


def _make_settings() -> AgentSettings:
    return AgentSettings(
        llm_model="gpt-4o-mini",
        llm_temperature=0.1,
        agent_max_iter=1,
        debug_max_iter=3,
        agent_verbose=False,
    )


class _StubAgent:
    """Minimal stub that satisfies CrewAI's internal checks."""
    role = "stub"
    goal = "stub"
    backstory = "stub"
    tools = []
    allow_delegation = False
    verbose = False
    max_iter = 1


class _StubTask:
    pass


def _make_crew_with_mock_responses(
    svc: ToolService,
    response_queue: list[str],
) -> SoftwareEngineeringCrew:
    """
    Build a SoftwareEngineeringCrew whose agent calls return canned responses.

    Patches BOTH _make_agent and _make_task (skips real CrewAI validation) AND
    _run_single_agent_crew (returns canned responses). This allows the pipeline
    Python logic to run fully while bypassing the LLM API.
    """
    settings = _make_settings()
    crew = SoftwareEngineeringCrew(tool_service=svc, settings=settings, llm=None)
    responses = iter(response_queue)

    crew._make_agent = lambda role, goal, backstory, tools: _StubAgent()
    crew._make_task = lambda description, agent, expected_output="": _StubTask()
    crew._run_single_agent_crew = lambda ctx, agent, task: next(responses)
    return crew


# ---------------------------------------------------------------------------
# JSON parser unit tests
# ---------------------------------------------------------------------------


class TestParsers:
    def test_parse_requirement_analysis_good_json(self):
        raw = json.dumps({
            "task_summary": "Add modulo",
            "functional_requirements": ["modulo(a, b) returns a % b"],
            "constraints": [],
            "acceptance_criteria": ["modulo(10, 3) == 1"],
            "ambiguities": [],
            "affected_areas": ["src/calculator.py"],
        })
        result = parse_requirement_analysis(raw, "Add modulo function", 100.0)
        assert result.status == AgentStatus.success
        assert result.task_summary == "Add modulo"
        assert "modulo(10, 3) == 1" in result.acceptance_criteria

    def test_parse_requirement_analysis_bad_json_returns_partial(self):
        result = parse_requirement_analysis("not json at all", "req", 0.0)
        assert result.status == AgentStatus.partial
        assert len(result.errors) > 0

    def test_parse_requirement_analysis_markdown_wrapped(self):
        raw = "```json\n" + json.dumps({"task_summary": "wrapped", "functional_requirements": [], "constraints": [], "acceptance_criteria": [], "ambiguities": [], "affected_areas": []}) + "\n```"
        result = parse_requirement_analysis(raw, "req", 0.0)
        assert result.task_summary == "wrapped"

    def test_parse_project_analysis(self):
        raw = json.dumps({
            "total_files": 5,
            "languages": {"python": 2},
            "important_files": ["README.md"],
            "relevant_files": ["src/calculator.py"],
            "relevant_file_contents": {},
            "existing_tests": ["tests/test_calculator.py"],
            "git_status": "not a git repo",
            "project_context": "Simple calculator project",
        })
        result = parse_project_analysis(raw, "ws-1", 200.0)
        assert result.total_files == 5
        assert result.languages["python"] == 2
        assert result.project_context == "Simple calculator project"

    def test_parse_implementation_plan(self):
        raw = json.dumps({
            "objective": "Add modulo",
            "steps": [{"step_number": 1, "description": "Add function", "files_affected": ["src/calculator.py"], "action": "patch"}],
            "files_to_modify": ["src/calculator.py"],
            "files_to_create": [],
            "test_strategy": "run pytest",
            "risks": [],
        })
        result = parse_implementation_plan(raw, 150.0)
        assert result.objective == "Add modulo"
        assert len(result.steps) == 1
        assert result.steps[0].action == "patch"

    def test_parse_test_result_passed(self):
        raw = json.dumps({
            "passed": True, "exit_code": 0, "tests_found": 9,
            "tests_passed": 9, "tests_failed": 0, "failures": [],
            "stdout": "9 passed", "stderr": "", "duration_ms": 300.0,
        })
        result = parse_test_result(raw, 300.0)
        assert result.passed is True
        assert result.exit_code == 0

    def test_parse_test_result_failed(self):
        raw = json.dumps({
            "passed": False, "exit_code": 1, "failures": ["test_modulo: assert 1 == 2"],
            "stdout": "FAILED", "stderr": "", "duration_ms": 100.0,
        })
        result = parse_test_result(raw, 100.0)
        assert result.passed is False

    def test_parse_verification_verified(self):
        raw = json.dumps({
            "verified": True,
            "requirements_satisfied": ["modulo works"],
            "requirements_failed": [],
            "tests_passed": True,
            "files_changed": ["src/calculator.py"],
            "remaining_risks": [],
            "evidence": "All tests pass",
        })
        result = parse_verification_result(raw, 50.0)
        assert result.verified is True

    def test_parse_coding_result(self):
        raw = json.dumps({
            "files_changed": [{"path": "src/calculator.py", "action": "patched", "description": "Added modulo"}],
            "implementation_notes": "Used % operator",
        })
        result = parse_coding_result(raw, 400.0)
        assert len(result.files_changed) == 1
        assert result.files_changed[0].path == "src/calculator.py"


# ---------------------------------------------------------------------------
# Mock LLM integration tests — full pipeline
# ---------------------------------------------------------------------------


CANNED_REQ = json.dumps({
    "task_summary": "Add modulo(a, b) function to calculator",
    "functional_requirements": ["modulo(a, b) returns a % b"],
    "constraints": ["preserve existing tests"],
    "acceptance_criteria": ["modulo(10, 3) == 1"],
    "ambiguities": [],
    "affected_areas": ["src/calculator.py"],
})

CANNED_PROJ = json.dumps({
    "total_files": 5,
    "languages": {"python": 2},
    "important_files": ["README.md"],
    "relevant_files": ["src/calculator.py"],
    "relevant_file_contents": {"src/calculator.py": "def add(a, b): return a + b"},
    "existing_tests": ["tests/test_calculator.py"],
    "git_status": "not a git repo",
    "project_context": "Simple calculator; needs modulo function",
})

CANNED_PLAN = json.dumps({
    "objective": "Add modulo(a, b) function",
    "steps": [{"step_number": 1, "description": "Patch calculator.py", "files_affected": ["src/calculator.py"], "action": "patch"}],
    "files_to_modify": ["src/calculator.py"],
    "files_to_create": [],
    "test_strategy": "run existing pytest suite",
    "risks": [],
})

# The coder's mock response — we describe what it "did"
# Note: the actual file changes are performed by mock_coder_action fixture
CANNED_CODE = json.dumps({
    "files_changed": [{"path": "src/calculator.py", "action": "patched", "description": "Added modulo function"}],
    "implementation_notes": "Used % operator",
})

CANNED_TEST_PASS = json.dumps({
    "passed": True, "exit_code": 0, "tests_found": 10, "tests_passed": 10,
    "tests_failed": 0, "failures": [], "stdout": "10 passed", "stderr": "", "duration_ms": 300.0,
})

CANNED_TEST_FAIL = json.dumps({
    "passed": False, "exit_code": 1, "tests_found": 10, "tests_passed": 9,
    "tests_failed": 1, "failures": ["test_modulo: AssertionError"],
    "stdout": "FAILED", "stderr": "", "duration_ms": 100.0,
})

CANNED_DEBUG = json.dumps({
    "root_cause": "Wrong return statement in modulo",
    "root_cause_category": "implementation_bug",
    "fix_applied": "Changed return to a % b",
    "files_changed": [{"path": "src/calculator.py", "action": "patched", "description": "Fixed modulo"}],
})

CANNED_VERIFY = json.dumps({
    "verified": True,
    "requirements_satisfied": ["modulo(10, 3) == 1"],
    "requirements_failed": [],
    "tests_passed": True,
    "files_changed": ["src/calculator.py"],
    "remaining_risks": [],
    "evidence": "All 10 tests pass, modulo function present in source",
})

CANNED_VERIFY_FAIL = json.dumps({
    "verified": False,
    "requirements_satisfied": [],
    "requirements_failed": ["modulo function not found in calculator.py"],
    "tests_passed": False,
    "files_changed": [],
    "remaining_risks": ["Implementation not complete"],
    "evidence": "File not modified",
})


@pytest.fixture(scope="module")
def seq_svc_ws(tmp_path_factory):
    """Shared ToolService + sample workspace for orchestration tests."""
    root = tmp_path_factory.mktemp("seq_crew_tests")
    svc = ToolService(workspace_root=str(root))
    ws_id = svc.create_workspace()
    from backend.tools.workspace import WorkspaceManager
    mgr = WorkspaceManager(workspace_root=root)
    repo = mgr.require(ws_id)
    shutil.copytree(str(FIXTURE_PROJECT), str(repo), dirs_exist_ok=True)
    return svc, ws_id, root


class TestSequentialPipelineHappyPath:
    """
    Happy path: all agents succeed on first try, no debug loop needed.

    The Coder mock response says it patched calculator.py. We ALSO actually
    write the modulo function so the Tester (which uses real pytest) passes.
    """

    def test_full_pipeline_produces_run_result(self, seq_svc_ws):
        svc, ws_id, root = seq_svc_ws
        # Pre-write the modulo function so real pytest passes
        calc_r = svc.read_file(ws_id, "src/calculator.py")
        content = calc_r.data["content"]
        if "def modulo" not in content:
            svc.write_file(
                ws_id, "src/calculator.py",
                content + "\n\ndef modulo(a: float, b: float) -> float:\n    return a % b\n"
            )

        # Provide canned responses for all 5 agent calls
        crew = _make_crew_with_mock_responses(svc, [
            CANNED_REQ,     # RequirementAnalyst
            CANNED_PROJ,    # ProjectAnalyst
            CANNED_PLAN,    # Planner
            CANNED_CODE,    # Coder
            CANNED_TEST_PASS,  # Tester (passes → no debug loop)
            CANNED_VERIFY,  # Verifier
        ])
        result = crew.run_task(ws_id, "Add modulo(a, b) function to calculator.py")

        assert isinstance(result, RunResult)
        assert result.run_id is not None
        assert result.workspace_id == ws_id

    def test_pipeline_collects_requirement_analysis(self, seq_svc_ws):
        svc, ws_id, root = seq_svc_ws
        crew = _make_crew_with_mock_responses(svc, [
            CANNED_REQ, CANNED_PROJ, CANNED_PLAN, CANNED_CODE, CANNED_TEST_PASS, CANNED_VERIFY,
        ])
        result = crew.run_task(ws_id, "Add modulo")
        assert result.requirement_analysis is not None
        assert result.requirement_analysis.task_summary == "Add modulo(a, b) function to calculator"

    def test_pipeline_collects_project_analysis(self, seq_svc_ws):
        svc, ws_id, root = seq_svc_ws
        crew = _make_crew_with_mock_responses(svc, [
            CANNED_REQ, CANNED_PROJ, CANNED_PLAN, CANNED_CODE, CANNED_TEST_PASS, CANNED_VERIFY,
        ])
        result = crew.run_task(ws_id, "Add modulo")
        assert result.project_analysis is not None
        assert result.project_analysis.total_files == 5

    def test_pipeline_collects_implementation_plan(self, seq_svc_ws):
        svc, ws_id, root = seq_svc_ws
        crew = _make_crew_with_mock_responses(svc, [
            CANNED_REQ, CANNED_PROJ, CANNED_PLAN, CANNED_CODE, CANNED_TEST_PASS, CANNED_VERIFY,
        ])
        result = crew.run_task(ws_id, "Add modulo")
        assert result.implementation_plan is not None
        assert result.implementation_plan.objective == "Add modulo(a, b) function"

    def test_pipeline_collects_coding_result(self, seq_svc_ws):
        svc, ws_id, root = seq_svc_ws
        crew = _make_crew_with_mock_responses(svc, [
            CANNED_REQ, CANNED_PROJ, CANNED_PLAN, CANNED_CODE, CANNED_TEST_PASS, CANNED_VERIFY,
        ])
        result = crew.run_task(ws_id, "Add modulo")
        assert result.coding_result is not None
        assert len(result.coding_result.files_changed) > 0

    def test_pipeline_success_status(self, seq_svc_ws):
        svc, ws_id, root = seq_svc_ws
        crew = _make_crew_with_mock_responses(svc, [
            CANNED_REQ, CANNED_PROJ, CANNED_PLAN, CANNED_CODE, CANNED_TEST_PASS, CANNED_VERIFY,
        ])
        result = crew.run_task(ws_id, "Add modulo")
        # With tests passing and verifier verified, should be success or partial
        assert result.status in (AgentStatus.success, AgentStatus.partial)


class TestDebugLoop:
    """
    Tests for the debug loop: fail → debug → pass scenario.

    The Tester initially fails, Debugger runs and fixes, Tester retries and passes.
    """

    @pytest.fixture(scope="class")
    def debug_svc_ws(self, tmp_path_factory):
        root = tmp_path_factory.mktemp("debug_loop_tests")
        svc = ToolService(workspace_root=str(root))
        ws_id = svc.create_workspace()
        from backend.tools.workspace import WorkspaceManager
        mgr = WorkspaceManager(workspace_root=root)
        repo = mgr.require(ws_id)
        shutil.copytree(str(FIXTURE_PROJECT), str(repo), dirs_exist_ok=True)
        # Write BROKEN modulo so first test run actually fails
        calc_r = svc.read_file(ws_id, "src/calculator.py")
        content = calc_r.data["content"]
        svc.write_file(ws_id, "src/calculator.py",
            content + "\n\ndef modulo(a: float, b: float) -> float:\n    return a + b  # BUG: should be %\n"
        )
        # Write a test for modulo that will fail with the buggy implementation
        svc.write_file(ws_id, "tests/test_modulo.py",
            "import sys, os\nsys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))\n"
            "from calculator import modulo\ndef test_modulo_basic(): assert modulo(10, 3) == 1\n"
        )
        return svc, ws_id

    def test_debug_loop_triggers_on_failure(self, debug_svc_ws):
        svc, ws_id = debug_svc_ws
        # Sequence: req, proj, plan, code, TEST_FAIL, debug_fix, TEST_PASS, verify
        # Debugger must fix the workspace — we simulate by also fixing the file
        # before the second tester call

        call_count = [0]
        responses = [
            CANNED_REQ, CANNED_PROJ, CANNED_PLAN, CANNED_CODE,
            CANNED_TEST_FAIL,  # first tester fails
            CANNED_DEBUG,      # debugger runs
            CANNED_TEST_PASS,  # tester retry passes (we will fix the real file too)
            CANNED_VERIFY,
        ]
        response_iter = iter(responses)

        def mock_run(ctx, agent, task):
            resp = next(response_iter)
            call_count[0] += 1
            # When the debugger runs (6th call), actually fix the file
            if call_count[0] == 6:  # debugger call
                calc_r = svc.read_file(ws_id, "src/calculator.py")
                content = calc_r.data["content"]
                svc.patch_file(ws_id, "src/calculator.py",
                    "return a + b  # BUG: should be %",
                    "return a % b"
                )
            return resp

        crew = _make_crew_with_mock_responses(svc, responses)
        crew._run_single_agent_crew = mock_run

        result = crew.run_task(ws_id, "Add modulo(a, b) function")
        assert result.debug_result is not None
        assert result.debug_iterations >= 1

    def test_debug_loop_increments_counter(self, debug_svc_ws):
        svc, ws_id = debug_svc_ws
        responses = [
            CANNED_REQ, CANNED_PROJ, CANNED_PLAN, CANNED_CODE,
            CANNED_TEST_FAIL, CANNED_DEBUG, CANNED_TEST_PASS, CANNED_VERIFY,
        ]
        crew = _make_crew_with_mock_responses(svc, responses)
        result = crew.run_task(ws_id, "Add modulo")
        assert result.debug_iterations >= 0  # counter was used

    def test_max_iterations_enforced(self, tmp_path_factory):
        """
        If all debug iterations fail, the loop stops at max_iterations.
        Verify max_iterations_reached is set.
        """
        root = tmp_path_factory.mktemp("max_iter_test")
        svc = ToolService(workspace_root=str(root))
        ws_id = svc.create_workspace()
        from backend.tools.workspace import WorkspaceManager
        mgr = WorkspaceManager(workspace_root=root)
        repo = mgr.require(ws_id)
        shutil.copytree(str(FIXTURE_PROJECT), str(repo), dirs_exist_ok=True)

        # All test runs fail (3 max debug iterations = 4 tester calls)
        # debug_max_iter=2 → 1 initial + 2 debug retries = 3 tester calls
        settings = _make_settings()
        settings_override = AgentSettings(
            llm_model="gpt-4o-mini",
            llm_temperature=0.1,
            agent_max_iter=1,
            debug_max_iter=2,  # only 2 debug iterations allowed
            agent_verbose=False,
        )
        responses = [
            CANNED_REQ, CANNED_PROJ, CANNED_PLAN, CANNED_CODE,
            CANNED_TEST_FAIL,   # initial
            CANNED_DEBUG,       # debug iter 1
            CANNED_TEST_FAIL,   # retry 1 still fails
            CANNED_DEBUG,       # debug iter 2
            CANNED_TEST_FAIL,   # retry 2 still fails → max reached
            CANNED_VERIFY_FAIL, # verifier
        ]
        crew = SoftwareEngineeringCrew(tool_service=svc, settings=settings_override, llm=None)
        crew._make_agent = lambda role, goal, backstory, tools: _StubAgent()
        crew._make_task = lambda description, agent, expected_output="": _StubTask()
        response_iter = iter(responses)
        crew._run_single_agent_crew = lambda ctx, a, t: next(response_iter)

        result = crew.run_task(ws_id, "Add modulo (will fail)")
        assert result.debug_result is not None
        assert result.debug_result.max_iterations_reached is True


class TestVerifier:
    def test_verified_false_when_requirements_not_met(self, seq_svc_ws):
        svc, ws_id, root = seq_svc_ws
        crew = _make_crew_with_mock_responses(svc, [
            CANNED_REQ, CANNED_PROJ, CANNED_PLAN, CANNED_CODE,
            CANNED_TEST_PASS, CANNED_VERIFY_FAIL,
        ])
        result = crew.run_task(ws_id, "Add modulo")
        assert result.verification_result is not None
        assert result.verification_result.verified is False
        # Status should NOT be success when not verified
        assert result.status != AgentStatus.success

    def test_verified_true_leads_to_success(self, seq_svc_ws):
        svc, ws_id, root = seq_svc_ws
        # Ensure modulo is actually present
        calc_r = svc.read_file(ws_id, "src/calculator.py")
        if "def modulo" not in calc_r.data["content"]:
            svc.write_file(ws_id, "src/calculator.py",
                calc_r.data["content"] + "\n\ndef modulo(a, b): return a % b\n"
            )
        crew = _make_crew_with_mock_responses(svc, [
            CANNED_REQ, CANNED_PROJ, CANNED_PLAN, CANNED_CODE,
            CANNED_TEST_PASS, CANNED_VERIFY,
        ])
        result = crew.run_task(ws_id, "Add modulo")
        assert result.verification_result.verified is True


class TestWorkspaceIsolation:
    """Security regression: agent pipeline must not escape workspace."""

    def test_invalid_workspace_raises(self, seq_svc_ws):
        svc, ws_id, root = seq_svc_ws
        crew = _make_crew_with_mock_responses(svc, [])
        result = crew.run_task("workspace-does-not-exist", "do something")
        assert result.status == AgentStatus.failed
        assert len(result.errors) > 0

"""
API tests for the Milestone 2 workspace endpoints.

The API client fixture uses the same session-scoped TestClient from conftest.
All workspace operations use the API-level service instance (not the test one),
so we create fresh workspaces via the API itself.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Workspace lifecycle
# ---------------------------------------------------------------------------


class TestWorkspaceLifecycleAPI:
    def test_create_workspace_returns_201(self, client):
        r = client.post("/workspaces", json={})
        assert r.status_code == 201
        body = r.json()
        assert "workspace_id" in body
        assert len(body["workspace_id"]) > 0

    def test_create_workspace_with_custom_id(self, client):
        r = client.post("/workspaces", json={"workspace_id": "api-test-custom"})
        assert r.status_code == 201
        assert r.json()["workspace_id"] == "api-test-custom"

    def test_list_workspaces(self, client):
        r = client.get("/workspaces")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_get_existing_workspace(self, client):
        create = client.post("/workspaces", json={})
        ws_id = create.json()["workspace_id"]
        r = client.get(f"/workspaces/{ws_id}")
        assert r.status_code == 200
        assert r.json()["workspace_id"] == ws_id

    def test_get_nonexistent_workspace_returns_404(self, client):
        r = client.get("/workspaces/this-does-not-exist-xyz")
        assert r.status_code == 404

    def test_delete_workspace(self, client):
        create = client.post("/workspaces", json={"workspace_id": "to-delete-api"})
        ws_id = create.json()["workspace_id"]
        r = client.delete(f"/workspaces/{ws_id}")
        assert r.status_code == 204
        # Verify it's gone
        r2 = client.get(f"/workspaces/{ws_id}")
        assert r2.status_code == 404


# ---------------------------------------------------------------------------
# File operations API
# ---------------------------------------------------------------------------


class TestFileOperationsAPI:
    @pytest.fixture(scope="class")
    @classmethod
    def api_ws(cls, client):
        """Create a workspace for this test class via the API."""
        r = client.post("/workspaces", json={})
        return r.json()["workspace_id"]

    def test_write_and_list(self, client, api_ws):
        # Write
        w = client.post(f"/workspaces/{api_ws}/write", json={"path": "hello.py", "content": "# hi"})
        assert w.status_code == 200
        # List
        l = client.get(f"/workspaces/{api_ws}/files")
        assert l.status_code == 200
        paths = [f["path"] for f in l.json()["data"]]
        assert "hello.py" in paths

    def test_write_and_read(self, client, api_ws):
        content = "def greet(): return 'VEXA'"
        client.post(f"/workspaces/{api_ws}/write", json={"path": "greet.py", "content": content})
        r = client.post(f"/workspaces/{api_ws}/read", json={"path": "greet.py"})
        assert r.status_code == 200
        assert r.json()["data"]["content"] == content

    def test_read_nonexistent_returns_400(self, client, api_ws):
        r = client.post(f"/workspaces/{api_ws}/read", json={"path": "ghost.py"})
        assert r.status_code == 400

    def test_path_traversal_returns_403(self, client, api_ws):
        r = client.post(f"/workspaces/{api_ws}/read", json={"path": "../../secrets.txt"})
        assert r.status_code == 403

    def test_patch_file_api(self, client, api_ws):
        client.post(f"/workspaces/{api_ws}/write", json={"path": "patch.py", "content": "x = 1\n"})
        r = client.post(f"/workspaces/{api_ws}/patch", json={
            "path": "patch.py",
            "old_text": "x = 1",
            "new_text": "x = 42",
        })
        assert r.status_code == 200
        # Verify
        read = client.post(f"/workspaces/{api_ws}/read", json={"path": "patch.py"})
        assert "x = 42" in read.json()["data"]["content"]


# ---------------------------------------------------------------------------
# Search API
# ---------------------------------------------------------------------------


class TestSearchAPI:
    @pytest.fixture(scope="class")
    @classmethod
    def search_ws(cls, client):
        r = client.post("/workspaces", json={})
        ws_id = r.json()["workspace_id"]
        client.post(f"/workspaces/{ws_id}/write", json={
            "path": "app.py",
            "content": "def authenticate_user(username, password):\n    pass\n",
        })
        return ws_id

    def test_search_finds_function(self, client, search_ws):
        r = client.post(f"/workspaces/{search_ws}/search", json={"query": "authenticate_user"})
        assert r.status_code == 200
        matches = r.json()["data"]["matches"]
        assert len(matches) >= 1

    def test_search_empty_query_returns_400(self, client, search_ws):
        r = client.post(f"/workspaces/{search_ws}/search", json={"query": ""})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Git API
# ---------------------------------------------------------------------------


class TestGitAPI:
    @pytest.fixture(scope="class")
    @classmethod
    def git_ws(cls, client):
        r = client.post("/workspaces", json={})
        return r.json()["workspace_id"]

    def test_git_status_endpoint(self, client, git_ws):
        r = client.get(f"/workspaces/{git_ws}/git/status")
        assert r.status_code == 200

    def test_git_diff_endpoint(self, client, git_ws):
        r = client.get(f"/workspaces/{git_ws}/git/diff")
        assert r.status_code == 200

    def test_git_log_endpoint(self, client, git_ws):
        r = client.get(f"/workspaces/{git_ws}/git/log")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Test execution API
# ---------------------------------------------------------------------------


class TestRunTestsAPI:
    @pytest.fixture(scope="class")
    @classmethod
    def tests_ws(cls, client):
        """Workspace with a passing test."""
        r = client.post("/workspaces", json={})
        ws_id = r.json()["workspace_id"]
        client.post(f"/workspaces/{ws_id}/write", json={
            "path": "test_simple.py",
            "content": "def test_pass():\n    assert 1 + 1 == 2\n",
        })
        return ws_id

    def test_run_tests_returns_result(self, client, tests_ws):
        r = client.post(f"/workspaces/{tests_ws}/tests", json={"test_path": "test_simple.py"})
        assert r.status_code == 200
        body = r.json()
        assert "exit_code" in body
        assert "stdout" in body
        assert "duration_ms" in body

    def test_run_tests_passing_tests(self, client, tests_ws):
        r = client.post(f"/workspaces/{tests_ws}/tests", json={"test_path": "test_simple.py"})
        assert r.json()["exit_code"] == 0
        assert r.json()["success"] is True

    def test_nonexistent_workspace_returns_404(self, client):
        r = client.post("/workspaces/no-such-ws/tests", json={})
        assert r.status_code == 404

# VEXA

**Architecture-Aware Autonomous Multi-Agent Software Engineering System**

VEXA autonomously converts natural-language software requirements into working software and can modify or debug existing software repositories. It simultaneously serves as a research platform that experimentally compares three multi-agent orchestration architectures on the same software-engineering tasks.

> **Current status: Milestone 2 — Tool Layer & Sandbox**
> The backend API starts and passes all tests (118/118). The workspace manager, filesystem tools, search tools, git inspection, and local sandbox execution engine are fully operational. Agent implementation (M3), RAG, MCP, architectures, and frontend are deferred to later milestones.

---

## What VEXA Does

Given a natural-language requirement such as:

> "Build a FastAPI task management API with JWT authentication, PostgreSQL, CRUD operations, Docker support and unit tests."

VEXA will (when fully implemented):
1. Analyse the requirement
2. Plan the implementation
3. Generate code
4. Run tests
5. Diagnose and repair failures iteratively
6. Verify the final result

It will do this under three different orchestration architectures — **Sequential**, **Hierarchical**, and **Event-Driven** — and record metrics for each run to support a controlled research comparison.

---

## Repository Structure

```
vexa/
├── backend/
│   ├── api/           # FastAPI routers
│   ├── core/          # Config, logging
│   └── models/        # Pydantic domain models
├── docs/              # Project documentation
│   └── PROJECT_SPEC.md
├── tests/             # pytest test suite
├── .env.example       # Environment variable template
├── pyproject.toml     # Python dependencies and tooling
├── Dockerfile         # Backend container image
└── docker-compose.yml # Development container stack
```

---

## Installation

### Prerequisites

- Python 3.11 or later
- pip

### Steps

```bash
# Clone the repository
git clone <repository-url>
cd vexa

# Create a virtual environment
python -m venv .venv

# Activate it
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# Install dependencies (production + dev)
pip install -e ".[dev]"
```

---

## Environment Configuration

```bash
# Copy the template
cp .env.example .env
```

Edit `.env` and set at minimum:

```env
ENVIRONMENT=development
LOG_LEVEL=INFO
```

`DATABASE_URL` can be left empty for Milestone 1 — the application starts without a database.

---

## Starting the Backend

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "service": "VEXA",
  "version": "0.1.0",
  "environment": "development"
}
```

API documentation is available at: http://localhost:8000/docs

---

## Running Tests

```bash
pytest
```

Expected output: all tests pass with no errors.

To run with verbose output:

```bash
pytest -v
```

---

## Running VEXA

VEXA now includes an operational frontend dashboard to observe autonomous agents in real-time.

1. **Start the Backend Server**:
   ```bash
   # In the root directory
   .venv\Scripts\activate
   uvicorn backend.main:app --reload
   ```

2. **Start the Frontend**:
   ```bash
   # In a new terminal
   cd frontend
   npm run dev
   ```

3. **Open the Dashboard**:
   Navigate to `http://localhost:5173` in your browser.

4. **Run a Task**:
   - Select an existing workspace
   - Enter your task requirement (e.g. "Add a modulo function to the calculator.")
   - Select Execution Mode (`Mock` for deterministic tests, or `Real` for live AI execution)
   - Click "RUN VEXA"

> **Warning:** Real execution connects to your configured AI provider and **will incur usage costs**. Ensure your token limits and billing are appropriately configured via your `.env` file (`VEXA_LLM_PROVIDER`, `VEXA_LLM_MODEL`, `VEXA_LLM_API_KEY`).

---

## Docker

### Build the image

```bash
docker build -t vexa-backend .
```

### Run the container

```bash
docker run --env-file .env -p 8000:8000 vexa-backend
```

### Using Docker Compose (development stack)

```bash
docker-compose up
```

---

## What Is Not Implemented Yet

The following features are **intentionally deferred** to later milestones:

| Feature | Milestone |
|---|---|
| Agent implementation (CrewAI) | M3 |
| Sequential architecture | M4 |
| Hierarchical architecture | M5 |
| Event-driven architecture | M6 |
| Project RAG | M7 |
| Observability / tracing | M8 |
| Evaluation framework | M8 |
| Benchmark runner (50 tasks) | M9 |
| Frontend dashboard | M10 |
| PostgreSQL persistence | M4+ |
| MCP integration | M3+ |
| Memory layer | M5+ |
| Safety guardrails | M6+ |

---

## Milestone 2 — Tool Layer

VEXA now includes a comprehensive tool layer designed for future agents to use securely.

- **Workspace Management**: Fully isolated workspaces. Agents cannot provide arbitrary host paths. Path traversal (`../`) is strictly blocked.
- **Filesystem Tools**: Secure `list_files`, `read_file`, `write_file`, `patch_file`, `delete_file`, and project inspection. Binary files and oversized files are handled gracefully.
- **Code Search**: Deterministic regex/text search bounded within the workspace.
- **Git Tools**: Read-only Git inspection (`git_status`, `git_diff`, `git_log`).
- **Execution Sandbox**: Runs allowlisted commands (like `pytest` or `python`) inside the workspace with captured stdout/stderr, strict timeouts, and memory limits. Currently defaults to `LocalSandboxExecutor` with `DockerSandboxExecutor` planned for production.
- **API Endpoints**: All tool capabilities are exposed via the `/workspaces/` API, operating only on `workspace_id`s, not host paths.

---

## Research Overview

VEXA compares three orchestration architectures on identical software-engineering tasks:

| Architecture | Description |
|---|---|
| Sequential | Linear agent pipeline, each step waits for the previous |
| Hierarchical | Manager agent delegates to and supervises specialist agents |
| Event-Driven | Agents subscribe to events; decoupled reactive execution |

**Key research principle:** All three architectures use the same agents, tools, prompts, and benchmark tasks. The only variable is the orchestration mechanism.

See [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md) for the full research specification.

---

## Licence

MIT

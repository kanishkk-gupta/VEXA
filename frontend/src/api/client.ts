// VEXA API Client

const API_BASE_URL = 'http://localhost:8000';

export interface Workspace {
  workspace_id: string;
  project_root: string;
  status: string;
  created_at: string;
}

export interface AgentEvent {
  timestamp: string;
  agent: string;
  event: string;
  detail: string;
  duration_ms: number | null;
}

export interface FileChange {
  path: string;
  action: string;
  diff: string;
}

export interface RunResult {
  run_id: string;
  workspace_id: string;
  requirement: string;
  architecture: string;
  execution_mode: string;
  provider: string | null;
  model: string | null;
  current_agent: string;
  started_at: string;
  finished_at: string | null;
  files_changed: string[] | FileChange[]; // Depending on if it's full result or context
  test_attempts: number;
  debug_iterations: number;
  errors: string[];
  usage_available: boolean;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  estimated_cost_usd: number | null;
  events: AgentEvent[];
}

export interface AgentRunResponse {
  run_id: string;
  workspace_id: string;
  execution_mode: string;
  architecture: string;
  status: string;
  total_duration_ms: number;
  files_changed: string[];
  debug_iterations: number;
  errors: string[];
  result: RunResult;
}

export interface AgentRunListResponse {
  runs: AgentRunResponse[];
}

export const VexaApi = {
  getWorkspaces: async (): Promise<{ workspaces: Workspace[] }> => {
    const response = await fetch(`${API_BASE_URL}/workspaces`);
    if (!response.ok) throw new Error('Failed to fetch workspaces');
    const data: string[] = await response.json();
    return {
      workspaces: data.map(id => ({
        workspace_id: id,
        project_root: '',
        status: '',
        created_at: ''
      }))
    };
  },

  createWorkspace: async (custom_id?: string, output_path?: string): Promise<Workspace> => {
    const response = await fetch(`${API_BASE_URL}/workspaces`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workspace_id: custom_id, output_path: output_path || null })
    });
    if (!response.ok) throw new Error('Failed to create workspace');
    const data = await response.json();
    return {
      workspace_id: data.workspace_id,
      project_root: data.output_path || '',
      status: '',
      created_at: ''
    };
  },

  submitAgentRunAsync: async (workspace_id: string, requirement: string, execution_mode: string): Promise<AgentRunResponse> => {
    const response = await fetch(`${API_BASE_URL}/agent-runs/async`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workspace_id, requirement, execution_mode })
    });
    if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || 'Failed to submit run');
    }
    return response.json();
  },

  getAgentRun: async (run_id: string): Promise<AgentRunResponse> => {
    const response = await fetch(`${API_BASE_URL}/agent-runs/${run_id}`);
    if (!response.ok) throw new Error('Failed to fetch run');
    return response.json();
  },

  listAgentRuns: async (): Promise<AgentRunListResponse> => {
    const response = await fetch(`${API_BASE_URL}/agent-runs`);
    if (!response.ok) throw new Error('Failed to fetch runs');
    return response.json();
  },
  
  getFileDiff: async (workspace_id: string): Promise<any> => {
    // Uses git diff endpoint
    const response = await fetch(`${API_BASE_URL}/workspaces/${workspace_id}/git/diff`);
    if (!response.ok) throw new Error('Failed to get diff');
    return response.json();
  },

  listFiles: async (workspace_id: string): Promise<any> => {
    const response = await fetch(`${API_BASE_URL}/workspaces/${workspace_id}/files`);
    if (!response.ok) throw new Error('Failed to list files');
    return response.json();
  },

  readFile: async (workspace_id: string, path: string): Promise<any> => {
    const response = await fetch(`${API_BASE_URL}/workspaces/${workspace_id}/read`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path })
    });
    if (!response.ok) throw new Error('Failed to read file');
    return response.json();
  }
};


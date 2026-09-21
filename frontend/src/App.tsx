import { useState, useEffect, useRef } from 'react';
import { Play, Activity, Server, FileCode, CheckCircle, XCircle, Loader2, GitCommit, RefreshCw, Terminal, LayoutPanelLeft, Code2, MonitorPlay, Settings } from 'lucide-react';
import { VexaApi } from './api/client';
import type { Workspace, AgentRunResponse } from './api/client';
import { CodeEditor } from './CodeEditor';
import { WebPreview } from './WebPreview';

const AGENTS = [
  'RequirementAnalyst',
  'ProjectAnalyst',
  'Planner',
  'Coder',
  'Tester',
  'Debugger',
  'Verifier'
];

function App() {
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [selectedWorkspace, setSelectedWorkspace] = useState<string>('');
  const [requirement, setRequirement] = useState('');
  const [executionMode, setExecutionMode] = useState<'mock' | 'real'>('mock');
  const [provider] = useState('openai');
  const [outputPath, setOutputPath] = useState('');
  const [isCreatingWorkspace, setIsCreatingWorkspace] = useState(false);
  const [workspaceCreateError, setWorkspaceCreateError] = useState<string | null>(null);
  const [llmModel, setLlmModel] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [fallbackApiKey, setFallbackApiKey] = useState('');
  
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [runState, setRunState] = useState<AgentRunResponse | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [diff, setDiff] = useState<string | null>(null);
  const [rightTab, setRightTab] = useState<'observability' | 'code' | 'preview'>('observability');

  const logsEndRef = useRef<HTMLDivElement>(null);

  // Initial load
  useEffect(() => {
    VexaApi.getWorkspaces().then(data => {
      setWorkspaces(data.workspaces);
      if (data.workspaces.length > 0) {
        setSelectedWorkspace(data.workspaces[0].workspace_id);
      }
    }).catch(console.error);
  }, []);

  // Polling loop
  useEffect(() => {
    if (!activeRunId) return;
    
    let isPolling = true;
    const poll = async () => {
      try {
        const run = await VexaApi.getAgentRun(activeRunId);
        if (isPolling) {
          setRunState(run);
          if (run.status === 'completed' || run.status === 'failed' || run.status === 'partial') {
            isPolling = false;
            setActiveRunId(null);
            // Fetch final diff
            if (run.workspace_id) {
               VexaApi.getFileDiff(run.workspace_id).then(res => setDiff(res.diff)).catch(console.error);
            }
          }
        }
      } catch (err) {
        console.error(err);
      }
    };

    const intervalId = setInterval(() => {
      if (isPolling) poll();
    }, 1000);

    // Initial immediate poll
    poll();

    return () => {
      isPolling = false;
      clearInterval(intervalId);
    };
  }, [activeRunId]);

  // Auto scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [runState?.result?.events]);

  const handleRun = async () => {
    if (!selectedWorkspace || !requirement) return;
    setIsSubmitting(true);
    setRunState(null);
    setDiff(null);
    try {
      // We pass execution_mode and overrides
      const res = await VexaApi.submitAgentRunAsync(
        selectedWorkspace, 
        requirement, 
        executionMode,
        llmModel || undefined,
        apiKey || undefined,
        fallbackApiKey || undefined
      );
      setActiveRunId(res.run_id);
    } catch (err: any) {
      alert(`Error starting run: ${err.message}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCreateWorkspace = async () => {
    setIsCreatingWorkspace(true);
    setWorkspaceCreateError(null);
    try {
      const ws = await VexaApi.createWorkspace(undefined, outputPath || undefined);
      setWorkspaces(prev => [...prev, ws]);
      setSelectedWorkspace(ws.workspace_id);
      setOutputPath('');
    } catch (err: any) {
      setWorkspaceCreateError(err.message || 'Failed to create workspace');
    } finally {
      setIsCreatingWorkspace(false);
    }
  };

  const getAgentState = (agentName: string) => {
    if (!runState) return 'pending';
    if (runState.status === 'failed' && runState.result?.current_agent === agentName) return 'failed';
    
    // Check events to see if it started or completed
    const events = runState.result?.events || [];
    const hasStarted = events.some(e => e.agent === agentName && e.event === 'started');
    const hasCompleted = events.some(e => e.agent === agentName && e.event === 'completed');
    
    if (hasCompleted) return 'completed';
    if (hasStarted) return 'running';
    return 'pending';
  };

  return (
    <div className="min-h-screen flex flex-col bg-vexa-bg text-vexa-text font-sans">
      
      {/* Top Bar */}
      <header className="flex items-center justify-between px-6 py-4 bg-vexa-panel border-b border-vexa-border">
        <div className="flex items-center space-x-4">
          <Activity className="w-6 h-6 text-vexa-accent" />
          <h1 className="text-xl font-bold tracking-tight">VEXA</h1>
          <span className="text-xs text-vexa-muted tracking-widest uppercase ml-4 hidden sm:inline-block">
            Architecture-Aware Autonomous Agent System
          </span>
        </div>
        <div className="flex items-center space-x-6 text-sm">
          <div className="flex items-center space-x-2">
            <span className="text-vexa-muted">Architecture:</span>
            <span className="font-mono bg-black/30 px-2 py-1 rounded">Sequential</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-vexa-muted">Mode:</span>
            <span className={`font-mono px-2 py-1 rounded flex items-center space-x-1 ${executionMode === 'real' ? 'text-vexa-warning bg-vexa-warning/10' : 'text-vexa-success bg-vexa-success/10'}`}>
              <div className={`w-2 h-2 rounded-full ${executionMode === 'real' ? 'bg-vexa-warning' : 'bg-vexa-success'}`} />
              <span>{executionMode.toUpperCase()}</span>
            </span>
          </div>
          <div className="flex items-center space-x-2">
            <Server className="w-4 h-4 text-vexa-success" />
            <span className="text-vexa-success">Connected</span>
          </div>
        </div>
      </header>

      <main className="flex-1 flex flex-col lg:flex-row overflow-hidden">
        
        {/* Left Column: Input Controls */}
        <div className="w-full lg:w-1/3 p-6 flex flex-col border-r border-vexa-border overflow-y-auto">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-vexa-muted mb-4">Task Specification</h2>
          
          <div className="space-y-4 flex-1">
            <div className="flex flex-col space-y-2">
              <label className="text-sm text-vexa-muted">Workspace</label>
              <select 
                className="bg-vexa-panel border border-vexa-border p-2 rounded focus:outline-none focus:border-vexa-accent transition-colors"
                value={selectedWorkspace}
                onChange={e => setSelectedWorkspace(e.target.value)}
                disabled={activeRunId !== null}
              >
                {(workspaces || []).map(ws => (
                  <option key={ws.workspace_id} value={ws.workspace_id}>{ws.workspace_id}</option>
                ))}
              </select>
            </div>

            {/* Create New Workspace */}
            <div className="flex flex-col space-y-2 border border-vexa-border rounded p-3 bg-black/10">
              <label className="text-xs font-semibold uppercase tracking-wider text-vexa-muted">Create New Workspace</label>
              <div className="flex flex-col space-y-1">
                <label className="text-xs text-vexa-muted">Output Folder <span className="text-vexa-muted/50">(optional — leave blank to use default)</span></label>
                <input
                  id="output-path-input"
                  type="text"
                  className="bg-vexa-bg border border-vexa-border p-2 rounded font-mono text-xs focus:outline-none focus:border-vexa-accent transition-colors"
                  placeholder="e.g. C:\Users\Kanishk\Projects\my-app"
                  value={outputPath}
                  onChange={e => setOutputPath(e.target.value)}
                  disabled={isCreatingWorkspace || activeRunId !== null}
                />
              </div>
              {workspaceCreateError && (
                <p className="text-xs text-red-400">{workspaceCreateError}</p>
              )}
              <button
                id="create-workspace-btn"
                className={`w-full py-1.5 rounded text-xs font-semibold flex items-center justify-center space-x-1 transition-colors ${
                  isCreatingWorkspace
                    ? 'bg-vexa-border text-vexa-muted cursor-not-allowed'
                    : 'bg-vexa-accent/20 hover:bg-vexa-accent/40 border border-vexa-accent text-vexa-accent'
                }`}
                onClick={handleCreateWorkspace}
                disabled={isCreatingWorkspace || activeRunId !== null}
              >
                {isCreatingWorkspace ? (
                  <><Loader2 className="w-3 h-3 animate-spin" /><span>Creating...</span></>
                ) : (
                  <span>+ New Workspace</span>
                )}
              </button>
            </div>

            <div className="flex flex-col space-y-2">
              <label className="text-sm text-vexa-muted">Requirement</label>
              <textarea 
                className="bg-vexa-panel border border-vexa-border p-3 rounded h-40 focus:outline-none focus:border-vexa-accent transition-colors resize-none font-mono text-sm"
                placeholder="Describe what you want VEXA to build or modify..."
                value={requirement}
                onChange={e => setRequirement(e.target.value)}
                disabled={activeRunId !== null}
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="flex flex-col space-y-2">
                <label className="text-sm text-vexa-muted">Execution Mode</label>
                <select 
                  className="bg-vexa-panel border border-vexa-border p-2 rounded focus:outline-none focus:border-vexa-accent"
                  value={executionMode}
                  onChange={e => setExecutionMode(e.target.value as any)}
                  disabled={activeRunId !== null}
                >
                  <option value="mock">Mock (Deterministic)</option>
                  <option value="real">Real (Live LLM)</option>
                </select>
              </div>
            </div>

            {/* LLM Configuration Panel */}
            <div className="flex flex-col space-y-3 mt-4 border border-vexa-border rounded p-4 bg-black/10">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-vexa-muted flex items-center space-x-2">
                <Settings className="w-3 h-3" />
                <span>LLM Configuration</span>
              </h3>
              
              <div className="flex flex-col space-y-1">
                <label className="text-xs text-vexa-muted flex justify-between">
                  <span>Model String</span>
                  <span className="text-[10px] text-vexa-muted/50">Overrides .env default</span>
                </label>
                <div className="relative">
                  <input
                    list="models-list"
                    className="w-full bg-vexa-bg border border-vexa-border p-2 rounded font-mono text-xs focus:outline-none focus:border-vexa-accent transition-colors"
                    placeholder="e.g. groq/openai/gpt-oss-120b"
                    value={llmModel}
                    onChange={e => setLlmModel(e.target.value)}
                    disabled={activeRunId !== null}
                  />
                  <datalist id="models-list">
                    <option value="groq/llama-3.1-70b-versatile">[Free] Groq - Llama 3.1 70B</option>
                    <option value="groq/llama-3.1-8b-instant">[Free] Groq - Llama 3.1 8B</option>
                    <option value="groq/openai/gpt-oss-20b">[Custom] Groq - GPT OSS 20B</option>
                    <option value="groq/openai/gpt-oss-120b">[Custom] Groq - GPT OSS 120B</option>
                    <option value="gpt-4o-mini">[Paid] OpenAI - GPT-4o Mini</option>
                    <option value="gpt-4o">[Paid] OpenAI - GPT-4o</option>
                    <option value="anthropic/claude-3-5-sonnet-20241022">[Paid] Anthropic - Claude 3.5 Sonnet</option>
                  </datalist>
                </div>
              </div>

              <div className="flex flex-col space-y-1">
                <label className="text-xs text-vexa-muted">Primary API Key</label>
                <input
                  type="password"
                  className="bg-vexa-bg border border-vexa-border p-2 rounded font-mono text-xs focus:outline-none focus:border-vexa-accent transition-colors"
                  placeholder="Leave blank to use .env key"
                  value={apiKey}
                  onChange={e => setApiKey(e.target.value)}
                  disabled={activeRunId !== null}
                />
              </div>

              <div className="flex flex-col space-y-1">
                <label className="text-xs text-vexa-muted">Fallback API Key (Mistral)</label>
                <input
                  type="password"
                  className="bg-vexa-bg border border-vexa-border p-2 rounded font-mono text-xs focus:outline-none focus:border-vexa-accent transition-colors"
                  placeholder="Leave blank to use .env key"
                  value={fallbackApiKey}
                  onChange={e => setFallbackApiKey(e.target.value)}
                  disabled={activeRunId !== null}
                />
              </div>
            </div>

          </div>

          <div className="mt-8">
            <button
              className={`w-full py-3 rounded font-bold flex items-center justify-center space-x-2 transition-colors ${
                !selectedWorkspace || !requirement || activeRunId || isSubmitting
                  ? 'bg-vexa-border text-vexa-muted cursor-not-allowed'
                  : 'bg-vexa-accent hover:bg-vexa-accent-hover text-white'
              }`}
              onClick={handleRun}
              disabled={!selectedWorkspace || !requirement || activeRunId !== null || isSubmitting}
            >
              {(isSubmitting || activeRunId) ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>VEXA IS RUNNING</span>
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  <span>RUN VEXA</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Right Column: Dynamic Tabs */}
        <div className="w-full lg:w-2/3 flex flex-col bg-black/20">
          
          {/* Tabs Bar */}
          <div className="flex border-b border-vexa-border bg-vexa-bg">
            <button 
              className={`flex-1 py-3 text-xs font-bold uppercase tracking-wider flex items-center justify-center space-x-2 transition-colors ${rightTab === 'observability' ? 'text-vexa-accent border-b-2 border-vexa-accent bg-vexa-panel' : 'text-vexa-muted hover:bg-vexa-panel hover:text-white'}`}
              onClick={() => setRightTab('observability')}
            >
              <LayoutPanelLeft className="w-4 h-4" />
              <span>Observability</span>
            </button>
            <button 
              className={`flex-1 py-3 text-xs font-bold uppercase tracking-wider flex items-center justify-center space-x-2 transition-colors ${rightTab === 'code' ? 'text-vexa-accent border-b-2 border-vexa-accent bg-vexa-panel' : 'text-vexa-muted hover:bg-vexa-panel hover:text-white'}`}
              onClick={() => setRightTab('code')}
            >
              <Code2 className="w-4 h-4" />
              <span>Code Editor</span>
            </button>
            <button 
              className={`flex-1 py-3 text-xs font-bold uppercase tracking-wider flex items-center justify-center space-x-2 transition-colors ${rightTab === 'preview' ? 'text-vexa-accent border-b-2 border-vexa-accent bg-vexa-panel' : 'text-vexa-muted hover:bg-vexa-panel hover:text-white'}`}
              onClick={() => setRightTab('preview')}
            >
              <MonitorPlay className="w-4 h-4" />
              <span>Web Preview</span>
            </button>
          </div>

          {rightTab === 'observability' && (
            <div className="flex flex-col flex-1 overflow-hidden">
              {/* Summary Strip */}
              <div className="grid grid-cols-6 border-b border-vexa-border bg-vexa-panel/50 shrink-0">
                <MetricBox label="STATUS" value={runState ? runState.status.toUpperCase() : 'READY'} 
                  color={runState?.status === 'failed' ? 'text-vexa-error' : runState?.status === 'completed' ? 'text-vexa-success' : runState?.status === 'running' ? 'text-vexa-accent' : ''} />
                <MetricBox label="TIME" value={runState ? `${(runState.total_duration_ms / 1000).toFixed(1)}s` : '-'} />
                <MetricBox label="DEBUG ITERS" value={runState?.result?.debug_iterations?.toString() || '0'} />
                <MetricBox label="TOKENS" value={runState?.result?.usage_available ? runState.result.total_tokens.toLocaleString() : 'N/A'} />
                <MetricBox label="COST" value={runState?.result?.usage_available && runState.result.estimated_cost_usd !== null ? `$${runState.result.estimated_cost_usd.toFixed(4)}` : 'N/A'} />
                <MetricBox label="FILES" value={runState?.result?.files_changed?.length?.toString() || '0'} />
              </div>

              {/* Main Dashboard Area */}
              <div className="flex-1 grid grid-cols-1 md:grid-cols-2 gap-4 p-4 overflow-y-auto">
                
                {/* Agent Pipeline */}
                <div className="bg-vexa-panel border border-vexa-border rounded p-4 flex flex-col">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-vexa-muted mb-4 flex items-center"><Activity className="w-4 h-4 mr-2" /> Agent Pipeline</h3>
                  <div className="flex-1 flex flex-col space-y-1">
                    {AGENTS.map((agent, idx) => {
                      const state = getAgentState(agent);
                      return (
                        <div key={agent} className="flex items-center relative py-2">
                          {/* Vertical line connector */}
                          {idx !== AGENTS.length - 1 && (
                            <div className={`absolute left-3 top-8 bottom-[-8px] w-0.5 ${state === 'completed' ? 'bg-vexa-success' : 'bg-vexa-border'}`} />
                          )}
                          
                          <div className={`z-10 w-6 h-6 rounded-full flex items-center justify-center mr-4 border-2 ${
                            state === 'completed' ? 'bg-vexa-bg border-vexa-success text-vexa-success' :
                            state === 'running' ? 'bg-vexa-accent border-vexa-accent text-white animate-pulse' :
                            state === 'failed' ? 'bg-vexa-bg border-vexa-error text-vexa-error' :
                            'bg-vexa-bg border-vexa-border text-vexa-muted'
                          }`}>
                            {state === 'completed' ? <CheckCircle className="w-4 h-4" /> :
                             state === 'failed' ? <XCircle className="w-4 h-4" /> :
                             state === 'running' ? <RefreshCw className="w-3 h-3 animate-spin" /> :
                             <div className="w-2 h-2 rounded-full bg-vexa-muted" />}
                          </div>
                          
                          <span className={`font-mono text-sm ${state === 'running' ? 'text-white' : state === 'pending' ? 'text-vexa-muted' : ''}`}>
                            {agent}
                          </span>
                          {state === 'running' && <span className="ml-auto text-xs text-vexa-accent font-bold tracking-wider">RUNNING</span>}
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Live Activity Log */}
                <div className="bg-vexa-panel border border-vexa-border rounded p-4 flex flex-col h-96 md:h-auto">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-vexa-muted mb-4 flex items-center"><Terminal className="w-4 h-4 mr-2" /> Live Activity</h3>
                  <div className="flex-1 bg-black/40 border border-vexa-border rounded p-3 overflow-y-auto font-mono text-xs">
                    {(!runState || !runState.result?.events || runState.result.events.length === 0) ? (
                      <div className="text-vexa-muted italic">Waiting for execution to begin...</div>
                    ) : (
                      <div className="space-y-2">
                        {runState.result.events.map((e, i) => (
                          <div key={i} className="flex space-x-3 items-start">
                            <span className="text-vexa-muted shrink-0">{new Date(e.timestamp).toLocaleTimeString([], {hour12: false})}</span>
                            <span className={
                              e.event === 'started' || e.event === 'completed' ? 'text-vexa-accent' :
                              e.event === 'tool_failed' ? 'text-vexa-error' :
                              e.event === 'test_started' ? 'text-vexa-warning' :
                              'text-white'
                            }>[{e.agent}]</span>
                            <span className="text-gray-300 break-words">{e.event} {e.detail ? `- ${e.detail}` : ''}</span>
                          </div>
                        ))}
                        <div ref={logsEndRef} />
                      </div>
                    )}
                  </div>
                </div>

                {/* File Changes */}
                <div className="bg-vexa-panel border border-vexa-border rounded p-4 flex flex-col col-span-1 md:col-span-2">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-vexa-muted mb-4 flex items-center"><GitCommit className="w-4 h-4 mr-2" /> File Changes & Verification</h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <h4 className="text-xs text-vexa-muted">Modified Files:</h4>
                      {(!runState?.result?.files_changed || runState.result.files_changed.length === 0) ? (
                        <div className="text-sm text-vexa-muted font-mono bg-black/40 p-2 rounded">No files changed yet</div>
                      ) : (
                        <ul className="space-y-1">
                          {runState.result.files_changed.map((f, i) => {
                            const pathName = typeof f === 'string' ? f : (f as any).path;
                            return (
                              <li key={i} className="text-sm font-mono flex items-center text-vexa-success">
                                <FileCode className="w-3 h-3 mr-2" /> {pathName}
                              </li>
                            );
                          })}
                        </ul>
                      )}

                      {diff && (
                        <div className="mt-4">
                          <h4 className="text-xs text-vexa-muted mb-2">Final Diff:</h4>
                          <pre className="bg-black/40 border border-vexa-border p-3 rounded text-xs font-mono overflow-x-auto whitespace-pre-wrap max-h-60">
                            {diff || "No git diff available."}
                          </pre>
                        </div>
                      )}
                    </div>

                    <div>
                      <h4 className="text-xs text-vexa-muted mb-2">Errors / Verification:</h4>
                      {(!runState?.result?.errors || runState.result.errors.length === 0) ? (
                        <div className="text-sm text-vexa-success flex items-center font-mono bg-black/40 p-2 rounded">
                          <CheckCircle className="w-4 h-4 mr-2" /> No errors reported
                        </div>
                      ) : (
                        <div className="space-y-2">
                          {runState.result.errors.map((err, i) => (
                            <div key={i} className="text-xs font-mono bg-vexa-error/10 text-vexa-error border border-vexa-error/20 p-2 rounded break-words">
                              {err}
                            </div>
                          ))}
                        </div>
                      )}
                      
                      {runState?.status === 'completed' && (
                         <div className="mt-4 bg-vexa-success/10 border border-vexa-success/30 p-3 rounded flex items-center text-vexa-success">
                           <CheckCircle className="w-5 h-5 mr-3" />
                           <span className="font-bold">VERIFICATION SUCCESSFUL</span>
                         </div>
                      )}
                    </div>
                  </div>
                </div>

              </div>
            </div>
          )}

          {rightTab === 'code' && (
            <div className="flex-1 overflow-hidden">
              <CodeEditor workspaceId={selectedWorkspace} />
            </div>
          )}

          {rightTab === 'preview' && (
            <div className="flex-1 overflow-hidden">
              <WebPreview workspaceId={selectedWorkspace} />
            </div>
          )}

        </div>
      </main>
    </div>
  );
}

function MetricBox({ label, value, color = '' }: { label: string, value: string, color?: string }) {
  return (
    <div className="p-3 border-r border-vexa-border last:border-r-0 flex flex-col justify-center">
      <div className="text-[10px] font-bold text-vexa-muted mb-1">{label}</div>
      <div className={`font-mono text-sm tracking-tight ${color}`}>{value}</div>
    </div>
  );
}

export default App;

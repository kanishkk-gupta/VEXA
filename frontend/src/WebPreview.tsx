import { RefreshCw } from 'lucide-react';
import { useState } from 'react';

interface WebPreviewProps {
  workspaceId: string;
}

export function WebPreview({ workspaceId }: WebPreviewProps) {
  const [key, setKey] = useState(0);

  // The backend serves the workspace root at this endpoint
  // By default, it loads index.html
  const previewUrl = `http://localhost:8000/workspaces/${workspaceId}/serve/`;

  const handleRefresh = () => {
    setKey(k => k + 1);
  };

  if (!workspaceId) {
    return (
      <div className="flex-1 flex items-center justify-center text-vexa-muted">
        Select a workspace to preview.
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-vexa-panel">
      {/* Browser Chrome Bar */}
      <div className="flex items-center space-x-2 bg-vexa-bg p-2 border-b border-vexa-border">
        <button 
          onClick={handleRefresh}
          className="p-1 hover:bg-vexa-panel rounded text-vexa-muted hover:text-white transition-colors"
          title="Refresh Preview"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
        <div className="flex-1 bg-black/40 rounded px-3 py-1 font-mono text-xs text-vexa-muted truncate border border-vexa-border">
          {previewUrl}
        </div>
      </div>
      
      {/* Iframe */}
      <div className="flex-1 bg-white">
        <iframe
          key={key}
          src={previewUrl}
          className="w-full h-full border-0"
          title="Web Preview"
          sandbox="allow-scripts allow-same-origin allow-forms"
        />
      </div>
    </div>
  );
}

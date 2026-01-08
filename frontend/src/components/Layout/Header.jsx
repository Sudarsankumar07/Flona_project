import React from 'react';
import { Film, Sparkles } from 'lucide-react';

export default function Header() {
  return (
    <header className="bg-surface border-b border-slate-700 px-6 py-4">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-primary rounded-lg">
            <Film className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white flex items-center gap-2">
              Smart B-Roll Inserter
              <Sparkles className="w-5 h-5 text-yellow-400" />
            </h1>
            <p className="text-sm text-slate-400">AI-Powered Video Editing</p>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <a 
            href="https://github.com" 
            target="_blank"
            rel="noopener noreferrer"
            className="text-slate-400 hover:text-white transition-colors text-sm"
          >
            Documentation
          </a>
        </div>
      </div>
    </header>
  );
}

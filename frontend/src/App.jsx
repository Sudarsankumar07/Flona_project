import React from 'react';
import { AppProvider, useApp } from './context/AppContext';
import Header from './components/Layout/Header';
import ARollUpload from './components/Upload/ARollUpload';
import BRollUpload from './components/Upload/BRollUpload';
import UrlInput from './components/Upload/UrlInput';
import ApiConfig from './components/Config/ApiConfig';
import ProcessButton from './components/Config/ProcessButton';
import TimelineViewer from './components/Output/TimelineViewer';
import InsertionList from './components/Output/InsertionList';
import TranscriptView from './components/Output/TranscriptView';
import { Upload, Link2 } from 'lucide-react';

function InputModeToggle() {
  const { state, dispatch } = useApp();
  const { inputMode } = state;
  
  return (
    <div className="bg-surface rounded-xl p-3 border border-slate-700">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-sm text-slate-400">Input Method:</span>
      </div>
      <div className="flex gap-2">
        <button
          onClick={() => dispatch({ type: 'SET_INPUT_MODE', payload: 'file' })}
          className={`flex-1 flex items-center justify-center gap-2 py-2 px-3 rounded-lg 
            text-sm font-medium transition-all ${
            inputMode === 'file'
              ? 'bg-primary text-white'
              : 'bg-slate-800 text-slate-400 hover:text-white hover:bg-slate-700'
          }`}
        >
          <Upload className="w-4 h-4" />
          Upload Files
        </button>
        <button
          onClick={() => dispatch({ type: 'SET_INPUT_MODE', payload: 'url' })}
          className={`flex-1 flex items-center justify-center gap-2 py-2 px-3 rounded-lg 
            text-sm font-medium transition-all ${
            inputMode === 'url'
              ? 'bg-accent text-white'
              : 'bg-slate-800 text-slate-400 hover:text-white hover:bg-slate-700'
          }`}
        >
          <Link2 className="w-4 h-4" />
          Video URLs
        </button>
      </div>
    </div>
  );
}

function InputSection() {
  const { state } = useApp();
  const { inputMode } = state;
  
  if (inputMode === 'url') {
    return <UrlInput />;
  }
  
  return (
    <>
      <ARollUpload />
      <BRollUpload />
    </>
  );
}

function AppContent() {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      
      <main className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          {/* Left Panel - Upload & Config */}
          <div className="lg:col-span-2 space-y-4">
            <InputModeToggle />
            <InputSection />
            <ApiConfig />
            <ProcessButton />
          </div>
          
          {/* Right Panel - Output */}
          <div className="lg:col-span-3 space-y-4">
            <TimelineViewer />
            <InsertionList />
            <TranscriptView />
          </div>
        </div>
      </main>
      
      {/* Footer */}
      <footer className="border-t border-slate-800 mt-8 py-4 text-center text-sm text-slate-500">
        Smart B-Roll Inserter • Powered by AI
      </footer>
    </div>
  );
}

function App() {
  return (
    <AppProvider>
      <AppContent />
    </AppProvider>
  );
}

export default App;

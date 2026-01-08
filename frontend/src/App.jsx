import React from 'react';
import { AppProvider } from './context/AppContext';
import Header from './components/Layout/Header';
import ARollUpload from './components/Upload/ARollUpload';
import BRollUpload from './components/Upload/BRollUpload';
import ApiConfig from './components/Config/ApiConfig';
import ProcessButton from './components/Config/ProcessButton';
import TimelineViewer from './components/Output/TimelineViewer';
import InsertionList from './components/Output/InsertionList';
import TranscriptView from './components/Output/TranscriptView';

function App() {
  return (
    <AppProvider>
      <div className="min-h-screen bg-background">
        <Header />
        
        <main className="max-w-7xl mx-auto px-4 py-6">
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
            {/* Left Panel - Upload & Config */}
            <div className="lg:col-span-2 space-y-4">
              <ARollUpload />
              <BRollUpload />
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
    </AppProvider>
  );
}

export default App;

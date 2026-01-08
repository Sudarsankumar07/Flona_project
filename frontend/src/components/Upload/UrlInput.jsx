import React, { useState, useCallback } from 'react';
import { Link2, Upload, X, Plus, Trash2, FileJson, CheckCircle, AlertCircle } from 'lucide-react';
import { useApp } from '../../context/AppContext';
import toast from 'react-hot-toast';

export default function UrlInput() {
  const { state, dispatch } = useApp();
  const { urlMode } = state;
  
  const [arollUrl, setArollUrl] = useState('');
  const [arollMetadata, setArollMetadata] = useState('');
  const [brollUrls, setBrollUrls] = useState([
    { id: 'broll_1', url: '', metadata: '' }
  ]);
  const [jsonInput, setJsonInput] = useState('');
  const [inputMode, setInputMode] = useState('manual'); // 'manual' or 'json'
  
  const addBrollUrl = () => {
    const newId = `broll_${brollUrls.length + 1}`;
    setBrollUrls([...brollUrls, { id: newId, url: '', metadata: '' }]);
  };
  
  const removeBrollUrl = (id) => {
    if (brollUrls.length > 1) {
      setBrollUrls(brollUrls.filter(b => b.id !== id));
    }
  };
  
  const updateBrollUrl = (id, field, value) => {
    setBrollUrls(brollUrls.map(b => 
      b.id === id ? { ...b, [field]: value } : b
    ));
  };
  
  const validateUrl = (url) => {
    try {
      new URL(url);
      return url.match(/\.(mp4|mov|avi|mkv|webm)(\?.*)?$/i) !== null;
    } catch {
      return false;
    }
  };
  
  const handleSaveUrls = () => {
    // Validate A-roll URL
    if (!arollUrl || !validateUrl(arollUrl)) {
      toast.error('Please enter a valid A-roll video URL (.mp4, .mov, etc.)');
      return;
    }
    
    // Validate B-roll URLs
    const validBrolls = brollUrls.filter(b => b.url && validateUrl(b.url));
    if (validBrolls.length === 0) {
      toast.error('Please enter at least one valid B-roll video URL');
      return;
    }
    
    // Save to context
    dispatch({
      type: 'SET_URL_DATA',
      payload: {
        aroll: { url: arollUrl, metadata: arollMetadata },
        brolls: validBrolls
      }
    });
    
    toast.success(`Saved ${validBrolls.length + 1} video URLs`);
  };
  
  const handleParseJson = () => {
    try {
      const data = JSON.parse(jsonInput);
      
      // Validate structure
      if (!data.a_roll?.url) {
        toast.error('JSON must contain a_roll with url');
        return;
      }
      
      if (!data.b_rolls || !Array.isArray(data.b_rolls) || data.b_rolls.length === 0) {
        toast.error('JSON must contain b_rolls array');
        return;
      }
      
      // Extract data
      setArollUrl(data.a_roll.url);
      setArollMetadata(data.a_roll.metadata || '');
      
      setBrollUrls(data.b_rolls.map((b, idx) => ({
        id: b.id || `broll_${idx + 1}`,
        url: b.url || '',
        metadata: b.metadata || ''
      })));
      
      setInputMode('manual');
      toast.success('JSON parsed successfully! Review and save the URLs.');
      
    } catch (e) {
      toast.error('Invalid JSON format');
    }
  };
  
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (event) => {
      setJsonInput(event.target.result);
      setInputMode('json');
    };
    reader.readAsText(file);
  };
  
  const clearUrls = () => {
    dispatch({ type: 'CLEAR_URL_DATA' });
    setArollUrl('');
    setArollMetadata('');
    setBrollUrls([{ id: 'broll_1', url: '', metadata: '' }]);
    setJsonInput('');
  };
  
  const isUrlsSaved = urlMode?.aroll?.url;
  
  return (
    <div className="bg-surface rounded-xl p-4 border border-slate-700">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Link2 className="w-5 h-5 text-accent" />
          <h3 className="font-semibold text-white">Video URLs</h3>
        </div>
        
        {/* Mode Toggle */}
        <div className="flex gap-1 bg-slate-800 rounded-lg p-1">
          <button
            onClick={() => setInputMode('manual')}
            className={`px-3 py-1 text-xs rounded transition-colors ${
              inputMode === 'manual' 
                ? 'bg-primary text-white' 
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Manual
          </button>
          <button
            onClick={() => setInputMode('json')}
            className={`px-3 py-1 text-xs rounded transition-colors ${
              inputMode === 'json' 
                ? 'bg-primary text-white' 
                : 'text-slate-400 hover:text-white'
            }`}
          >
            JSON
          </button>
        </div>
      </div>
      
      {/* Status indicator */}
      {isUrlsSaved && (
        <div className="flex items-center gap-2 mb-4 p-2 bg-green-500/10 border border-green-500/30 rounded-lg">
          <CheckCircle className="w-4 h-4 text-green-500" />
          <span className="text-sm text-green-400">
            URLs saved: 1 A-roll + {urlMode.brolls?.length || 0} B-rolls
          </span>
          <button
            onClick={clearUrls}
            className="ml-auto text-xs text-red-400 hover:text-red-300"
          >
            Clear
          </button>
        </div>
      )}
      
      {inputMode === 'manual' ? (
        <div className="space-y-4">
          {/* A-Roll URL */}
          <div className="space-y-2">
            <label className="text-sm text-slate-400">A-Roll Video URL</label>
            <input
              type="url"
              value={arollUrl}
              onChange={(e) => setArollUrl(e.target.value)}
              placeholder="https://example.com/a_roll.mp4"
              className={`w-full bg-slate-800 border rounded-lg px-3 py-2 text-sm text-white
                placeholder-slate-500 focus:outline-none focus:border-primary
                ${arollUrl && !validateUrl(arollUrl) ? 'border-red-500' : 'border-slate-600'}`}
            />
            <input
              type="text"
              value={arollMetadata}
              onChange={(e) => setArollMetadata(e.target.value)}
              placeholder="Description (optional)"
              className="w-full bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-sm text-white
                placeholder-slate-500 focus:outline-none focus:border-primary"
            />
          </div>
          
          {/* B-Roll URLs */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm text-slate-400">B-Roll Video URLs</label>
              <button
                onClick={addBrollUrl}
                className="flex items-center gap-1 text-xs text-primary hover:text-primary/80"
              >
                <Plus className="w-3 h-3" />
                Add B-Roll
              </button>
            </div>
            
            <div className="space-y-3 max-h-60 overflow-y-auto pr-2">
              {brollUrls.map((broll, idx) => (
                <div key={broll.id} className="p-3 bg-slate-800/50 rounded-lg space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-medium text-secondary">B-Roll {idx + 1}</span>
                    {brollUrls.length > 1 && (
                      <button
                        onClick={() => removeBrollUrl(broll.id)}
                        className="text-slate-500 hover:text-red-400"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                  <input
                    type="url"
                    value={broll.url}
                    onChange={(e) => updateBrollUrl(broll.id, 'url', e.target.value)}
                    placeholder="https://example.com/broll.mp4"
                    className={`w-full bg-slate-700 border rounded px-2 py-1.5 text-xs text-white
                      placeholder-slate-500 focus:outline-none focus:border-primary
                      ${broll.url && !validateUrl(broll.url) ? 'border-red-500' : 'border-slate-600'}`}
                  />
                  <input
                    type="text"
                    value={broll.metadata}
                    onChange={(e) => updateBrollUrl(broll.id, 'metadata', e.target.value)}
                    placeholder="Description (helps AI match better)"
                    className="w-full bg-slate-700 border border-slate-600 rounded px-2 py-1.5 text-xs text-white
                      placeholder-slate-500 focus:outline-none focus:border-primary"
                  />
                </div>
              ))}
            </div>
          </div>
          
          {/* Save Button */}
          <button
            onClick={handleSaveUrls}
            className="w-full py-2 bg-accent hover:bg-accent/80 text-white rounded-lg 
              font-medium transition-colors flex items-center justify-center gap-2"
          >
            <CheckCircle className="w-4 h-4" />
            Save URLs
          </button>
        </div>
      ) : (
        <div className="space-y-4">
          {/* JSON Input */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm text-slate-400">Paste video_url.json content</label>
              <label className="flex items-center gap-1 text-xs text-primary hover:text-primary/80 cursor-pointer">
                <FileJson className="w-3 h-3" />
                Upload JSON
                <input
                  type="file"
                  accept=".json"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </label>
            </div>
            <textarea
              value={jsonInput}
              onChange={(e) => setJsonInput(e.target.value)}
              placeholder={`{
  "a_roll": {
    "url": "https://...",
    "metadata": "description"
  },
  "b_rolls": [
    { "id": "broll_1", "url": "https://...", "metadata": "..." }
  ]
}`}
              rows={8}
              className="w-full bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-xs text-white
                placeholder-slate-500 focus:outline-none focus:border-primary font-mono"
            />
          </div>
          
          {/* Parse Button */}
          <button
            onClick={handleParseJson}
            disabled={!jsonInput.trim()}
            className="w-full py-2 bg-accent hover:bg-accent/80 disabled:bg-slate-700 
              disabled:text-slate-500 text-white rounded-lg font-medium transition-colors 
              flex items-center justify-center gap-2"
          >
            <Upload className="w-4 h-4" />
            Parse JSON
          </button>
        </div>
      )}
    </div>
  );
}

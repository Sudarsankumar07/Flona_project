import React from 'react';
import { useApp } from '../../context/AppContext';
import InsertionCard from './InsertionCard';
import { Download, FileJson, Video } from 'lucide-react';

export default function InsertionList() {
  const { state } = useApp();
  const { timeline } = state;
  
  if (!timeline || !timeline.insertions || timeline.insertions.length === 0) {
    return null;
  }
  
  const handleDownloadJson = () => {
    const blob = new Blob([JSON.stringify(timeline, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'timeline.json';
    a.click();
    URL.revokeObjectURL(url);
  };
  
  return (
    <div className="bg-surface rounded-xl border border-slate-700 p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="font-semibold text-white">B-Roll Insertions</h3>
          <p className="text-sm text-slate-400">
            {timeline.insertions.length} insertions in {timeline.aroll_duration?.toFixed(1)}s video
          </p>
        </div>
        
        {/* Download Button */}
        <button
          onClick={handleDownloadJson}
          className="flex items-center gap-2 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 
                   rounded-lg text-sm text-slate-300 transition-colors"
        >
          <Download className="w-4 h-4" />
          <span>Export JSON</span>
        </button>
      </div>
      
      {/* Insertion Cards */}
      <div className="space-y-3 max-h-96 overflow-y-auto pr-2">
        {timeline.insertions.map((insertion, index) => (
          <InsertionCard 
            key={index} 
            insertion={insertion} 
            index={index} 
          />
        ))}
      </div>
      
      {/* Summary Stats */}
      <div className="mt-4 pt-4 border-t border-slate-700 grid grid-cols-3 gap-4">
        <div className="text-center">
          <div className="text-2xl font-bold text-primary">
            {timeline.insertions.length}
          </div>
          <div className="text-xs text-slate-400">Insertions</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-green-400">
            {calculateCoverage(timeline)}%
          </div>
          <div className="text-xs text-slate-400">Coverage</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-yellow-400">
            {calculateAvgConfidence(timeline)}%
          </div>
          <div className="text-xs text-slate-400">Avg Confidence</div>
        </div>
      </div>
    </div>
  );
}

function calculateCoverage(timeline) {
  if (!timeline.insertions || !timeline.aroll_duration) return 0;
  const totalBrollTime = timeline.insertions.reduce((sum, ins) => sum + ins.duration_sec, 0);
  return Math.round((totalBrollTime / timeline.aroll_duration) * 100);
}

function calculateAvgConfidence(timeline) {
  if (!timeline.insertions || timeline.insertions.length === 0) return 0;
  const avgConf = timeline.insertions.reduce((sum, ins) => sum + (ins.confidence || 0), 0) 
                  / timeline.insertions.length;
  return Math.round(avgConf * 100);
}

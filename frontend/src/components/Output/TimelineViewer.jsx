import React from 'react';
import { useApp } from '../../context/AppContext';

export default function TimelineViewer() {
  const { state } = useApp();
  const { timeline, aroll } = state;
  
  if (!timeline) {
    return (
      <div className="bg-surface rounded-xl border border-slate-700 p-6">
        <div className="text-center text-slate-500 py-8">
          <p className="mb-2">No timeline generated yet</p>
          <p className="text-sm">Upload videos and click "Generate Timeline"</p>
        </div>
      </div>
    );
  }
  
  const { aroll_duration, insertions } = timeline;
  
  // Calculate timeline scale
  const timelineWidth = 100; // percentage
  
  return (
    <div className="bg-surface rounded-xl border border-slate-700 p-4">
      <h3 className="font-semibold text-white mb-4">Timeline Preview</h3>
      
      {/* Timeline Track */}
      <div className="relative">
        {/* A-Roll Track */}
        <div className="mb-2">
          <div className="text-xs text-slate-400 mb-1">A-Roll</div>
          <div className="h-12 bg-slate-700 rounded-lg relative overflow-hidden">
            {/* Base A-roll */}
            <div className="absolute inset-0 bg-gradient-to-r from-blue-600/30 to-blue-500/30 flex items-center px-3">
              <span className="text-xs text-blue-300 truncate">
                {aroll?.name || 'a_roll.mp4'}
              </span>
            </div>
          </div>
        </div>
        
        {/* B-Roll Track */}
        <div className="mb-2">
          <div className="text-xs text-slate-400 mb-1">B-Roll Insertions</div>
          <div className="h-12 bg-slate-800 rounded-lg relative overflow-hidden">
            {insertions.map((insertion, index) => {
              const leftPercent = (insertion.start_sec / aroll_duration) * 100;
              const widthPercent = (insertion.duration_sec / aroll_duration) * 100;
              
              // Color based on confidence
              const confidence = insertion.confidence || 0.5;
              const bgColor = confidence >= 0.7 
                ? 'bg-green-500' 
                : confidence >= 0.4 
                  ? 'bg-yellow-500' 
                  : 'bg-orange-500';
              
              return (
                <div
                  key={index}
                  className={`absolute h-full ${bgColor} rounded flex items-center justify-center cursor-pointer
                             hover:brightness-110 transition-all group`}
                  style={{
                    left: `${leftPercent}%`,
                    width: `${Math.max(widthPercent, 3)}%`,
                  }}
                  title={`${insertion.broll_filename} (${(confidence * 100).toFixed(0)}%)`}
                >
                  <span className="text-xs text-white font-medium truncate px-1">
                    B{index + 1}
                  </span>
                  
                  {/* Tooltip */}
                  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 
                                  bg-slate-900 text-white text-xs rounded px-2 py-1
                                  opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-10">
                    {insertion.broll_filename}
                    <br />
                    {insertion.start_sec.toFixed(1)}s - {(insertion.start_sec + insertion.duration_sec).toFixed(1)}s
                  </div>
                </div>
              );
            })}
          </div>
        </div>
        
        {/* Time Markers */}
        <div className="flex justify-between text-xs text-slate-500 px-1">
          <span>0:00</span>
          <span>{formatTime(aroll_duration / 4)}</span>
          <span>{formatTime(aroll_duration / 2)}</span>
          <span>{formatTime((aroll_duration * 3) / 4)}</span>
          <span>{formatTime(aroll_duration)}</span>
        </div>
      </div>
      
      {/* Summary */}
      <div className="mt-4 flex items-center gap-4 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded bg-blue-500/30" />
          <span className="text-slate-400">A-Roll</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded bg-green-500" />
          <span className="text-slate-400">High Confidence</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded bg-yellow-500" />
          <span className="text-slate-400">Medium</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded bg-orange-500" />
          <span className="text-slate-400">Low</span>
        </div>
      </div>
    </div>
  );
}

function formatTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

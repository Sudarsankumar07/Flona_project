import React, { useState } from 'react';
import { useApp } from '../../context/AppContext';
import { MessageSquare, ChevronDown, ChevronUp } from 'lucide-react';

export default function TranscriptView() {
  const { state } = useApp();
  const { timeline } = state;
  const [isExpanded, setIsExpanded] = useState(false);
  
  if (!timeline || !timeline.transcript_segments) {
    return null;
  }
  
  const segments = timeline.transcript_segments;
  const displaySegments = isExpanded ? segments : segments.slice(0, 3);
  
  return (
    <div className="bg-surface rounded-xl border border-slate-700 p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <MessageSquare className="w-5 h-5 text-slate-400" />
          <h3 className="font-semibold text-white">Transcript</h3>
          <span className="text-xs bg-slate-700 px-2 py-0.5 rounded-full text-slate-400">
            {segments.length} segments
          </span>
        </div>
      </div>
      
      {/* Segments List */}
      <div className="space-y-2">
        {displaySegments.map((segment, index) => {
          // Check if this segment has a B-roll insertion
          const hasInsertion = timeline.insertions?.some(
            ins => ins.transcript_segment_id === segment.id
          );
          
          return (
            <div 
              key={segment.id}
              className={`
                p-3 rounded-lg border transition-colors
                ${hasInsertion 
                  ? 'bg-primary/10 border-primary/30' 
                  : 'bg-slate-800/50 border-slate-700'}
              `}
            >
              <div className="flex items-start gap-3">
                {/* Timestamp */}
                <div className="text-xs font-mono text-slate-500 w-20 flex-shrink-0">
                  {formatTime(segment.start)} - {formatTime(segment.end)}
                </div>
                
                {/* Text */}
                <p className={`text-sm flex-1 ${hasInsertion ? 'text-slate-200' : 'text-slate-400'}`}>
                  {segment.text}
                </p>
                
                {/* B-roll indicator */}
                {hasInsertion && (
                  <span className="text-xs bg-primary/20 text-primary px-2 py-0.5 rounded">
                    B-Roll
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>
      
      {/* Expand/Collapse Button */}
      {segments.length > 3 && (
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="w-full mt-3 py-2 text-sm text-slate-400 hover:text-white 
                   flex items-center justify-center gap-1 transition-colors"
        >
          {isExpanded ? (
            <>
              <ChevronUp className="w-4 h-4" />
              Show Less
            </>
          ) : (
            <>
              <ChevronDown className="w-4 h-4" />
              Show All ({segments.length - 3} more)
            </>
          )}
        </button>
      )}
    </div>
  );
}

function formatTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

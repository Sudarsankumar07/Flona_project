import React from 'react';
import { Film, Clock, Target, MessageSquare } from 'lucide-react';

export default function InsertionCard({ insertion, index }) {
  const confidence = insertion.confidence || 0.5;
  const confidencePercent = (confidence * 100).toFixed(0);
  
  // Color based on confidence
  const confidenceColor = confidence >= 0.7 
    ? 'text-green-400 bg-green-400/10 border-green-400/30'
    : confidence >= 0.4 
      ? 'text-yellow-400 bg-yellow-400/10 border-yellow-400/30'
      : 'text-orange-400 bg-orange-400/10 border-orange-400/30';
  
  return (
    <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700 hover:border-slate-600 transition-colors">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="p-1.5 bg-primary/20 rounded">
            <Film className="w-4 h-4 text-primary" />
          </div>
          <span className="font-medium text-white">Insertion {index + 1}</span>
        </div>
        
        {/* Confidence Badge */}
        <div className={`px-2 py-0.5 rounded-full text-xs font-medium border ${confidenceColor}`}>
          {confidencePercent}% match
        </div>
      </div>
      
      {/* B-Roll Info */}
      <div className="space-y-2 text-sm">
        <div className="flex items-center gap-2 text-slate-300">
          <Film className="w-4 h-4 text-slate-500" />
          <span className="font-mono">{insertion.broll_filename}</span>
        </div>
        
        <div className="flex items-center gap-2 text-slate-400">
          <Clock className="w-4 h-4 text-slate-500" />
          <span>
            {insertion.start_sec.toFixed(1)}s → {(insertion.start_sec + insertion.duration_sec).toFixed(1)}s
            <span className="text-slate-500 ml-1">
              ({insertion.duration_sec.toFixed(1)}s)
            </span>
          </span>
        </div>
        
        {/* Transcript Context */}
        {insertion.transcript_text && (
          <div className="mt-2 pt-2 border-t border-slate-700">
            <div className="flex items-start gap-2">
              <MessageSquare className="w-4 h-4 text-slate-500 mt-0.5 flex-shrink-0" />
              <p className="text-xs text-slate-400 italic line-clamp-2">
                "{insertion.transcript_text}"
              </p>
            </div>
          </div>
        )}
        
        {/* Reason */}
        {insertion.reason && (
          <div className="mt-2 pt-2 border-t border-slate-700">
            <div className="flex items-start gap-2">
              <Target className="w-4 h-4 text-slate-500 mt-0.5 flex-shrink-0" />
              <p className="text-xs text-slate-500 line-clamp-2">
                {insertion.reason}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

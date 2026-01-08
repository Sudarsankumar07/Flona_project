import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Film, X, Clock, GripVertical, FileText } from 'lucide-react';
import { useApp } from '../../context/AppContext';

export default function BRollUpload() {
  const { state, dispatch } = useApp();
  const { brolls } = state;
  
  const onDrop = useCallback((acceptedFiles) => {
    acceptedFiles.forEach((file, index) => {
      const id = `broll_${Date.now()}_${index}`;
      const preview = URL.createObjectURL(file);
      
      // Get video duration using a separate blob URL
      const tempUrl = URL.createObjectURL(file);
      const video = document.createElement('video');
      video.preload = 'metadata';
      video.onloadedmetadata = () => {
        const duration = video.duration;
        URL.revokeObjectURL(tempUrl); // Only revoke the temp URL, not the preview
        dispatch({
          type: 'ADD_BROLL',
          payload: {
            id,
            file,
            preview,
            duration,
            name: file.name,
            metadata: '',
          }
        });
      };
      video.src = tempUrl;
    });
  }, [dispatch]);
  
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.mov', '.avi', '.mkv', '.webm']
    },
    multiple: true,
  });
  
  const removeBroll = (id) => {
    const broll = brolls.find(b => b.id === id);
    if (broll?.preview) {
      URL.revokeObjectURL(broll.preview);
    }
    dispatch({ type: 'REMOVE_BROLL', payload: id });
  };
  
  const updateMetadata = (id, metadata) => {
    dispatch({
      type: 'UPDATE_BROLL',
      payload: { id, metadata }
    });
  };
  
  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };
  
  return (
    <div className="bg-surface rounded-xl p-4 border border-slate-700">
      <div className="flex items-center gap-2 mb-3">
        <Film className="w-5 h-5 text-secondary" />
        <h3 className="font-semibold text-white">B-Roll Videos</h3>
        <span className="text-xs bg-slate-700 px-2 py-0.5 rounded-full text-slate-300">
          {brolls.length} clips
        </span>
      </div>
      
      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-lg p-6 text-center cursor-pointer
          transition-all duration-200 mb-4
          ${isDragActive 
            ? 'border-secondary bg-secondary/10' 
            : 'border-slate-600 hover:border-slate-500 hover:bg-slate-800/50'
          }
        `}
      >
        <input {...getInputProps()} />
        <Upload className="w-8 h-8 text-slate-400 mx-auto mb-2" />
        <p className="text-slate-300 text-sm">
          {isDragActive ? 'Drop B-roll clips here' : 'Add B-roll clips'}
        </p>
        <p className="text-xs text-slate-500">Multiple files supported</p>
      </div>
      
      {/* B-roll Grid */}
      {brolls.length > 0 && (
        <div className="space-y-3 max-h-80 overflow-y-auto pr-2">
          {brolls.map((broll, index) => (
            <div 
              key={broll.id}
              className="flex gap-3 bg-slate-800/50 rounded-lg p-2 animate-slide-up"
            >
              {/* Thumbnail */}
              <div className="relative w-24 h-16 flex-shrink-0 rounded overflow-hidden">
                <video
                  src={broll.preview}
                  className="w-full h-full object-cover"
                  muted
                  onMouseEnter={(e) => e.target.play()}
                  onMouseLeave={(e) => { e.target.pause(); e.target.currentTime = 0; }}
                />
                <div className="absolute bottom-0 right-0 bg-black/70 px-1 text-xs text-white">
                  {formatDuration(broll.duration)}
                </div>
              </div>
              
              {/* Info */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs font-medium text-primary">B-Roll {index + 1}</span>
                  <span className="text-xs text-slate-500 truncate">{broll.name}</span>
                </div>
                <input
                  type="text"
                  placeholder="Add description (optional)..."
                  value={broll.metadata}
                  onChange={(e) => updateMetadata(broll.id, e.target.value)}
                  className="w-full text-xs bg-slate-700 border border-slate-600 rounded px-2 py-1 
                           text-slate-300 placeholder-slate-500 focus:outline-none focus:border-primary"
                />
              </div>
              
              {/* Remove Button */}
              <button
                onClick={() => removeBroll(broll.id)}
                className="p-1 hover:bg-slate-700 rounded transition-colors self-start"
              >
                <X className="w-4 h-4 text-slate-400 hover:text-red-400" />
              </button>
            </div>
          ))}
        </div>
      )}
      
      {brolls.length === 0 && (
        <p className="text-center text-slate-500 text-sm py-4">
          No B-roll clips added yet
        </p>
      )}
    </div>
  );
}

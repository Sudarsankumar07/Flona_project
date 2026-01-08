import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Video, X, Clock } from 'lucide-react';
import { useApp } from '../../context/AppContext';

export default function ARollUpload() {
  const { state, dispatch } = useApp();
  const { aroll } = state;
  
  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      const preview = URL.createObjectURL(file);
      
      // Get video duration using a separate blob URL
      const tempUrl = URL.createObjectURL(file);
      const video = document.createElement('video');
      video.preload = 'metadata';
      video.onloadedmetadata = () => {
        const duration = video.duration;
        URL.revokeObjectURL(tempUrl); // Only revoke the temp URL, not the preview
        dispatch({
          type: 'SET_AROLL',
          payload: {
            file,
            preview,
            duration,
            name: file.name,
          }
        });
      };
      video.src = tempUrl;
    }
  }, [dispatch]);
  
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.mov', '.avi', '.mkv', '.webm']
    },
    maxFiles: 1,
    multiple: false,
  });
  
  const removeAroll = () => {
    if (aroll?.preview) {
      URL.revokeObjectURL(aroll.preview);
    }
    dispatch({ type: 'SET_AROLL', payload: null });
  };
  
  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };
  
  return (
    <div className="bg-surface rounded-xl p-4 border border-slate-700">
      <div className="flex items-center gap-2 mb-3">
        <Video className="w-5 h-5 text-primary" />
        <h3 className="font-semibold text-white">A-Roll Video</h3>
        <span className="text-xs text-slate-400 ml-auto">Main talking head</span>
      </div>
      
      {!aroll ? (
        <div
          {...getRootProps()}
          className={`
            border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
            transition-all duration-200
            ${isDragActive 
              ? 'border-primary bg-primary/10' 
              : 'border-slate-600 hover:border-slate-500 hover:bg-slate-800/50'
            }
          `}
        >
          <input {...getInputProps()} />
          <Upload className="w-10 h-10 text-slate-400 mx-auto mb-3" />
          <p className="text-slate-300 mb-1">
            {isDragActive ? 'Drop your video here' : 'Drag & drop your A-roll video'}
          </p>
          <p className="text-sm text-slate-500">or click to browse</p>
          <p className="text-xs text-slate-600 mt-2">MP4, MOV, AVI, MKV, WebM</p>
        </div>
      ) : (
        <div className="relative rounded-lg overflow-hidden bg-slate-800">
          <video
            src={aroll.preview}
            className="w-full h-48 object-cover"
            muted
            onMouseEnter={(e) => e.target.play()}
            onMouseLeave={(e) => { e.target.pause(); e.target.currentTime = 0; }}
          />
          <button
            onClick={removeAroll}
            className="absolute top-2 right-2 p-1.5 bg-red-500 rounded-full hover:bg-red-600 transition-colors"
          >
            <X className="w-4 h-4 text-white" />
          </button>
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-3">
            <p className="text-white text-sm font-medium truncate">{aroll.name}</p>
            <div className="flex items-center gap-1 text-xs text-slate-300 mt-1">
              <Clock className="w-3 h-3" />
              <span>{formatDuration(aroll.duration)}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

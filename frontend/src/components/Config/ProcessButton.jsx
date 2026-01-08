import React, { useState, useEffect } from 'react';
import { Play, Loader2, CheckCircle, AlertCircle, RefreshCw } from 'lucide-react';
import { useApp } from '../../context/AppContext';
import { uploadVideos, uploadFromUrls, configureApi, startProcessing, getStatus, getTimeline } from '../../services/api';
import toast from 'react-hot-toast';

const STEPS = [
  { id: 'upload', name: 'Uploading Videos' },
  { id: 'configure', name: 'Configuring API' },
  { id: 'transcribe', name: 'Transcribing A-Roll' },
  { id: 'caption', name: 'Captioning B-Rolls' },
  { id: 'embed', name: 'Generating Embeddings' },
  { id: 'match', name: 'Finding Matches' },
  { id: 'timeline', name: 'Creating Timeline' },
];

export default function ProcessButton() {
  const { state, dispatch } = useApp();
  const { aroll, brolls, urlMode, inputMode, config, processing } = state;
  const [currentStepIndex, setCurrentStepIndex] = useState(-1);
  
  // Check if we can process based on input mode
  const canProcessFiles = inputMode === 'file' && aroll && brolls.length > 0;
  const canProcessUrls = inputMode === 'url' && urlMode?.aroll?.url && urlMode?.brolls?.length > 0;
  const hasApiConfig = config.provider === 'offline' || config.apiKey.length > 10;
  const canProcess = (canProcessFiles || canProcessUrls) && hasApiConfig;
  
  const isProcessing = processing.status === 'processing' || processing.status === 'uploading';
  
  const handleProcess = async () => {
    if (!canProcess || isProcessing) return;
    
    dispatch({ 
      type: 'SET_PROCESSING', 
      payload: { status: 'uploading', progress: 0, currentStep: 'Uploading videos...' } 
    });
    setCurrentStepIndex(0);
    
    try {
      // Step 1: Upload videos (based on input mode)
      if (inputMode === 'url') {
        toast.loading('Downloading videos from URLs...', { id: 'process' });
        await uploadFromUrls(urlMode);
      } else {
        toast.loading('Uploading videos...', { id: 'process' });
        await uploadVideos(
          aroll.file,
          brolls.map(b => b.file),
          brolls.map(b => ({ id: b.id, metadata: b.metadata }))
        );
      }
      
      setCurrentStepIndex(1);
      dispatch({ 
        type: 'SET_PROCESSING', 
        payload: { progress: 15, currentStep: 'Configuring API...' } 
      });
      
      // Step 2: Configure API
      await configureApi(config.provider, config.apiKey || null);
      
      setCurrentStepIndex(2);
      dispatch({ 
        type: 'SET_PROCESSING', 
        payload: { status: 'processing', progress: 25, currentStep: 'Starting pipeline...' } 
      });
      
      // Step 3: Start processing
      toast.loading('Processing videos...', { id: 'process' });
      
      const processResult = await startProcessing({
        similarity_threshold: 0.15,
        max_insertions: 6,
        min_gap_seconds: 5.0,
      });
      
      console.log('Process result:', processResult);
      
      // Poll for status
      const jobId = processResult.job_id;
      
      if (!jobId) {
        // If no job_id, the server might have processed synchronously
        // Try to get the timeline directly
        console.log('No job_id, trying to get timeline directly...');
        try {
          const timeline = await getTimeline();
          dispatch({ type: 'SET_TIMELINE', payload: timeline });
          dispatch({ 
            type: 'SET_PROCESSING', 
            payload: { status: 'completed', progress: 100, currentStep: 'Complete!' } 
          });
          toast.success('Timeline generated successfully!', { id: 'process' });
          return;
        } catch (e) {
          throw new Error('No job ID returned from server and no timeline available');
        }
      }
      
      let completed = false;
      let pollAttempts = 0;
      const maxPollAttempts = 60; // 2 minutes max
      
      while (!completed && pollAttempts < maxPollAttempts) {
        pollAttempts++;
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        try {
          const statusResult = await getStatus(jobId);
          
          // Update progress based on status
          const stepMapping = {
            'transcribing': { index: 2, progress: 35 },
            'captioning': { index: 3, progress: 50 },
            'embedding': { index: 4, progress: 65 },
            'matching': { index: 5, progress: 80 },
            'timeline': { index: 6, progress: 90 },
          };
          
          const stepKey = Object.keys(stepMapping).find(k => 
            statusResult.current_step?.toLowerCase().includes(k)
          );
          
          if (stepKey) {
            setCurrentStepIndex(stepMapping[stepKey].index);
            dispatch({ 
              type: 'SET_PROCESSING', 
              payload: { 
                progress: stepMapping[stepKey].progress, 
                currentStep: statusResult.current_step 
              } 
            });
          }
          
          if (statusResult.status === 'completed') {
            completed = true;
          } else if (statusResult.status === 'failed') {
            throw new Error(statusResult.error || 'Processing failed');
          }
        } catch (pollError) {
          // If job not found, check if timeline exists
          if (pollError.response?.status === 404) {
            try {
              const timeline = await getTimeline();
              dispatch({ type: 'SET_TIMELINE', payload: timeline });
              dispatch({ 
                type: 'SET_PROCESSING', 
                payload: { status: 'completed', progress: 100, currentStep: 'Complete!' } 
              });
              toast.success('Timeline generated successfully!', { id: 'process' });
              return;
            } catch (e) {
              // Continue polling
            }
          }
        }
      }
      
      // Get final timeline
      const timeline = await getTimeline();
      
      dispatch({ type: 'SET_TIMELINE', payload: timeline });
      dispatch({ 
        type: 'SET_PROCESSING', 
        payload: { status: 'completed', progress: 100, currentStep: 'Complete!' } 
      });
      
      toast.success('Timeline generated successfully!', { id: 'process' });
      
    } catch (error) {
      console.error('Processing error:', error);
      dispatch({ 
        type: 'SET_PROCESSING', 
        payload: { 
          status: 'failed', 
          error: error.message || 'Processing failed',
          currentStep: 'Failed'
        } 
      });
      toast.error(error.message || 'Processing failed', { id: 'process' });
    }
  };
  
  const handleReset = () => {
    dispatch({ 
      type: 'SET_PROCESSING', 
      payload: { status: 'idle', progress: 0, currentStep: '', error: null } 
    });
    dispatch({ type: 'SET_TIMELINE', payload: null });
    setCurrentStepIndex(-1);
  };
  
  return (
    <div className="space-y-4">
      {/* Process Button */}
      <button
        onClick={handleProcess}
        disabled={!canProcess || isProcessing}
        className={`
          w-full py-3 px-4 rounded-xl font-semibold text-white
          flex items-center justify-center gap-2 transition-all
          ${canProcess && !isProcessing
            ? 'bg-primary hover:bg-primary-hover shadow-lg shadow-primary/25'
            : 'bg-slate-700 cursor-not-allowed opacity-50'
          }
        `}
      >
        {isProcessing ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            <span>Processing...</span>
          </>
        ) : processing.status === 'completed' ? (
          <>
            <CheckCircle className="w-5 h-5" />
            <span>Complete!</span>
          </>
        ) : processing.status === 'failed' ? (
          <>
            <AlertCircle className="w-5 h-5" />
            <span>Failed - Click to Retry</span>
          </>
        ) : (
          <>
            <Play className="w-5 h-5" />
            <span>Generate Timeline</span>
          </>
        )}
      </button>
      
      {/* Validation Messages */}
      {!aroll && (
        <p className="text-xs text-yellow-400 flex items-center gap-1">
          <AlertCircle className="w-3 h-3" />
          Upload an A-roll video
        </p>
      )}
      {aroll && brolls.length === 0 && (
        <p className="text-xs text-yellow-400 flex items-center gap-1">
          <AlertCircle className="w-3 h-3" />
          Add at least one B-roll clip
        </p>
      )}
      {aroll && brolls.length > 0 && config.provider !== 'offline' && config.apiKey.length < 10 && (
        <p className="text-xs text-yellow-400 flex items-center gap-1">
          <AlertCircle className="w-3 h-3" />
          Enter API key or switch to Offline mode
        </p>
      )}
      
      {/* Progress Steps */}
      {isProcessing && (
        <div className="space-y-2 animate-slide-up">
          <div className="flex justify-between text-xs text-slate-400">
            <span>{processing.currentStep}</span>
            <span>{processing.progress}%</span>
          </div>
          
          {/* Progress Bar */}
          <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-primary to-secondary transition-all duration-500"
              style={{ width: `${processing.progress}%` }}
            />
          </div>
          
          {/* Step Indicators */}
          <div className="flex justify-between mt-3">
            {STEPS.map((step, index) => (
              <div
                key={step.id}
                className={`flex flex-col items-center ${index <= currentStepIndex ? 'text-primary' : 'text-slate-600'}`}
                title={step.name}
              >
                <div className={`
                  w-2 h-2 rounded-full 
                  ${index < currentStepIndex ? 'bg-primary' : 
                    index === currentStepIndex ? 'bg-primary animate-pulse' : 'bg-slate-600'}
                `} />
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Reset Button */}
      {(processing.status === 'completed' || processing.status === 'failed') && (
        <button
          onClick={handleReset}
          className="w-full py-2 text-sm text-slate-400 hover:text-white flex items-center justify-center gap-2"
        >
          <RefreshCw className="w-4 h-4" />
          Process Again
        </button>
      )}
    </div>
  );
}

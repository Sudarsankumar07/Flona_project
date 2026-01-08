import React, { createContext, useContext, useReducer } from 'react';

const AppContext = createContext();

const initialState = {
  // Upload state
  aroll: null, // { file, preview, duration }
  brolls: [], // [{ id, file, preview, metadata, duration }]
  
  // Config state
  config: {
    provider: 'offline', // 'gemini', 'openai', 'openrouter', 'offline'
    apiKey: '',
    isConfigured: false,
  },
  
  // Processing state
  processing: {
    status: 'idle', // 'idle', 'uploading', 'processing', 'completed', 'failed'
    progress: 0,
    currentStep: '',
    jobId: null,
    error: null,
  },
  
  // Timeline output
  timeline: null,
};

function appReducer(state, action) {
  switch (action.type) {
    case 'SET_AROLL':
      return { ...state, aroll: action.payload };
    
    case 'ADD_BROLL':
      return { 
        ...state, 
        brolls: [...state.brolls, action.payload] 
      };
    
    case 'UPDATE_BROLL':
      return {
        ...state,
        brolls: state.brolls.map(b => 
          b.id === action.payload.id ? { ...b, ...action.payload } : b
        )
      };
    
    case 'REMOVE_BROLL':
      return {
        ...state,
        brolls: state.brolls.filter(b => b.id !== action.payload)
      };
    
    case 'CLEAR_BROLLS':
      return { ...state, brolls: [] };
    
    case 'SET_CONFIG':
      return {
        ...state,
        config: { ...state.config, ...action.payload }
      };
    
    case 'SET_PROCESSING':
      return {
        ...state,
        processing: { ...state.processing, ...action.payload }
      };
    
    case 'SET_TIMELINE':
      return { ...state, timeline: action.payload };
    
    case 'RESET':
      return initialState;
    
    default:
      return state;
  }
}

export function AppProvider({ children }) {
  const [state, dispatch] = useReducer(appReducer, initialState);
  
  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within AppProvider');
  }
  return context;
}

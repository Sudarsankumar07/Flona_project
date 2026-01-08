import React, { useState } from 'react';
import { Settings, Key, Cpu, Cloud, Zap, ChevronDown, Check, AlertCircle } from 'lucide-react';
import { useApp } from '../../context/AppContext';

const PROVIDERS = [
  {
    id: 'offline',
    name: 'Offline Mode',
    icon: Cpu,
    description: 'No API key needed. Uses local models.',
    color: 'text-green-400',
    bgColor: 'bg-green-400/10',
  },
  {
    id: 'openrouter',
    name: 'OpenRouter',
    icon: Zap,
    description: 'Access multiple AI models with one key.',
    color: 'text-yellow-400',
    bgColor: 'bg-yellow-400/10',
  },
  {
    id: 'gemini',
    name: 'Google Gemini',
    icon: Cloud,
    description: 'Google\'s powerful AI models.',
    color: 'text-blue-400',
    bgColor: 'bg-blue-400/10',
  },
  {
    id: 'openai',
    name: 'OpenAI',
    icon: Cloud,
    description: 'GPT-4 and Whisper models.',
    color: 'text-purple-400',
    bgColor: 'bg-purple-400/10',
  },
];

export default function ApiConfig() {
  const { state, dispatch } = useApp();
  const { config } = state;
  const [isExpanded, setIsExpanded] = useState(true);
  const [showApiKey, setShowApiKey] = useState(false);
  
  const selectedProvider = PROVIDERS.find(p => p.id === config.provider);
  
  const handleProviderChange = (providerId) => {
    dispatch({
      type: 'SET_CONFIG',
      payload: { 
        provider: providerId,
        isConfigured: providerId === 'offline',
        apiKey: providerId === 'offline' ? '' : config.apiKey,
      }
    });
  };
  
  const handleApiKeyChange = (apiKey) => {
    dispatch({
      type: 'SET_CONFIG',
      payload: { 
        apiKey,
        isConfigured: apiKey.length > 0 || config.provider === 'offline',
      }
    });
  };
  
  const needsApiKey = config.provider !== 'offline';
  const isConfigValid = config.provider === 'offline' || config.apiKey.length > 10;
  
  return (
    <div className="bg-surface rounded-xl border border-slate-700 overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center gap-2 p-4 hover:bg-slate-800/50 transition-colors"
      >
        <Settings className="w-5 h-5 text-slate-400" />
        <span className="font-semibold text-white">API Configuration</span>
        
        {/* Status Badge */}
        <div className={`ml-auto flex items-center gap-2 text-xs px-2 py-1 rounded-full
          ${isConfigValid ? 'bg-green-400/10 text-green-400' : 'bg-yellow-400/10 text-yellow-400'}`}
        >
          {isConfigValid ? (
            <>
              <Check className="w-3 h-3" />
              <span>Ready</span>
            </>
          ) : (
            <>
              <AlertCircle className="w-3 h-3" />
              <span>Needs API Key</span>
            </>
          )}
        </div>
        
        <ChevronDown className={`w-5 h-5 text-slate-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`} />
      </button>
      
      {/* Content */}
      {isExpanded && (
        <div className="px-4 pb-4 space-y-4 animate-slide-up">
          {/* Provider Selection */}
          <div className="grid grid-cols-2 gap-2">
            {PROVIDERS.map((provider) => {
              const Icon = provider.icon;
              const isSelected = config.provider === provider.id;
              
              return (
                <button
                  key={provider.id}
                  onClick={() => handleProviderChange(provider.id)}
                  className={`
                    p-3 rounded-lg border-2 text-left transition-all
                    ${isSelected 
                      ? `border-primary ${provider.bgColor}` 
                      : 'border-slate-700 hover:border-slate-600 bg-slate-800/30'
                    }
                  `}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <Icon className={`w-4 h-4 ${isSelected ? provider.color : 'text-slate-400'}`} />
                    <span className={`text-sm font-medium ${isSelected ? 'text-white' : 'text-slate-300'}`}>
                      {provider.name}
                    </span>
                  </div>
                  <p className="text-xs text-slate-500 line-clamp-1">
                    {provider.description}
                  </p>
                </button>
              );
            })}
          </div>
          
          {/* API Key Input */}
          {needsApiKey && (
            <div className="space-y-2">
              <label className="flex items-center gap-2 text-sm text-slate-400">
                <Key className="w-4 h-4" />
                API Key
              </label>
              <div className="relative">
                <input
                  type={showApiKey ? 'text' : 'password'}
                  placeholder={`Enter your ${selectedProvider?.name} API key...`}
                  value={config.apiKey}
                  onChange={(e) => handleApiKeyChange(e.target.value)}
                  className="w-full bg-slate-800 border border-slate-600 rounded-lg px-4 py-2.5
                           text-slate-200 placeholder-slate-500 focus:outline-none focus:border-primary
                           text-sm pr-20"
                />
                <button
                  onClick={() => setShowApiKey(!showApiKey)}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-xs text-slate-400 
                           hover:text-white px-2 py-1 rounded"
                >
                  {showApiKey ? 'Hide' : 'Show'}
                </button>
              </div>
              
              {/* Help Links */}
              <p className="text-xs text-slate-500">
                Get your API key from{' '}
                {config.provider === 'openrouter' && (
                  <a href="https://openrouter.ai/keys" target="_blank" rel="noopener noreferrer" 
                     className="text-primary hover:underline">openrouter.ai/keys</a>
                )}
                {config.provider === 'gemini' && (
                  <a href="https://aistudio.google.com/app/apikey" target="_blank" rel="noopener noreferrer"
                     className="text-primary hover:underline">Google AI Studio</a>
                )}
                {config.provider === 'openai' && (
                  <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener noreferrer"
                     className="text-primary hover:underline">OpenAI Platform</a>
                )}
              </p>
            </div>
          )}
          
          {/* Offline Mode Info */}
          {config.provider === 'offline' && (
            <div className="bg-green-400/10 border border-green-400/20 rounded-lg p-3">
              <p className="text-sm text-green-400 flex items-center gap-2">
                <Cpu className="w-4 h-4" />
                Offline mode uses local AI models
              </p>
              <p className="text-xs text-slate-400 mt-1">
                First run will download ~500MB of model files. Processing may be slower than cloud APIs.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

import axios from 'axios';

const API_BASE = '/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 300000, // 5 minutes for video processing
});

// Upload videos (file mode)
export const uploadVideos = async (arollFile, brollFiles, brollMetadata = []) => {
  const formData = new FormData();
  
  if (arollFile) {
    formData.append('aroll', arollFile);
  }
  
  brollFiles.forEach((file, index) => {
    formData.append('brolls', file);
  });
  
  if (brollMetadata.length > 0) {
    formData.append('broll_metadata', JSON.stringify(brollMetadata));
  }
  
  const response = await api.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.data;
};

// Upload videos from URLs (url mode)
export const uploadFromUrls = async (urlData) => {
  const response = await api.post('/upload-urls', {
    a_roll: urlData.aroll,
    b_rolls: urlData.brolls
  });
  return response.data;
};

// Configure API settings
export const configureApi = async (provider, apiKey = null) => {
  const response = await api.post('/configure', {
    provider,
    api_key: apiKey,
  });
  return response.data;
};

// Start processing
export const startProcessing = async (settings = {}) => {
  console.log('Calling /process with settings:', settings);
  const response = await api.post('/process', { settings });
  console.log('Response from /process:', response);
  console.log('Response data:', response.data);
  return response.data;
};

// Get processing status
export const getStatus = async (jobId) => {
  const response = await api.get(`/status/${jobId}`);
  return response.data;
};

// Get timeline
export const getTimeline = async () => {
  const response = await api.get('/timeline');
  return response.data;
};

// Check backend health
export const checkHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

// Check if offline models are available
export const checkOfflineModels = async () => {
  const response = await api.get('/offline-status');
  return response.data;
};

// Download offline models
export const downloadOfflineModels = async () => {
  const response = await api.post('/download-models');
  return response.data;
};

export default api;

import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'
import { Toaster } from 'react-hot-toast'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
    <Toaster 
      position="top-right"
      toastOptions={{
        style: {
          background: '#1e293b',
          color: '#f8fafc',
          border: '1px solid #334155'
        },
        success: {
          iconTheme: {
            primary: '#22c55e',
            secondary: '#f8fafc'
          }
        },
        error: {
          iconTheme: {
            primary: '#ef4444',
            secondary: '#f8fafc'
          }
        }
      }}
    />
  </React.StrictMode>,
)

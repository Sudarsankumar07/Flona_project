# Smart B-Roll Inserter - Frontend Design Document

## 1. Overview

A React-based web application that allows users to upload A-roll and B-roll videos, configure API settings, and generate an intelligent B-roll insertion timeline.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FRONTEND (React + Vite)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│  │   Upload Panel  │  │  Config Panel   │  │      Output Panel           │ │
│  │                 │  │                 │  │                             │ │
│  │  ┌───────────┐  │  │  • API Provider │  │  ┌─────────────────────┐   │ │
│  │  │  A-Roll   │  │  │  • API Key      │  │  │  Timeline Viewer    │   │ │
│  │  │  (1 video)│  │  │  • Offline Mode │  │  │                     │   │ │
│  │  └───────────┘  │  │                 │  │  │  [====|B1|===|B2|=] │   │ │
│  │                 │  │  ┌───────────┐  │  │  └─────────────────────┘   │ │
│  │  ┌───────────┐  │  │  │  Process  │  │  │                             │ │
│  │  │  B-Rolls  │  │  │  │  Button   │  │  │  ┌─────────────────────┐   │ │
│  │  │ (multiple)│  │  │  └───────────┘  │  │  │  Insertion Details  │   │ │
│  │  └───────────┘  │  │                 │  │  │  • Timestamp        │   │ │
│  │                 │  │                 │  │  │  • B-roll clip      │   │ │
│  └─────────────────┘  └─────────────────┘  │  │  • Confidence       │   │ │
│                                            │  └─────────────────────┘   │ │
│                                            └─────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ HTTP/REST API
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          BACKEND (FastAPI)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  POST /api/upload          - Upload videos                                  │
│  POST /api/configure       - Set API provider & key                         │
│  POST /api/process         - Run the pipeline                               │
│  GET  /api/status          - Check processing status                        │
│  GET  /api/timeline        - Get generated timeline                         │
│  GET  /api/download/{file} - Download output files                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── Layout/
│   │   │   ├── Header.jsx
│   │   │   └── Footer.jsx
│   │   ├── Upload/
│   │   │   ├── ARollUpload.jsx      # Single video dropzone
│   │   │   ├── BRollUpload.jsx      # Multiple video dropzone
│   │   │   └── VideoPreview.jsx     # Video thumbnail preview
│   │   ├── Config/
│   │   │   ├── ApiConfig.jsx        # API provider selection
│   │   │   └── ProcessButton.jsx    # Start processing
│   │   ├── Output/
│   │   │   ├── TimelineViewer.jsx   # Visual timeline
│   │   │   ├── InsertionCard.jsx    # Individual insertion detail
│   │   │   └── TranscriptView.jsx   # Transcript segments
│   │   └── common/
│   │       ├── ProgressBar.jsx
│   │       ├── StatusBadge.jsx
│   │       └── Modal.jsx
│   ├── hooks/
│   │   ├── useVideoUpload.js
│   │   ├── useProcessing.js
│   │   └── useTimeline.js
│   ├── services/
│   │   └── api.js                   # Backend API calls
│   ├── App.jsx
│   ├── main.jsx
│   └── index.css
├── package.json
└── vite.config.js
```

---

## 4. User Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   Upload    │────▶│  Configure   │────▶│   Process   │────▶│    View      │
│   Videos    │     │   API Key    │     │   Pipeline  │     │   Timeline   │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
      │                    │                    │                    │
      ▼                    ▼                    ▼                    ▼
  • Drag & drop       • Select provider    • Show progress      • Visual timeline
  • A-roll (1)        • Enter API key      • Real-time status   • Insertion cards
  • B-rolls (N)       • Or skip for        • Error handling     • Download JSON
  • Preview videos      offline mode                            • Export video
```

---

## 5. UI Design Specifications

### 5.1 Color Palette
```css
--primary:     #6366f1;  /* Indigo - main actions */
--secondary:   #8b5cf6;  /* Violet - accents */
--success:     #22c55e;  /* Green - completed */
--warning:     #f59e0b;  /* Amber - warnings */
--error:       #ef4444;  /* Red - errors */
--background:  #0f172a;  /* Slate 900 - dark mode */
--surface:     #1e293b;  /* Slate 800 - cards */
--text:        #f8fafc;  /* Slate 50 - primary text */
--text-muted:  #94a3b8;  /* Slate 400 - secondary text */
```

### 5.2 Layout Grid
```
┌────────────────────────────────────────────────────────────────┐
│                         HEADER                                  │
├───────────────────────┬────────────────────────────────────────┤
│                       │                                        │
│    UPLOAD PANEL       │           OUTPUT PANEL                 │
│    (Left - 40%)       │           (Right - 60%)                │
│                       │                                        │
│  ┌─────────────────┐  │  ┌──────────────────────────────────┐ │
│  │   A-ROLL        │  │  │         TIMELINE VIEWER          │ │
│  │   Drop Zone     │  │  │                                  │ │
│  └─────────────────┘  │  └──────────────────────────────────┘ │
│                       │                                        │
│  ┌─────────────────┐  │  ┌──────────────────────────────────┐ │
│  │   B-ROLLS       │  │  │      INSERTION DETAILS           │ │
│  │   Drop Zone     │  │  │                                  │ │
│  │   (grid view)   │  │  │  Card 1 | Card 2 | Card 3 ...   │ │
│  └─────────────────┘  │  │                                  │ │
│                       │  └──────────────────────────────────┘ │
│  ┌─────────────────┐  │                                        │
│  │   API CONFIG    │  │  ┌──────────────────────────────────┐ │
│  │   + Process Btn │  │  │      TRANSCRIPT VIEW             │ │
│  └─────────────────┘  │  │                                  │ │
│                       │  └──────────────────────────────────┘ │
├───────────────────────┴────────────────────────────────────────┤
│                         FOOTER                                  │
└────────────────────────────────────────────────────────────────┘
```

---

## 6. API Endpoints Design

### 6.1 Upload Videos
```
POST /api/upload
Content-Type: multipart/form-data

Request:
  - aroll: File (single video)
  - brolls: File[] (multiple videos)
  - broll_metadata: JSON[] (optional descriptions)

Response:
{
  "success": true,
  "aroll": {
    "filename": "a_roll.mp4",
    "duration": 40.1,
    "path": "/uploads/aroll/a_roll.mp4"
  },
  "brolls": [
    {"id": "broll_1", "filename": "broll_1.mp4", "duration": 5.2},
    ...
  ]
}
```

### 6.2 Configure API
```
POST /api/configure
Content-Type: application/json

Request:
{
  "provider": "openrouter" | "gemini" | "openai" | "offline",
  "api_key": "sk-..." (optional for offline)
}

Response:
{
  "success": true,
  "provider": "openrouter",
  "mode": "online" | "offline"
}
```

### 6.3 Process Pipeline
```
POST /api/process
Content-Type: application/json

Request:
{
  "settings": {
    "similarity_threshold": 0.35,
    "max_insertions": 6,
    "min_gap_seconds": 5.0
  }
}

Response:
{
  "job_id": "uuid-xxx",
  "status": "processing"
}
```

### 6.4 Get Status
```
GET /api/status/{job_id}

Response:
{
  "job_id": "uuid-xxx",
  "status": "processing" | "completed" | "failed",
  "progress": 65,
  "current_step": "Transcribing A-roll...",
  "steps": [
    {"name": "Download", "status": "completed"},
    {"name": "Transcribe", "status": "in_progress"},
    {"name": "Caption B-rolls", "status": "pending"},
    ...
  ]
}
```

### 6.5 Get Timeline
```
GET /api/timeline

Response:
{
  "aroll_filename": "a_roll.mp4",
  "aroll_duration": 40.1,
  "total_insertions": 4,
  "insertions": [...],
  "transcript_segments": [...]
}
```

---

## 7. State Management

```javascript
// Global App State
{
  // Upload State
  uploads: {
    aroll: { file: File, preview: string, status: 'idle'|'uploading'|'done' },
    brolls: [{ id: string, file: File, preview: string, metadata: string }]
  },
  
  // Config State
  config: {
    provider: 'openrouter' | 'gemini' | 'openai' | 'offline',
    apiKey: string,
    isConfigured: boolean
  },
  
  // Processing State
  processing: {
    status: 'idle' | 'processing' | 'completed' | 'failed',
    progress: number,
    currentStep: string,
    jobId: string
  },
  
  // Output State
  timeline: {
    arollDuration: number,
    insertions: [],
    transcriptSegments: []
  }
}
```

---

## 8. Offline Mode Flow

```
User selects "Offline Mode" (no API key)
            │
            ▼
┌─────────────────────────────────────┐
│  Check if offline models exist      │
│  - sentence-transformers            │
│  - BLIP vision model                │
└─────────────────────────────────────┘
            │
            ▼
    Models exist?
    ┌───┴───┐
   Yes     No
    │       │
    ▼       ▼
  Start   Download models
  Process  (show progress)
            │
            ▼
         Start Process
```

---

## 9. Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| UI Framework | React 18 | Component-based UI |
| Build Tool | Vite | Fast development & build |
| Styling | Tailwind CSS | Utility-first CSS |
| State | React Context + useReducer | Global state management |
| HTTP Client | Axios | API communication |
| Video Preview | HTML5 Video | Video thumbnails |
| Icons | Lucide React | Modern icon set |
| Notifications | React Hot Toast | Toast messages |

---

## 10. Responsive Breakpoints

```css
/* Mobile First */
sm: 640px   /* Small devices */
md: 768px   /* Tablets */
lg: 1024px  /* Laptops */
xl: 1280px  /* Desktops */
2xl: 1536px /* Large screens */
```

Mobile layout: Stacked panels (Upload → Config → Output)
Desktop layout: Side-by-side (Upload+Config | Output)

---

## 11. Error Handling

| Error Type | User Message | Action |
|------------|--------------|--------|
| Upload failed | "Failed to upload video. Please try again." | Retry button |
| Invalid API key | "API key is invalid. Please check and try again." | Re-enter key |
| Rate limited | "API rate limited. Switching to offline mode..." | Auto-fallback |
| Processing failed | "Processing failed: {reason}" | Show logs, retry |
| Network error | "Connection lost. Please check your internet." | Retry when online |

---

## 12. Performance Considerations

1. **Video Previews**: Generate thumbnails on upload, not full video playback
2. **Lazy Loading**: Load timeline components only when processing complete
3. **Debouncing**: Debounce API calls for status polling (every 2s)
4. **Chunked Upload**: For large videos, use chunked upload
5. **Web Workers**: Offload heavy computations to web workers


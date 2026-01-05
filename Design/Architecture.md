# Smart B-Roll Inserter - Project Architecture

> **Document Purpose**: This file outlines the complete system architecture, data flow, and implementation strategy for the Smart B-Roll Inserter project.

---

## 1. Project Understanding

### The Problem
UGC/talking-head videos are often visually monotonous. Editors manually insert **B-roll clips** at relevant moments to enhance engagement. This project aims to **automate the planning process**.

### The Goal
Build a system that:
1. **Transcribes** A-roll video with timestamps
2. **Understands** what each B-roll clip shows (text description)
3. **Matches** semantically relevant B-roll to A-roll moments
4. **Outputs** a structured JSON timeline plan
5. *(Optional)* Renders the final stitched video

### Constraints
- A-roll length: **30–90 seconds**
- Number of B-roll clips: **6**
- B-roll insertions: **3–6**

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (React)                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ Upload      │  │ Transcript  │  │ Timeline    │  │ Video Preview           │ │
│  │ Panel       │  │ View        │  │ View        │  │ (Optional)              │ │
│  └──────┬──────┘  └──────▲──────┘  └──────▲──────┘  └───────────▲─────────────┘ │
└─────────┼────────────────┼────────────────┼─────────────────────┼───────────────┘
          │                │                │                     │
          │ REST API       │                │                     │
          ▼                │                │                     │
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             BACKEND (FastAPI)                                   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         1. INGESTION LAYER                              │   │
│  │   ┌─────────────────────┐      ┌─────────────────────┐                  │   │
│  │   │  A-Roll Ingest      │      │  B-Roll Ingest      │                  │   │
│  │   │  (Single video)     │      │  (Multiple clips)   │                  │   │
│  │   └──────────┬──────────┘      └──────────┬──────────┘                  │   │
│  └──────────────┼────────────────────────────┼─────────────────────────────┘   │
│                 │                            │                                  │
│                 ▼                            ▼                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      2. UNDERSTANDING LAYER                             │   │
│  │                                                                         │   │
│  │   ┌─────────────────────┐      ┌─────────────────────┐                  │   │
│  │   │  Transcriber        │      │  B-Roll Captioner   │                  │   │
│  │   │  (Whisper API)      │      │  (Vision LLM/GPT-4V)│                  │   │
│  │   │                     │      │                     │                  │   │
│  │   │  Video → Audio →    │      │  Video frames →     │                  │   │
│  │   │  Timestamped Text   │      │  Text Description   │                  │   │
│  │   └──────────┬──────────┘      └──────────┬──────────┘                  │   │
│  └──────────────┼────────────────────────────┼─────────────────────────────┘   │
│                 │                            │                                  │
│                 ▼                            ▼                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                       3. EMBEDDINGS LAYER                               │   │
│  │                                                                         │   │
│  │   ┌───────────────────────────────────────────────────────────────┐     │   │
│  │   │              Embedding Generator (OpenAI/Sentence-BERT)       │     │   │
│  │   │                                                               │     │   │
│  │   │   Transcript Segments → Vectors    B-Roll Descriptions → Vec  │     │   │
│  │   └───────────────────────────────────────────────────────────────┘     │   │
│  └─────────────────────────────────┬───────────────────────────────────────┘   │
│                                    │                                           │
│                                    ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                       4. MATCHING LAYER (Core AI)                       │   │
│  │                                                                         │   │
│  │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │   │                    Semantic Matcher                             │   │   │
│  │   │   • Cosine similarity between A-roll segments & B-roll          │   │   │
│  │   │   • Threshold filtering (confidence > 0.7)                      │   │   │
│  │   │   • Avoid emotional/critical moments                            │   │   │
│  │   │   • Frequency control (not too many insertions)                 │   │   │
│  │   └─────────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────┬───────────────────────────────────────┘   │
│                                    │                                           │
│                                    ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                       5. PLANNING LAYER                                 │   │
│  │                                                                         │   │
│  │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │   │                   Timeline Planner                              │   │   │
│  │   │   • Generate insertion points (start_sec, duration_sec)         │   │   │
│  │   │   • Attach B-roll IDs                                           │   │   │
│  │   │   • Generate reasoning (LLM explanation)                        │   │   │
│  │   │   • Output: timeline.json                                       │   │   │
│  │   └─────────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────┬───────────────────────────────────────┘   │
│                                    │                                           │
│                                    ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                   6. RENDERING LAYER (Optional)                         │   │
│  │                                                                         │   │
│  │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │   │                   Video Renderer (ffmpeg)                       │   │   │
│  │   │   • Overlay B-roll visuals at specified timestamps              │   │   │
│  │   │   • Keep A-roll audio intact                                    │   │   │
│  │   │   • Output: final_video.mp4                                     │   │   │
│  │   └─────────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │         ARTIFACTS             │
                    │  • transcripts/               │
                    │  • broll_descriptions/        │
                    │  • timeline.json              │
                    │  • final_video.mp4 (optional) │
                    └───────────────────────────────┘
```

---

## 3. Data Flow Diagram

```
A-Roll Video          B-Roll Clips (6)
     │                      │
     ▼                      ▼
┌─────────┐           ┌─────────┐
│ Extract │           │ Extract │
│ Audio   │           │ Frames  │
└────┬────┘           └────┬────┘
     │                     │
     ▼                     ▼
┌─────────┐           ┌─────────┐
│ Whisper │           │ GPT-4V  │
│ ASR     │           │ Vision  │
└────┬────┘           └────┬────┘
     │                     │
     ▼                     ▼
┌─────────────┐      ┌─────────────┐
│ Transcript  │      │ Description │
│ [{start,    │      │ [{id,       │
│   end,      │      │   text}]    │
│   text}]    │      │             │
└──────┬──────┘      └──────┬──────┘
       │                    │
       ▼                    ▼
   ┌───────────────────────────┐
   │    Embedding Generation   │
   │    (text-embedding-ada)   │
   └─────────────┬─────────────┘
                 │
                 ▼
   ┌───────────────────────────┐
   │   Cosine Similarity       │
   │   Matrix Computation      │
   └─────────────┬─────────────┘
                 │
                 ▼
   ┌───────────────────────────┐
   │   Smart Matching Rules    │
   │   • threshold > 0.7       │
   │   • min gap 8-10 sec      │
   │   • avoid critical moments│
   └─────────────┬─────────────┘
                 │
                 ▼
   ┌───────────────────────────┐
   │   Timeline JSON Output    │
   └───────────────────────────┘
```

---

## 4. Project Folder Structure

```
smart-broll-inserter/
│
├── backend/
│   ├── app.py                    # FastAPI entry point
│   ├── config.py                 # API keys & settings
│   ├── requirements.txt          # Python dependencies
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── aroll_ingest.py       # Handle A-roll upload
│   │   └── broll_ingest.py       # Handle B-roll uploads
│   │
│   ├── transcription/
│   │   ├── __init__.py
│   │   └── transcriber.py        # A-roll → timestamped transcript
│   │
│   ├── understanding/
│   │   ├── __init__.py
│   │   ├── broll_caption.py      # B-roll → text description
│   │   └── embeddings.py         # Text → vector embeddings
│   │
│   ├── matching/
│   │   ├── __init__.py
│   │   └── matcher.py            # Semantic matching logic
│   │
│   ├── planning/
│   │   ├── __init__.py
│   │   └── timeline_planner.py   # Final timeline JSON
│   │
│   ├── rendering/
│   │   ├── __init__.py
│   │   └── video_renderer.py     # ffmpeg stitching (optional)
│   │
│   └── schemas/
│       ├── __init__.py
│       └── timeline_schema.py    # Pydantic output models
│
├── frontend/
│   ├── package.json
│   ├── src/
│   │   ├── App.jsx
│   │   ├── index.jsx
│   │   ├── components/
│   │   │   ├── UploadPanel.jsx
│   │   │   ├── TranscriptView.jsx
│   │   │   └── TimelineView.jsx
│   │   └── styles/
│   │       └── index.css
│   └── public/
│       └── index.html
│
├── artifacts/
│   ├── uploads/
│   │   ├── aroll/
│   │   └── broll/
│   ├── transcripts/
│   ├── captions/
│   └── output/
│       └── timeline.json
│
├── Design/
│   ├── Initial Design 1.md
│   ├── Initial project design.md
│   └── Architecture.md           # (THIS FILE)
│
├── .env.example
├── .gitignore
└── README.md
```

---

## 5. Tech Stack

| Layer | Technology | Reason |
|-------|------------|--------|
| **Backend** | FastAPI | Async support, auto-docs, fast development |
| **Transcription** | OpenAI Whisper API | Accurate, word-level timestamps |
| **B-Roll Understanding** | GPT-4 Vision | Best for video frame descriptions |
| **Embeddings** | `text-embedding-3-small` | Cost-effective semantic matching |
| **Similarity** | NumPy/SciPy | Fast cosine similarity calculation |
| **Frontend** | React + Tailwind CSS | Modern UI, rapid development |
| **Video Rendering** | ffmpeg-python | Python bindings for ffmpeg |
| **Storage** | Local filesystem | Simple for demo scope |

---

## 6. API Endpoints Design

### Backend REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload/aroll` | Upload A-roll video |
| `POST` | `/api/upload/broll` | Upload B-roll clips |
| `POST` | `/api/process` | Trigger full pipeline |
| `GET` | `/api/transcript` | Get A-roll transcript |
| `GET` | `/api/broll-descriptions` | Get B-roll descriptions |
| `GET` | `/api/timeline` | Get generated timeline JSON |
| `POST` | `/api/render` | Render final video (optional) |
| `GET` | `/api/status` | Check processing status |

---

## 7. Output Schema

### Timeline JSON Structure

```json
{
  "aroll_duration": 65.0,
  "aroll_filename": "speaker_video.mp4",
  "transcript": [
    {
      "start": 0.0,
      "end": 4.2,
      "text": "Hey everyone, today I want to show you..."
    },
    {
      "start": 4.2,
      "end": 7.8,
      "text": "This coffee maker gives a rich crema"
    }
  ],
  "broll_clips": [
    {
      "broll_id": "broll_01",
      "filename": "coffee_pour.mp4",
      "duration": 5.0,
      "description": "Close-up of coffee pouring into a white cup"
    }
  ],
  "insertions": [
    {
      "start_sec": 12.5,
      "duration_sec": 2.5,
      "broll_id": "broll_03",
      "confidence": 0.81,
      "reason": "Speaker mentions coffee texture; B-roll shows crema close-up"
    },
    {
      "start_sec": 28.0,
      "duration_sec": 3.0,
      "broll_id": "broll_01",
      "confidence": 0.76,
      "reason": "Speaker discusses pouring technique; B-roll demonstrates pour"
    }
  ],
  "generated_at": "2026-01-06T10:30:00Z"
}
```

---

## 8. Matching Logic Rules

The matching algorithm follows these rules:

1. **Semantic Threshold**: Only match if cosine similarity > 0.7
2. **Minimum Gap**: At least 8-10 seconds between insertions
3. **Avoid Critical Moments**: Skip segments with emotional keywords (e.g., "important", "key point", "listen")
4. **Duration Limits**: B-roll insertion duration: 2-4 seconds
5. **No Overlap**: B-roll insertions cannot overlap
6. **Limit Total**: Maximum 6 insertions for 90-second video

### Pseudo-code

```python
for segment in transcript_segments:
    if is_visually_useful(segment) and not is_critical_moment(segment):
        best_broll = find_best_match(segment, broll_embeddings)
        if best_broll.similarity > THRESHOLD:
            if time_since_last_insertion > MIN_GAP:
                schedule_insertion(segment.start, best_broll)
```

---

## 9. Implementation Timeline

### Day 1 – Foundations

| Task | Component | Status |
|------|-----------|--------|
| Setup project structure | All | ⬜ |
| A-roll upload endpoint | Ingestion | ⬜ |
| Audio extraction | Transcription | ⬜ |
| Whisper transcription | Transcription | ⬜ |
| B-roll upload endpoint | Ingestion | ⬜ |
| B-roll frame extraction | Understanding | ⬜ |
| GPT-4V description generation | Understanding | ⬜ |

### Day 2 – Intelligence

| Task | Component | Status |
|------|-----------|--------|
| Embedding generation | Understanding | ⬜ |
| Cosine similarity matching | Matching | ⬜ |
| Smart matching rules | Matching | ⬜ |
| Timeline JSON generation | Planning | ⬜ |
| API integration | Backend | ⬜ |

### Day 3 – UI & Polish

| Task | Component | Status |
|------|-----------|--------|
| React project setup | Frontend | ⬜ |
| Upload panel component | Frontend | ⬜ |
| Transcript view | Frontend | ⬜ |
| Timeline visualization | Frontend | ⬜ |
| Video rendering (ffmpeg) | Rendering | ⬜ |
| Testing & documentation | All | ⬜ |

---

## 10. Environment Variables

```env
# .env.example

# OpenAI API
OPENAI_API_KEY=sk-your-api-key-here

# Server Config
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000

# Upload Limits
MAX_AROLL_SIZE_MB=100
MAX_BROLL_SIZE_MB=50

# Processing Config
SIMILARITY_THRESHOLD=0.7
MIN_INSERTION_GAP_SEC=8
MAX_INSERTIONS=6
BROLL_DURATION_MIN=2.0
BROLL_DURATION_MAX=4.0
```

---

## 11. Key Dependencies

### Backend (Python)

```txt
fastapi==0.109.0
uvicorn==0.27.0
python-multipart==0.0.6
openai==1.12.0
moviepy==1.0.3
ffmpeg-python==0.2.0
numpy==1.26.0
scipy==1.12.0
pydantic==2.6.0
python-dotenv==1.0.0
```

### Frontend (Node.js)

```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "axios": "^1.6.0",
    "tailwindcss": "^3.4.0"
  }
}
```

---

## 12. Success Criteria

The project is successful if:

- ✅ A-roll transcription works with accurate timestamps
- ✅ B-roll clips have meaningful text descriptions
- ✅ Matching logic produces sensible insertions
- ✅ Timeline JSON is valid and human-readable
- ✅ Frontend allows upload and visualization
- ✅ (Bonus) Final video renders correctly

---

## 13. What Evaluators Care About

| ✅ They ARE judging | ❌ They are NOT judging |
|---------------------|------------------------|
| Clear reasoning | Perfect UI design |
| Smart matching logic | Fancy video transitions |
| Clean code structure | Edge case handling |
| Language → Video timeline connection | Performance optimization |
| System design clarity | 100% test coverage |

---

> **Final Note**: If the **timeline JSON makes sense to a human editor**, the solution is correct. Think like an editor, implement like an engineer.

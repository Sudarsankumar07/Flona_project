# Smart B-Roll Inserter

> Automatically plan B-roll insertions for UGC/talking-head videos using AI

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- ffmpeg installed and in PATH
- **API key (Gemini FREE or OpenAI)** OR **Run in Offline Mode** (see below)

### Installation

```bash
# Clone and enter project
cd Flona_project

# Setup backend
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env and add your API key
```

### Get Free API Keys
See [API_CREDITS_GUIDE.md](API_CREDITS_GUIDE.md) for free API setup.

**TL;DR**: Use Google Gemini - it's 100% FREE!

### 🌐 Offline Mode (No API Keys Required!)

**NEW**: Run the entire pipeline without any API keys using local HuggingFace models!

```bash
# Edit .env file
API_PROVIDER=offline
TRANSCRIPTION_PROVIDER=offline
OFFLINE_VISION_MODEL=blip
OFFLINE_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

**What's included:**
- **Vision**: BLIP model (~1GB) for B-roll captioning
- **Embeddings**: sentence-transformers (384-dim) for semantic matching
- **Transcription**: Optional Whisper model for A-roll transcription

**Recommended**: Run on Google Colab with GPU for faster inference (see [colab_runner.ipynb](colab_runner.ipynb))

**Test offline models:**
```bash
python backend/test_offline_models.py
```

---

## 🎬 Usage

### Option 1: Using video_url.json (Recommended)

Put your video URLs in `video_url.json`:

```json
{
  "a_roll": {
    "url": "https://example.com/aroll.mp4",
    "metadata": "Description of A-roll content"
  },
  "b_rolls": [
    {
      "id": "broll_1",
      "url": "https://example.com/broll1.mp4",
      "metadata": "Description of B-roll 1"
    }
  ]
}
```

Then run:

```bash
# Option A: Run the CLI pipeline directly
cd backend
python run_pipeline.py

# Option B: Use the API server
python app.py
# Then POST to http://localhost:8000/api/process-from-urls
```

### Option 2: Upload files via API

```bash
# Start server
cd backend
python app.py

# Upload A-roll
curl -X POST -F "file=@aroll.mp4" http://localhost:8000/api/upload/aroll

# Upload B-rolls
curl -X POST -F "files=@broll1.mp4" -F "files=@broll2.mp4" http://localhost:8000/api/upload/broll

# Process
curl -X POST http://localhost:8000/api/process

# Get timeline
curl http://localhost:8000/api/timeline
```

---

## 📁 Project Structure

```
Flona_project/
├── video_url.json             # ← Your input video URLs
├── backend/
│   ├── app.py                 # FastAPI main app
│   ├── run_pipeline.py        # CLI to run full pipeline
│   ├── config.py              # Configuration
│   ├── requirements.txt       # Dependencies
│   ├── .env.example           # Environment template
│   │
│   ├── ingestion/             # File upload & URL download
│   ├── transcription/         # Speech-to-text
│   ├── understanding/         # B-roll captioning & embeddings
│   ├── matching/              # Semantic matching logic
│   ├── planning/              # Timeline generation
│   ├── rendering/             # Video rendering (optional)
│   └── schemas/               # Data models
│
├── artifacts/                 # Generated files
│   ├── uploads/               # Downloaded videos
│   ├── transcripts/           # Transcription JSONs
│   ├── captions/              # B-roll captions
│   └── output/                # Timeline JSONs & rendered videos
│
└── API_CREDITS_GUIDE.md       # Free API guide
```

---

## 🔌 API Endpoints

### Main Endpoint (URL-based)
- `POST /api/process-from-urls` - **Process videos from video_url.json** ⭐

### Upload (alternative)
- `POST /api/upload/aroll` - Upload A-roll video
- `POST /api/upload/broll` - Upload B-roll clips

### Process
- `POST /api/transcribe` - Transcribe A-roll
- `POST /api/caption-broll` - Generate B-roll descriptions
- `POST /api/process` - Run full pipeline (after upload)

### Results
- `GET /api/timeline` - Get timeline JSON
- `GET /api/transcript` - Get transcript
- `GET /api/status` - Get processing status

### Render (Optional)
- `POST /api/render` - Render final video

---

## 📤 Example Output

```json
{
  "aroll_duration": 65.0,
  "total_insertions": 4,
  "insertions": [
    {
      "start_sec": 12.5,
      "duration_sec": 2.5,
      "broll_id": "broll_03",
      "broll_filename": "coffee_pour.mp4",
      "confidence": 0.81,
      "reason": "Speaker mentions 'rich crema' matching B-roll of coffee crema"
    }
  ]
}
```

---

## ⚙️ Configuration

Edit `backend/.env`:

```env
# Option 1: Use Gemini (FREE)
API_PROVIDER=gemini
GEMINI_API_KEY=your_key_here

# Option 2: Use OpenAI
# API_PROVIDER=openai
# OPENAI_API_KEY=your_key_here

# Option 3: Use Offline Mode (No API keys!)
# API_PROVIDER=offline
# TRANSCRIPTION_PROVIDER=offline
# OFFLINE_VISION_MODEL=blip
# OFFLINE_EMBEDDING_MODEL=all-MiniLM-L6-v2
# OFFLINE_WHISPER_MODEL=base

# Matching settings
SIMILARITY_THRESHOLD=0.65
MIN_GAP_SECONDS=8.0
MAX_INSERTIONS=6
```

---

## 🎬 How It Works

1. **Upload** A-roll (talking-head) + B-roll clips
2. **Transcribe** A-roll → timestamped text segments
3. **Caption** B-roll clips using vision AI
4. **Embed** all text into vectors
5. **Match** semantically similar content
6. **Plan** optimal insertion points
7. **Output** structured timeline JSON
8. **(Optional)** Render final video

---

## 📝 License

MIT License

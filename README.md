# Smart B-Roll Inserter

> Automatically plan B-roll insertions for UGC/talking-head videos using AI

![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20Mac-blue)
![Python](https://img.shields.io/badge/Python-3.9+-green)
![Node.js](https://img.shields.io/badge/Node.js-16+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🚀 Quick Start

### One-Command Setup

**Windows (PowerShell):**
```powershell
.\run.ps1
```

**Windows (CMD):**
```cmd
run.bat
```

**Linux/Mac:**
```bash
chmod +x run.sh
./run.sh
```

This will:
- ✅ Check Python & Node.js installation
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Create `.env` configuration file
- ✅ Start Backend API (http://localhost:8000)
- ✅ Start Frontend UI (http://localhost:3000)

---

## 📋 Prerequisites

- **Python 3.9+** - [Download](https://www.python.org/downloads/)
- **Node.js 16+** - [Download](https://nodejs.org/)
- **FFmpeg** - [Download](https://ffmpeg.org/download.html) (required for video processing)

### FFmpeg Installation

**Windows:**
```powershell
# Using Chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html and add to PATH
```

**Linux:**
```bash
sudo apt install ffmpeg
```

**Mac:**
```bash
brew install ffmpeg
```

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| 🎬 **Video Upload** | Upload A-roll (main video) and B-roll clips via drag & drop |
| 🔗 **URL Support** | Download videos directly from URLs |
| 🎤 **Transcription** | Automatic speech-to-text using Whisper |
| 🖼️ **B-Roll Captioning** | AI-generated descriptions for B-roll clips |
| 🔍 **Smart Matching** | Keyword-based matching for accurate B-roll placement |
| 📊 **Timeline Viewer** | Visual timeline showing insertion points |
| 🎥 **Video Rendering** | Export final video with B-roll insertions |

---

## 🖥️ Web Interface

After running the startup script, open **http://localhost:3000** in your browser:

1. **Upload A-Roll** - Your main talking-head video
2. **Upload B-Rolls** - Supporting footage clips
3. **Configure API** - Choose AI provider (optional)
4. **Process** - Click to analyze and generate timeline
5. **View Results** - See transcript and B-roll insertion points

---

## 📁 Project Structure

```
Flona_project/
├── run.sh                     # Linux/Mac startup script
├── run.bat                    # Windows CMD startup script
├── run.ps1                    # Windows PowerShell startup script
├── video_url.json             # Input video URLs (optional)
│
├── backend/                   # Python FastAPI Backend
│   ├── app.py                 # Main API server
│   ├── config.py              # Configuration settings
│   ├── requirements.txt       # Python dependencies
│   ├── .env                   # API keys (created on first run)
│   │
│   ├── ingestion/             # Video upload & URL download
│   ├── transcription/         # Speech-to-text (Whisper)
│   ├── understanding/         # B-roll captioning & embeddings
│   ├── matching/              # B-roll matching algorithms
│   │   ├── matcher.py         # Semantic similarity matcher
│   │   └── keyword_matcher.py # Keyword-based matcher
│   ├── planning/              # Timeline generation
│   ├── rendering/             # Video rendering (FFmpeg)
│   └── schemas/               # Data models
│
├── frontend/                  # React + Vite Frontend
│   ├── src/
│   │   ├── App.jsx            # Main application
│   │   ├── components/        # UI components
│   │   ├── context/           # React context
│   │   └── services/          # API service layer
│   └── package.json           # Node dependencies
│
├── artifacts/                 # Generated files (auto-created)
│   ├── uploads/
│   │   ├── aroll/             # Uploaded A-roll videos
│   │   └── broll/             # Uploaded B-roll clips
│   ├── transcripts/           # Transcription JSON files
│   ├── captions/              # B-roll captions JSON
│   ├── embeddings/            # Text embeddings
│   ├── matching/              # Match results JSON
│   └── output/                # Timeline JSON & rendered videos
│
└── Design/                    # Design documentation
```

---

## ⚙️ Configuration

Edit `backend/.env` to configure API providers:

```env
# =============================================================================
# API Provider: "gemini", "openai", "openrouter", or "offline"
# =============================================================================
API_PROVIDER=offline

# =============================================================================
# Google Gemini (FREE - Recommended)
# Get key: https://aistudio.google.com/apikey
# =============================================================================
GEMINI_API_KEY=your_gemini_key_here

# =============================================================================
# OpenAI (Paid)
# =============================================================================
OPENAI_API_KEY=your_openai_key_here

# =============================================================================
# OpenRouter (Multiple models via one API)
# Get key: https://openrouter.ai/keys
# =============================================================================
OPENROUTER_API_KEY=your_openrouter_key_here

# =============================================================================
# Matching Settings
# =============================================================================
SIMILARITY_THRESHOLD=0.65
MIN_GAP_SECONDS=3.0
MAX_INSERTIONS=6
```

### API Options

| Provider | Cost | Best For |
|----------|------|----------|
| **Offline** | Free | No API key needed, uses local models |
| **Gemini** | Free | Best free option with high accuracy |
| **OpenRouter** | Pay-per-use | Access to GPT-4, Claude, etc. |
| **OpenAI** | Pay-per-use | Premium quality |

---

## 🔌 API Endpoints

### Main Processing
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/process` | Process uploaded videos |
| `POST` | `/api/process-from-urls` | Process from video_url.json |

### Upload
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload/aroll` | Upload A-roll video |
| `POST` | `/api/upload/broll` | Upload B-roll clips |

### Results
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/timeline` | Get timeline JSON |
| `GET` | `/api/transcript` | Get transcript |
| `GET` | `/api/status` | Get processing status |

### Render
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/render` | Render final video |

**API Documentation:** http://localhost:8000/docs

---

## 📤 Output Format

The timeline JSON output:

```json
{
  "aroll_video": "artifacts/uploads/aroll/video.mp4",
  "aroll_duration": 45.0,
  "total_insertions": 5,
  "insertions": [
    {
      "timestamp": 1.0,
      "duration": 3.0,
      "broll_id": "broll_4",
      "broll_file": "artifacts/uploads/broll/broll_4.mp4",
      "transcript_text": "food quality and safety",
      "broll_description": "plates of food on a table",
      "match_score": 0.85,
      "reason": "Keyword match: food quality → plates of food"
    }
  ]
}
```

---

## 🎬 How It Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Upload    │────▶│  Transcribe │────▶│   Caption   │
│   Videos    │     │   A-Roll    │     │   B-Rolls   │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Render    │◀────│   Generate  │◀────│    Match    │
│   Video     │     │   Timeline  │     │   Content   │
└─────────────┘     └─────────────┘     └─────────────┘
```

1. **Upload** - A-roll (main video) + B-roll clips
2. **Transcribe** - Convert speech to timestamped text
3. **Caption** - Generate descriptions for B-roll clips
4. **Match** - Find semantic matches using keywords
5. **Timeline** - Generate insertion points with timing
6. **Render** - (Optional) Export final video

---

## 📚 Additional Guides

- [API_CREDITS_GUIDE.md](API_CREDITS_GUIDE.md) - Free API key setup
- [AI_PLANNING_GUIDE.md](AI_PLANNING_GUIDE.md) - AI-powered planning
- [OFFLINE_MODE_GUIDE.md](OFFLINE_MODE_GUIDE.md) - Running without APIs

---

## 🐛 Troubleshooting

### Backend won't start
```bash
# Check Python version
python --version  # Should be 3.9+

# Reinstall dependencies
cd backend
pip install -r requirements.txt
```

### Frontend won't start
```bash
# Check Node version
node --version  # Should be 16+

# Reinstall dependencies
cd frontend
rm -rf node_modules
npm install
```

### No B-roll insertions appearing
- Check that B-roll clips have been uploaded
- Verify transcription completed successfully
- Check `artifacts/matching/matching_results.json` for match scores

### FFmpeg not found
- Ensure FFmpeg is installed and in your system PATH
- Restart terminal after installation

---

## 📝 License

MIT License - Feel free to use and modify!

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

<p align="center">
  Made with ❤️ for content creators
</p>

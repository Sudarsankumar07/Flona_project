"""
Configuration module for Smart B-Roll Inserter
Supports both OpenAI and Google Gemini APIs
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# BASE PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
BACKEND_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
UPLOADS_DIR = ARTIFACTS_DIR / "uploads"
AROLL_DIR = UPLOADS_DIR / "aroll"
BROLL_DIR = UPLOADS_DIR / "broll"
TRANSCRIPTS_DIR = ARTIFACTS_DIR / "transcripts"
CAPTIONS_DIR = ARTIFACTS_DIR / "captions"
OUTPUT_DIR = ARTIFACTS_DIR / "output"

# Create directories if they don't exist
for dir_path in [AROLL_DIR, BROLL_DIR, TRANSCRIPTS_DIR, CAPTIONS_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# API CONFIGURATION
# =============================================================================

# API Provider Selection: "openai", "gemini", "openrouter", or "offline"
API_PROVIDER = os.getenv("API_PROVIDER", "gemini").lower()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_WHISPER_MODEL = os.getenv("OPENAI_WHISPER_MODEL", "whisper-1")

# Google Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")  # For AI planning
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")

# OpenRouter Configuration (access to many models via one API)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")  # Fast, cheap, great for planning

# Offline Model Configuration
OFFLINE_VISION_MODEL = os.getenv("OFFLINE_VISION_MODEL", "blip")  # "blip", "git", "moondream"
OFFLINE_EMBEDDING_MODEL = os.getenv("OFFLINE_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OFFLINE_WHISPER_MODEL = os.getenv("OFFLINE_WHISPER_MODEL", "base")  # "tiny", "base", "small", "medium"
OFFLINE_FRAMES_PER_VIDEO = int(os.getenv("OFFLINE_FRAMES_PER_VIDEO", "3"))  # Number of frames to extract per video

# =============================================================================
# TRANSCRIPTION SETTINGS
# =============================================================================
TRANSCRIPTION_PROVIDER = os.getenv("TRANSCRIPTION_PROVIDER", "openai").lower()  # "openai", "gemini", or "offline"

# =============================================================================
# MATCHING SETTINGS
# =============================================================================
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.65"))
MIN_GAP_SECONDS = float(os.getenv("MIN_GAP_SECONDS", "8.0"))
MAX_INSERTIONS = int(os.getenv("MAX_INSERTIONS", "6"))
MIN_INSERTIONS = int(os.getenv("MIN_INSERTIONS", "3"))
BROLL_MIN_DURATION = float(os.getenv("BROLL_MIN_DURATION", "2.0"))
BROLL_MAX_DURATION = float(os.getenv("BROLL_MAX_DURATION", "4.0"))

# Keywords that indicate critical speaking moments (avoid B-roll here)
CRITICAL_KEYWORDS = [
    "important", "key point", "listen", "remember", "note this",
    "crucial", "essential", "must", "never forget", "pay attention"
]

# =============================================================================
# VIDEO SETTINGS
# =============================================================================
SUPPORTED_VIDEO_FORMATS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
MAX_AROLL_DURATION = 90  # seconds
MIN_AROLL_DURATION = 30  # seconds
MAX_FILE_SIZE_MB = 100

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_vision_provider():
    """Get the active vision API provider"""
    if API_PROVIDER == "offline":
        return "offline"
    elif API_PROVIDER == "openai" and OPENAI_API_KEY:
        return "openai"
    elif API_PROVIDER == "gemini" and GEMINI_API_KEY:
        return "gemini"
    elif GEMINI_API_KEY:
        return "gemini"
    elif OPENAI_API_KEY:
        return "openai"
    else:
        # Default to offline if no API keys
        return "offline"

def get_embedding_provider():
    """Get the active embedding API provider"""
    # For embeddings, prefer the same provider as vision
    vision_provider = get_vision_provider()
    return vision_provider

def validate_config():
    """Validate that required configuration is present"""
    errors = []
    
    # Offline mode doesn't require API keys
    if API_PROVIDER == "offline":
        return True
    
    if not OPENAI_API_KEY and not GEMINI_API_KEY:
        errors.append("No API key configured. Set OPENAI_API_KEY or GEMINI_API_KEY in .env file, or use API_PROVIDER=offline")
    
    if TRANSCRIPTION_PROVIDER == "openai" and not OPENAI_API_KEY:
        errors.append("TRANSCRIPTION_PROVIDER is 'openai' but OPENAI_API_KEY is not set")
    
    if errors:
        raise ValueError("\n".join(errors))
    
    return True

def get_config_summary():
    """Get a summary of current configuration"""
    return {
        "api_provider": API_PROVIDER,
        "vision_provider": get_vision_provider() if (OPENAI_API_KEY or GEMINI_API_KEY) else None,
        "transcription_provider": TRANSCRIPTION_PROVIDER,
        "openai_configured": bool(OPENAI_API_KEY),
        "gemini_configured": bool(GEMINI_API_KEY),
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "min_gap_seconds": MIN_GAP_SECONDS,
        "max_insertions": MAX_INSERTIONS,
    }

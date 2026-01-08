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
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"
MATCHING_DIR = ARTIFACTS_DIR / "matching"
OUTPUT_DIR = ARTIFACTS_DIR / "output"

# Create all directories if they don't exist
ALL_DIRS = [
    AROLL_DIR, 
    BROLL_DIR, 
    TRANSCRIPTS_DIR, 
    CAPTIONS_DIR, 
    EMBEDDINGS_DIR,
    MATCHING_DIR,
    OUTPUT_DIR
]
for dir_path in ALL_DIRS:
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
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")  # Default model for planning
OPENROUTER_VISION_MODEL = os.getenv("OPENROUTER_VISION_MODEL", "openai/gpt-4o")  # For vision tasks

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
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.15"))  # Lowered for offline embeddings
MIN_GAP_SECONDS = float(os.getenv("MIN_GAP_SECONDS", "3.0"))  # Minimum gap between insertions (reduced for more coverage)
MAX_INSERTIONS = int(os.getenv("MAX_INSERTIONS", "6"))
MIN_INSERTIONS = int(os.getenv("MIN_INSERTIONS", "2"))
BROLL_MIN_DURATION = float(os.getenv("BROLL_MIN_DURATION", "2.0"))
BROLL_MAX_DURATION = float(os.getenv("BROLL_MAX_DURATION", "4.0"))

# Keywords that indicate critical speaking moments (avoid B-roll here)
# Reduced list to avoid over-filtering
CRITICAL_KEYWORDS = [
    "listen carefully", "remember this", "never forget", "pay attention"
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
    elif API_PROVIDER == "openrouter" and OPENROUTER_API_KEY:
        return "openrouter"
    elif API_PROVIDER == "openai" and OPENAI_API_KEY:
        return "openai"
    elif API_PROVIDER == "gemini" and GEMINI_API_KEY:
        return "gemini"
    elif OPENROUTER_API_KEY:
        return "openrouter"
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
    
    if not OPENAI_API_KEY and not GEMINI_API_KEY and not OPENROUTER_API_KEY:
        errors.append("No API key configured. Set OPENAI_API_KEY, GEMINI_API_KEY, or OPENROUTER_API_KEY in .env file, or use API_PROVIDER=offline")
    
    if TRANSCRIPTION_PROVIDER == "openai" and not OPENAI_API_KEY:
        errors.append("TRANSCRIPTION_PROVIDER is 'openai' but OPENAI_API_KEY is not set")
    
    if TRANSCRIPTION_PROVIDER == "openrouter" and not OPENROUTER_API_KEY:
        errors.append("TRANSCRIPTION_PROVIDER is 'openrouter' but OPENROUTER_API_KEY is not set")
    
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

"""
Smart B-Roll Inserter - FastAPI Backend
Main application entry point with all API endpoints
"""

import time
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# Import all modules
from config import (
    validate_config,
    get_config_summary,
    OUTPUT_DIR,
    AROLL_DIR,
    BROLL_DIR,
)
from schemas import (
    TimelinePlan,
    TranscriptResponse,
    BRollCaptionsResponse,
    UploadResponse,
)
from ingestion import ArollIngestor, BrollIngestor
from transcription import Transcriber
from understanding import BRollCaptioner, EmbeddingGenerator
from matching import SemanticMatcher
from planning import TimelinePlanner
from rendering import VideoRenderer


# Application state
class AppState:
    """Holds application state and cached results"""
    def __init__(self):
        self.aroll_path: Optional[str] = None
        self.aroll_duration: Optional[float] = None
        self.aroll_filename: Optional[str] = None
        self.transcript_segments: list = []
        self.broll_descriptions: list = []
        self.transcript_embeddings: dict = {}
        self.broll_embeddings: dict = {}
        self.similarity_matrix: dict = {}
        self.selected_matches: list = []
        self.timeline: Optional[TimelinePlan] = None
        self.processing_start_time: Optional[float] = None


state = AppState()


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    print("=" * 50)
    print("Smart B-Roll Inserter Backend Starting...")
    print("=" * 50)
    
    try:
        validate_config()
        config = get_config_summary()
        print(f"Vision Provider: {config.get('vision_provider', 'Not configured')}")
        print(f"OpenAI Configured: {config.get('openai_configured')}")
        print(f"Gemini Configured: {config.get('gemini_configured')}")
    except ValueError as e:
        print(f"Configuration Warning: {e}")
    
    print("=" * 50)
    
    yield
    
    # Shutdown
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Smart B-Roll Inserter API",
    description="Automatically plan B-roll insertions for UGC videos using AI",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Smart B-Roll Inserter API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "upload_aroll": "POST /api/upload/aroll",
            "upload_broll": "POST /api/upload/broll",
            "transcribe": "POST /api/transcribe",
            "caption_broll": "POST /api/caption-broll",
            "process": "POST /api/process",
            "timeline": "GET /api/timeline",
            "render": "POST /api/render"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/api/config")
async def get_config():
    """Get current configuration (without sensitive data)"""
    return get_config_summary()


@app.get("/api/status")
async def get_status():
    """Get current processing status"""
    return {
        "aroll_uploaded": state.aroll_path is not None,
        "aroll_filename": state.aroll_filename,
        "aroll_duration": state.aroll_duration,
        "transcript_ready": len(state.transcript_segments) > 0,
        "transcript_segments": len(state.transcript_segments),
        "broll_count": len(state.broll_descriptions),
        "broll_captions_ready": len(state.broll_descriptions) > 0,
        "embeddings_ready": len(state.transcript_embeddings) > 0,
        "matches_found": len(state.selected_matches),
        "timeline_ready": state.timeline is not None
    }


# ============================================================================
# Upload Endpoints
# ============================================================================

@app.post("/api/upload/aroll", response_model=UploadResponse)
async def upload_aroll(file: UploadFile = File(...)):
    """
    Upload A-roll (talking-head) video
    
    - Validates video format and duration (30-90 seconds)
    - Stores file for processing
    """
    ingestor = ArollIngestor()
    
    filepath, duration = await ingestor.ingest(file)
    
    # Update state
    state.aroll_path = filepath
    state.aroll_duration = duration
    state.aroll_filename = file.filename
    
    # Reset dependent state
    state.transcript_segments = []
    state.transcript_embeddings = {}
    state.similarity_matrix = {}
    state.selected_matches = []
    state.timeline = None
    
    return UploadResponse(
        success=True,
        message=f"A-roll uploaded successfully",
        aroll_filename=file.filename,
        aroll_duration=duration
    )


@app.post("/api/upload/broll", response_model=UploadResponse)
async def upload_broll(files: List[UploadFile] = File(...)):
    """
    Upload multiple B-roll video clips
    
    - Accepts up to 6 B-roll clips
    - Validates video formats
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 B-roll clips allowed")
    
    ingestor = BrollIngestor()
    
    # Clear existing B-rolls
    ingestor.clear_brolls()
    
    broll_info = await ingestor.ingest_multiple(files)
    
    # Reset dependent state
    state.broll_descriptions = []
    state.broll_embeddings = {}
    state.similarity_matrix = {}
    state.selected_matches = []
    state.timeline = None
    
    return UploadResponse(
        success=True,
        message=f"Uploaded {len(broll_info)} B-roll clips",
        broll_files=[b["filename"] for b in broll_info],
        broll_count=len(broll_info)
    )


# ============================================================================
# Processing Endpoints
# ============================================================================

@app.post("/api/transcribe", response_model=TranscriptResponse)
async def transcribe_aroll():
    """
    Transcribe A-roll video to timestamped segments
    
    Uses OpenAI Whisper or Google Gemini based on configuration
    """
    if not state.aroll_path:
        raise HTTPException(status_code=400, detail="No A-roll uploaded. Upload A-roll first.")
    
    transcriber = Transcriber()
    
    # Check for existing transcript
    existing = transcriber.load_transcript(state.aroll_path)
    if existing:
        state.transcript_segments = existing
        return TranscriptResponse(
            success=True,
            message="Loaded existing transcript",
            aroll_filename=state.aroll_filename,
            duration=state.aroll_duration,
            segments=existing,
            segment_count=len(existing)
        )
    
    # Transcribe
    segments = await transcriber.transcribe(state.aroll_path)
    state.transcript_segments = segments
    
    return TranscriptResponse(
        success=True,
        message="Transcription complete",
        aroll_filename=state.aroll_filename,
        duration=state.aroll_duration,
        segments=segments,
        segment_count=len(segments)
    )


@app.post("/api/caption-broll", response_model=BRollCaptionsResponse)
async def caption_broll():
    """
    Generate AI descriptions for all B-roll clips
    
    Uses GPT-4 Vision or Gemini Vision based on configuration
    """
    ingestor = BrollIngestor()
    brolls = ingestor.get_all_brolls()
    
    if not brolls:
        raise HTTPException(status_code=400, detail="No B-roll uploaded. Upload B-roll clips first.")
    
    captioner = BRollCaptioner()
    
    # Check for existing captions
    existing = captioner.load_captions()
    if existing and len(existing) == len(brolls):
        state.broll_descriptions = existing
        return BRollCaptionsResponse(
            success=True,
            message="Loaded existing captions",
            captions=existing,
            caption_count=len(existing)
        )
    
    # Generate captions
    descriptions = await captioner.caption_all(brolls)
    state.broll_descriptions = descriptions
    
    return BRollCaptionsResponse(
        success=True,
        message="B-roll captions generated",
        captions=descriptions,
        caption_count=len(descriptions)
    )


@app.post("/api/process")
async def process_full_pipeline():
    """
    Run the complete processing pipeline:
    1. Transcribe A-roll (if not done)
    2. Caption B-roll (if not done)
    3. Generate embeddings
    4. Compute similarity matrix
    5. Find best matches
    6. Generate timeline plan
    
    Returns the complete timeline JSON
    """
    state.processing_start_time = time.time()
    
    # Step 1: Check uploads
    if not state.aroll_path:
        raise HTTPException(status_code=400, detail="No A-roll uploaded")
    
    ingestor = BrollIngestor()
    brolls = ingestor.get_all_brolls()
    if not brolls:
        raise HTTPException(status_code=400, detail="No B-roll uploaded")
    
    # Step 2: Transcribe if needed
    if not state.transcript_segments:
        transcriber = Transcriber()
        existing = transcriber.load_transcript(state.aroll_path)
        if existing:
            state.transcript_segments = existing
        else:
            state.transcript_segments = await transcriber.transcribe(state.aroll_path)
    
    # Step 3: Caption B-roll if needed
    if not state.broll_descriptions:
        captioner = BRollCaptioner()
        existing = captioner.load_captions()
        if existing and len(existing) == len(brolls):
            state.broll_descriptions = existing
        else:
            state.broll_descriptions = await captioner.caption_all(brolls)
    
    # Step 4: Generate embeddings
    embedder = EmbeddingGenerator()
    
    state.transcript_embeddings = await embedder.embed_transcript_segments(
        state.transcript_segments
    )
    
    state.broll_embeddings = await embedder.embed_broll_descriptions(
        state.broll_descriptions
    )
    
    # Step 5: Compute similarity matrix
    state.similarity_matrix = embedder.compute_similarity_matrix(
        state.transcript_embeddings,
        state.broll_embeddings
    )
    
    # Step 6: Find best matches
    matcher = SemanticMatcher()
    state.selected_matches = matcher.find_best_matches(
        state.transcript_segments,
        state.broll_descriptions,
        state.similarity_matrix
    )
    
    # Step 7: Generate timeline
    planner = TimelinePlanner()
    state.timeline = planner.generate_timeline(
        aroll_filename=state.aroll_filename,
        aroll_duration=state.aroll_duration,
        segments=state.transcript_segments,
        broll_descriptions=state.broll_descriptions,
        selected_matches=state.selected_matches,
        processing_start_time=state.processing_start_time
    )
    
    # Validate timeline
    validation = planner.validate_timeline(state.timeline)
    
    return {
        "success": True,
        "message": "Processing complete",
        "timeline": state.timeline.model_dump(),
        "validation": validation,
        "processing_time_sec": state.timeline.processing_time_sec
    }


# ============================================================================
# Results Endpoints
# ============================================================================

@app.get("/api/transcript")
async def get_transcript():
    """Get the current transcript"""
    if not state.transcript_segments:
        raise HTTPException(status_code=404, detail="No transcript available. Run transcription first.")
    
    return {
        "aroll_filename": state.aroll_filename,
        "duration": state.aroll_duration,
        "segments": [s.model_dump() for s in state.transcript_segments],
        "segment_count": len(state.transcript_segments)
    }


@app.get("/api/broll-captions")
async def get_broll_captions():
    """Get the current B-roll captions"""
    if not state.broll_descriptions:
        raise HTTPException(status_code=404, detail="No captions available. Run captioning first.")
    
    return {
        "captions": [d.model_dump() for d in state.broll_descriptions],
        "caption_count": len(state.broll_descriptions)
    }


@app.get("/api/timeline")
async def get_timeline():
    """Get the generated timeline plan"""
    if not state.timeline:
        # Try to load from file
        planner = TimelinePlanner()
        state.timeline = planner.load_latest_timeline()
    
    if not state.timeline:
        raise HTTPException(status_code=404, detail="No timeline available. Run processing first.")
    
    return state.timeline.model_dump()


@app.get("/api/timeline/simple")
async def get_simple_timeline():
    """Get a simplified version of the timeline"""
    if not state.timeline:
        planner = TimelinePlanner()
        state.timeline = planner.load_latest_timeline()
    
    if not state.timeline:
        raise HTTPException(status_code=404, detail="No timeline available")
    
    planner = TimelinePlanner()
    return planner.export_simple_timeline(state.timeline)


@app.get("/api/similarity-matrix")
async def get_similarity_matrix():
    """Get the computed similarity matrix"""
    if not state.similarity_matrix:
        raise HTTPException(status_code=404, detail="No similarity matrix available")
    
    return {
        "matrix": state.similarity_matrix,
        "transcript_count": len(state.transcript_embeddings),
        "broll_count": len(state.broll_embeddings)
    }


# ============================================================================
# Rendering Endpoints
# ============================================================================

@app.post("/api/render")
async def render_video():
    """
    Render the final video with B-roll insertions
    
    Uses ffmpeg to overlay B-roll onto A-roll while keeping A-roll audio
    """
    if not state.timeline:
        raise HTTPException(status_code=400, detail="No timeline available. Run processing first.")
    
    if not state.aroll_path:
        raise HTTPException(status_code=400, detail="A-roll video not found")
    
    renderer = VideoRenderer()
    
    try:
        # Use the simpler rendering approach
        output_path = renderer.render_simple(
            aroll_path=state.aroll_path,
            timeline=state.timeline
        )
        
        return {
            "success": True,
            "message": "Video rendered successfully",
            "output_path": output_path,
            "preview": renderer.preview_timeline(state.timeline)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rendering failed: {str(e)}")


@app.get("/api/render/preview")
async def get_render_preview():
    """Get a text preview of what the rendered video will look like"""
    if not state.timeline:
        raise HTTPException(status_code=404, detail="No timeline available")
    
    renderer = VideoRenderer()
    preview = renderer.preview_timeline(state.timeline)
    
    return {"preview": preview}


@app.get("/api/download/timeline")
async def download_timeline():
    """Download the timeline JSON file"""
    timeline_path = OUTPUT_DIR / "timeline_latest.json"
    
    if not timeline_path.exists():
        raise HTTPException(status_code=404, detail="No timeline file available")
    
    return FileResponse(
        path=str(timeline_path),
        filename="timeline.json",
        media_type="application/json"
    )


@app.get("/api/download/video")
async def download_rendered_video():
    """Download the rendered video if available"""
    # Find most recent rendered video
    rendered_videos = list(OUTPUT_DIR.glob("rendered_*.mp4"))
    
    if not rendered_videos:
        raise HTTPException(status_code=404, detail="No rendered video available")
    
    # Get most recent
    latest = max(rendered_videos, key=lambda p: p.stat().st_mtime)
    
    return FileResponse(
        path=str(latest),
        filename=latest.name,
        media_type="video/mp4"
    )


# ============================================================================
# Reset Endpoint
# ============================================================================

@app.post("/api/reset")
async def reset_state():
    """Reset all application state and clear uploaded files"""
    global state
    
    # Clear files
    aroll_ingestor = ArollIngestor()
    broll_ingestor = BrollIngestor()
    
    aroll_ingestor.clear_aroll()
    broll_ingestor.clear_brolls()
    
    # Reset state
    state = AppState()
    
    return {"success": True, "message": "Application state reset"}


# ============================================================================
# Run with Uvicorn
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

"""
Smart B-Roll Inserter - FastAPI Backend
Main application entry point with all API endpoints
"""

import time
from pathlib import Path
from typing import List, Optional
from pathlib import Path
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
from ingestion import ArollIngestor, BrollIngestor, VideoDownloader
from transcription import Transcriber
from understanding import BRollCaptioner, EmbeddingGenerator
from matching import SemanticMatcher, KeywordMatcher
from planning import TimelinePlanner
from planning.ai_planner import AIInsertionPlanner
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
            "process_from_urls": "POST /api/process-from-urls  ← Use this! (reads video_url.json)",
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
    
    # Step 6: Find best matches using KeywordMatcher
    print("📝 Using Keyword Matcher for B-roll insertions...")
    keyword_matcher = KeywordMatcher()
    state.selected_matches = keyword_matcher.find_matches(
        segments=state.transcript_segments,
        broll_descriptions=state.broll_descriptions,
        max_insertions=6,
        min_gap_seconds=3.0
    )
    print(f"✓ Keyword Matcher returned {len(state.selected_matches)} insertions")
    
    # Log each insertion for debugging
    for ins in state.selected_matches:
        print(f"  - {ins.start_sec}s: {ins.broll_id} ({ins.confidence:.2f})")
    
    # Step 7: Generate timeline
    planner = TimelinePlanner()
    state.timeline = planner.create_timeline(
        aroll_filename=state.aroll_filename,
        aroll_duration=state.aroll_duration,
        matches=state.selected_matches,
        transcript_segments=state.transcript_segments,
        broll_descriptions=state.broll_descriptions
    )
    
    # Save the timeline
    planner.save_timeline(state.timeline)
    
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
# URL-Based Processing (from video_url.json)
# ============================================================================

@app.post("/api/process-from-urls")
async def process_from_urls(json_path: Optional[str] = None):
    """
    Complete pipeline using videos from video_url.json
    
    1. Downloads A-roll and B-roll videos from URLs
    2. Transcribes A-roll
    3. Captions B-roll clips
    4. Generates embeddings
    5. Matches semantically
    6. Creates timeline plan
    
    Args:
        json_path: Optional custom path to video_url.json
    """
    state.processing_start_time = time.time()
    
    print("=" * 50)
    print("PROCESSING FROM URL JSON")
    print("=" * 50)
    
    # Step 1: Download videos from URLs
    print("\n[1/6] Downloading videos from URLs...")
    downloader = VideoDownloader()
    
    try:
        download_results = await downloader.download_from_json(json_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download videos: {str(e)}")
    
    if not download_results["aroll"]:
        raise HTTPException(status_code=400, detail="No A-roll URL found in JSON")
    
    if not download_results["brolls"]:
        raise HTTPException(status_code=400, detail="No B-roll URLs found in JSON")
    
    # Update state with downloaded files
    state.aroll_path = download_results["aroll"]["path"]
    state.aroll_duration = downloader.get_video_duration(state.aroll_path)
    state.aroll_filename = Path(state.aroll_path).name
    
    print(f"  A-roll: {state.aroll_filename} ({state.aroll_duration:.1f}s)")
    print(f"  B-rolls: {len(download_results['brolls'])} clips downloaded")
    
    # Step 2: Transcribe A-roll
    print("\n[2/6] Transcribing A-roll...")
    transcriber = Transcriber()
    
    existing_transcript = transcriber.load_transcript(state.aroll_path)
    if existing_transcript:
        state.transcript_segments = existing_transcript
        print(f"  Loaded existing transcript ({len(existing_transcript)} segments)")
    else:
        state.transcript_segments = await transcriber.transcribe(state.aroll_path)
        print(f"  Transcribed: {len(state.transcript_segments)} segments")
    
    # Step 3: Prepare B-roll info and caption them
    print("\n[3/6] Captioning B-roll clips...")
    
    # Build broll_files list for captioner
    broll_files = []
    for broll in download_results["brolls"]:
        duration = downloader.get_video_duration(broll["path"])
        broll_files.append({
            "broll_id": broll["broll_id"],
            "filename": broll["filename"],
            "filepath": broll["path"],
            "duration": duration,
            "metadata": broll.get("metadata", "")
        })
    
    captioner = BRollCaptioner()
    
    # Check for existing captions
    existing_captions = captioner.load_captions()
    if existing_captions and len(existing_captions) == len(broll_files):
        state.broll_descriptions = existing_captions
        print(f"  Loaded existing captions ({len(existing_captions)} clips)")
    else:
        # Use metadata if available, otherwise generate with AI
        state.broll_descriptions = await caption_with_metadata(captioner, broll_files)
        print(f"  Captioned: {len(state.broll_descriptions)} clips")
    
    # Step 4: Generate embeddings
    print("\n[4/6] Generating embeddings...")
    embedder = EmbeddingGenerator()
    
    state.transcript_embeddings = await embedder.embed_transcript_segments(
        state.transcript_segments
    )
    print(f"  Transcript embeddings: {len(state.transcript_embeddings)}")
    
    state.broll_embeddings = await embedder.embed_broll_descriptions(
        state.broll_descriptions
    )
    print(f"  B-roll embeddings: {len(state.broll_embeddings)}")
    
    # Step 5: Compute similarity and match
    print("\n[5/6] Finding semantic matches...")
    state.similarity_matrix = embedder.compute_similarity_matrix(
        state.transcript_embeddings,
        state.broll_embeddings
    )
    
    # Use KeywordMatcher for better results
    print("📝 Using Keyword Matcher for B-roll insertions...")
    keyword_matcher = KeywordMatcher()
    state.selected_matches = keyword_matcher.find_matches(
        segments=state.transcript_segments,
        broll_descriptions=state.broll_descriptions,
        max_insertions=6,
        min_gap_seconds=3.0
    )
    print(f"  Selected matches: {len(state.selected_matches)}")
    
    # Log each insertion for debugging
    for ins in state.selected_matches:
        print(f"    - {ins.start_sec}s: {ins.broll_id} ({ins.confidence:.2f})")
    
    # Step 6: Generate timeline
    print("\n[6/6] Generating timeline plan...")
    planner = TimelinePlanner()
    state.timeline = planner.create_timeline(
        aroll_filename=state.aroll_filename,
        aroll_duration=state.aroll_duration,
        matches=state.selected_matches,
        transcript_segments=state.transcript_segments,
        broll_descriptions=state.broll_descriptions
    )
    
    # Save timeline
    planner.save_timeline(state.timeline)
    
    # Validate
    validation = planner.validate_timeline(state.timeline)
    
    print("\n" + "=" * 50)
    print("PROCESSING COMPLETE!")
    print(f"  Total time: {state.timeline.processing_time_sec:.1f}s")
    print(f"  Insertions: {state.timeline.total_insertions}")
    print("=" * 50)
    
    return {
        "success": True,
        "message": "Processing complete",
        "downloads": {
            "aroll": download_results["aroll"]["path"],
            "broll_count": len(download_results["brolls"])
        },
        "timeline": state.timeline.model_dump(),
        "validation": validation,
        "processing_time_sec": state.timeline.processing_time_sec
    }


async def caption_with_metadata(captioner: BRollCaptioner, broll_files: list) -> list:
    """
    Caption B-roll clips, using metadata from JSON if available
    Falls back to AI captioning if no metadata
    """
    from schemas import BRollDescription
    
    descriptions = []
    
    for broll in broll_files:
        metadata = broll.get("metadata", "").strip()
        
        if metadata:
            # Use provided metadata as description
            descriptions.append(BRollDescription(
                broll_id=broll["broll_id"],
                filename=broll["filename"],
                description=metadata,
                duration=broll["duration"],
                filepath=broll["filepath"]
            ))
            print(f"    {broll['broll_id']}: Using metadata")
        else:
            # Generate with AI
            desc = await captioner.caption_single(
                filepath=broll["filepath"],
                broll_id=broll["broll_id"],
                filename=broll["filename"],
                duration=broll["duration"]
            )
            descriptions.append(desc)
            print(f"    {broll['broll_id']}: AI generated")
    
    # Save captions
    captioner._save_captions(descriptions)
    
    return descriptions


# ============================================================================
# Frontend API Endpoints
# ============================================================================

import uuid
import os
from pydantic import BaseModel
from typing import Dict, Any

# Store for background jobs
processing_jobs: Dict[str, Dict[str, Any]] = {}


class ConfigureRequest(BaseModel):
    provider: str
    api_key: Optional[str] = None


class ProcessRequest(BaseModel):
    settings: Optional[Dict[str, Any]] = None


class UrlUploadRequest(BaseModel):
    a_roll: Dict[str, str]  # { url, metadata }
    b_rolls: List[Dict[str, str]]  # [{ id, url, metadata }]


@app.post("/api/upload-urls")
async def upload_from_urls(request: UrlUploadRequest):
    """
    Download and process videos from URLs
    Similar to video_url.json functionality
    """
    from ingestion.url_downloader import VideoDownloader
    
    print(f"URL Upload request received:")
    print(f"  A-roll URL: {request.a_roll.get('url', 'None')[:50]}...")
    print(f"  B-rolls: {len(request.b_rolls)} URLs")
    
    result = {
        "success": True,
        "aroll": None,
        "brolls": []
    }
    
    downloader = VideoDownloader()
    
    try:
        # Download A-roll
        if request.a_roll.get('url'):
            aroll_path = await downloader.download_video(
                url=request.a_roll['url'],
                save_dir=downloader.aroll_dir,
                filename="a_roll.mp4"
            )
            
            # Get duration
            duration = downloader.get_video_duration(aroll_path)
            if duration == 0:
                # Fallback using cv2
                try:
                    import cv2
                    cap = cv2.VideoCapture(aroll_path)
                    if cap.isOpened():
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        duration = frame_count / fps if fps > 0 else 60.0
                        cap.release()
                except:
                    duration = 60.0
            
            state.aroll_path = aroll_path
            state.aroll_duration = duration
            state.aroll_filename = "a_roll.mp4"
            
            result["aroll"] = {
                "filename": "a_roll.mp4",
                "duration": duration,
                "path": aroll_path,
                "metadata": request.a_roll.get('metadata', '')
            }
            print(f"  A-roll downloaded: {aroll_path}")
        
        # Download B-rolls
        for idx, broll in enumerate(request.b_rolls):
            if broll.get('url'):
                broll_id = broll.get('id', f'broll_{idx + 1}')
                filename = f"{broll_id}.mp4"
                
                try:
                    broll_path = await downloader.download_video(
                        url=broll['url'],
                        save_dir=downloader.broll_dir,
                        filename=filename
                    )
                    
                    duration = downloader.get_video_duration(broll_path)
                    if duration == 0:
                        duration = 5.0
                    
                    result["brolls"].append({
                        "id": broll_id,
                        "filename": filename,
                        "duration": duration,
                        "path": broll_path,
                        "metadata": broll.get('metadata', '')
                    })
                    print(f"  B-roll downloaded: {filename}")
                except Exception as e:
                    print(f"  Warning: B-roll {broll_id} download failed: {e}")
        
        return result
        
    except Exception as e:
        print(f"URL upload error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"URL download failed: {str(e)}")


@app.post("/api/upload")
async def upload_videos(
    aroll: UploadFile = File(None),
    brolls: List[UploadFile] = File(default=[]),
    broll_metadata: Optional[str] = None
):
    """
    Upload A-roll and B-roll videos together
    Used by frontend for combined upload
    """
    import json
    
    print(f"Upload request received:")
    print(f"  A-roll: {aroll.filename if aroll else 'None'}")
    print(f"  B-rolls: {len(brolls)} files")
    print(f"  Metadata: {broll_metadata}")
    
    result = {
        "success": True,
        "aroll": None,
        "brolls": []
    }
    
    # Handle A-roll upload
    if aroll and aroll.filename:
        aroll_ingestor = ArollIngestor()
        try:
            filepath, duration = await aroll_ingestor.ingest(aroll)
            state.aroll_path = filepath
            state.aroll_duration = duration
            state.aroll_filename = aroll.filename
            
            result["aroll"] = {
                "filename": aroll.filename,
                "duration": duration,
                "path": filepath
            }
            print(f"  A-roll uploaded: {aroll.filename}")
        except Exception as e:
            print(f"  A-roll upload failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"A-roll upload failed: {str(e)}")
    
    # Handle B-roll uploads
    if brolls:
        broll_ingestor = BrollIngestor()
        
        # Parse metadata if provided
        metadata_list = []
        if broll_metadata:
            try:
                metadata_list = json.loads(broll_metadata)
            except:
                metadata_list = []
        
        for idx, broll_file in enumerate(brolls):
            try:
                broll_id = f"broll_{idx + 1}"
                filepath, duration = await broll_ingestor.ingest(broll_file, broll_id=broll_id)
                
                # Get metadata for this broll
                meta = ""
                if idx < len(metadata_list):
                    meta = metadata_list[idx].get("metadata", "")
                
                result["brolls"].append({
                    "id": broll_id,
                    "filename": broll_file.filename,
                    "duration": duration,
                    "path": filepath,
                    "metadata": meta
                })
            except Exception as e:
                print(f"Warning: B-roll {broll_file.filename} upload failed: {e}")
    
    return result


@app.post("/api/configure")
async def configure_api(request: ConfigureRequest):
    """
    Configure API provider and key
    Updates .env file with new settings
    """
    import os
    from dotenv import set_key
    
    env_path = Path(__file__).parent / ".env"
    
    # Update environment variables
    os.environ["API_PROVIDER"] = request.provider
    os.environ["TRANSCRIPTION_PROVIDER"] = request.provider
    
    if request.api_key:
        if request.provider == "openrouter":
            os.environ["OPENROUTER_API_KEY"] = request.api_key
        elif request.provider == "gemini":
            os.environ["GEMINI_API_KEY"] = request.api_key
        elif request.provider == "openai":
            os.environ["OPENAI_API_KEY"] = request.api_key
    
    # Reload config
    from importlib import reload
    import config
    reload(config)
    
    return {
        "success": True,
        "provider": request.provider,
        "mode": "offline" if request.provider == "offline" else "online"
    }


@app.post("/api/process")
async def start_processing(request: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Start the processing pipeline in background
    Returns job_id for status polling
    """
    import asyncio
    
    job_id = str(uuid.uuid4())
    
    print(f"Starting processing job: {job_id}")
    print(f"Settings: {request.settings}")
    
    # Initialize job status
    processing_jobs[job_id] = {
        "status": "processing",
        "progress": 0,
        "current_step": "Starting...",
        "error": None
    }
    
    # Run the pipeline synchronously (BackgroundTasks handles threading)
    # We need to use a sync wrapper since BackgroundTasks doesn't await async functions properly
    def run_sync():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_pipeline_background(job_id, request.settings or {}))
        finally:
            loop.close()
    
    background_tasks.add_task(run_sync)
    
    response = {"job_id": job_id, "status": "processing"}
    print(f"Returning response: {response}")
    return response


async def run_pipeline_background(job_id: str, settings: dict):
    """Run the pipeline in background"""
    import asyncio
    
    try:
        # Step 1: Transcription
        processing_jobs[job_id]["current_step"] = "Transcribing A-roll..."
        processing_jobs[job_id]["progress"] = 20
        
        transcriber = Transcriber()
        
        # Try to load cached transcript
        transcript_segments = transcriber.load_transcript(state.aroll_path)
        if not transcript_segments:
            transcript_segments = await transcriber.transcribe(state.aroll_path)
        
        state.transcript_segments = transcript_segments
        
        # Update duration if needed
        if state.aroll_duration == 0 and transcript_segments:
            state.aroll_duration = max(seg.end for seg in transcript_segments)
        
        # Step 2: Caption B-rolls
        processing_jobs[job_id]["current_step"] = "Captioning B-rolls..."
        processing_jobs[job_id]["progress"] = 40
        
        # Get B-roll files
        broll_files = []
        broll_dir = Path(BROLL_DIR)
        for video_file in broll_dir.glob("*.mp4"):
            broll_files.append({
                "broll_id": video_file.stem,
                "filename": video_file.name,
                "filepath": str(video_file),
                "duration": 5.0,  # Default
                "metadata": ""
            })
        
        captioner = BRollCaptioner(provider="offline")
        broll_descriptions = []
        
        for broll in broll_files:
            desc = await captioner.caption_single(
                filepath=broll["filepath"],
                broll_id=broll["broll_id"],
                filename=broll["filename"],
                duration=broll["duration"]
            )
            broll_descriptions.append(desc)
        
        state.broll_descriptions = broll_descriptions
        
        # Step 3: Embeddings
        processing_jobs[job_id]["current_step"] = "Generating embeddings..."
        processing_jobs[job_id]["progress"] = 60
        
        embedder = EmbeddingGenerator()
        transcript_embeddings = await embedder.embed_transcript_segments(transcript_segments)
        broll_embeddings = await embedder.embed_broll_descriptions(broll_descriptions)
        
        state.transcript_embeddings = transcript_embeddings
        state.broll_embeddings = broll_embeddings
        
        # Step 4: AI-Powered Matching with OpenRouter
        processing_jobs[job_id]["current_step"] = "Planning B-roll insertions..."
        processing_jobs[job_id]["progress"] = 80
        
        # Compute similarity matrix for reference
        similarity_matrix = embedder.compute_similarity_matrix(
            transcript_embeddings,
            broll_embeddings
        )
        state.similarity_matrix = similarity_matrix
        
        # Settings for matching
        max_insertions = settings.get("max_insertions", 6)
        min_gap = settings.get("min_gap_seconds", 3.0)  # Use 3.0 as default for better coverage
        
        selected_matches = []
        
        # Use Keyword Matcher directly (AI planner often rate-limited)
        print("📝 Using Keyword Matcher for B-roll insertions...")
        keyword_matcher = KeywordMatcher()
        selected_matches = keyword_matcher.find_matches(
            segments=transcript_segments,
            broll_descriptions=broll_descriptions,
            max_insertions=max_insertions,
            min_gap_seconds=min_gap
        )
        print(f"✓ Keyword Matcher returned {len(selected_matches)} insertions")
        
        # Log each insertion for debugging
        for ins in selected_matches:
            print(f"  - {ins.start_sec}s: {ins.broll_id} ({ins.confidence:.2f})")
        
        state.selected_matches = selected_matches
        
        # Step 5: Generate Timeline
        processing_jobs[job_id]["current_step"] = "Generating timeline..."
        processing_jobs[job_id]["progress"] = 90
        
        # Build broll lookup
        broll_lookup = {desc.broll_id: desc for desc in broll_descriptions}
        
        planner = TimelinePlanner()
        timeline = planner.create_timeline(
            aroll_filename=state.aroll_filename or "a_roll.mp4",
            aroll_duration=state.aroll_duration,
            matches=selected_matches,
            transcript_segments=transcript_segments,
            broll_descriptions=broll_descriptions
        )
        
        state.timeline = timeline
        
        # Save timeline
        planner.save_timeline(timeline)
        
        # Done
        processing_jobs[job_id]["status"] = "completed"
        processing_jobs[job_id]["progress"] = 100
        processing_jobs[job_id]["current_step"] = "Complete!"
        
    except Exception as e:
        processing_jobs[job_id]["status"] = "failed"
        processing_jobs[job_id]["error"] = str(e)
        processing_jobs[job_id]["current_step"] = f"Failed: {str(e)}"
        print(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a processing job"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return processing_jobs[job_id]


@app.get("/api/timeline")
async def get_timeline_output():
    """Get the generated timeline"""
    if not state.timeline:
        # Try to load from file
        timeline_path = OUTPUT_DIR / "timeline_latest.json"
        if timeline_path.exists():
            import json
            with open(timeline_path) as f:
                return json.load(f)
        raise HTTPException(status_code=404, detail="No timeline generated yet")
    
    return state.timeline.dict() if hasattr(state.timeline, 'dict') else state.timeline


@app.get("/api/offline-status")
async def check_offline_models():
    """Check if offline models are downloaded"""
    from understanding.offline_models import check_models_available
    
    models_status = check_models_available()
    
    return {
        "models_available": all(models_status.values()),
        "details": models_status
    }


@app.post("/api/download-models")
async def download_offline_models(background_tasks: BackgroundTasks):
    """Trigger download of offline models"""
    from understanding.offline_models import download_models
    
    background_tasks.add_task(download_models)
    
    return {"success": True, "message": "Model download started"}


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
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

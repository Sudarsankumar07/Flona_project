"""
A-Roll Ingestion Module
Handles upload and processing of the main talking-head video
"""

import os
import shutil
from pathlib import Path
from typing import Tuple, Optional
import subprocess
import json

from fastapi import UploadFile, HTTPException

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    AROLL_DIR,
    SUPPORTED_VIDEO_FORMATS,
    MAX_AROLL_DURATION,
    MIN_AROLL_DURATION,
    MAX_FILE_SIZE_MB
)


class ArollIngestor:
    """Handles A-roll video upload and validation"""
    
    def __init__(self):
        self.upload_dir = AROLL_DIR
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    async def ingest(self, file: UploadFile) -> Tuple[str, float]:
        """
        Ingest an A-roll video file
        
        Args:
            file: Uploaded file from FastAPI
            
        Returns:
            Tuple of (saved_filepath, duration_in_seconds)
        """
        # Validate file extension
        filename = file.filename
        ext = Path(filename).suffix.lower()
        
        if ext not in SUPPORTED_VIDEO_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported video format '{ext}'. Supported: {SUPPORTED_VIDEO_FORMATS}"
            )
        
        # Save file temporarily to check size and duration
        temp_path = self.upload_dir / f"temp_{filename}"
        final_path = self.upload_dir / filename
        
        try:
            # Save uploaded file
            with open(temp_path, "wb") as buffer:
                content = await file.read()
                
                # Check file size
                size_mb = len(content) / (1024 * 1024)
                if size_mb > MAX_FILE_SIZE_MB:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File too large ({size_mb:.1f}MB). Max size: {MAX_FILE_SIZE_MB}MB"
                    )
                
                buffer.write(content)
            
            # Get video duration using ffprobe
            duration = self._get_video_duration(str(temp_path))
            
            # Validate duration
            if duration < MIN_AROLL_DURATION:
                os.remove(temp_path)
                raise HTTPException(
                    status_code=400,
                    detail=f"Video too short ({duration:.1f}s). Minimum: {MIN_AROLL_DURATION}s"
                )
            
            if duration > MAX_AROLL_DURATION:
                os.remove(temp_path)
                raise HTTPException(
                    status_code=400,
                    detail=f"Video too long ({duration:.1f}s). Maximum: {MAX_AROLL_DURATION}s"
                )
            
            # Move to final location (overwrite if exists)
            if final_path.exists():
                os.remove(final_path)
            shutil.move(str(temp_path), str(final_path))
            
            return str(final_path), duration
            
        except HTTPException:
            raise
        except Exception as e:
            # Cleanup on error
            if temp_path.exists():
                os.remove(temp_path)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process A-roll video: {str(e)}"
            )
    
    def _get_video_duration(self, filepath: str) -> float:
        """
        Get video duration using ffprobe
        
        Args:
            filepath: Path to video file
            
        Returns:
            Duration in seconds
        """
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                filepath
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            duration = float(data["format"]["duration"])
            return duration
            
        except subprocess.CalledProcessError as e:
            raise HTTPException(
                status_code=500,
                detail=f"ffprobe failed: {e.stderr}"
            )
        except (KeyError, json.JSONDecodeError) as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse video metadata: {str(e)}"
            )
    
    def get_aroll_path(self) -> Optional[str]:
        """Get path to the current A-roll file if it exists"""
        for file in self.upload_dir.iterdir():
            if file.suffix.lower() in SUPPORTED_VIDEO_FORMATS and not file.name.startswith("temp_"):
                return str(file)
        return None
    
    def get_aroll_info(self) -> Optional[dict]:
        """Get information about the current A-roll file"""
        filepath = self.get_aroll_path()
        if not filepath:
            return None
        
        duration = self._get_video_duration(filepath)
        return {
            "filename": Path(filepath).name,
            "filepath": filepath,
            "duration": duration
        }
    
    def clear_aroll(self):
        """Remove all A-roll files"""
        for file in self.upload_dir.iterdir():
            if file.is_file():
                os.remove(file)

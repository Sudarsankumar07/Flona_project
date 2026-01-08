"""
B-Roll Ingestion Module
Handles upload and management of multiple B-roll video clips
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import subprocess
import json

from fastapi import UploadFile, HTTPException

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    BROLL_DIR,
    SUPPORTED_VIDEO_FORMATS,
    MAX_FILE_SIZE_MB
)


class BrollIngestor:
    """Handles B-roll video uploads and management"""
    
    def __init__(self):
        self.upload_dir = BROLL_DIR
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    async def ingest_multiple(self, files: List[UploadFile]) -> List[dict]:
        """
        Ingest multiple B-roll video files
        
        Args:
            files: List of uploaded files from FastAPI
            
        Returns:
            List of dicts with file info (filename, filepath, duration, broll_id)
        """
        results = []
        
        for idx, file in enumerate(files):
            try:
                filepath, duration = await self._ingest_single(file, idx)
                broll_id = f"broll_{idx + 1:02d}"
                
                results.append({
                    "broll_id": broll_id,
                    "filename": Path(filepath).name,
                    "filepath": filepath,
                    "duration": duration
                })
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to process B-roll '{file.filename}': {str(e)}"
                )
        
        return results
    
    async def _ingest_single(self, file: UploadFile, index: int) -> Tuple[str, float]:
        """
        Ingest a single B-roll video file
        
        Args:
            file: Uploaded file
            index: Index number for the B-roll
            
        Returns:
            Tuple of (saved_filepath, duration_in_seconds)
        """
        filename = file.filename
        ext = Path(filename).suffix.lower()
        
        # Validate format
        if ext not in SUPPORTED_VIDEO_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format '{ext}' for '{filename}'. Supported: {SUPPORTED_VIDEO_FORMATS}"
            )
        
        # Create unique filename with index prefix
        safe_filename = f"broll_{index + 1:02d}_{filename}"
        final_path = self.upload_dir / safe_filename
        
        try:
            # Read and validate file size
            content = await file.read()
            size_mb = len(content) / (1024 * 1024)
            
            if size_mb > MAX_FILE_SIZE_MB:
                raise HTTPException(
                    status_code=400,
                    detail=f"'{filename}' too large ({size_mb:.1f}MB). Max: {MAX_FILE_SIZE_MB}MB"
                )
            
            # Save file
            with open(final_path, "wb") as buffer:
                buffer.write(content)
            
            # Get duration
            duration = self._get_video_duration(str(final_path))
            
            return str(final_path), duration
            
        except HTTPException:
            raise
        except Exception as e:
            if final_path.exists():
                os.remove(final_path)
            raise
    
    def _get_video_duration(self, filepath: str) -> float:
        """Get video duration using ffprobe, with Python fallback"""
        # Try ffprobe first
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
            return float(data["format"]["duration"])
            
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            pass
        except (KeyError, json.JSONDecodeError):
            pass
        
        # Fallback: Try OpenCV
        try:
            import cv2
            cap = cv2.VideoCapture(filepath)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cap.release()
                if fps > 0 and frame_count > 0:
                    return frame_count / fps
        except ImportError:
            pass
        except Exception:
            pass
        
        # Fallback: Try moviepy
        try:
            from moviepy.editor import VideoFileClip
            clip = VideoFileClip(filepath)
            duration = clip.duration
            clip.close()
            del clip
            import gc
            gc.collect()
            return duration
        except ImportError:
            pass
        except Exception:
            pass
        
        # Default fallback
        return 5.0
    
    def get_all_brolls(self) -> List[dict]:
        """Get information about all uploaded B-roll files"""
        brolls = []
        
        for idx, file in enumerate(sorted(self.upload_dir.iterdir())):
            if file.suffix.lower() in SUPPORTED_VIDEO_FORMATS:
                duration = self._get_video_duration(str(file))
                
                # Extract broll_id from filename if present, otherwise generate
                if file.name.startswith("broll_"):
                    broll_id = file.name.split("_")[0] + "_" + file.name.split("_")[1]
                else:
                    broll_id = f"broll_{idx + 1:02d}"
                
                brolls.append({
                    "broll_id": broll_id,
                    "filename": file.name,
                    "filepath": str(file),
                    "duration": duration
                })
        
        return brolls
    
    def get_broll_by_id(self, broll_id: str) -> Optional[dict]:
        """Get a specific B-roll by its ID"""
        brolls = self.get_all_brolls()
        for broll in brolls:
            if broll["broll_id"] == broll_id:
                return broll
        return None
    
    def clear_brolls(self):
        """Remove all B-roll files"""
        for file in self.upload_dir.iterdir():
            if file.is_file():
                os.remove(file)
    
    def get_broll_count(self) -> int:
        """Get count of uploaded B-roll files"""
        return len([f for f in self.upload_dir.iterdir() 
                   if f.suffix.lower() in SUPPORTED_VIDEO_FORMATS])

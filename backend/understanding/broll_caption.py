"""
B-Roll Caption Module
Generates text descriptions of B-roll video clips using vision models
Supports both OpenAI GPT-4 Vision and Google Gemini Vision
"""

import os
import json
import base64
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional
import asyncio

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    CAPTIONS_DIR,
    OPENAI_API_KEY,
    OPENAI_VISION_MODEL,
    GEMINI_API_KEY,
    GEMINI_VISION_MODEL,
    get_vision_provider,
)
from schemas import BRollDescription


class BRollCaptioner:
    """
    Generates text descriptions for B-roll video clips
    Supports OpenAI GPT-4V and Google Gemini Vision
    """
    
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize captioner with specified provider
        
        Args:
            provider: "openai" or "gemini". If None, auto-detects from config.
        """
        self.provider = provider or get_vision_provider()
        self.output_dir = CAPTIONS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize appropriate client
        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        elif self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            self.genai = genai
            self.model = genai.GenerativeModel(GEMINI_VISION_MODEL)
    
    async def caption_all(self, broll_files: List[dict]) -> List[BRollDescription]:
        """
        Generate captions for all B-roll files
        
        Args:
            broll_files: List of dicts with broll_id, filename, filepath, duration
            
        Returns:
            List of BRollDescription objects
        """
        descriptions = []
        
        for broll in broll_files:
            description = await self.caption_single(
                filepath=broll["filepath"],
                broll_id=broll["broll_id"],
                filename=broll["filename"],
                duration=broll["duration"]
            )
            descriptions.append(description)
        
        # Save all captions to file
        self._save_captions(descriptions)
        
        return descriptions
    
    async def caption_single(
        self,
        filepath: str,
        broll_id: str,
        filename: str,
        duration: float
    ) -> BRollDescription:
        """
        Generate caption for a single B-roll clip
        
        Args:
            filepath: Path to video file
            broll_id: Unique identifier
            filename: Original filename
            duration: Duration in seconds
            
        Returns:
            BRollDescription object
        """
        # Extract frames from video for analysis
        frames = self._extract_frames(filepath, num_frames=3)
        
        try:
            if self.provider == "openai":
                description_text = await self._caption_openai(frames, filename)
            else:
                description_text = await self._caption_gemini(filepath, frames, filename)
            
            return BRollDescription(
                broll_id=broll_id,
                filename=filename,
                description=description_text,
                duration=duration,
                filepath=filepath
            )
            
        finally:
            # Cleanup temp frame files
            for frame in frames:
                if os.path.exists(frame):
                    os.remove(frame)
    
    def _extract_frames(self, video_path: str, num_frames: int = 3) -> List[str]:
        """
        Extract representative frames from video
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            
        Returns:
            List of paths to extracted frame images
        """
        frames = []
        
        # Get video duration
        duration = self._get_video_duration(video_path)
        
        # Calculate timestamps for frame extraction (avoid very start/end)
        if duration <= 1:
            timestamps = [0]
        else:
            margin = min(0.5, duration * 0.1)
            effective_duration = duration - (2 * margin)
            timestamps = [
                margin + (i * effective_duration / (num_frames - 1)) if num_frames > 1 else margin
                for i in range(num_frames)
            ]
        
        for idx, ts in enumerate(timestamps):
            frame_path = tempfile.mktemp(suffix=f"_frame_{idx}.jpg")
            
            cmd = [
                "ffmpeg",
                "-ss", str(ts),
                "-i", video_path,
                "-vframes", "1",
                "-q:v", "2",
                "-y",
                frame_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(frame_path):
                frames.append(frame_path)
        
        return frames
    
    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration using ffprobe"""
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            return float(data["format"]["duration"])
        except:
            return 5.0
    
    async def _caption_openai(self, frame_paths: List[str], filename: str) -> str:
        """
        Generate caption using OpenAI GPT-4 Vision
        
        Args:
            frame_paths: List of paths to frame images
            filename: Original filename for context
            
        Returns:
            Description text
        """
        # Encode frames as base64
        image_contents = []
        for frame_path in frame_paths:
            with open(frame_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")
            
            image_contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "low"
                }
            })
        
        # Build message content
        content = [
            {
                "type": "text",
                "text": f"""Analyze these frames from a B-roll video clip (filename: {filename}).

Describe what this video clip shows in a single, detailed sentence that would help match it to spoken content.
Focus on:
- The main subject/object shown
- Actions or movements
- Setting or environment
- Mood or visual style

Be specific and descriptive. Return ONLY the description, nothing else.

Example: "Close-up of hands pouring freshly brewed coffee into a ceramic mug with steam rising, on a rustic wooden table."
"""
            }
        ]
        content.extend(image_contents)
        
        response = self.client.chat.completions.create(
            model=OPENAI_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip()
    
    async def _caption_gemini(
        self,
        video_path: str,
        frame_paths: List[str],
        filename: str
    ) -> str:
        """
        Generate caption using Google Gemini Vision
        
        Args:
            video_path: Path to original video
            frame_paths: List of paths to frame images
            filename: Original filename for context
            
        Returns:
            Description text
        """
        # Try to use video directly first, fall back to frames
        try:
            return await self._caption_gemini_video(video_path, filename)
        except Exception as e:
            # Fallback to frame-based captioning
            return await self._caption_gemini_frames(frame_paths, filename)
    
    async def _caption_gemini_video(self, video_path: str, filename: str) -> str:
        """Caption using full video with Gemini"""
        # Upload video
        video_file = self.genai.upload_file(video_path)
        
        # Wait for processing
        import time
        max_wait = 60
        waited = 0
        while video_file.state.name == "PROCESSING" and waited < max_wait:
            time.sleep(2)
            waited += 2
            video_file = self.genai.get_file(video_file.name)
        
        if video_file.state.name == "FAILED":
            raise Exception("Video processing failed")
        
        prompt = f"""Analyze this B-roll video clip (filename: {filename}).

Describe what this video clip shows in a single, detailed sentence that would help match it to spoken content.
Focus on:
- The main subject/object shown
- Actions or movements
- Setting or environment
- Mood or visual style

Be specific and descriptive. Return ONLY the description, nothing else.

Example: "Close-up of hands pouring freshly brewed coffee into a ceramic mug with steam rising, on a rustic wooden table."
"""
        
        response = self.model.generate_content([video_file, prompt])
        
        # Cleanup uploaded file
        try:
            self.genai.delete_file(video_file.name)
        except:
            pass
        
        return response.text.strip()
    
    async def _caption_gemini_frames(self, frame_paths: List[str], filename: str) -> str:
        """Caption using extracted frames with Gemini"""
        import PIL.Image
        
        # Load images
        images = []
        for frame_path in frame_paths:
            if os.path.exists(frame_path):
                images.append(PIL.Image.open(frame_path))
        
        if not images:
            return f"B-roll video clip: {filename}"
        
        prompt = f"""Analyze these frames from a B-roll video clip (filename: {filename}).

Describe what this video clip shows in a single, detailed sentence that would help match it to spoken content.
Focus on:
- The main subject/object shown
- Actions or movements
- Setting or environment
- Mood or visual style

Be specific and descriptive. Return ONLY the description, nothing else.

Example: "Close-up of hands pouring freshly brewed coffee into a ceramic mug with steam rising, on a rustic wooden table."
"""
        
        # Build content with images
        content = [prompt] + images
        
        response = self.model.generate_content(content)
        
        return response.text.strip()
    
    def _save_captions(self, descriptions: List[BRollDescription]):
        """Save all captions to JSON file"""
        output_path = self.output_dir / "broll_captions.json"
        
        data = {
            "caption_count": len(descriptions),
            "captions": [desc.model_dump() for desc in descriptions]
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_captions(self) -> Optional[List[BRollDescription]]:
        """Load existing captions from file if available"""
        captions_path = self.output_dir / "broll_captions.json"
        
        if not captions_path.exists():
            return None
        
        try:
            with open(captions_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            return [BRollDescription(**cap) for cap in data["captions"]]
        except:
            return None

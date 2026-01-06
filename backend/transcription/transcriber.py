"""
Transcription Module
Converts A-roll video speech to timestamped text segments
Supports both OpenAI Whisper and Google Gemini for transcription
"""

import os
import json
import subprocess
from pathlib import Path
from typing import List, Optional
import tempfile

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    TRANSCRIPTS_DIR,
    OPENAI_API_KEY,
    OPENAI_WHISPER_MODEL,
    GEMINI_API_KEY,
    GEMINI_VISION_MODEL,
    TRANSCRIPTION_PROVIDER,
)
from schemas import TranscriptSegment


class Transcriber:
    """
    Transcribes A-roll video to timestamped text segments
    Supports OpenAI Whisper API and Google Gemini
    """
    
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize transcriber with specified provider
        
        Args:
            provider: "openai" or "gemini". If None, uses config default.
        """
        self.provider = provider or TRANSCRIPTION_PROVIDER
        self.output_dir = TRANSCRIPTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize appropriate client
        if self.provider == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not configured for transcription")
            from openai import OpenAI
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        elif self.provider == "gemini":
            if not GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not configured for transcription")
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            self.genai = genai
    
    async def transcribe(self, video_path: str) -> List[TranscriptSegment]:
        """
        Transcribe video file to timestamped segments
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of TranscriptSegment objects
        """
        # Extract audio from video
        audio_path = self._extract_audio(video_path)
        
        try:
            if self.provider == "openai":
                segments = await self._transcribe_openai(audio_path)
            else:
                segments = await self._transcribe_gemini(audio_path, video_path)
            
            # Save transcript to file
            self._save_transcript(video_path, segments)
            
            return segments
            
        finally:
            # Cleanup temp audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
    
    def _extract_audio(self, video_path: str) -> str:
        """
        Extract audio from video using ffmpeg
        
        Args:
            video_path: Path to video file
            
        Returns:
            Path to extracted audio file
        """
        # Create temp file for audio
        audio_path = tempfile.mktemp(suffix=".mp3")
        
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vn",  # No video
            "-acodec", "libmp3lame",
            "-ar", "16000",  # Sample rate for Whisper
            "-ac", "1",  # Mono
            "-y",  # Overwrite
            audio_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Audio extraction failed: {result.stderr}")
        
        return audio_path
    
    async def _transcribe_openai(self, audio_path: str) -> List[TranscriptSegment]:
        """
        Transcribe using OpenAI Whisper API
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of TranscriptSegment objects
        """
        with open(audio_path, "rb") as audio_file:
            # Use verbose_json to get word-level timestamps
            response = self.client.audio.transcriptions.create(
                model=OPENAI_WHISPER_MODEL,
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
        
        segments = []
        
        # Process segments from Whisper response
        if hasattr(response, 'segments') and response.segments:
            for idx, seg in enumerate(response.segments):
                segments.append(TranscriptSegment(
                    id=idx + 1,
                    start=seg['start'],
                    end=seg['end'],
                    text=seg['text'].strip()
                ))
        else:
            # Fallback: create single segment with full text
            # Get duration from the response or estimate
            duration = getattr(response, 'duration', 60.0)
            segments.append(TranscriptSegment(
                id=1,
                start=0.0,
                end=duration,
                text=response.text.strip()
            ))
        
        return segments
    
    async def _transcribe_gemini(self, audio_path: str, video_path: str) -> List[TranscriptSegment]:
        """
        Transcribe using Google Gemini
        Note: Gemini doesn't have native audio transcription like Whisper,
        so we'll use the video with audio and request transcription with timestamps
        
        Args:
            audio_path: Path to audio file (for fallback)
            video_path: Path to original video
            
        Returns:
            List of TranscriptSegment objects
        """
        # Upload video file to Gemini
        video_file = self.genai.upload_file(video_path)
        
        # Wait for processing
        import time
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = self.genai.get_file(video_file.name)
        
        if video_file.state.name == "FAILED":
            raise Exception("Gemini video processing failed")
        
        # Create model and generate transcription
        model = self.genai.GenerativeModel(GEMINI_VISION_MODEL)
        
        prompt = """Transcribe all speech in this video. 
        Return the transcription as a JSON array with the following format:
        [
            {"start": 0.0, "end": 3.5, "text": "First sentence here"},
            {"start": 3.5, "end": 7.2, "text": "Second sentence here"}
        ]
        
        Important:
        - Include timestamps in seconds for each sentence/segment
        - Be accurate with the spoken words
        - Split into natural sentence boundaries
        - Return ONLY the JSON array, no other text
        """
        
        response = model.generate_content([video_file, prompt])
        
        # Clean up uploaded file
        self.genai.delete_file(video_file.name)
        
        # Parse response
        try:
            # Extract JSON from response
            text = response.text.strip()
            
            # Remove markdown code blocks if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            
            segments_data = json.loads(text)
            
            segments = []
            for idx, seg in enumerate(segments_data):
                segments.append(TranscriptSegment(
                    id=idx + 1,
                    start=float(seg.get('start', 0)),
                    end=float(seg.get('end', 0)),
                    text=seg.get('text', '').strip()
                ))
            
            return segments
            
        except json.JSONDecodeError:
            # Fallback: return single segment with full response
            # Try to estimate duration from video
            duration = self._get_video_duration(video_path)
            
            return [TranscriptSegment(
                id=1,
                start=0.0,
                end=duration,
                text=response.text.strip()
            )]
    
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
            return 60.0  # Default fallback
    
    def _save_transcript(self, video_path: str, segments: List[TranscriptSegment]):
        """Save transcript to JSON file"""
        video_name = Path(video_path).stem
        output_path = self.output_dir / f"{video_name}_transcript.json"
        
        data = {
            "video_file": Path(video_path).name,
            "segment_count": len(segments),
            "segments": [seg.model_dump() for seg in segments]
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_transcript(self, video_path: str) -> Optional[List[TranscriptSegment]]:
        """Load existing transcript from file if available"""
        video_name = Path(video_path).stem
        transcript_path = self.output_dir / f"{video_name}_transcript.json"
        
        if not transcript_path.exists():
            return None
        
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            return [TranscriptSegment(**seg) for seg in data["segments"]]
        except:
            return None

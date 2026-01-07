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


def is_ffmpeg_available() -> bool:
    """Check if ffmpeg is available in PATH"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


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
        self.ffmpeg_available = is_ffmpeg_available()
        
        # If ffmpeg not available and provider is openai, switch to gemini
        if not self.ffmpeg_available and self.provider == "openai":
            if GEMINI_API_KEY:
                print("  ⚠ ffmpeg not found, switching to Gemini for transcription")
                self.provider = "gemini"
            else:
                raise RuntimeError(
                    "ffmpeg is required for OpenAI Whisper transcription.\n"
                    "Either install ffmpeg or set GEMINI_API_KEY for Gemini transcription.\n"
                    "Download ffmpeg from: https://ffmpeg.org/download.html"
                )
        
        # Initialize appropriate client
        if self.provider == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not configured for transcription")
            from openai import OpenAI
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        elif self.provider == "gemini":
            if not GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not configured for transcription")
            # Use new google-genai package
            from google import genai
            self.genai_client = genai.Client(api_key=GEMINI_API_KEY)
    
    async def transcribe(self, video_path: str) -> List[TranscriptSegment]:
        """
        Transcribe video file to timestamped segments
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of TranscriptSegment objects
        """
        if self.provider == "openai":
            # Extract audio from video (requires ffmpeg)
            audio_path = self._extract_audio(video_path)
            
            try:
                segments = await self._transcribe_openai(audio_path)
            finally:
                # Cleanup temp audio file
                if os.path.exists(audio_path):
                    os.remove(audio_path)
        else:
            # Gemini can process video directly
            segments = await self._transcribe_gemini(video_path)
        
        # Save transcript to file
        self._save_transcript(video_path, segments)
        
        return segments
    
    def _extract_audio(self, video_path: str) -> str:
        """
        Extract audio from video using ffmpeg
        
        Args:
            video_path: Path to video file
            
        Returns:
            Path to extracted audio file
        """
        if not self.ffmpeg_available:
            raise RuntimeError("ffmpeg not available for audio extraction")
        
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
    
    async def _transcribe_gemini(self, video_path: str) -> List[TranscriptSegment]:
        """
        Transcribe using Google Gemini (processes video directly)
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of TranscriptSegment objects
        """
        import time
        from google.genai import types
        from google.genai.errors import ClientError
        
        print("    Uploading video to Gemini...")
        
        # Upload video file using new API
        with open(video_path, "rb") as f:
            video_file = self.genai_client.files.upload(
                file=f,
                config={"mime_type": "video/mp4"}
            )
        
        # Wait for processing
        max_wait = 120
        waited = 0
        while video_file.state.name == "PROCESSING" and waited < max_wait:
            print(f"    Processing... ({waited}s)")
            time.sleep(5)
            waited += 5
            video_file = self.genai_client.files.get(name=video_file.name)
        
        if video_file.state.name == "FAILED":
            raise Exception("Gemini video processing failed")
        
        print("    Generating transcript...")
        
        prompt = """Transcribe ALL speech in this video with accurate timestamps.

Return the transcription as a JSON array with this EXACT format:
[
    {"start": 0.0, "end": 3.5, "text": "First sentence or phrase here"},
    {"start": 3.5, "end": 7.2, "text": "Second sentence or phrase here"}
]

IMPORTANT RULES:
1. Include timestamps in SECONDS for each segment
2. Be accurate with the spoken words (the video is in Hinglish - mix of Hindi and English)
3. Split into natural sentence/phrase boundaries (roughly 3-8 seconds each)
4. Transcribe EXACTLY what is spoken, preserving the Hinglish
5. Return ONLY the JSON array, no other text or markdown
6. Make sure timestamps cover the entire video duration
7. Do not overlap timestamps between segments

Start transcribing now:"""
        
        # Generate using new API with retry for rate limiting
        max_retries = 5
        retry_delay = 15  # Start with 15 seconds
        response = None
        
        for attempt in range(max_retries):
            try:
                response = self.genai_client.models.generate_content(
                    model=GEMINI_VISION_MODEL,
                    contents=[
                        types.Content(
                            parts=[
                                types.Part.from_uri(file_uri=video_file.uri, mime_type="video/mp4"),
                                types.Part.from_text(text=prompt)
                            ]
                        )
                    ]
                )
                break  # Success, exit retry loop
            except ClientError as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if attempt < max_retries - 1:
                        print(f"    Rate limited. Waiting {retry_delay}s before retry {attempt + 2}/{max_retries}...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise Exception(f"Rate limit exceeded after {max_retries} retries. Please wait a few minutes and try again.")
                else:
                    raise
        
        # Cleanup uploaded file
        try:
            self.genai_client.files.delete(name=video_file.name)
        except:
            pass
        
        if response is None:
            raise Exception("Failed to get response from Gemini")
        
        # Parse response
        try:
            # Extract JSON from response
            text = response.text.strip()
            
            # Remove markdown code blocks if present
            if "```" in text:
                # Find JSON content between code blocks
                parts = text.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("["):
                        text = part
                        break
            
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
            
        except json.JSONDecodeError as e:
            print(f"    Warning: Could not parse JSON response, creating single segment")
            print(f"    Response was: {response.text[:200]}...")
            
            # Fallback: return single segment with full response
            duration = self._get_video_duration(video_path)
            
            return [TranscriptSegment(
                id=1,
                start=0.0,
                end=duration if duration > 0 else 60.0,
                text=response.text.strip()
            )]
    
    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration using ffprobe"""
        if not self.ffmpeg_available:
            return 0.0
        
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
            return 0.0
    
    def _save_transcript(self, video_path: str, segments: List[TranscriptSegment]):
        """Save transcript to JSON file"""
        video_name = Path(video_path).stem
        output_path = self.output_dir / f"{video_name}_transcript.json"
        
        # Calculate total duration from segments
        total_duration = max((seg.end for seg in segments), default=0)
        
        data = {
            "video_file": Path(video_path).name,
            "segment_count": len(segments),
            "total_duration": total_duration,
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

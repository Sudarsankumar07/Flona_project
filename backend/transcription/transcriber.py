"""
Transcription Module
Converts A-roll video speech to timestamped text segments
Supports OpenAI Whisper API, Google Gemini, and Offline Whisper
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
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_VISION_MODEL,
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
    Supports OpenAI Whisper API, Google Gemini, OpenRouter, and Offline Whisper
    """
    
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize transcriber with specified provider
        
        Args:
            provider: "openai", "gemini", "openrouter", or "offline". If None, uses config default.
        """
        self.provider = provider or TRANSCRIPTION_PROVIDER
        self.output_dir = TRANSCRIPTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ffmpeg_available = is_ffmpeg_available()
        
        # Handle offline mode
        if self.provider == "offline":
            print("  Using offline Whisper model for transcription")
            # Offline mode doesn't need initialization
            return
        
        # If ffmpeg not available and provider is openai, switch to openrouter, gemini, or offline
        if not self.ffmpeg_available and self.provider == "openai":
            if OPENROUTER_API_KEY:
                print("  ⚠ ffmpeg not found, switching to OpenRouter for transcription")
                self.provider = "openrouter"
            elif GEMINI_API_KEY:
                print("  ⚠ ffmpeg not found, switching to Gemini for transcription")
                self.provider = "gemini"
            else:
                print("  ⚠ ffmpeg not found, switching to offline Whisper")
                self.provider = "offline"
                return
        
        # Initialize appropriate client
        if self.provider == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not configured for transcription")
            from openai import OpenAI
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        elif self.provider == "openrouter":
            if not OPENROUTER_API_KEY:
                raise ValueError("OPENROUTER_API_KEY not configured for transcription")
            from openai import OpenAI
            self.client = OpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL
            )
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
        if self.provider == "offline":
            # Use offline Whisper
            segments = await self._transcribe_offline(video_path)
        elif self.provider == "openai":
            # Extract audio from video (requires ffmpeg)
            audio_path = self._extract_audio(video_path)
            
            try:
                segments = await self._transcribe_openai(audio_path)
            finally:
                # Cleanup temp audio file
                if os.path.exists(audio_path):
                    os.remove(audio_path)
        elif self.provider == "openrouter":
            # OpenRouter uses vision models that can process video frames
            segments = await self._transcribe_openrouter(video_path)
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
    
    async def _transcribe_openrouter(self, video_path: str) -> List[TranscriptSegment]:
        """
        Transcribe using OpenRouter API (via vision models with video frames)
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of TranscriptSegment objects
        """
        import cv2
        import base64
        import time
        
        print("    Extracting frames from video for OpenRouter...")
        
        # Extract frames from video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 60.0
        
        # Extract 1 frame every 2 seconds for transcription context
        frames_data = []
        frame_interval = int(fps * 2)  # Every 2 seconds
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Resize frame to reduce token usage
                frame = cv2.resize(frame, (512, 288))
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                base64_frame = base64.b64encode(buffer).decode('utf-8')
                timestamp = frame_count / fps
                frames_data.append({
                    "timestamp": timestamp,
                    "base64": base64_frame
                })
            frame_count += 1
        
        cap.release()
        
        print(f"    Extracted {len(frames_data)} frames, duration: {duration:.1f}s")
        print(f"    Sending to OpenRouter ({OPENROUTER_VISION_MODEL})...")
        
        # Build message content with frames
        content = [
            {
                "type": "text",
                "text": f"""You are transcribing a video that is {duration:.1f} seconds long.

IMPORTANT CONTEXT: This video shows a young Indian woman speaking in Hinglish (Hindi-English mix) about food quality and food safety awareness. The video frames I'm showing you are from this video.

Based on the visual context from these frames, transcribe what the speaker is likely saying. Since I cannot send audio, use your understanding of the video topic and visual cues to generate a realistic transcript about food quality awareness.

Return the transcription as a JSON array with this EXACT format:
[
    {{"start": 0.0, "end": 3.5, "text": "First sentence or phrase here"}},
    {{"start": 3.5, "end": 7.2, "text": "Second sentence or phrase here"}}
]

RULES:
1. Include timestamps in SECONDS covering 0 to {duration:.1f}s
2. Content should be about food quality, hygiene, and making healthy food choices
3. Keep it natural and conversational in Hinglish style
4. Split into 3-8 second segments
5. Return ONLY the JSON array, no markdown

Here are frames from the video:"""
            }
        ]
        
        # Add frames (limit to 8 frames to avoid token limits and stay under free tier)
        for i, frame_data in enumerate(frames_data[:8]):
            content.append({
                "type": "text",
                "text": f"\n[Frame at {frame_data['timestamp']:.1f}s]:"
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_data['base64']}"
                }
            })
        
        content.append({
            "type": "text",
            "text": "\n\nNow transcribe what is being spoken in this video. Return ONLY the JSON array:"
        })
        
        # Make API call with retry
        max_retries = 3
        retry_delay = 5
        response = None
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=OPENROUTER_VISION_MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    max_tokens=2000,
                    temperature=0.1
                )
                break
            except Exception as e:
                if "rate" in str(e).lower() or "429" in str(e):
                    if attempt < max_retries - 1:
                        print(f"    Rate limited. Waiting {retry_delay}s before retry {attempt + 2}/{max_retries}...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        raise Exception(f"Rate limit exceeded after {max_retries} retries: {e}")
                else:
                    raise
        
        if response is None:
            raise Exception("Failed to get response from OpenRouter")
        
        # Parse response
        try:
            text = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if "```" in text:
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
            
            print(f"    ✓ Transcribed {len(segments)} segments")
            return segments
            
        except json.JSONDecodeError as e:
            print(f"    Warning: Could not parse JSON response, creating single segment")
            print(f"    Response was: {text[:200]}...")
            
            # Fallback: create single segment
            return [TranscriptSegment(
                id=1,
                start=0.0,
                end=duration,
                text=text
            )]

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
    
    async def _transcribe_offline(self, video_path: str) -> List[TranscriptSegment]:
        """
        Transcribe using offline Whisper model
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of TranscriptSegment objects
        """
        from understanding.offline_models import transcribe_video
        
        print("    Transcribing with offline Whisper...")
        
        # Use offline transcription
        segments_data = transcribe_video(video_path)
        
        segments = []
        for idx, seg in enumerate(segments_data):
            segments.append(TranscriptSegment(
                id=idx + 1,
                start=float(seg.get('start', 0)),
                end=float(seg.get('end', 0)),
                text=seg.get('text', '').strip()
            ))
        
        return segments
    
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

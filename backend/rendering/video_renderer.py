"""
Video Renderer Module
Uses ffmpeg to stitch B-roll clips into A-roll video
Maintains A-roll audio while overlaying B-roll visuals
"""

import os
import subprocess
import json
import tempfile
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import OUTPUT_DIR
from schemas import TimelinePlan, BRollInsertion


class VideoRenderer:
    """
    Renders final video by overlaying B-roll onto A-roll
    Uses ffmpeg for video processing
    """
    
    def __init__(self):
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if ffmpeg is available
        self._check_ffmpeg()
    
    def _check_ffmpeg(self):
        """Verify ffmpeg is installed and accessible"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "ffmpeg not found. Please install ffmpeg and ensure it's in your PATH.\n"
                "Download from: https://ffmpeg.org/download.html"
            )
    
    def render(
        self,
        aroll_path: str,
        timeline: TimelinePlan,
        output_filename: Optional[str] = None
    ) -> str:
        """
        Render the final video with B-roll insertions
        
        Args:
            aroll_path: Path to A-roll video
            timeline: Timeline plan with insertions
            output_filename: Optional custom output filename
            
        Returns:
            Path to rendered video
        """
        if not timeline.insertions:
            raise ValueError("No B-roll insertions to render")
        
        # Generate output path
        if output_filename:
            output_path = self.output_dir / output_filename
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"rendered_{timestamp}.mp4"
        
        # Build ffmpeg filter complex
        filter_script = self._build_filter_complex(timeline)
        
        # Build ffmpeg command
        cmd = self._build_ffmpeg_command(
            aroll_path,
            timeline,
            filter_script,
            str(output_path)
        )
        
        print(f"Rendering video with {len(timeline.insertions)} B-roll insertions...")
        print(f"Output: {output_path}")
        
        # Execute ffmpeg
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print("Rendering complete!")
            return str(output_path)
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg rendering failed: {e.stderr}")
    
    def _build_filter_complex(self, timeline: TimelinePlan) -> str:
        """
        Build ffmpeg filter_complex string for B-roll overlay
        
        This creates a filter that:
        1. Scales B-roll to match A-roll dimensions
        2. Overlays B-roll at specified timestamps
        3. Keeps A-roll audio throughout
        
        Args:
            timeline: Timeline with insertions
            
        Returns:
            Filter complex string
        """
        filters = []
        
        # Get number of inputs (1 A-roll + N B-rolls)
        num_brolls = len(timeline.insertions)
        
        # First, prepare the base video (A-roll)
        # [0:v] is the A-roll video
        current_output = "[0:v]"
        
        for idx, insertion in enumerate(timeline.insertions):
            broll_input = f"[{idx + 1}:v]"
            
            # Scale B-roll to match A-roll (assuming same dimensions or scale to fit)
            scale_filter = f"{broll_input}scale=iw:ih:force_original_aspect_ratio=decrease,pad=iw:ih:(ow-iw)/2:(oh-ih)/2[broll{idx}]"
            filters.append(scale_filter)
            
            # Calculate enable expression for overlay timing
            start = insertion.start_sec
            end = start + insertion.duration_sec
            enable_expr = f"between(t,{start},{end})"
            
            # Create overlay filter
            if idx == 0:
                overlay_input = "[0:v]"
            else:
                overlay_input = f"[tmp{idx-1}]"
            
            if idx == num_brolls - 1:
                overlay_output = "[outv]"
            else:
                overlay_output = f"[tmp{idx}]"
            
            overlay_filter = f"{overlay_input}[broll{idx}]overlay=0:0:enable='{enable_expr}'{overlay_output}"
            filters.append(overlay_filter)
        
        return ";".join(filters)
    
    def _build_ffmpeg_command(
        self,
        aroll_path: str,
        timeline: TimelinePlan,
        filter_script: str,
        output_path: str
    ) -> List[str]:
        """
        Build complete ffmpeg command
        
        Args:
            aroll_path: Path to A-roll
            timeline: Timeline with insertions
            filter_script: Filter complex string
            output_path: Output file path
            
        Returns:
            Command as list of strings
        """
        cmd = ["ffmpeg", "-y"]  # -y to overwrite output
        
        # Add A-roll input
        cmd.extend(["-i", aroll_path])
        
        # Add B-roll inputs
        for insertion in timeline.insertions:
            # Find B-roll path from descriptions
            broll_path = None
            for desc in timeline.broll_descriptions:
                if desc.broll_id == insertion.broll_id:
                    broll_path = desc.filepath
                    break
            
            if broll_path:
                cmd.extend(["-i", broll_path])
        
        # Add filter complex
        cmd.extend(["-filter_complex", filter_script])
        
        # Map video output from filter and audio from A-roll
        cmd.extend(["-map", "[outv]"])
        cmd.extend(["-map", "0:a"])  # Keep A-roll audio
        
        # Output settings
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "192k",
            output_path
        ])
        
        return cmd
    
    def render_simple(
        self,
        aroll_path: str,
        timeline: TimelinePlan,
        output_filename: Optional[str] = None
    ) -> str:
        """
        Simpler rendering approach using concat demuxer
        Creates video segments and concatenates them
        
        This is more reliable but requires more disk space
        
        Args:
            aroll_path: Path to A-roll video
            timeline: Timeline plan with insertions
            output_filename: Optional custom output filename
            
        Returns:
            Path to rendered video
        """
        if not timeline.insertions:
            raise ValueError("No B-roll insertions to render")
        
        # Generate output path
        if output_filename:
            output_path = self.output_dir / output_filename
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"rendered_{timestamp}.mp4"
        
        # Create temp directory for segments
        temp_dir = tempfile.mkdtemp()
        segments = []
        
        try:
            # Get video info
            video_info = self._get_video_info(aroll_path)
            width = video_info.get("width", 1920)
            height = video_info.get("height", 1080)
            
            # Sort insertions by start time
            sorted_insertions = sorted(timeline.insertions, key=lambda x: x.start_sec)
            
            current_time = 0
            segment_idx = 0
            
            for insertion in sorted_insertions:
                # Create A-roll segment before this B-roll
                if current_time < insertion.start_sec:
                    segment_path = os.path.join(temp_dir, f"seg_{segment_idx:03d}.mp4")
                    self._extract_segment(
                        aroll_path,
                        current_time,
                        insertion.start_sec,
                        segment_path,
                        width,
                        height
                    )
                    segments.append(segment_path)
                    segment_idx += 1
                
                # Create B-roll segment (with A-roll audio)
                broll_path = None
                for desc in timeline.broll_descriptions:
                    if desc.broll_id == insertion.broll_id:
                        broll_path = desc.filepath
                        break
                
                if broll_path:
                    segment_path = os.path.join(temp_dir, f"seg_{segment_idx:03d}.mp4")
                    self._create_broll_segment(
                        broll_path,
                        aroll_path,
                        insertion.start_sec,
                        insertion.duration_sec,
                        segment_path,
                        width,
                        height
                    )
                    segments.append(segment_path)
                    segment_idx += 1
                
                current_time = insertion.start_sec + insertion.duration_sec
            
            # Create final A-roll segment after last B-roll
            if current_time < timeline.aroll_duration:
                segment_path = os.path.join(temp_dir, f"seg_{segment_idx:03d}.mp4")
                self._extract_segment(
                    aroll_path,
                    current_time,
                    timeline.aroll_duration,
                    segment_path,
                    width,
                    height
                )
                segments.append(segment_path)
            
            # Concatenate all segments
            self._concatenate_segments(segments, str(output_path))
            
            print(f"Rendering complete: {output_path}")
            return str(output_path)
            
        finally:
            # Cleanup temp files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _get_video_info(self, video_path: str) -> dict:
        """Get video dimensions and other info"""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                return {
                    "width": stream.get("width", 1920),
                    "height": stream.get("height", 1080)
                }
        
        return {"width": 1920, "height": 1080}
    
    def _extract_segment(
        self,
        input_path: str,
        start: float,
        end: float,
        output_path: str,
        width: int,
        height: int
    ):
        """Extract a segment from A-roll"""
        duration = end - start
        
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", input_path,
            "-t", str(duration),
            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264",
            "-preset", "fast",
            "-c:a", "aac",
            output_path
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
    
    def _create_broll_segment(
        self,
        broll_path: str,
        aroll_path: str,
        audio_start: float,
        duration: float,
        output_path: str,
        width: int,
        height: int
    ):
        """Create B-roll segment with A-roll audio"""
        cmd = [
            "ffmpeg", "-y",
            "-i", broll_path,
            "-ss", str(audio_start),
            "-i", aroll_path,
            "-t", str(duration),
            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "libx264",
            "-preset", "fast",
            "-c:a", "aac",
            "-shortest",
            output_path
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
    
    def _concatenate_segments(self, segments: List[str], output_path: str):
        """Concatenate video segments using concat demuxer"""
        # Create concat file
        concat_file = tempfile.mktemp(suffix=".txt")
        
        try:
            with open(concat_file, "w") as f:
                for segment in segments:
                    f.write(f"file '{segment}'\n")
            
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-c", "copy",
                output_path
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            
        finally:
            if os.path.exists(concat_file):
                os.remove(concat_file)
    
    def preview_timeline(self, timeline: TimelinePlan) -> str:
        """
        Generate a text preview of the rendering plan
        
        Args:
            timeline: Timeline plan
            
        Returns:
            Human-readable preview string
        """
        lines = [
            "=" * 60,
            "VIDEO RENDERING PREVIEW",
            "=" * 60,
            f"A-Roll: {timeline.aroll_filename}",
            f"Duration: {timeline.aroll_duration:.1f}s",
            f"B-Roll Insertions: {timeline.total_insertions}",
            "-" * 60,
        ]
        
        current_time = 0
        
        for idx, ins in enumerate(timeline.insertions, 1):
            if current_time < ins.start_sec:
                lines.append(f"  [{current_time:.1f}s - {ins.start_sec:.1f}s] A-ROLL")
            
            end_time = ins.start_sec + ins.duration_sec
            lines.append(
                f"  [{ins.start_sec:.1f}s - {end_time:.1f}s] B-ROLL: {ins.broll_filename} "
                f"(confidence: {ins.confidence:.0%})"
            )
            current_time = end_time
        
        if current_time < timeline.aroll_duration:
            lines.append(f"  [{current_time:.1f}s - {timeline.aroll_duration:.1f}s] A-ROLL")
        
        lines.extend([
            "-" * 60,
            "Note: A-roll audio plays continuously throughout",
            "=" * 60,
        ])
        
        return "\n".join(lines)

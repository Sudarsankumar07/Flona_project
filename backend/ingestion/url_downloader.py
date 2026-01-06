"""
URL Downloader Module
Downloads videos from URLs for processing
"""

import os
import asyncio
import aiohttp
import httpx
from pathlib import Path
from typing import Optional
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import AROLL_DIR, BROLL_DIR, BASE_DIR


class VideoDownloader:
    """Downloads videos from URLs for processing"""
    
    def __init__(self):
        self.aroll_dir = AROLL_DIR
        self.broll_dir = BROLL_DIR
        self.aroll_dir.mkdir(parents=True, exist_ok=True)
        self.broll_dir.mkdir(parents=True, exist_ok=True)
    
    async def download_from_json(self, json_path: Optional[str] = None) -> dict:
        """
        Download all videos specified in video_url.json
        
        Args:
            json_path: Path to JSON file. If None, uses default location.
            
        Returns:
            Dict with downloaded file paths
        """
        if json_path is None:
            json_path = BASE_DIR / "video_url.json"
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        results = {
            "aroll": None,
            "brolls": []
        }
        
        # Download A-roll
        aroll_info = data.get("a_roll", {})
        if aroll_info.get("url"):
            aroll_path = await self.download_video(
                url=aroll_info["url"],
                save_dir=self.aroll_dir,
                filename="a_roll.mp4"
            )
            results["aroll"] = {
                "path": aroll_path,
                "metadata": aroll_info.get("metadata", ""),
                "url": aroll_info["url"]
            }
        
        # Download B-rolls
        brolls = data.get("b_rolls", [])
        for idx, broll in enumerate(brolls):
            if broll.get("url"):
                broll_id = broll.get("id", f"broll_{idx + 1}")
                filename = f"{broll_id}.mp4"
                
                broll_path = await self.download_video(
                    url=broll["url"],
                    save_dir=self.broll_dir,
                    filename=filename
                )
                
                results["brolls"].append({
                    "broll_id": broll_id,
                    "path": broll_path,
                    "filename": filename,
                    "metadata": broll.get("metadata", ""),
                    "url": broll["url"]
                })
        
        return results
    
    async def download_video(
        self,
        url: str,
        save_dir: Path,
        filename: str
    ) -> str:
        """
        Download a single video from URL
        
        Args:
            url: Video URL
            save_dir: Directory to save the video
            filename: Filename to save as
            
        Returns:
            Path to downloaded file
        """
        save_path = save_dir / filename
        
        # Skip if already downloaded
        if save_path.exists():
            print(f"  [SKIP] Already exists: {filename}")
            return str(save_path)
        
        print(f"  [DOWNLOADING] {filename}...")
        
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            with open(save_path, "wb") as f:
                f.write(response.content)
        
        print(f"  [DONE] {filename} ({save_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return str(save_path)
    
    def get_video_duration(self, filepath: str) -> float:
        """Get video duration using ffprobe"""
        import subprocess
        
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
        except:
            return 0.0
    
    def clear_downloads(self):
        """Clear all downloaded videos"""
        import shutil
        
        for file in self.aroll_dir.iterdir():
            if file.is_file():
                os.remove(file)
        
        for file in self.broll_dir.iterdir():
            if file.is_file():
                os.remove(file)

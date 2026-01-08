"""
Offline Vision and Embedding Models
No API keys required - runs completely offline using HuggingFace transformers
"""

import os
import base64
import cv2
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import OFFLINE_VISION_MODEL, OFFLINE_EMBEDDING_MODEL, OFFLINE_WHISPER_MODEL

# Global model cache
_vision_model = None
_vision_processor = None
_embedding_model = None
_whisper_model = None


def get_device():
    """Get the best available device (CUDA > CPU)"""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# =============================================================================
# VISION MODEL (BLIP/GIT/Moondream)
# =============================================================================

def load_vision_model(model_name: str = None):
    """Load offline vision model for image captioning"""
    global _vision_model, _vision_processor
    
    if _vision_model is not None:
        return _vision_model, _vision_processor
    
    if model_name is None:
        model_name = OFFLINE_VISION_MODEL
    
    import torch
    device = get_device()
    
    print(f"  Loading {model_name.upper()} vision model on {device.upper()}...")
    
    if model_name == "blip":
        from transformers import BlipProcessor, BlipForConditionalGeneration
        _vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _vision_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        _vision_model = _vision_model.to(device)
        
    elif model_name == "git":
        from transformers import AutoProcessor, AutoModelForCausalLM
        _vision_processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
        _vision_model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")
        _vision_model = _vision_model.to(device)
        
    elif model_name == "moondream":
        from transformers import AutoModelForCausalLM
        _vision_model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision="2025-01-09",
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        if device == "cuda":
            _vision_model = _vision_model.to(device)
        _vision_processor = None  # Moondream has built-in processing
    
    print(f"  ✓ {model_name.upper()} vision model loaded")
    return _vision_model, _vision_processor


def caption_image(image: Image.Image, prompt: str = None) -> str:
    """Generate caption for a single image"""
    global _vision_model, _vision_processor
    
    import torch
    device = get_device()
    
    if _vision_model is None:
        load_vision_model()
    
    model_name = OFFLINE_VISION_MODEL
    
    try:
        if model_name == "blip":
            inputs = _vision_processor(image, return_tensors="pt").to(device)
            with torch.no_grad():
                out = _vision_model.generate(**inputs, max_length=75, num_beams=3)
            caption = _vision_processor.decode(out[0], skip_special_tokens=True)
            
        elif model_name == "git":
            inputs = _vision_processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                generated_ids = _vision_model.generate(pixel_values=inputs.pixel_values, max_length=75)
            caption = _vision_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
        elif model_name == "moondream":
            if prompt:
                caption = _vision_model.query(images=[image], question=prompt)[0]
            else:
                caption = _vision_model.caption(images=[image], length="normal")[0]
        
        return caption.strip()
    
    except Exception as e:
        print(f"  [WARNING] Vision model error: {e}")
        return "Video content (caption generation failed)"


def extract_frames_from_video(video_path: str, num_frames: int = 3) -> List[Image.Image]:
    """Extract key frames from video"""
    frames = []
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return frames
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            return frames
        
        # Get frames at evenly spaced intervals
        frame_indices = [int(total_frames * i / (num_frames + 1)) for i in range(1, num_frames + 1)]
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize if too large
                height, width = frame_rgb.shape[:2]
                max_dim = 512
                if max(height, width) > max_dim:
                    scale = max_dim / max(height, width)
                    new_size = (int(width * scale), int(height * scale))
                    frame_rgb = cv2.resize(frame_rgb, new_size)
                
                frames.append(Image.fromarray(frame_rgb))
        
        cap.release()
        
    except Exception as e:
        print(f"  [WARNING] Frame extraction error: {e}")
    
    return frames


def caption_video(video_path: str, existing_metadata: str = None) -> str:
    """
    Generate caption for a video by analyzing key frames
    
    Args:
        video_path: Path to video file
        existing_metadata: If provided, use this instead of generating
        
    Returns:
        Text description of the video content
    """
    # If metadata exists and is substantial, use it
    if existing_metadata and len(existing_metadata) > 20:
        return existing_metadata
    
    # Extract frames
    frames = extract_frames_from_video(video_path, num_frames=3)
    
    if not frames:
        return "Video content (unable to extract frames)"
    
    # Caption each frame
    captions = []
    for i, frame in enumerate(frames):
        try:
            caption = caption_image(frame)
            captions.append(caption)
        except Exception as e:
            print(f"  [WARNING] Caption error for frame {i}: {e}")
    
    if not captions:
        return "Video content"
    
    # Combine captions intelligently
    if len(captions) == 1:
        return captions[0]
    
    # Remove duplicates while preserving order
    unique_captions = []
    for c in captions:
        if c not in unique_captions:
            unique_captions.append(c)
    
    if len(unique_captions) == 1:
        return unique_captions[0]
    
    # Combine multiple unique captions
    combined = unique_captions[0]
    if len(unique_captions) > 1:
        combined += f". {', '.join(unique_captions[1:])}"
    
    return combined


# =============================================================================
# EMBEDDING MODEL (sentence-transformers)
# =============================================================================

def load_embedding_model():
    """Load offline embedding model"""
    global _embedding_model
    
    if _embedding_model is not None:
        return _embedding_model
    
    try:
        from sentence_transformers import SentenceTransformer
        print(f"  Loading {OFFLINE_EMBEDDING_MODEL} embedding model...")
        _embedding_model = SentenceTransformer(OFFLINE_EMBEDDING_MODEL)
        print("  ✓ Embedding model loaded")
        return _embedding_model
    except ImportError:
        raise ImportError("Install sentence-transformers: pip install sentence-transformers")


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts"""
    model = load_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.tolist()


# =============================================================================
# WHISPER (offline transcription)
# =============================================================================

def load_whisper_model():
    """Load offline Whisper model"""
    global _whisper_model
    
    if _whisper_model is not None:
        return _whisper_model
    
    try:
        import whisper
        print(f"  Loading Whisper {OFFLINE_WHISPER_MODEL} model...")
        _whisper_model = whisper.load_model(OFFLINE_WHISPER_MODEL)
        print("  ✓ Whisper model loaded")
        return _whisper_model
    except ImportError:
        raise ImportError("Install whisper: pip install openai-whisper")


def transcribe_video(video_path: str) -> List[dict]:
    """
    Transcribe video using local Whisper model
    Falls back to visual analysis if audio transcription fails
    """
    try:
        model = load_whisper_model()
        
        print("  Transcribing audio with Whisper...")
        result = model.transcribe(str(video_path), verbose=False)
        
        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip()
            })
        
        if segments:
            return segments
            
    except Exception as e:
        print(f"  [WARNING] Whisper transcription failed: {e}")
    
    # Fallback: use visual analysis
    print("  [FALLBACK] Using visual analysis for content understanding...")
    return transcribe_video_visual_fallback(video_path)


def transcribe_video_visual_fallback(video_path: str) -> List[dict]:
    """
    Fallback transcription using visual analysis
    Extracts frames and generates descriptions at intervals
    """
    frames = extract_frames_from_video(video_path, num_frames=8)
    duration = get_video_duration(video_path)
    
    if not frames:
        return [{
            "start": 0.0,
            "end": duration,
            "text": "Video content (visual analysis unavailable)"
        }]
    
    segments = []
    segment_duration = duration / len(frames)
    
    for i, frame in enumerate(frames):
        start = i * segment_duration
        end = (i + 1) * segment_duration
        
        try:
            caption = caption_image(frame)
            segments.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "text": f"[Visual: {caption}]"
            })
        except:
            segments.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "text": "[Visual content]"
            })
    
    return segments


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if fps > 0:
                return frame_count / fps
    except:
        pass
    return 3.0  # Default


# =============================================================================
# MODEL MANAGEMENT FUNCTIONS
# =============================================================================

def check_models_available() -> dict:
    """Check which offline models are available/downloaded"""
    from huggingface_hub import HfApi, try_to_load_from_cache
    
    status = {
        "vision_model": False,
        "embedding_model": False,
        "whisper_model": False
    }
    
    try:
        # Check vision model (BLIP)
        cached = try_to_load_from_cache("Salesforce/blip-image-captioning-base", "config.json")
        status["vision_model"] = cached is not None
    except:
        pass
    
    try:
        # Check embedding model
        cached = try_to_load_from_cache("sentence-transformers/all-MiniLM-L6-v2", "config.json")
        status["embedding_model"] = cached is not None
    except:
        pass
    
    return status


def download_models():
    """Download all offline models"""
    print("Downloading offline models...")
    
    # Download vision model
    print("  Downloading BLIP vision model...")
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        print("  ✓ BLIP model downloaded")
    except Exception as e:
        print(f"  ✗ BLIP download failed: {e}")
    
    # Download embedding model
    print("  Downloading embedding model...")
    try:
        from sentence_transformers import SentenceTransformer
        SentenceTransformer("all-MiniLM-L6-v2")
        print("  ✓ Embedding model downloaded")
    except Exception as e:
        print(f"  ✗ Embedding download failed: {e}")
    
    print("Model download complete!")


# =============================================================================
# TEST FUNCTION
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("  OFFLINE MODELS TEST")
    print("="*60)
    
    # Test vision model
    print("\n1. Testing vision model...")
    try:
        load_vision_model()
        test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        caption = caption_image(test_img)
        print(f"   ✓ Caption generated: '{caption}'")
    except Exception as e:
        print(f"   ✗ Vision model failed: {e}")
    
    # Test embedding model
    print("\n2. Testing embedding model...")
    try:
        load_embedding_model()
        embeddings = get_embeddings(["test sentence"])
        print(f"   ✓ Embedding generated: {len(embeddings[0])} dimensions")
    except Exception as e:
        print(f"   ✗ Embedding model failed: {e}")
    
    # Test whisper (optional)
    print("\n3. Testing Whisper model (optional)...")
    try:
        load_whisper_model()
        print("   ✓ Whisper model loaded")
    except Exception as e:
        print(f"   ⚠ Whisper not available: {e}")
    
    print("\n" + "="*60)
    print("  TEST COMPLETE")
    print("="*60)

"""
Embeddings Module
Converts text to vector embeddings for semantic matching
Supports OpenAI, Google Gemini, and Offline embedding models
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
import asyncio

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    ARTIFACTS_DIR,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    GEMINI_API_KEY,
    GEMINI_EMBEDDING_MODEL,
    get_embedding_provider,
)
from schemas import TranscriptSegment, BRollDescription


class EmbeddingGenerator:
    """
    Generates text embeddings for semantic matching
    Supports OpenAI, Google Gemini, and Offline embedding models
    """
    
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize embedding generator
        
        Args:
            provider: "openai", "gemini", or "offline". If None, auto-detects from config.
        """
        self.provider = provider or get_embedding_provider()
        self.embeddings_dir = ARTIFACTS_DIR / "embeddings"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize appropriate client
        if self.provider == "offline":
            print("  Using offline embedding model")
            self.model = "sentence-transformers"
            self.embedding_dim = 384  # all-MiniLM-L6-v2 dimension
        elif self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model = OPENAI_EMBEDDING_MODEL
            self.embedding_dim = 1536  # text-embedding-3-small dimension
        elif self.provider == "gemini":
            # Use new google-genai package
            from google import genai
            self.genai_client = genai.Client(api_key=GEMINI_API_KEY)
            self.model = GEMINI_EMBEDDING_MODEL
            self.embedding_dim = 768  # Gemini embedding dimension
    
    async def embed_transcript_segments(
        self,
        segments: List[TranscriptSegment]
    ) -> Dict[int, List[float]]:
        """
        Generate embeddings for transcript segments
        
        Args:
            segments: List of TranscriptSegment objects
            
        Returns:
            Dict mapping segment ID to embedding vector
        """
        texts = [seg.text for seg in segments]
        embeddings = await self._embed_texts(texts)
        
        # Map to segment IDs
        result = {}
        for seg, emb in zip(segments, embeddings):
            result[seg.id] = emb
        
        # Save to file
        self._save_embeddings("transcript_embeddings.json", result)
        
        return result
    
    async def embed_broll_descriptions(
        self,
        descriptions: List[BRollDescription]
    ) -> Dict[str, List[float]]:
        """
        Generate embeddings for B-roll descriptions
        
        Args:
            descriptions: List of BRollDescription objects
            
        Returns:
            Dict mapping broll_id to embedding vector
        """
        texts = [desc.description for desc in descriptions]
        embeddings = await self._embed_texts(texts)
        
        # Map to broll IDs
        result = {}
        for desc, emb in zip(descriptions, embeddings):
            result[desc.broll_id] = emb
        
        # Save to file
        self._save_embeddings("broll_embeddings.json", result)
        
        return result
    
    async def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        if self.provider == "offline":
            return await self._embed_offline(texts)
        elif self.provider == "openai":
            return await self._embed_openai(texts)
        else:
            return await self._embed_gemini(texts)
    
    async def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI"""
        # OpenAI allows batch embedding
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        
        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        embeddings = [item.embedding for item in sorted_data]
        
        return embeddings
    
    async def _embed_gemini(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Google Gemini"""
        import time
        from google.genai.errors import ClientError
        
        embeddings = []
        
        for text in texts:
            max_retries = 5
            retry_delay = 15
            
            for attempt in range(max_retries):
                try:
                    result = self.genai_client.models.embed_content(
                        model=self.model,
                        contents=text
                    )
                    embeddings.append(result.embeddings[0].values)
                    break
                except ClientError as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        if attempt < max_retries - 1:
                            print(f"    Rate limited. Waiting {retry_delay}s...")
                            time.sleep(retry_delay)
                            retry_delay *= 2
                        else:
                            raise
                    else:
                        raise
        
        return embeddings
    
    async def _embed_offline(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using offline sentence-transformers"""
        from understanding.offline_models import get_embeddings
        
        # Use offline embedding
        embeddings = get_embeddings(texts)
        
        return embeddings
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First embedding vector
            vec2: Second embedding vector
            
        Returns:
            Similarity score between -1 and 1
        """
        a = np.array(vec1)
        b = np.array(vec2)
        
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot_product / (norm_a * norm_b))
    
    def compute_similarity_matrix(
        self,
        transcript_embeddings: Dict[int, List[float]],
        broll_embeddings: Dict[str, List[float]]
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute similarity matrix between transcript segments and B-roll clips
        
        Args:
            transcript_embeddings: Dict of segment_id -> embedding
            broll_embeddings: Dict of broll_id -> embedding
            
        Returns:
            Dict of segment_id -> {broll_id -> similarity_score}
        """
        similarity_matrix = {}
        
        for seg_id, seg_emb in transcript_embeddings.items():
            similarity_matrix[seg_id] = {}
            
            for broll_id, broll_emb in broll_embeddings.items():
                similarity = self.cosine_similarity(seg_emb, broll_emb)
                similarity_matrix[seg_id][broll_id] = similarity
        
        # Save similarity matrix
        self._save_embeddings("similarity_matrix.json", similarity_matrix)
        
        return similarity_matrix
    
    def _save_embeddings(self, filename: str, data: Dict):
        """Save embeddings to JSON file"""
        output_path = self.embeddings_dir / filename
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj
        
        serializable_data = convert_to_serializable(data)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=2)
    
    def load_embeddings(self, filename: str) -> Optional[Dict]:
        """Load embeddings from JSON file"""
        embeddings_path = self.embeddings_dir / filename
        
        if not embeddings_path.exists():
            return None
        
        try:
            with open(embeddings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Convert string keys back to int for segment IDs
            if filename == "transcript_embeddings.json":
                return {int(k): v for k, v in data.items()}
            elif filename == "similarity_matrix.json":
                return {int(k): v for k, v in data.items()}
            
            return data
        except:
            return None

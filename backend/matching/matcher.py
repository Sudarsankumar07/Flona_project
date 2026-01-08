"""
Semantic Matcher Module
Core intelligence for matching A-roll transcript segments with B-roll clips
Uses embedding similarity and smart filtering rules
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    ARTIFACTS_DIR,
    SIMILARITY_THRESHOLD,
    MIN_GAP_SECONDS,
    MAX_INSERTIONS,
    MIN_INSERTIONS,
    BROLL_MIN_DURATION,
    BROLL_MAX_DURATION,
    CRITICAL_KEYWORDS,
)
from schemas import TranscriptSegment, BRollDescription, MatchResult


class SemanticMatcher:
    """
    Matches transcript segments with B-roll clips using semantic similarity
    Applies smart filtering rules to ensure quality insertions
    """
    
    def __init__(self):
        self.output_dir = ARTIFACTS_DIR / "matching"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Matching parameters (can be overridden)
        self.similarity_threshold = SIMILARITY_THRESHOLD
        self.min_gap_seconds = MIN_GAP_SECONDS
        self.max_insertions = MAX_INSERTIONS
        self.min_insertions = MIN_INSERTIONS
        self.broll_min_duration = BROLL_MIN_DURATION
        self.broll_max_duration = BROLL_MAX_DURATION
        self.critical_keywords = CRITICAL_KEYWORDS
    
    def find_best_matches(
        self,
        segments: List[TranscriptSegment],
        broll_descriptions: List[BRollDescription],
        similarity_matrix: Dict[int, Dict[str, float]]
    ) -> List[MatchResult]:
        """
        Find the best B-roll matches for transcript segments
        
        Args:
            segments: List of transcript segments
            broll_descriptions: List of B-roll descriptions
            similarity_matrix: Pre-computed similarity scores
            
        Returns:
            List of MatchResult objects for suitable segments
        """
        # Create lookup for B-roll info
        broll_lookup = {desc.broll_id: desc for desc in broll_descriptions}
        
        # Score and rank all segments
        segment_scores = []
        
        for segment in segments:
            # Check if segment is suitable for B-roll insertion
            is_suitable, skip_reason = self._is_segment_suitable(segment)
            
            if not is_suitable:
                segment_scores.append(MatchResult(
                    segment_id=segment.id,
                    segment_text=segment.text,
                    segment_start=segment.start,
                    segment_end=segment.end,
                    best_broll_id=None,
                    similarity_score=0.0,
                    is_suitable=False,
                    skip_reason=skip_reason
                ))
                continue
            
            # Find best matching B-roll
            best_broll_id, best_score = self._find_best_broll(
                segment.id,
                similarity_matrix,
                broll_lookup
            )
            
            # Check if score meets threshold
            if best_score < self.similarity_threshold:
                segment_scores.append(MatchResult(
                    segment_id=segment.id,
                    segment_text=segment.text,
                    segment_start=segment.start,
                    segment_end=segment.end,
                    best_broll_id=best_broll_id,
                    similarity_score=best_score,
                    is_suitable=False,
                    skip_reason=f"Similarity {best_score:.2f} below threshold {self.similarity_threshold}"
                ))
            else:
                segment_scores.append(MatchResult(
                    segment_id=segment.id,
                    segment_text=segment.text,
                    segment_start=segment.start,
                    segment_end=segment.end,
                    best_broll_id=best_broll_id,
                    similarity_score=best_score,
                    is_suitable=True,
                    skip_reason=None
                ))
        
        # Apply selection rules
        selected_matches = self._select_insertions(segment_scores)
        
        # Save matching results
        self._save_matching_results(segment_scores, selected_matches)
        
        return selected_matches
    
    def _is_segment_suitable(self, segment: TranscriptSegment) -> Tuple[bool, Optional[str]]:
        """
        Check if a transcript segment is suitable for B-roll insertion
        
        Args:
            segment: Transcript segment to check
            
        Returns:
            Tuple of (is_suitable, skip_reason)
        """
        text = segment.text.lower()
        
        # Rule 1: Skip very short segments (less than 2 seconds)
        duration = segment.end - segment.start
        if duration < 2.0:
            return False, "Segment too short (< 2 seconds)"
        
        # Rule 2: Skip segments with critical keywords (emotional/important moments)
        for keyword in self.critical_keywords:
            if keyword.lower() in text:
                return False, f"Critical moment detected ('{keyword}')"
        
        # Rule 3: Skip very short text (likely filler words)
        word_count = len(text.split())
        if word_count < 3:
            return False, "Too few words (< 3 words)"
        
        # Rule 4: Skip segments that seem like direct address to camera
        direct_address_patterns = [
            r"\b(look at this|watch this|let me show you)\b",
            r"\b(as you can see|you'll notice)\b"
        ]
        for pattern in direct_address_patterns:
            if re.search(pattern, text):
                return False, "Direct address detected - viewer should see speaker"
        
        return True, None
    
    def _find_best_broll(
        self,
        segment_id: int,
        similarity_matrix: Dict[int, Dict[str, float]],
        broll_lookup: Dict[str, BRollDescription]
    ) -> Tuple[str, float]:
        """
        Find the best matching B-roll for a segment
        
        Args:
            segment_id: ID of the transcript segment
            similarity_matrix: Pre-computed similarities
            broll_lookup: B-roll info lookup
            
        Returns:
            Tuple of (best_broll_id, similarity_score)
        """
        if segment_id not in similarity_matrix:
            return None, 0.0
        
        segment_similarities = similarity_matrix[segment_id]
        
        # Find B-roll with highest similarity
        best_broll_id = None
        best_score = 0.0
        
        for broll_id, score in segment_similarities.items():
            if score > best_score:
                best_score = score
                best_broll_id = broll_id
        
        return best_broll_id, best_score
    
    def _select_insertions(self, all_matches: List[MatchResult]) -> List[MatchResult]:
        """
        Select final insertions from all potential matches
        Applies frequency control and spacing rules
        
        Args:
            all_matches: All MatchResult objects
            
        Returns:
            Selected matches for final timeline
        """
        # Filter to suitable matches only
        suitable_matches = [m for m in all_matches if m.is_suitable]
        
        if not suitable_matches:
            return []
        
        # Sort by similarity score (highest first)
        sorted_matches = sorted(
            suitable_matches,
            key=lambda x: x.similarity_score,
            reverse=True
        )
        
        selected = []
        used_broll_ids = set()
        
        for match in sorted_matches:
            # Check if we have enough insertions
            if len(selected) >= self.max_insertions:
                break
            
            # Check minimum gap from previous insertions
            too_close = False
            for prev_match in selected:
                gap = abs(match.segment_start - prev_match.segment_start)
                if gap < self.min_gap_seconds:
                    too_close = True
                    break
            
            if too_close:
                continue
            
            # Optionally: avoid reusing same B-roll (can be relaxed if few B-rolls)
            # Commenting out to allow reuse if needed
            # if match.best_broll_id in used_broll_ids:
            #     continue
            
            selected.append(match)
            if match.best_broll_id:
                used_broll_ids.add(match.best_broll_id)
        
        # Sort by timestamp for chronological order
        selected.sort(key=lambda x: x.segment_start)
        
        return selected
    
    def calculate_insertion_duration(
        self,
        segment: TranscriptSegment,
        broll: BRollDescription
    ) -> float:
        """
        Calculate optimal B-roll insertion duration
        
        Args:
            segment: Transcript segment
            broll: B-roll description
            
        Returns:
            Duration in seconds
        """
        segment_duration = segment.end - segment.start
        broll_duration = broll.duration
        
        # Target duration: 60-80% of segment duration
        target_duration = segment_duration * 0.7
        
        # Clamp to limits
        duration = max(self.broll_min_duration, min(target_duration, self.broll_max_duration))
        
        # Don't exceed actual B-roll duration
        duration = min(duration, broll_duration)
        
        # Don't exceed segment duration
        duration = min(duration, segment_duration - 0.5)  # Leave buffer
        
        return round(duration, 1)
    
    def generate_reason(
        self,
        segment: TranscriptSegment,
        broll: BRollDescription,
        similarity_score: float
    ) -> str:
        """
        Generate human-readable reason for B-roll insertion
        
        Args:
            segment: Transcript segment
            broll: B-roll description
            similarity_score: Matching score
            
        Returns:
            Explanation string
        """
        # Extract key concepts from segment text
        segment_keywords = self._extract_keywords(segment.text)
        broll_keywords = self._extract_keywords(broll.description)
        
        # Find common concepts
        common = segment_keywords.intersection(broll_keywords)
        
        if common:
            common_str = ", ".join(list(common)[:3])
            return (
                f"Speaker mentions '{segment.text[:50]}...' which relates to "
                f"B-roll showing '{broll.description[:50]}...' "
                f"(shared concepts: {common_str}, confidence: {similarity_score:.0%})"
            )
        else:
            return (
                f"B-roll showing '{broll.description[:60]}...' provides visual context "
                f"for spoken content about '{segment.text[:40]}...' "
                f"(semantic match confidence: {similarity_score:.0%})"
            )
    
    def _extract_keywords(self, text: str) -> set:
        """Extract meaningful keywords from text"""
        # Simple keyword extraction (could be enhanced with NLP)
        stopwords = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who',
            'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
            'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'or',
            'of', 'to', 'for', 'with', 'on', 'at', 'by', 'from', 'in', 'into'
        }
        
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter stopwords
        keywords = {w for w in words if w not in stopwords}
        
        return keywords
    
    def _save_matching_results(
        self,
        all_matches: List[MatchResult],
        selected_matches: List[MatchResult]
    ):
        """Save matching results to JSON file"""
        output_path = self.output_dir / "matching_results.json"
        
        data = {
            "total_segments": len(all_matches),
            "suitable_segments": len([m for m in all_matches if m.is_suitable]),
            "selected_insertions": len(selected_matches),
            "all_matches": [m.model_dump() for m in all_matches],
            "selected_matches": [m.model_dump() for m in selected_matches]
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

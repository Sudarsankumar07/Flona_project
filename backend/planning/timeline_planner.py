"""
Timeline Planner Module
Generates the final structured timeline JSON with all B-roll insertions
This is the main deliverable of the project
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import time

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    OUTPUT_DIR,
    BROLL_MIN_DURATION,
    BROLL_MAX_DURATION,
)
from schemas import (
    TranscriptSegment,
    BRollDescription,
    BRollInsertion,
    TimelinePlan,
    MatchResult,
)
from matching import SemanticMatcher


class TimelinePlanner:
    """
    Generates the final timeline plan with B-roll insertions
    Combines all processing results into structured output
    """
    
    def __init__(self):
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.matcher = SemanticMatcher()
    
    def generate_timeline(
        self,
        aroll_filename: str,
        aroll_duration: float,
        segments: List[TranscriptSegment],
        broll_descriptions: List[BRollDescription],
        selected_matches: List[MatchResult],
        processing_start_time: Optional[float] = None
    ) -> TimelinePlan:
        """
        Generate the complete timeline plan
        
        Args:
            aroll_filename: Name of the A-roll file
            aroll_duration: Duration of A-roll in seconds
            segments: All transcript segments
            broll_descriptions: All B-roll descriptions
            selected_matches: Matches selected for insertion
            processing_start_time: Start time for calculating processing duration
            
        Returns:
            Complete TimelinePlan object
        """
        # Create B-roll lookup
        broll_lookup = {desc.broll_id: desc for desc in broll_descriptions}
        segment_lookup = {seg.id: seg for seg in segments}
        
        # Generate insertions
        insertions = []
        
        for match in selected_matches:
            if not match.best_broll_id or not match.is_suitable:
                continue
            
            broll = broll_lookup.get(match.best_broll_id)
            segment = segment_lookup.get(match.segment_id)
            
            if not broll or not segment:
                continue
            
            # Calculate insertion timing
            # Insert B-roll slightly after segment starts (0.3s delay)
            start_sec = segment.start + 0.3
            
            # Calculate duration
            duration_sec = self.matcher.calculate_insertion_duration(segment, broll)
            
            # Generate explanation
            reason = self.matcher.generate_reason(segment, broll, match.similarity_score)
            
            insertion = BRollInsertion(
                start_sec=round(start_sec, 2),
                duration_sec=duration_sec,
                broll_id=broll.broll_id,
                broll_filename=broll.filename,
                confidence=round(match.similarity_score, 2),
                reason=reason,
                transcript_segment_id=segment.id,
                transcript_text=segment.text
            )
            
            insertions.append(insertion)
        
        # Sort insertions by start time
        insertions.sort(key=lambda x: x.start_sec)
        
        # Calculate processing time
        processing_time = None
        if processing_start_time:
            processing_time = round(time.time() - processing_start_time, 2)
        
        # Create timeline plan
        timeline = TimelinePlan(
            aroll_filename=aroll_filename,
            aroll_duration=round(aroll_duration, 2),
            total_insertions=len(insertions),
            insertions=insertions,
            transcript_segments=segments,
            broll_descriptions=broll_descriptions,
            processing_time_sec=processing_time
        )
        
        # Save timeline to file
        self._save_timeline(timeline)
        
        return timeline
    
    def _save_timeline(self, timeline: TimelinePlan):
        """Save timeline to JSON file"""
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"timeline_{timestamp}.json"
        output_path = self.output_dir / filename
        
        # Also save as latest.json for easy access
        latest_path = self.output_dir / "timeline_latest.json"
        
        # Convert to dict for JSON serialization
        data = timeline.model_dump()
        
        # Add metadata
        data["_metadata"] = {
            "generated_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "filename": filename
        }
        
        # Save timestamped version
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Save latest version
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Timeline saved to: {output_path}")
        print(f"Latest timeline: {latest_path}")
    
    def load_latest_timeline(self) -> Optional[TimelinePlan]:
        """Load the most recent timeline from file"""
        latest_path = self.output_dir / "timeline_latest.json"
        
        if not latest_path.exists():
            return None
        
        try:
            with open(latest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Remove metadata before parsing
            data.pop("_metadata", None)
            
            return TimelinePlan(**data)
        except Exception as e:
            print(f"Error loading timeline: {e}")
            return None
    
    def export_simple_timeline(self, timeline: TimelinePlan) -> dict:
        """
        Export a simplified version of the timeline
        Useful for frontend display or quick review
        
        Args:
            timeline: Full timeline plan
            
        Returns:
            Simplified dict with essential information
        """
        return {
            "aroll_duration": timeline.aroll_duration,
            "total_insertions": timeline.total_insertions,
            "insertions": [
                {
                    "start": ins.start_sec,
                    "duration": ins.duration_sec,
                    "broll": ins.broll_filename,
                    "confidence": f"{ins.confidence:.0%}",
                    "context": ins.transcript_text[:50] + "..." if len(ins.transcript_text) > 50 else ins.transcript_text
                }
                for ins in timeline.insertions
            ]
        }
    
    def validate_timeline(self, timeline: TimelinePlan) -> Dict[str, any]:
        """
        Validate timeline for common issues
        
        Args:
            timeline: Timeline to validate
            
        Returns:
            Validation result with any warnings
        """
        issues = []
        warnings = []
        
        # Check number of insertions
        if timeline.total_insertions < 1:
            issues.append("No B-roll insertions found")
        elif timeline.total_insertions > 6:
            warnings.append(f"High number of insertions ({timeline.total_insertions})")
        
        # Check for overlapping insertions
        for i, ins1 in enumerate(timeline.insertions):
            for ins2 in timeline.insertions[i+1:]:
                end1 = ins1.start_sec + ins1.duration_sec
                if end1 > ins2.start_sec:
                    issues.append(
                        f"Overlapping insertions: {ins1.broll_id} ends at {end1:.1f}s "
                        f"but {ins2.broll_id} starts at {ins2.start_sec:.1f}s"
                    )
        
        # Check insertion timing
        for ins in timeline.insertions:
            if ins.start_sec < 0:
                issues.append(f"Insertion {ins.broll_id} has negative start time")
            
            end_time = ins.start_sec + ins.duration_sec
            if end_time > timeline.aroll_duration:
                warnings.append(
                    f"Insertion {ins.broll_id} extends beyond A-roll duration "
                    f"({end_time:.1f}s > {timeline.aroll_duration:.1f}s)"
                )
            
            if ins.duration_sec < BROLL_MIN_DURATION:
                warnings.append(f"Insertion {ins.broll_id} very short ({ins.duration_sec:.1f}s)")
            
            if ins.duration_sec > BROLL_MAX_DURATION:
                warnings.append(f"Insertion {ins.broll_id} quite long ({ins.duration_sec:.1f}s)")
        
        # Check confidence scores
        low_confidence = [ins for ins in timeline.insertions if ins.confidence < 0.5]
        if low_confidence:
            warnings.append(f"{len(low_confidence)} insertions have low confidence (<50%)")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "summary": {
                "total_insertions": timeline.total_insertions,
                "aroll_duration": timeline.aroll_duration,
                "avg_confidence": sum(i.confidence for i in timeline.insertions) / max(len(timeline.insertions), 1)
            }
        }

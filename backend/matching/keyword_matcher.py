"""
Keyword-Based Matcher Module
Matches B-roll clips to transcript segments using keyword analysis
Works without external APIs - fallback when AI planner is unavailable
"""

import re
from typing import List, Dict, Tuple
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from schemas import TranscriptSegment, BRollDescription, BRollInsertion


class KeywordMatcher:
    """
    Matches transcript segments with B-roll clips using keyword analysis
    Provides reliable insertions when AI/embedding approaches fail
    """
    
    # Define keyword mappings for common topics
    KEYWORD_MAPPINGS = {
        # Food-related keywords
        'food': ['food', 'foods', 'eat', 'eating', 'meal', 'dish', 'plate', 'container', 'kitchen', 'cook', 'cooking', 'served', 'serving'],
        'fruit': ['fruit', 'fruits', 'veggies', 'vegetables', 'fresh', 'seasonal', 'bowl', 'apple', 'orange', 'banana', 'buying'],
        'pastry': ['pastry', 'pastries', 'bread', 'bakery', 'baked', 'sweet', 'dessert', 'cake', 'filled'],
        'packaged': ['packaged', 'package', 'container', 'plastic', 'box', 'sealed', 'expiry', 'label', 'date'],
        'street_food': ['street', 'stand', 'vendor', 'market', 'stall', 'scene'],
        'healthy': ['healthy', 'nutritious', 'nutrition', 'quality', 'safe', 'safety', 'chemical', 'important', 'mindful'],
        'hygiene': ['hygiene', 'wash', 'clean', 'soap', 'water', 'hands', 'scrub'],
        'table': ['table', 'counter', 'sitting', 'plates'],
    }
    
    # Direct topic-to-broll mappings based on description content
    TOPIC_BROLL_HINTS = {
        'food quality': ['food', 'plate', 'container', 'kitchen'],
        'eating': ['food', 'plate', 'eat'],
        'expiry date': ['container', 'plastic', 'packaged'],
        'packaged foods': ['container', 'plastic', 'packaged'],
        'spoilage': ['food', 'container'],
        'hygiene': ['water', 'clean'],
        'cooking': ['kitchen', 'plate', 'food'],
        'fruits': ['fruit', 'bowl', 'fresh'],
        'veggies': ['fruit', 'fresh', 'bowl'],
        'nutritious': ['fruit', 'food', 'fresh'],
        'healthy': ['fruit', 'food', 'fresh'],
    }
    
    # Specific segment-to-broll mappings (very direct matches)
    # Keys are phrases to look for in transcript, values are keywords to match in B-roll
    DIRECT_MAPPINGS = [
        (['food quality', 'food safety', 'about food'], ['food', 'plate', 'kitchen', 'served']),
        (['what you\'re eating', 'what you are eating', 'is it healthy'], ['food', 'plate']),
        (['expiry date', 'packaged foods', 'packaged food'], ['container', 'plastic']),
        (['spoilage', 'smells weird', 'don\'t eat'], ['food', 'container']),
        (['hygiene', 'wash your hands', 'cooking or eating'], ['kitchen', 'food', 'plate']),
        (['soap and water', 'scrub', '20 seconds'], ['water', 'bowl']),
        (['fruits and veggies', 'fruits', 'vegetables', 'fresh and seasonal'], ['fruit', 'bowl']),
        (['nutritious', 'harmful chemicals'], ['fruit', 'fresh']),
        (['stay healthy', 'be mindful'], ['food', 'fruit', 'plate']),
        (['street food', 'food stand'], ['street', 'stand', 'vendor']),
        (['pastries', 'bakery', 'dessert'], ['pastry', 'pastries', 'filled']),
    ]
    
    def __init__(self):
        self.min_gap_seconds = 5.0
        self.max_insertions = 6
    
    def find_matches(
        self,
        segments: List[TranscriptSegment],
        broll_descriptions: List[BRollDescription],
        max_insertions: int = 6,
        min_gap_seconds: float = 5.0
    ) -> List[BRollInsertion]:
        """
        Find B-roll matches for transcript segments using keyword analysis
        
        Args:
            segments: List of transcript segments
            broll_descriptions: List of B-roll descriptions
            max_insertions: Maximum number of insertions
            min_gap_seconds: Minimum gap between insertions
            
        Returns:
            List of BRollInsertion objects
        """
        self.min_gap_seconds = min_gap_seconds
        self.max_insertions = max_insertions
        
        # Analyze B-rolls for keywords
        broll_keywords = self._analyze_brolls(broll_descriptions)
        
        # Score each segment-broll pair
        matches = []
        used_brolls = set()  # Track used B-rolls to ensure variety
        
        for segment in segments:
            segment_text = segment.text.lower()
            segment_keywords = self._extract_keywords(segment_text)
            
            # Also check for topic hints
            topic_hints = self._get_topic_hints(segment_text)
            
            best_match = None
            best_score = 0.0
            
            for broll in broll_descriptions:
                broll_kw = broll_keywords.get(broll.broll_id, set())
                
                # Calculate base score
                score = self._calculate_match_score(segment_keywords, broll_kw)
                
                # Boost score if topic hints match
                for hint_keyword in topic_hints:
                    if hint_keyword in broll.description.lower():
                        score += 0.15
                
                # Slight penalty if B-roll already used (encourage variety)
                if broll.broll_id in used_brolls:
                    score *= 0.7
                
                if score > best_score:
                    best_score = score
                    best_match = broll
            
            # Lower threshold to get more matches
            if best_match and best_score > 0.05:
                matches.append({
                    'segment': segment,
                    'broll': best_match,
                    'score': best_score
                })
                used_brolls.add(best_match.broll_id)
        
        # Sort by score and apply spacing rules
        matches.sort(key=lambda x: x['score'], reverse=True)
        
        # Select final insertions with spacing constraints
        selected = self._apply_spacing_rules(matches)
        
        # Convert to BRollInsertion objects
        insertions = []
        for match in selected:
            segment = match['segment']
            broll = match['broll']
            
            # Calculate insertion point (middle of segment)
            insertion_start = segment.start + (segment.end - segment.start) / 4
            duration = min(3.0, broll.duration, segment.end - insertion_start)
            
            insertion = BRollInsertion(
                start_sec=round(insertion_start, 1),
                duration_sec=round(duration, 1),
                broll_id=broll.broll_id,
                broll_filename=broll.filename,
                confidence=round(min(0.95, match['score']), 2),
                reason=f"Keyword match: {self._get_matching_keywords(segment.text, broll.description)}",
                transcript_segment_id=segment.id,
                transcript_text=segment.text
            )
            insertions.append(insertion)
        
        # Sort by start time
        insertions.sort(key=lambda x: x.start_sec)
        
        print(f"✓ Keyword matcher found {len(insertions)} insertions")
        return insertions
    
    def _get_topic_hints(self, text: str) -> List[str]:
        """Get relevant keywords from topic hints"""
        hints = []
        text_lower = text.lower()
        
        # Check standard topic hints
        for topic, keywords in self.TOPIC_BROLL_HINTS.items():
            if topic in text_lower:
                hints.extend(keywords)
        
        # Check direct mappings (more specific)
        for phrases, broll_keywords in self.DIRECT_MAPPINGS:
            for phrase in phrases:
                if phrase in text_lower:
                    hints.extend(broll_keywords)
                    break
        
        return list(set(hints))  # Remove duplicates
    
    def _analyze_brolls(self, brolls: List[BRollDescription]) -> Dict[str, set]:
        """Extract keywords from B-roll descriptions"""
        broll_keywords = {}
        
        for broll in brolls:
            keywords = self._extract_keywords(broll.description.lower())
            broll_keywords[broll.broll_id] = keywords
        
        return broll_keywords
    
    def _extract_keywords(self, text: str) -> set:
        """Extract relevant keywords from text"""
        text = text.lower()
        found_keywords = set()
        
        # Check against all keyword categories
        for category, keywords in self.KEYWORD_MAPPINGS.items():
            for keyword in keywords:
                if keyword in text:
                    found_keywords.add(category)
                    found_keywords.add(keyword)
        
        # Also add significant nouns from the text
        words = re.findall(r'\b[a-z]{3,}\b', text)
        for word in words:
            if word not in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 
                           'was', 'one', 'our', 'out', 'has', 'have', 'been', 'will', 'more', 'when',
                           'who', 'oil', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'this', 'that',
                           'with', 'they', 'from', 'what', 'about', 'which', 'their', 'there', 'these',
                           'would', 'make', 'like', 'just', 'know', 'take', 'into', 'year', 'your',
                           'good', 'some', 'could', 'them', 'other', 'than', 'then', 'look', 'only',
                           'come', 'over', 'such', 'also', 'back', 'after', 'thing', 'gonna', 'super',
                           'really', 'actually', 'right', 'yeah']:
                found_keywords.add(word)
        
        return found_keywords
    
    def _calculate_match_score(self, segment_keywords: set, broll_keywords: set) -> float:
        """Calculate match score between segment and B-roll keywords"""
        if not segment_keywords or not broll_keywords:
            return 0.0
        
        # Find intersection
        common = segment_keywords & broll_keywords
        
        if not common:
            return 0.0
        
        # Calculate Jaccard similarity with boost for category matches
        category_boost = 0
        for category in self.KEYWORD_MAPPINGS.keys():
            if category in common:
                category_boost += 0.2
        
        jaccard = len(common) / len(segment_keywords | broll_keywords)
        
        return min(1.0, jaccard + category_boost)
    
    def _apply_spacing_rules(self, matches: List[Dict]) -> List[Dict]:
        """Apply minimum spacing rules between insertions - time-aware selection"""
        if not matches:
            return []
        
        # Remove duplicates (same segment might have been added multiple times)
        seen_segments = set()
        unique_matches = []
        for m in matches:
            if m['segment'].id not in seen_segments:
                seen_segments.add(m['segment'].id)
                unique_matches.append(m)
        
        # Sort by time
        time_sorted = sorted(unique_matches, key=lambda x: x['segment'].start)
        
        selected = []
        last_end_time = -self.min_gap_seconds
        used_brolls_count = {}  # Track how many times each B-roll is used
        
        for match in time_sorted:
            if len(selected) >= self.max_insertions:
                break
            
            segment = match['segment']
            broll_id = match['broll'].broll_id
            
            # Check if this segment meets the gap requirement
            if segment.start >= last_end_time + self.min_gap_seconds:
                # Only add if score is meaningful (very low threshold)
                if match['score'] > 0.05:
                    # Limit same B-roll usage to 2 times max
                    if used_brolls_count.get(broll_id, 0) < 2:
                        selected.append(match)
                        last_end_time = segment.end
                        used_brolls_count[broll_id] = used_brolls_count.get(broll_id, 0) + 1
        
        return selected
    
    def _get_matching_keywords(self, transcript_text: str, broll_description: str) -> str:
        """Get a brief description of matching keywords"""
        transcript_kw = self._extract_keywords(transcript_text.lower())
        broll_kw = self._extract_keywords(broll_description.lower())
        
        common = transcript_kw & broll_kw
        
        # Filter to most relevant keywords
        relevant = [kw for kw in common if kw in self.KEYWORD_MAPPINGS.keys() or len(kw) > 4]
        
        if relevant:
            return ', '.join(list(relevant)[:3])
        elif common:
            return ', '.join(list(common)[:3])
        else:
            return "visual context"

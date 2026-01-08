"""
AI-Powered Insertion Planner
Uses Gemini/OpenAI/OpenRouter to intelligently decide B-roll insertions based on semantic understanding
This replaces pure embedding similarity with AI reasoning
"""

import json
import os
from typing import List, Dict, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import get_vision_provider, GEMINI_API_KEY, OPENAI_API_KEY
from schemas import TranscriptSegment, BRollDescription, BRollInsertion

# OpenRouter API key (can also be in .env)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")


class AIInsertionPlanner:
    """
    Uses AI (Gemini/OpenAI/OpenRouter) to analyze transcript and B-roll descriptions
    and suggest optimal insertion points with reasoning
    """
    
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize AI planner with specified provider
        
        Args:
            provider: "gemini", "openai", or "openrouter". If None, auto-detect from config
        """
        self.provider = provider or get_vision_provider()
        
        if self.provider == "gemini":
            self._init_gemini()
        elif self.provider == "openai":
            self._init_openai()
        elif self.provider == "openrouter":
            self._init_openrouter()
        else:
            raise ValueError(f"AI provider must be 'gemini', 'openai', or 'openrouter', got: {self.provider}")
    
    def _init_gemini(self):
        """Initialize Gemini client"""
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            
            # Try models in order of preference (updated for 2025)
            models_to_try = [
                "gemini-2.5-flash",      # Best balance: stable, 65K output
                "gemini-2.0-flash",       # Fast and reliable
                "gemini-2.5-pro",         # Highest quality (more expensive)
                "gemini-2.0-flash-exp",   # Experimental fallback
            ]
            
            for model_name in models_to_try:
                try:
                    self.model = genai.GenerativeModel(model_name)
                    # Test with simple prompt
                    response = self.model.generate_content(
                        "Say OK",
                        generation_config={"max_output_tokens": 5}
                    )
                    if response.text:
                        self.model_name = model_name
                        print(f"✓ Using Gemini model: {model_name}")
                        return
                except:
                    continue
            
            raise ValueError("No available Gemini models found. Check quota/API key.")
            
        except ImportError:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model_name = "gpt-4o-mini"  # Cost-effective for this task
            print(f"✓ Using OpenAI model: {self.model_name}")
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")
    
    def _init_openrouter(self):
        """Initialize OpenRouter client (uses OpenAI SDK)"""
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not found in .env file")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY
            )
            # Use a fast, capable model via OpenRouter
            # Good options: openai/gpt-4o-mini, meta-llama/llama-3.1-70b-instruct, anthropic/claude-3-haiku
            self.model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
            print(f"✓ Using OpenRouter model: {self.model_name}")
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")
    
    def plan_insertions(
        self,
        segments: List[TranscriptSegment],
        broll_descriptions: List[BRollDescription],
        max_insertions: int = 6,
        min_gap_seconds: float = 8.0
    ) -> List[BRollInsertion]:
        """
        Use AI to analyze transcript and B-rolls and suggest optimal insertions
        
        Args:
            segments: Transcript segments from A-roll
            broll_descriptions: Available B-roll clips with descriptions
            max_insertions: Maximum number of B-roll insertions allowed
            min_gap_seconds: Minimum gap between insertions
            
        Returns:
            List of B-roll insertions with AI reasoning
        """
        print(f"\n🤖 Using AI ({self.provider}) to plan insertions...")
        
        # Build the prompt
        prompt = self._build_insertion_prompt(
            segments, broll_descriptions, max_insertions, min_gap_seconds
        )
        
        # Get AI response
        if self.provider == "gemini":
            response_text = self._query_gemini(prompt)
        elif self.provider == "openrouter":
            response_text = self._query_openrouter(prompt)
        else:
            response_text = self._query_openai(prompt)
        
        # Parse AI response to extract insertions
        insertions = self._parse_ai_response(response_text, broll_descriptions)
        
        print(f"✓ AI suggested {len(insertions)} insertions")
        return insertions
    
    def _build_insertion_prompt(
        self,
        segments: List[TranscriptSegment],
        broll_descriptions: List[BRollDescription],
        max_insertions: int,
        min_gap_seconds: float
    ) -> str:
        """Build the prompt for AI to analyze and suggest insertions"""
        
        # Format transcript
        transcript_text = "\n".join([
            f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}"
            for seg in segments
        ])
        
        # Format B-rolls
        broll_text = "\n".join([
            f"{broll.id}: {broll.description} (Duration: {broll.duration:.1f}s)"
            for broll in broll_descriptions
        ])
        
        prompt = f"""You are an expert video editor. Analyze this A-roll transcript and suggest the BEST B-roll insertions to enhance the video.

TRANSCRIPT:
{transcript_text}

AVAILABLE B-ROLLS:
{broll_text}

RULES:
1. Maximum {max_insertions} insertions
2. Minimum {min_gap_seconds} seconds between insertions
3. Each B-roll can be used multiple times if relevant
4. Insert B-rolls during natural pauses or when speaker mentions related topics
5. NEVER insert during critical speaking moments (important points, calls-to-action)
6. Prioritize visual relevance over word matching

OUTPUT FORMAT (JSON only, no explanation):
{{
  "insertions": [
    {{
      "segment_id": 2,
      "start_sec": 4.5,
      "duration_sec": 3.0,
      "broll_id": "broll_1",
      "confidence": 0.85,
      "reason": "Brief explanation why this B-roll fits here"
    }}
  ]
}}

Analyze carefully and respond with ONLY the JSON object."""
        
        return prompt
    
    def _query_gemini(self, prompt: str) -> str:
        """Query Gemini API"""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,  # Lower = more focused
                    "max_output_tokens": 2048
                }
            )
            return response.text
        except Exception as e:
            print(f"❌ Gemini API error: {e}")
            raise
    
    def _query_openai(self, prompt: str) -> str:
        """Query OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert video editor who suggests B-roll insertions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2048
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ OpenAI API error: {e}")
            raise
    
    def _query_openrouter(self, prompt: str) -> str:
        """Query OpenRouter API (uses OpenAI SDK format)"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert video editor who suggests B-roll insertions. Respond with JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2048,
                extra_headers={
                    "HTTP-Referer": "https://github.com/Sudarsankumar07/Flona_project",
                    "X-Title": "Smart B-Roll Inserter"
                }
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ OpenRouter API error: {e}")
            raise
    
    def _parse_ai_response(
        self,
        response_text: str,
        broll_descriptions: List[BRollDescription]
    ) -> List[BRollInsertion]:
        """Parse AI response JSON into BRollInsertion objects"""
        
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_text = response_text.strip()
            if json_text.startswith("```"):
                # Remove markdown code block markers
                lines = json_text.split("\n")
                json_text = "\n".join(lines[1:-1]) if len(lines) > 2 else json_text
            
            # Parse JSON
            data = json.loads(json_text)
            insertions_data = data.get("insertions", [])
            
            # Convert to BRollInsertion objects
            insertions = []
            for ins in insertions_data:
                # Find matching B-roll
                broll = next(
                    (b for b in broll_descriptions if b.id == ins["broll_id"]),
                    None
                )
                if not broll:
                    print(f"⚠ Warning: B-roll {ins['broll_id']} not found, skipping")
                    continue
                
                insertion = BRollInsertion(
                    start_sec=float(ins["start_sec"]),
                    duration_sec=min(float(ins["duration_sec"]), broll.duration),
                    broll_id=ins["broll_id"],
                    broll_filename=broll.filename,
                    confidence=float(ins.get("confidence", 0.8)),
                    reason=ins.get("reason", "AI suggested insertion")
                )
                insertions.append(insertion)
            
            # Sort by start time
            insertions.sort(key=lambda x: x.start_sec)
            
            return insertions
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse AI response as JSON: {e}")
            print(f"Response was: {response_text[:200]}...")
            return []
        except Exception as e:
            print(f"❌ Error parsing AI response: {e}")
            return []


if __name__ == "__main__":
    """Test the AI planner with sample data"""
    
    # Sample transcript segments
    segments = [
        TranscriptSegment(id=1, start=0.0, end=3.0, text="Welcome to my coffee brewing tutorial"),
        TranscriptSegment(id=2, start=3.5, end=7.0, text="First, we'll grind fresh coffee beans"),
        TranscriptSegment(id=3, start=8.0, end=12.0, text="The grind size is very important for espresso"),
    ]
    
    # Sample B-rolls
    brolls = [
        BRollDescription(
            id="broll_1",
            filename="coffee_beans.mp4",
            duration=5.0,
            description="Close-up of fresh coffee beans in a grinder"
        ),
        BRollDescription(
            id="broll_2",
            filename="espresso_machine.mp4",
            duration=4.0,
            description="Espresso machine brewing coffee with steam"
        ),
    ]
    
    # Test planner
    planner = AIInsertionPlanner(provider="gemini")  # or "openai"
    insertions = planner.plan_insertions(segments, brolls)
    
    print("\n" + "=" * 70)
    print("AI SUGGESTED INSERTIONS:")
    print("=" * 70)
    for ins in insertions:
        print(f"\n{ins.start_sec:.1f}s: {ins.broll_id}")
        print(f"  Duration: {ins.duration_sec:.1f}s")
        print(f"  Confidence: {ins.confidence:.2f}")
        print(f"  Reason: {ins.reason}")

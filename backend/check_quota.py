"""
Quick script to check if Gemini API quota is available
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

def check_gemini_quota():
    """Check if Gemini API quota is available"""
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_VISION_MODEL", "gemini-2.0-flash")
    
    if not api_key:
        print("❌ No GEMINI_API_KEY found in .env")
        return False
    
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        
        # Try a simple text generation (much less tokens than video)
        response = client.models.generate_content(
            model=model,
            contents="Say 'Hello' in one word."
        )
        
        print(f"✓ Gemini API is working!")
        print(f"  Model: {model}")
        print(f"  Response: {response.text.strip()}")
        return True
        
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            print(f"❌ Gemini API quota exhausted")
            print(f"  Model: {model}")
            print("\nOptions:")
            print("  1. Wait for quota reset (usually resets daily)")
            print("  2. Create new API key: https://aistudio.google.com/app/apikey")
            print("  3. Use OpenAI API (requires ffmpeg)")
        else:
            print(f"❌ Error: {e}")
        return False


def check_openai_quota():
    """Check if OpenAI API is configured and working"""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key or api_key == "your_openai_api_key_here":
        print("❌ No valid OPENAI_API_KEY found in .env")
        return False
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'Hello' in one word."}],
            max_tokens=10
        )
        
        print(f"✓ OpenAI API is working!")
        print(f"  Response: {response.choices[0].message.content.strip()}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("API QUOTA CHECK")
    print("=" * 50)
    print()
    
    print("Checking Gemini API...")
    gemini_ok = check_gemini_quota()
    print()
    
    print("Checking OpenAI API...")
    openai_ok = check_openai_quota()
    print()
    
    print("=" * 50)
    if gemini_ok:
        print("✓ Ready to run: python run_pipeline.py")
    elif openai_ok:
        print("⚠ Gemini quota exhausted, but OpenAI is available")
        print("  Update .env to use API_PROVIDER=openai")
        print("  Note: OpenAI requires ffmpeg for transcription")
    else:
        print("❌ Neither API is available")
        print("\nTo fix:")
        print("  1. Wait for Gemini quota reset (daily)")
        print("  2. Create new Gemini API key")
        print("  3. Add OpenAI API key to .env")

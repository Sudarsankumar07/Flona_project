"""
Test script to check which Gemini models are available with your API key
Run this to find the best model for insertion planning
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    print("❌ No GEMINI_API_KEY found in .env file")
    print("Please add your Gemini API key to .env:")
    print("GEMINI_API_KEY=your_key_here")
    sys.exit(1)

print("=" * 70)
print("GEMINI MODEL AVAILABILITY TEST")
print("=" * 70)
print(f"\nAPI Key: {GEMINI_API_KEY[:20]}..." if len(GEMINI_API_KEY) > 20 else GEMINI_API_KEY)
print()

try:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
except ImportError:
    print("❌ google-generativeai not installed")
    print("Run: pip install google-generativeai")
    sys.exit(1)

# Test models in order of preference (best to basic)
test_models = [
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
    "gemini-pro",
]

print("Testing available models...\n")
print("-" * 70)

available_models = []
best_model = None

for model_name in test_models:
    try:
        print(f"\n🔍 Testing: {model_name}")
        model = genai.GenerativeModel(model_name)
        
        # Simple test prompt
        response = model.generate_content(
            "Say 'OK' if you can read this.",
            generation_config={"max_output_tokens": 10}
        )
        
        if response.text:
            print(f"   ✅ AVAILABLE - Response: {response.text.strip()}")
            available_models.append(model_name)
            if best_model is None:
                best_model = model_name
        else:
            print(f"   ❌ No response")
            
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            print(f"   ⚠️  QUOTA EXCEEDED - {error_msg[:100]}")
        elif "404" in error_msg or "NOT_FOUND" in error_msg:
            print(f"   ⚠️  MODEL NOT FOUND")
        elif "403" in error_msg or "PERMISSION_DENIED" in error_msg:
            print(f"   ⚠️  PERMISSION DENIED")
        else:
            print(f"   ❌ ERROR - {error_msg[:100]}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if available_models:
    print(f"\n✅ {len(available_models)} model(s) available:")
    for model in available_models:
        marker = " ⭐ RECOMMENDED" if model == best_model else ""
        print(f"   • {model}{marker}")
    
    print(f"\n🎯 BEST MODEL: {best_model}")
    print("\nUpdate your .env file:")
    print(f"   API_PROVIDER=gemini")
    print(f"   GEMINI_API_KEY={GEMINI_API_KEY[:20]}...")
    print(f"   GEMINI_VISION_MODEL={best_model}")
    
else:
    print("\n❌ No models available with this API key")
    print("\nPossible issues:")
    print("   1. API key is invalid")
    print("   2. All models have quota exhausted")
    print("   3. Region restrictions")
    print("\nTry:")
    print("   • Get a new API key from https://aistudio.google.com/apikey")
    print("   • Use a different Google account")
    print("   • Switch to OpenAI API")

print("\n" + "=" * 70)

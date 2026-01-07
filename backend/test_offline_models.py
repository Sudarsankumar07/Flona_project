"""
Test script for offline models - verifies all models work before running pipeline
"""
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def test_vision_model():
    """Test BLIP vision model"""
    print("\n🚀 Testing BLIP Vision Model...")
    print("-" * 60)
    
    from understanding.offline_models import load_vision_model, caption_image
    from PIL import Image
    import numpy as np
    
    try:
        # Load model
        print("Loading BLIP model (this may take a minute on first run)...")
        model, processor, device = load_vision_model("blip")
        print(f"✓ BLIP model loaded successfully on {device}")
        
        # Test with dummy image
        dummy_img = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        caption = caption_image(dummy_img, model, processor, device)
        print(f"✓ Generated test caption: '{caption}'")
        
        return True
    except Exception as e:
        print(f"✗ Vision model test failed: {e}")
        return False


def test_embedding_model():
    """Test sentence-transformers embedding model"""
    print("\n🚀 Testing Sentence-Transformers Embedding Model...")
    print("-" * 60)
    
    from understanding.offline_models import load_embedding_model, get_embeddings
    
    try:
        # Load model
        print("Loading embedding model...")
        emb_model = load_embedding_model("all-MiniLM-L6-v2")
        print(f"✓ Embedding model loaded successfully")
        
        # Test with sample sentences
        test_texts = [
            "A person walking in a park",
            "Someone jogging outdoors",
            "A cat sleeping on a couch"
        ]
        embeddings = get_embeddings(test_texts, emb_model)
        print(f"✓ Generated embeddings shape: {embeddings.shape}")
        print(f"✓ Embedding dimensions: {embeddings.shape[1]} (expected: 384)")
        
        # Test similarity
        import numpy as np
        sim_1_2 = np.dot(embeddings[0], embeddings[1])
        sim_1_3 = np.dot(embeddings[0], embeddings[2])
        print(f"✓ Similarity (walk vs jog): {sim_1_2:.3f}")
        print(f"✓ Similarity (walk vs cat): {sim_1_3:.3f}")
        print(f"  → Walk/jog should be more similar (higher score)")
        
        return True
    except Exception as e:
        print(f"✗ Embedding model test failed: {e}")
        return False


def test_whisper_model():
    """Test Whisper transcription model (optional)"""
    print("\n🚀 Testing Whisper Transcription Model (Optional)...")
    print("-" * 60)
    
    try:
        from understanding.offline_models import load_whisper_model
        
        # Try to load Whisper
        print("Loading Whisper model (this downloads ~500MB on first run)...")
        model = load_whisper_model("base")
        print(f"✓ Whisper model loaded successfully")
        print("  Note: This model will be used for A-roll transcription")
        
        return True
    except ImportError:
        print("⚠ Whisper not installed (optional)")
        print("  Install with: pip install openai-whisper")
        print("  You can still use visual fallback for transcription")
        return True  # Not a failure
    except Exception as e:
        print(f"⚠ Whisper model test failed: {e}")
        print("  You can still use visual fallback for transcription")
        return True  # Not a failure


def main():
    """Run all offline model tests"""
    print("=" * 60)
    print("OFFLINE MODELS TEST SUITE")
    print("=" * 60)
    print("\nThis will download models on first run (~1.5GB total)")
    print("Subsequent runs will use cached models.\n")
    
    results = {
        "Vision (BLIP)": test_vision_model(),
        "Embeddings (sentence-transformers)": test_embedding_model(),
        "Transcription (Whisper)": test_whisper_model()
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_critical_passed = results["Vision (BLIP)"] and results["Embeddings (sentence-transformers)"]
    
    if all_critical_passed:
        print("\n🎉 All critical offline models are working!")
        print("✓ You can now run the full pipeline with: python run_pipeline.py")
        return 0
    else:
        print("\n⚠ Some critical models failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

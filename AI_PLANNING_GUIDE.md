# AI-Powered Insertion Planning Guide

## Overview

The system uses a **hybrid architecture** combining the best of both worlds:

1. **Local Models** (BLIP + Whisper) - Free captioning & transcription
2. **AI Planning** (Gemini/OpenAI) - Intelligent insertion decisions

### 🏗️ Architecture Flow

```
B-roll Videos → BLIP (local) → Captions
A-roll Video → Whisper (local) → Transcript
                    ↓
        Both sent to Gemini AI
                    ↓
    AI analyzes & plans insertions
                    ↓
        Timeline JSON output
```

### ✅ Benefits

- **Cost-Effective**: Only use API for insertion planning, not captioning/transcription
- **Accurate**: BLIP generates detailed B-roll descriptions
- **Cross-Language**: AI understands Urdu/Hindi transcript + English captions
- **Context-Aware**: AI makes intelligent decisions, not just similarity matching

### ⚠️ vs Pure Offline Mode

**Pure Offline** (OLD):
- BLIP captions → Embeddings → Similarity matching
- ❌ Can't handle cross-language
- ❌ No reasoning/context

**Hybrid AI** (NEW):
- BLIP captions → Gemini analyzes → Smart insertions
- ✅ Cross-language support
- ✅ Reasoning included

---

## 🚀 Quick Start: AI Planning Mode

### Step 1: Get Gemini API Key (Free!)

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Click "Create API Key"
3. Copy your key

### Step 2: Test Available Models

```bash
cd backend

# Add your API key to .env
echo "GEMINI_API_KEY=your_key_here" >> .env

# Test which models work
python check_gemini_models.py
```

**Expected Output:**
```
🔍 Testing: gemini-2.0-flash-exp
   ✅ AVAILABLE - Response: OK

🎯 BEST MODEL: gemini-2.0-flash-exp
```

### Step 3: Configure for AI Mode

Edit `backend/.env`:

```env
# Hybrid Architecture
API_PROVIDER=gemini
GEMINI_API_KEY=your_actual_key_here

# Offline models for captioning & transcription
TRANSCRIPTION_PROVIDER=offline
OFFLINE_WHISPER_MODEL=base
OFFLINE_VISION_MODEL=blip

# Pipeline Settings
SIMILARITY_THRESHOLD=0.65
MIN_GAP_SECONDS=8.0
MAX_INSERTIONS=6
```

### Step 4: Run Pipeline

```bash
python run_pipeline.py
```

**New Output:**
```
[3/6] CAPTIONING B-ROLL CLIPS
------------------------------------------------------------
  ✓ broll_1: Generated caption with BLIP
  ✓ broll_2: Generated caption with BLIP
  ...

[4/6] AI-POWERED INSERTION PLANNING
------------------------------------------------------------
🤖 Using AI (gemini) to plan insertions...
✓ Using Gemini model: gemini-2.0-flash-exp
✓ AI suggested 4 insertions
```

---

## 🎯 How AI Planning Works

### Hybrid Architecture (Correct)
```
B-roll Videos
  ↓ (BLIP vision model - offline)
B-roll Captions: "Mumbai street food stall, utensils visible..."
  
A-roll Video  
  ↓ (Whisper - offline)
Transcript: "آپ کو پتہ ہے؟ مومبہی جسی سٹی میں..."

Both → Gemini AI
  ↓ (AI understands both languages + context)
  ↓ (AI reasons: "Mumbai mentioned → street food B-roll fits")
  
Insertions with reasoning ✓
```

### What AI Receives

The AI gets:
```
TRANSCRIPT (from Whisper):
[0.0s - 1.0s] آپ کو پتہ ہے?
[1.0s - 4.2s] مومبہی جسی سٹی میں...

B-ROLL CAPTIONS (from BLIP):
broll_1: "An empty street food stall with cooking utensils on the counter, signboards visible in background" (5.2s)
broll_2: "Food containers on a table near window with natural daylight" (4.8s)

AI Task: Decide where to insert each B-roll
```

**AI Response:**
```json
{
  "insertions": [
    {
      "start_sec": 1.5,
      "broll_id": "broll_1",
      "reason": "Speaker mentions Mumbai city, showing street food context"
    }
  ]
}
```

---

## 📊 Mode Comparison

| Feature | AI Planning | Offline Embeddings |
|---------|-------------|-------------------|
| **Language Handling** | ✅ Excellent (cross-language) | ⚠️ Same language only |
| **Accuracy** | ⭐⭐⭐⭐⭐ (95%) | ⭐⭐⭐ (75%) |
| **Context Awareness** | ✅ Understands semantics | ❌ Vector similarity only |
| **Speed** | 🐇 Fast (2-5s) | 🐇 Fast (3-10s) |
| **Cost** | 💰 Free (Gemini) / Paid (OpenAI) | 💰 Free |
| **Quota Limits** | 15 RPM (Gemini Free) | ∞ Unlimited |
| **Internet Required** | ✅ Yes | ❌ No (after download) |

---

## 🔧 Configuration Options

### Using OpenAI Instead of Gemini

```env
API_PROVIDER=openai
OPENAI_API_KEY=sk-proj-...
```

**OpenAI Models:**
- `gpt-4o-mini` (default) - Fast and cheap
- `gpt-4o` - Best quality, more expensive

### Fallback Strategy

If AI planning fails, the system automatically falls back to offline mode:

```python
try:
    ai_planner = AIInsertionPlanner(provider="gemini")
    insertions = ai_planner.plan_insertions(...)
except Exception as e:
    print("AI planning failed, using offline embeddings...")
    # Falls back to BLIP + sentence-transformers
```

---

## 🐛 Troubleshooting

### "No available Gemini models found"

**Cause:** API key quota exhausted or invalid

**Solution:**
```bash
# Check your models
python backend/check_gemini_models.py

# Get a new API key
# Visit: https://aistudio.google.com/apikey
```

### "429 RESOURCE_EXHAUSTED"

**Cause:** Hit Gemini free tier limit (15 requests/minute)

**Solutions:**
1. Wait 1 minute and retry
2. Use a different API key
3. Upgrade to paid tier
4. Switch to offline mode

### "Still getting 0 insertions"

**Cause:** AI might not find good matches

**Solution:** Lower the confidence threshold:
```env
SIMILARITY_THRESHOLD=0.5  # AI uses this as minimum confidence
```

Or check the AI's reasoning in logs to understand why it didn't suggest insertions.

---

## 📈 Example: AI Planning Output

**Input:**
- Transcript: Urdu/Hindi speech about food quality
- B-rolls: 6 clips with English descriptions

**AI Analysis:**
```json
{
  "insertions": [
    {
      "segment_id": 2,
      "start_sec": 4.5,
      "duration_sec": 3.0,
      "broll_id": "broll_1",
      "confidence": 0.87,
      "reason": "Speaker mentions Mumbai street food, showing empty stall context"
    },
    {
      "segment_id": 5,
      "start_sec": 14.2,
      "duration_sec": 2.5,
      "broll_id": "broll_3",
      "confidence": 0.82,
      "reason": "Discussion of food hygiene, showing uncovered food at stall"
    }
  ]
}
```

**Output Timeline:**
```
[4.5s - 7.5s] broll_1.mp4 (confidence: 87%)
    Context: "मोमबहی جसی سٹی میں..."
    Reason: Speaker mentions Mumbai street food

[14.2s - 16.7s] broll_3.mp4 (confidence: 82%)
    Context: "हाइजീन की बात..."
    Reason: Discussion of food hygiene
```

---

## 🎓 Best Practices

### 1. Provide Good B-roll Descriptions

**Bad:**
```json
"metadata": "Food video"
```

**Good:**
```json
"metadata": "Mumbai street food context shot with closed stalls, utensils visible, establishing urban food culture"
```

The more detailed your metadata, the better the AI can match!

### 2. Use Transcription Provider Strategically

**For Urdu/Hindi content:**
```env
TRANSCRIPTION_PROVIDER=offline  # Whisper understands Hindi/Urdu
OFFLINE_WHISPER_MODEL=base
```

**For English content:**
```env
TRANSCRIPTION_PROVIDER=gemini  # Faster for English
```

### 3. Test Before Full Pipeline

```bash
# Test AI planning module directly
cd backend/planning
python ai_planner.py
```

### 4. Monitor API Usage

Check Gemini usage at: https://aistudio.google.com/app/apikey

---

## 🚀 Next Steps

1. **Run the test script** to verify your API key works
2. **Update your .env** to use AI planning
3. **Run the pipeline** with your videos
4. **Check the timeline** to see AI reasoning

**Questions?** Check [README.md](../README.md) or [OFFLINE_MODE_GUIDE.md](../OFFLINE_MODE_GUIDE.md)

---

## 💡 Summary

**Use AI Planning When:**
- ✅ Cross-language matching needed (Urdu → English)
- ✅ Want intelligent, context-aware insertions
- ✅ Have Gemini/OpenAI API key
- ✅ Internet connection available

**Use Offline Mode When:**
- ✅ No internet / API quota concerns
- ✅ Same language matching (English → English)
- ✅ Privacy concerns
- ✅ High-volume batch processing

**Best of Both:**
```env
# Use offline transcription + AI planning
TRANSCRIPTION_PROVIDER=offline
API_PROVIDER=gemini
```

This gives you free transcription + intelligent AI planning! 🎉

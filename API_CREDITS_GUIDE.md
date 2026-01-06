# 🆓 Free API Credits Guide for Smart B-Roll Inserter

This guide explains how to get **free API credits** for both **OpenAI** and **Google Gemini** to run the Smart B-Roll Inserter project.

---

## 📋 Quick Comparison

| Feature | OpenAI | Google Gemini |
|---------|--------|---------------|
| Free Tier | $5-18 credits (new users) | **Completely FREE** |
| Vision Model | GPT-4o ($5/1M tokens) | Gemini 1.5 Flash (FREE) |
| Whisper (Audio) | $0.006/minute | Use Gemini for transcription |
| Embeddings | $0.02/1M tokens | FREE embeddings |
| Best For | High accuracy | Free usage |

---

## 🌟 Option 1: Google Gemini (RECOMMENDED - 100% FREE)

Google Gemini offers **generous free tiers** that are perfect for this project.

### Step 1: Get Your API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the generated key

### Step 2: Free Limits (Very Generous!)

| Model | Free Limit | Use Case |
|-------|------------|----------|
| Gemini 1.5 Flash | **1,500 requests/day** | Vision captioning |
| Gemini 1.5 Pro | **50 requests/day** | Complex analysis |
| Text Embedding | **1,500 requests/day** | Semantic matching |

**This is MORE than enough for the project!**

### Step 3: Configure in .env

```env
API_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_VISION_MODEL=gemini-1.5-flash
GEMINI_EMBEDDING_MODEL=models/text-embedding-004
```

### Gemini Free Tier Benefits:
- ✅ **No credit card required**
- ✅ **1,500 requests/day** for vision and embeddings
- ✅ **Video understanding** (can process video files directly)
- ✅ **High quality results**

---

## 🔷 Option 2: OpenAI (Free Credits for New Users)

OpenAI provides free credits for new accounts.

### Step 1: Create Account & Get Credits

1. Go to [OpenAI Platform](https://platform.openai.com/signup)
2. Create a new account
3. Verify your phone number
4. **New users get $5-18 in free credits** (varies by region/time)

### Step 2: Get Your API Key

1. Go to [API Keys](https://platform.openai.com/api-keys)
2. Click **"Create new secret key"**
3. Name it (e.g., "broll-inserter")
4. Copy the key immediately (you won't see it again!)

### Step 3: Check Your Credits

1. Go to [Usage Page](https://platform.openai.com/usage)
2. See your remaining credits and usage

### Step 4: Configure in .env

```env
API_PROVIDER=openai
OPENAI_API_KEY=sk-your_api_key_here
OPENAI_VISION_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_WHISPER_MODEL=whisper-1
```

### OpenAI Pricing (After Free Credits):

| Model | Price | Typical Usage |
|-------|-------|---------------|
| GPT-4o | $5/1M input tokens | ~$0.01 per B-roll caption |
| text-embedding-3-small | $0.02/1M tokens | ~$0.001 per embedding |
| Whisper | $0.006/minute | ~$0.01 for 90s video |

**Estimated cost per video:** ~$0.05-0.10

---

## 🎯 Recommended Setup for This Project

### For Maximum Free Usage (Use Gemini):

```env
# .env file
API_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_key_here

# Transcription - Use Gemini (free) instead of Whisper
TRANSCRIPTION_PROVIDER=gemini
```

### For Best Quality (Use OpenAI):

```env
# .env file  
API_PROVIDER=openai
OPENAI_API_KEY=your_openai_key_here

# Whisper for better transcription
TRANSCRIPTION_PROVIDER=openai
```

### Hybrid Setup (Best of Both):

```env
# Use Gemini for vision (free), OpenAI Whisper for transcription
API_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_key_here

TRANSCRIPTION_PROVIDER=openai
OPENAI_API_KEY=your_openai_key_here
```

---

## 💡 Tips to Minimize API Usage

1. **Cache Results**: The system caches transcripts and captions - don't regenerate unnecessarily
2. **Use Smaller Models**: `gemini-1.5-flash` is faster and uses less quota than `gemini-1.5-pro`
3. **Limit B-roll Clips**: 6 clips is optimal; more clips = more API calls
4. **Short A-roll Videos**: 30-60 seconds is ideal

---

## 🔗 Quick Links

### Google Gemini
- [Get API Key](https://aistudio.google.com/app/apikey)
- [Pricing](https://ai.google.dev/pricing)
- [Documentation](https://ai.google.dev/docs)

### OpenAI
- [Sign Up](https://platform.openai.com/signup)
- [API Keys](https://platform.openai.com/api-keys)
- [Usage & Billing](https://platform.openai.com/usage)
- [Pricing](https://openai.com/pricing)

---

## ⚠️ Important Notes

1. **API keys are secret** - Never commit them to Git or share publicly
2. **Use .env file** - Keep keys in `.env` (already in `.gitignore`)
3. **Monitor usage** - Check your usage dashboard regularly
4. **Rate limits** - Both providers have rate limits; the code handles them gracefully

---

## 🚀 Quick Start

```bash
# 1. Copy the example env file
cp backend/.env.example backend/.env

# 2. Add your API key(s) to backend/.env

# 3. For 100% free usage, use Gemini:
#    Set API_PROVIDER=gemini
#    Set GEMINI_API_KEY=your_key

# 4. Run the backend
cd backend
pip install -r requirements.txt
python app.py
```

**That's it!** With Gemini, you can run this project completely free! 🎉

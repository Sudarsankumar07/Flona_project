# Smart B-Roll Inserter – End-to-End Design & Implementation Plan

> **Purpose of this file**
> This is a **design + execution document**, not a README. It explains **what to build, why to build it, and how to build it**, step by step, exactly as expected in the assignment. You can follow this file during development over **3 days**.

---

## 1. Problem Understanding (What are we solving?)

UGC / talking-head videos often feel boring if they only show a person speaking. Editors manually insert **B-roll clips** (product shots, lifestyle visuals, screen recordings) at the right moments to improve engagement.

### Goal

Build a system that:

1. Understands **what is being said** in an A-roll video and *when*
2. Understands **what each B-roll clip shows**
3. Automatically decides:

   * **Where** to insert B-roll
   * **Which** B-roll to insert
   * **Why** it fits that moment
4. Outputs a **structured timeline plan (JSON)**
5. Optionally renders the final video

This is **planning + reasoning**, not just video stitching.

---

## 2. High-Level System Overview

The system has **4 major layers**:

1. **Input Layer** – Upload videos
2. **Understanding Layer** – Convert video → text meaning
3. **Decision Layer** – Match A-roll moments with B-roll
4. **Output Layer** – Timeline JSON + optional final video

---

## 3. Project Folder Structure (Design View)

```text
smart-broll-inserter/
│
├── backend/
│   ├── app.py                    # FastAPI entry point
│   ├── config.py                 # API keys & settings
│   │
│   ├── ingestion/
│   │   ├── aroll_ingest.py        # Handle A-roll upload
│   │   └── broll_ingest.py        # Handle B-roll uploads
│   │
│   ├── transcription/
│   │   └── transcriber.py         # A-roll → timestamped transcript
│   │
│   ├── understanding/
│   │   ├── broll_caption.py       # B-roll → text description
│   │   └── embeddings.py          # Text → vector embeddings
│   │
│   ├── matching/
│   │   └── matcher.py             # Semantic matching logic
│   │
│   ├── planning/
│   │   └── timeline_planner.py    # Final timeline JSON
│   │
│   ├── rendering/ (optional)
│   │   └── video_renderer.py      # ffmpeg stitching
│   │
│   └── schemas/
│       └── timeline_schema.py     # Output structure
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── UploadPanel.jsx
│   │   │   ├── TranscriptView.jsx
│   │   │   └── TimelineView.jsx
│   │   └── App.jsx
│
├── artifacts/
│   ├── transcripts/
│   ├── captions/
│   └── timeline.json
│
└── design/
    └── system_design.md           # (THIS FILE)
```

---

## 4. Detailed Component Design

### 4.1 A‑Roll Understanding (Speech → Meaning)

**Input:** A-roll video (30–90 sec)

**Steps:**

1. Extract audio from video
2. Convert speech → text
3. Split transcript into **timestamped segments** (sentence-level)

**Output Example:**

```json
[
  {
    "start": 4.2,
    "end": 7.8,
    "text": "This coffee maker gives a rich crema"
  }
]
```

**Why this matters:**

* Without timestamps, we cannot decide *when* to insert B-roll

---

### 4.2 B‑Roll Understanding (Video → Description)

Each B-roll clip must have a **text meaning**.

**Possible sources:**

* File name (fallback)
* Manually added metadata
* Auto-generated captions using vision + LLM

**Output Example:**

```json
{
  "broll_id": "broll_03",
  "description": "Close-up of coffee pouring with visible crema"
}
```

**Key Rule:**

> If the system cannot explain what a B-roll clip shows in text, it cannot match it intelligently.

---

### 4.3 Embeddings Layer (Text → Vectors)

Convert both:

* A-roll transcript segments
* B-roll descriptions

into **vector embeddings**.

**Why?**

* Embeddings allow **semantic similarity** (meaning-based matching)
* Avoids keyword-only logic

**Example:**

* A-roll: "rich crema forms on top"
* B-roll: "espresso crema close-up"
  → High similarity score

---

### 4.4 Matching Logic (Core Intelligence)

This is the **heart of the project**.

#### Rules to implement:

1. Do NOT insert B-roll too frequently
2. Avoid emotional / critical speaking moments
3. Prefer moments where visuals **add value**
4. Choose highest semantic similarity B-roll
5. Enforce duration limits (2–4 seconds)

**Pseudo logic:**

```text
for each transcript segment:
    if segment is visually useful:
        find best matching b-roll
        if similarity > threshold:
            schedule insertion
```

---

### 4.5 Timeline Planning (Final Output)

**Final Output: JSON plan**

```json
{
  "aroll_duration": 65,
  "insertions": [
    {
      "start_sec": 12.5,
      "duration_sec": 2.5,
      "broll_id": "broll_03",
      "confidence": 0.81,
      "reason": "Speaker talks about coffee texture"
    }
  ]
}
```

This JSON is **the main deliverable**.

---

### 4.6 Optional Rendering (ffmpeg)

If implemented:

* Keep **A-roll audio untouched**
* Overlay B-roll video visually
* Focus on correctness, not transitions

This step is optional but adds bonus signal.

---

## 5. Frontend Design (Minimal but Functional)

### Features:

* Upload A-roll
* Upload multiple B-roll clips
* Trigger processing
* View:

  * Transcript with timestamps
  * Proposed B-roll insertions

**No editing required**, only visualization.

---

## 6. 3-Day Execution Plan

### Day 1 – Foundations

* A-roll upload + transcription
* B-roll upload + descriptions
* Save artifacts

### Day 2 – Intelligence

* Embeddings
* Semantic matching
* Timeline planner

### Day 3 – UI + Polish

* React UI
* Timeline visualization
* Optional video rendering

---

## 7. What Evaluators Care About

They are NOT judging:

* Perfect UI
* Fancy transitions

They ARE judging:

* Clear reasoning
* Smart matching logic
* Clean structure
* Ability to connect language → video time

---

## 8. Final Note

If your **timeline JSON makes sense to a human editor**, your solution is correct.

> Think like an editor, implement like an engineer.

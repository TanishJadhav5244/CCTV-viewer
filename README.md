# CCTV Deep Learning Analytics

A professional-grade, AI-powered surveillance analytics system combining **YOLOv8 instance segmentation** and **OpenAI CLIP** multimodal embeddings for natural-language object search over CCTV footage.

## What It Does

You point it at any video (file, RTSP stream, or webcam). It runs AI on every frame to detect objects, saves a cropped image of each one, and encodes them into search vectors. Then you can search your footage using plain English — *"person with red bag"*, *"black car near entrance"* — and it finds the most visually matching detections instantly.

---

## Features

- 🎯 **YOLOv8-seg** — real-time instance segmentation of people, vehicles, objects
- 🔍 **CLIP Text Search** — find objects with natural language: *"man in red shirt"*, *"black car"*
- ⚡ **FAISS Vector Index** — instant cosine-similarity search across thousands of detections
- 📊 **Live Progress Dashboard** — frames processed, FPS, detection count, animated progress bar
- 📈 **Analytics** — object class distribution chart, total detections, index size
- 🖥️ **Premium Dark UI** — glassmorphism dashboard, no framework required
- 🧩 **Zero Infrastructure** — SQLite + FAISS, runs fully locally

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (Browser)                        │
│  index.html  +  style.css  +  app.js                        │
│                                                             │
│  ┌──────────┐ ┌──────────────┐ ┌──────────┐ ┌───────────┐  │
│  │  Search  │ │ Process Video│ │Analytics │ │Detections │  │
│  │          │ │  + Progress  │ │  Chart   │ │  Gallery  │  │
│  └──────────┘ └──────────────┘ └──────────┘ └───────────┘  │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP REST (FastAPI @ :8000)
┌─────────────────────▼───────────────────────────────────────┐
│                    BACKEND (Python)                          │
│                                                             │
│  main.py ──────── API routes + static file serving          │
│      │                                                      │
│      ├── processor.py ── background thread, reads video     │
│      │       │           frames via OpenCV                  │
│      │       ├── YOLOv8 ── detects & crops each object      │
│      │       └── CLIP ──── encodes crop → 512-dim vector    │
│      │                                                      │
│      └── database.py                                        │
│              ├── SQLite ── label, confidence, frame_no,     │
│              │             timestamp, bbox, crop_path       │
│              └── FAISS ──  512-dim cosine similarity index  │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Flow — One Video Frame

```
Frame N from video
    │
    ▼ YOLOv8
  [ person 92% ]  [ car 87% ]  [ bag 74% ]
    │                  │              │
    ▼ (each object)    ▼              ▼
  Crop JPEG → saved to data/crops/
    │
    ▼ CLIP
  512-dim float32 vector  (semantic meaning of the image)
    │
    ├──► FAISS.add(vector)        ← now searchable by text
    └──► SQLite INSERT(metadata)  ← label, conf, timestamp, path
```

---

## How Text Search Works

```
You type: "person in red shirt"
    │
    ▼ CLIP encodes text → 512-dim vector
    │
    ▼ FAISS cosine similarity search
      against ALL stored crop vectors
    │
    ▼ Top-K closest matches returned
      (crops CLIP thinks visually match your query)
```

CLIP was trained on **400M image-text pairs** — it understands that *"red shirt"* should match a crop of a person wearing red, with no manual tagging needed.

---

## What the Detections Tab Shows

Each card in the Detections tab = **one detected object** from one frame:

| Field | Description |
|---|---|
| **Image** | JPEG crop of just that object, cut from the frame |
| **Label** | Object class — `person`, `car`, `bicycle`, etc. |
| **Confidence** | YOLOv8's certainty — e.g. `92.3%` |
| **Timestamp** | Real clock time when that frame was processed |

The FAISS vectors are **not shown** here — they are invisible and only used when you search.

---

## Recent Changes

### Frontend Progress Fixes (v1.1)
- **Progress visible immediately** — shows as soon as ▶ Start is clicked, no delay
- **Fixed poller stopping too early** — added `_seenRunning` guard so the status poller doesn't quit before the backend thread switches to `"running"`
- **Progress bar** — now 10px tall (was 4px, essentially invisible) with cyan glow
- **Percentage label** — shows `45%` above the bar during processing
- **Indeterminate animation** — sliding bar for live streams/webcams with no known total frames
- **FPS formatting** — shown to 1 decimal place
- **Frame counts** — formatted with `.toLocaleString()` for large numbers

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **GPU users**: Replace `faiss-cpu` with `faiss-gpu` in `requirements.txt` for faster indexing.

### 2. Configure (optional)

```bash
copy .env.example .env
# Edit .env to adjust VIDEO_SOURCE, model names, confidence threshold, etc.
```

### 3. Run

```bash
python run.py
```

Dashboard opens at **http://localhost:8000**

---

## Project Structure

```
CCTV seg/
├── backend/
│   ├── main.py          # FastAPI app & REST API
│   ├── processor.py     # YOLOv8 frame processing loop
│   ├── embedder.py      # CLIP image/text encoding
│   ├── database.py      # SQLite + FAISS storage layer
│   ├── search.py        # Similarity search helper
│   └── config.py        # Settings & paths
├── frontend/
│   ├── index.html       # Dashboard SPA
│   ├── style.css        # Glassmorphism dark UI
│   └── app.js           # Dashboard logic & API polling
├── data/
│   ├── crops/           # Detected object JPEG crops
│   ├── db/              # SQLite DB + FAISS index files
│   └── videos/          # Input video files
├── demo/                # Put your demo video here
├── requirements.txt
├── .env.example
└── run.py               # One-command launcher
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/process` | Start video processing (background thread) |
| `POST` | `/stop` | Stop processing loop |
| `GET` | `/status` | Live progress: frames, FPS, detections, status |
| `POST` | `/search` | Text query → CLIP → FAISS → top-K crops |
| `GET` | `/detections` | Recent N detections from SQLite |
| `GET` | `/stats` | Total counts + per-label breakdown |
| `GET` | `/docs` | Interactive Swagger API docs |

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `VIDEO_SOURCE` | `demo/sample.mp4` | Input source (file, RTSP, webcam index) |
| `YOLO_MODEL` | `yolov8n-seg.pt` | YOLOv8 model (`n`=fast, `x`=accurate) |
| `CLIP_MODEL` | `ViT-B-32` | CLIP model variant |
| `CONFIDENCE_THRESHOLD` | `0.4` | Minimum detection confidence |
| `FRAME_SKIP` | `5` | Process every N-th frame |

---

## Hardware Requirements

| Mode | Requirement |
|---|---|
| CPU (demo/testing) | 16 GB RAM, any modern CPU |
| GPU (real-time) | NVIDIA RTX 3060+ recommended |
| Edge AI | NVIDIA Jetson Orin |

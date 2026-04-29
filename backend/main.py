import sys
import os
import shutil
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

from config import CROPS_DIR, DATA_DIR, BASE_DIR, VIDEOS_DIR
from database import DetectionDB
from processor import VideoProcessor
from search import search_by_text

app = FastAPI(title="CCTV Deep Learning Analytics", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared state
db = DetectionDB()
processor = VideoProcessor(db)
_processing_thread: Optional[threading.Thread] = None

# ── Static files ──────────────────────────────────────────────────────────────
app.mount("/crops", StaticFiles(directory=str(CROPS_DIR)), name="crops")
FRONTEND_DIR = BASE_DIR / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/demo")
def serve_demo():
    return FileResponse(str(FRONTEND_DIR / "demo.html"))


# ── Models ────────────────────────────────────────────────────────────────────
class ProcessRequest(BaseModel):
    source: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 12


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file and start processing it automatically."""
    global _processing_thread
    if processor.progress["status"] == "running":
        raise HTTPException(status_code=400, detail="Processing already in progress.")

    # Save uploaded file
    dest = VIDEOS_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    source = str(dest)

    def run():
        processor.process(source)

    _processing_thread = threading.Thread(target=run, daemon=True)
    _processing_thread.start()
    return {"status": "started", "source": source, "filename": file.filename}


@app.post("/process")
def start_processing(req: ProcessRequest, background_tasks: BackgroundTasks):
    """Start processing from a path / RTSP URL / webcam index."""
    global _processing_thread
    if processor.progress["status"] == "running":
        raise HTTPException(status_code=400, detail="Processing already in progress.")

    source = req.source
    if not source.startswith("rtsp://") and not source.isdigit():
        p = Path(source)
        if not p.is_absolute():
            p = BASE_DIR / source
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {p}")
        source = str(p)

    def run():
        processor.process(source)

    _processing_thread = threading.Thread(target=run, daemon=True)
    _processing_thread.start()
    return {"status": "started", "source": source}


@app.post("/stop")
def stop_processing():
    processor.stop()
    return {"status": "stopping"}


@app.get("/status")
def get_status():
    return processor.progress


@app.post("/search")
def search(req: SearchRequest):
    stats = db.get_stats()
    if stats["index_size"] == 0:
        return {"results": [], "message": "No detections indexed yet. Process a video first."}
    results = search_by_text(req.query, db, top_k=req.top_k)
    return {"query": req.query, "results": results}


@app.get("/detections")
def get_recent_detections(limit: int = 50):
    return db.get_recent_detections(limit=limit)


@app.get("/stats")
def get_stats():
    return db.get_stats()


@app.on_event("shutdown")
def on_shutdown():
    processor.stop()
    db.close()

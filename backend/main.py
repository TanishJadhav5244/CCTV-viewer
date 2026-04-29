import sys
import os
import io
import csv
import shutil
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
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
app.mount("/data/videos", StaticFiles(directory=str(VIDEOS_DIR)), name="videos")


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
    label: Optional[str] = None   # e.g. 'person'
    color: Optional[str] = None   # e.g. 'red'


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
        print(f"[Debug] Checking path: {p}")
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
    results = search_by_text(
        req.query, db,
        top_k=req.top_k,
        label=req.label,
        color=req.color,
    )
    return {"query": req.query, "results": results}


@app.get("/detections")
def get_recent_detections(
    limit: int = 50,
    label: Optional[str] = None,
    color: Optional[str] = None,
    source: Optional[str] = None,
    gender: Optional[str] = None,
    has_hat: Optional[bool] = None,
    has_bag: Optional[bool] = None,
):
    return db.get_recent_detections(
        limit=limit,
        label=label,
        color=color,
        source=source,
        gender=gender,
        has_hat=has_hat,
        has_bag=has_bag,
    )


@app.get("/stats")
def get_stats():
    return db.get_stats()


@app.get("/sources")
def get_sources():
    """Return distinct video sources with detection counts."""
    return db.get_sources()


@app.get("/sessions")
def get_sessions():
    """Return all processing sessions ordered by most recent first."""
    return db.get_sessions()


@app.get("/persons")
def get_persons():
    """Return person detections grouped by attribute signature for Re-ID gallery."""
    return db.get_person_groups()


@app.get("/timeline")
def get_timeline(source: str):
    """Return detections for a specific video source, ordered by video_time."""
    rows = db.get_timeline(source)
    if not rows:
        return {"source": source, "detections": [], "duration": 0}
    duration = max((r["video_time"] or 0) for r in rows)
    return {"source": source, "detections": rows, "duration": round(duration, 2)}


@app.get("/export/detections")
def export_detections(source: Optional[str] = None):
    """Stream all detections as a CSV file download."""
    rows = db.get_all_detections_for_export(source=source)

    def generate():
        buf = io.StringIO()
        fields = ["id", "timestamp", "video_src", "video_time", "frame_no",
                  "label", "confidence", "crop_path", "bbox", "attributes"]
        writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        yield buf.getvalue()
        for r in rows:
            buf = io.StringIO()
            # Flatten nested objects to strings
            r["bbox"] = str(r.get("bbox", ""))
            r["attributes"] = str(r.get("attributes", ""))
            writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
            writer.writerow(r)
            yield buf.getvalue()

    filename = "detections.csv" if not source else f"detections_{Path(source).stem}.csv"
    return StreamingResponse(
        generate(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.on_event("shutdown")
def on_shutdown():
    processor.stop()
    db.close()

import sys
import os
import io
import csv
import shutil
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, Response
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

    # ── File-type validation ───────────────────────────────────────────────
    ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv",
                          ".flv", ".webm", ".m4v", ".ts", ".mts"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    if processor.progress["status"] == "running":
        raise HTTPException(status_code=400, detail="Processing already in progress. Stop the current job first.")

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
    """Return full processor progress including ETA, resolution and detection rate."""
    p = processor.progress.copy()
    # Compute a human-friendly ETA string
    eta = p.get("eta_seconds")
    if eta is not None and eta > 0:
        m, s = divmod(int(eta), 60)
        p["eta_label"] = f"{m}m {s}s" if m else f"{s}s"
    else:
        p["eta_label"] = None
    return p


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


@app.get("/frame")
def get_latest_frame():
    """Return the latest processed frame as JPEG for live preview."""
    frame_bytes = processor.get_latest_frame()
    if frame_bytes is None:
        raise HTTPException(status_code=204, detail="No frame available")
    return Response(content=frame_bytes, media_type="image/jpeg")


@app.delete("/sessions/{session_id}")
def delete_session(session_id: int):
    """Delete a processing session and all its associated detections."""
    deleted = db.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


@app.delete("/detections/clear")
def clear_detections(confirm: bool = Query(False)):
    """Wipe ALL detections and reset the vector index. Requires confirm=true."""
    if not confirm:
        raise HTTPException(status_code=400, detail="Pass ?confirm=true to proceed")
    db.clear_all()
    return {"status": "cleared"}


# ── Phase 5: Reverse Image Search & Dashboard ─────────────────────────────────

@app.get("/search/similar/{detection_id}")
def find_similar(detection_id: int, top_k: int = 12):
    """Find detections visually similar to a given detection by its FAISS index."""
    # FAISS index is 0-based; detection IDs are 1-based
    faiss_idx = detection_id - 1
    if faiss_idx < 0 or faiss_idx >= db.index.ntotal:
        raise HTTPException(status_code=404, detail="Detection not in vector index")
    import numpy as np
    import faiss
    vec = db.index.reconstruct(faiss_idx).reshape(1, -1).astype("float32")
    faiss.normalize_L2(vec)
    results = db.search(vec, top_k=top_k + 1)
    # Exclude the source detection itself
    results = [r for r in results if r.get("id") != detection_id][:top_k]
    return {"source_id": detection_id, "results": results}


@app.post("/search/image")
async def search_by_image(file: UploadFile = File(...), top_k: int = 12):
    """Upload an image crop and find visually similar detections."""
    from embedder import embedder
    from PIL import Image
    import io as _io
    embedder.load()
    raw = await file.read()
    img = Image.open(_io.BytesIO(raw)).convert("RGB")
    vec = embedder.embed_image_pil(img)
    results = db.search(vec, top_k=top_k)
    return {"results": results}


@app.get("/heatmap")
def get_heatmap(source: Optional[str] = None, label: Optional[str] = None, grid: int = 20):
    """Return a spatial detection heatmap as a grid of (col, row, count) cells."""
    import sqlite3 as _sq
    import json as _json
    conn = _sq.connect(str(db.sqlite_path))
    cur = conn.cursor()
    where = []
    params: list = []
    if source:
        where.append("video_src = ?"); params.append(source)
    if label:
        where.append("label = ?"); params.append(label)
    where.append("bbox IS NOT NULL AND bbox != ''")
    sql = "SELECT bbox FROM detections"
    if where:
        sql += " WHERE " + " AND ".join(where)
    cur.execute(sql, params)
    rows = cur.fetchall()

    # ── Detect actual video resolution dynamically ────────────────────────────
    # Try to get width/height from the latest processing session for this source
    vid_w, vid_h = 0, 0
    if source:
        cur.execute(
            "SELECT video_width, video_height FROM sessions WHERE source = ? ORDER BY id DESC LIMIT 1",
            (source,)
        )
        row = cur.fetchone()
        if row and row[0] and row[1]:
            vid_w, vid_h = row[0], row[1]

    conn.close()

    # If still unknown, auto-detect from bbox extents (fallback)
    if vid_w == 0 or vid_h == 0:
        max_x2, max_y2 = 0, 0
        for (bbox_str,) in rows:
            try:
                x1, y1, x2, y2 = _json.loads(bbox_str)
                max_x2 = max(max_x2, x2)
                max_y2 = max(max_y2, y2)
            except Exception:
                continue
        vid_w = max_x2 if max_x2 > 0 else 1920
        vid_h = max_y2 if max_y2 > 0 else 1080

    cells: dict = {}
    total = 0
    for (bbox_str,) in rows:
        try:
            x1, y1, x2, y2 = _json.loads(bbox_str)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            # Normalise to actual video resolution; clamp to [0,1)
            nx = max(0.0, min(0.9999, cx / vid_w))
            ny = max(0.0, min(0.9999, cy / vid_h))
            col = int(nx * grid)
            row = int(ny * grid)
            cells[(col, row)] = cells.get((col, row), 0) + 1
            total += 1
        except Exception:
            continue

    result = [{"col": c, "row": r, "count": cnt} for (c, r), cnt in cells.items()]
    return {"grid": grid, "total": total, "cells": result, "video_width": vid_w, "video_height": vid_h}


@app.get("/dashboard")
def get_dashboard():
    """Rich overview: stats, label breakdown, recent sessions, top labels."""
    import sqlite3 as _sq
    stats = db.get_stats()
    sessions = db.get_sessions()
    conn = _sq.connect(str(db.sqlite_path))
    cur = conn.cursor()

    # Detections in last 5 sessions
    cur.execute("""
        SELECT d.label, COUNT(*) as cnt
        FROM detections d
        ORDER BY d.id DESC LIMIT 1000
    """)
    recent_labels = {r[0]: r[1] for r in cur.fetchall()}

    # Unique sources
    cur.execute("SELECT COUNT(DISTINCT video_src) FROM detections WHERE video_src IS NOT NULL")
    unique_sources = cur.fetchone()[0] or 0

    # Average confidence
    cur.execute("SELECT AVG(confidence) FROM detections")
    avg_conf = round((cur.fetchone()[0] or 0) * 100, 1)

    # Detections today
    cur.execute("""
        SELECT COUNT(*) FROM detections
        WHERE date(timestamp) = date('now')
    """)
    today_count = cur.fetchone()[0] or 0

    conn.close()

    recent_sessions = sessions[:5]
    return {
        "total_detections": stats["total_detections"],
        "index_size": stats["index_size"],
        "label_counts": stats["label_counts"],
        "unique_sources": unique_sources,
        "avg_confidence_pct": avg_conf,
        "today_count": today_count,
        "recent_sessions": recent_sessions,
        "processing_status": processor.progress.get("status", "idle"),
    }


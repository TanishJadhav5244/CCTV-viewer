"""
api/index.py — Vercel serverless entrypoint for CCTV Analytics.

Only read-only endpoints are active here.
Video processing (upload/process) requires the local backend.
"""
import sys
import os
from pathlib import Path
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent      # project root  (f:/mini/CCTV seg)
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(ROOT))

os.environ.setdefault("SERVERLESS", "1")

# ── FastAPI app ───────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(
    title="CCTV Deep Learning Analytics",
    version="2.0.0",
    description="Serverless read-only API. Video processing runs locally.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── DB bootstrap (config + sqlite + faiss only — no torch/cv2) ───────────────
DB_AVAILABLE = False
db = None

try:
    from config import CROPS_DIR, VIDEOS_DIR
    from database import DetectionDB
    for d in [CROPS_DIR, VIDEOS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    db = DetectionDB()
    DB_AVAILABLE = True
except Exception as _e:
    print(f"[api] DB bootstrap error: {_e}")


def _db():
    """Return db or raise 503."""
    if not DB_AVAILABLE or db is None:
        raise HTTPException(status_code=503, detail="Database not available in this deployment.")
    return db


# ── Models ────────────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    top_k: int = 12
    label: Optional[str] = None
    color: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "ok",
        "mode": "serverless",
        "db": DB_AVAILABLE,
        "note": "Video processing endpoints are disabled in serverless mode.",
    }


@app.get("/health")
def health():
    return {"status": "ok", "db_available": DB_AVAILABLE}


@app.get("/stats")
def get_stats():
    return _db().get_stats()


@app.get("/sources")
def get_sources():
    return _db().get_sources()


@app.get("/sessions")
def get_sessions():
    return _db().get_sessions()


@app.get("/persons")
def get_persons():
    return _db().get_person_groups()


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
    return _db().get_recent_detections(
        limit=limit, label=label, color=color,
        source=source, gender=gender,
        has_hat=has_hat, has_bag=has_bag,
    )


@app.get("/timeline")
def get_timeline(source: str):
    database = _db()
    rows = database.get_timeline(source)
    if not rows:
        return {"source": source, "detections": [], "duration": 0}
    duration = max((r["video_time"] or 0) for r in rows)
    return {"source": source, "detections": rows, "duration": round(duration, 2)}


@app.get("/dashboard")
def get_dashboard():
    import sqlite3 as _sq
    database = _db()
    stats = database.get_stats()
    sessions = database.get_sessions()
    conn = _sq.connect(str(database.sqlite_path))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(DISTINCT video_src) FROM detections WHERE video_src IS NOT NULL")
    unique_sources = cur.fetchone()[0] or 0
    cur.execute("SELECT AVG(confidence) FROM detections")
    avg_conf = round((cur.fetchone()[0] or 0) * 100, 1)
    cur.execute("SELECT COUNT(*) FROM detections WHERE date(timestamp) = date('now')")
    today_count = cur.fetchone()[0] or 0
    conn.close()
    return {
        "total_detections": stats["total_detections"],
        "index_size": stats["index_size"],
        "label_counts": stats["label_counts"],
        "unique_sources": unique_sources,
        "avg_confidence_pct": avg_conf,
        "today_count": today_count,
        "recent_sessions": sessions[:5],
        "processing_status": "idle (serverless)",
    }


@app.get("/heatmap")
def get_heatmap(source: Optional[str] = None, label: Optional[str] = None, grid: int = 20):
    import sqlite3 as _sq, json as _json
    database = _db()
    conn = _sq.connect(str(database.sqlite_path))
    cur = conn.cursor()
    where, params = [], []
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
    vid_w, vid_h = 1920, 1080
    if source:
        cur.execute(
            "SELECT video_width, video_height FROM processing_sessions WHERE source = ? ORDER BY id DESC LIMIT 1",
            (source,)
        )
        row = cur.fetchone()
        if row and row[0] and row[1]:
            vid_w, vid_h = row[0], row[1]
    conn.close()
    cells: dict = {}
    total = 0
    for (bbox_str,) in rows:
        try:
            x1, y1, x2, y2 = _json.loads(bbox_str)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            nx = max(0.0, min(0.9999, cx / vid_w))
            ny = max(0.0, min(0.9999, cy / vid_h))
            col, row_ = int(nx * grid), int(ny * grid)
            cells[(col, row_)] = cells.get((col, row_), 0) + 1
            total += 1
        except Exception:
            continue
    return {
        "grid": grid, "total": total,
        "cells": [{"col": c, "row": r, "count": cnt} for (c, r), cnt in cells.items()],
        "video_width": vid_w, "video_height": vid_h,
    }


@app.post("/search")
def search(req: SearchRequest):
    database = _db()
    try:
        from search import search_by_text
        stats = database.get_stats()
        if stats["index_size"] == 0:
            return {"results": [], "message": "No detections indexed yet. Process a video locally first."}
        results = search_by_text(req.query, database, top_k=req.top_k, label=req.label, color=req.color)
        return {"query": req.query, "results": results}
    except Exception as e:
        return JSONResponse(status_code=503, content={"detail": f"Search unavailable: {e}"})


@app.get("/search/similar/{detection_id}")
def find_similar(detection_id: int, top_k: int = 12):
    database = _db()
    faiss_idx = detection_id - 1
    if faiss_idx < 0 or faiss_idx >= database.index.ntotal:
        raise HTTPException(status_code=404, detail="Detection not in vector index")
    import numpy as np, faiss
    vec = database.index.reconstruct(faiss_idx).reshape(1, -1).astype("float32")
    faiss.normalize_L2(vec)
    results = database.search(vec, top_k=top_k + 1)
    results = [r for r in results if r.get("id") != detection_id][:top_k]
    return {"source_id": detection_id, "results": results}


# ── Disabled endpoints ────────────────────────────────────────────────────────
_DISABLED = JSONResponse(
    status_code=501,
    content={"detail": "Video processing is not available in serverless mode. Run the backend locally."}
)


@app.post("/upload")
async def upload_disabled():
    return _DISABLED


@app.post("/process")
async def process_disabled():
    return _DISABLED


@app.post("/stop")
async def stop_disabled():
    return _DISABLED


@app.get("/status")
def status_info():
    return {"status": "idle", "note": "Serverless mode — video processing not available."}


@app.post("/search/image")
async def image_search_disabled():
    return _DISABLED


@app.get("/frame")
async def frame_disabled():
    return _DISABLED

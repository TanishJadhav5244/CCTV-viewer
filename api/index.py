"""
api/index.py — Vercel serverless entrypoint for CCTV Analytics.

Design rules for Vercel Python runtime:
  - NO faiss, torch, cv2, ultralytics, open_clip at import time (too large / won't build)
  - NO StaticFiles mounts (read-only filesystem)
  - Frontend is served as static files directly by Vercel (see vercel.json)
  - Only pure-Python, lightweight deps: fastapi, sqlite3, json, pathlib
  - All DB / FAISS access is wrapped in try/except so missing files return graceful errors
"""

import sys
import os
import json
import sqlite3
from pathlib import Path
from typing import Optional, List

# ── Path bootstrap ────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent      # project root
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(ROOT))

os.environ.setdefault("SERVERLESS", "1")

# ── FastAPI ───────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, Query
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


# ── Lightweight SQLite helper (no faiss, no torch) ────────────────────────────
# We read the DB file directly — if it doesn't exist we return empty results.
_TMP_DIR = Path("/tmp/cctv_data")

def _sqlite_path() -> Optional[Path]:
    """Return path to SQLite DB, checking /tmp (Vercel) and local data/ dir."""
    candidates = [
        _TMP_DIR / "db" / "detections.db",
        ROOT / "data" / "db" / "detections.db",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _get_conn():
    """Open a read-only sqlite3 connection or raise 503."""
    p = _sqlite_path()
    if p is None:
        raise HTTPException(
            status_code=503,
            detail="No detection database found. Run the backend locally and upload data.",
        )
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_dict(row) -> dict:
    d = dict(row)
    for key in ("bbox", "attributes"):
        if key in d and d[key]:
            try:
                d[key] = json.loads(d[key])
            except Exception:
                pass
    return d


# ── Models ────────────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    top_k: int = 12
    label: Optional[str] = None
    color: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    db_found = _sqlite_path() is not None
    return {
        "status": "ok",
        "mode": "serverless",
        "db_available": db_found,
        "note": "Video processing is not available in serverless mode. Run the backend locally to process videos.",
    }


@app.get("/health")
def health():
    return {"status": "ok", "db_available": _sqlite_path() is not None}


@app.get("/status")
def status_info():
    return {
        "status": "idle",
        "note": "Serverless mode — video processing not available. Run the local backend for processing.",
    }


@app.get("/stats")
def get_stats():
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM detections")
        total = cur.fetchone()[0]
        cur.execute("SELECT label, COUNT(*) cnt FROM detections GROUP BY label ORDER BY cnt DESC")
        counts = {r[0]: r[1] for r in cur.fetchall()}
        conn.close()
        return {"total_detections": total, "label_counts": counts, "index_size": total}
    except HTTPException:
        raise
    except Exception as e:
        return {"total_detections": 0, "label_counts": {}, "index_size": 0, "error": str(e)}


@app.get("/sources")
def get_sources():
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT video_src, COUNT(*) cnt FROM detections "
            "WHERE video_src IS NOT NULL GROUP BY video_src ORDER BY cnt DESC"
        )
        rows = cur.fetchall()
        conn.close()
        return [{"source": r[0], "count": r[1]} for r in rows]
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=503, content={"detail": str(e)})


@app.get("/sessions")
def get_sessions():
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("SELECT * FROM processing_sessions ORDER BY id DESC")
        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except HTTPException:
        raise
    except Exception:
        return []


@app.get("/persons")
def get_persons():
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT
                id, crop_path, video_src, video_time, confidence, attributes,
                json_extract(attributes, '$.upper_color') AS upper_color,
                json_extract(attributes, '$.lower_color') AS lower_color,
                json_extract(attributes, '$.gender')      AS gender,
                json_extract(attributes, '$.has_hat')     AS has_hat,
                json_extract(attributes, '$.has_bag')     AS has_bag
            FROM detections WHERE label = 'person' ORDER BY id ASC
        """)
        rows = cur.fetchall()
        conn.close()

        groups: dict = {}
        for r in rows:
            key = (
                r["upper_color"] or "unknown",
                r["lower_color"] or "unknown",
                r["gender"]      or "unknown",
                bool(r["has_hat"]),
                bool(r["has_bag"]),
            )
            if key not in groups:
                groups[key] = {
                    "upper_color": r["upper_color"] or "unknown",
                    "lower_color": r["lower_color"] or "unknown",
                    "gender":      r["gender"]      or "unknown",
                    "has_hat":     bool(r["has_hat"]),
                    "has_bag":     bool(r["has_bag"]),
                    "crop_path":   r["crop_path"],
                    "video_src":   r["video_src"],
                    "video_time":  r["video_time"],
                    "count":       0,
                    "ids":         [],
                }
            groups[key]["count"] += 1
            groups[key]["ids"].append(r["id"])

        return sorted(groups.values(), key=lambda g: g["count"], reverse=True)
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=503, content={"detail": str(e)})


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
    try:
        conn = _get_conn()
        cur = conn.cursor()
        query = "SELECT * FROM detections"
        params: list = []
        where: list = []
        if label:  where.append("label = ?");     params.append(label)
        if source: where.append("video_src = ?"); params.append(source)
        if color:
            where.append("(json_extract(attributes,'$.upper_color')=? OR json_extract(attributes,'$.lower_color')=?)")
            params.extend([color, color])
        if gender:
            where.append("json_extract(attributes,'$.gender')=?"); params.append(gender)
        if has_hat is not None:
            where.append("json_extract(attributes,'$.has_hat')=?"); params.append(1 if has_hat else 0)
        if has_bag is not None:
            where.append("json_extract(attributes,'$.has_bag')=?"); params.append(1 if has_bag else 0)
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        cur.execute(query, params)
        rows = cur.fetchall()
        conn.close()
        return [_row_to_dict(r) for r in rows]
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=503, content={"detail": str(e)})


@app.get("/timeline")
def get_timeline(source: str):
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, video_time, label, confidence, crop_path, attributes "
            "FROM detections WHERE video_src = ? ORDER BY video_time ASC",
            (source,)
        )
        rows = cur.fetchall()
        conn.close()
        result = []
        for r in rows:
            result.append({
                "id": r["id"], "video_time": r["video_time"],
                "label": r["label"], "confidence": round(float(r["confidence"]), 4),
                "crop_path": r["crop_path"],
                "attributes": json.loads(r["attributes"] or "{}"),
            })
        duration = max((r["video_time"] or 0 for r in rows), default=0)
        return {"source": source, "detections": result, "duration": round(float(duration), 2)}
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=503, content={"detail": str(e)})


@app.get("/dashboard")
def get_dashboard():
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM detections")
        total = cur.fetchone()[0]
        cur.execute("SELECT label, COUNT(*) cnt FROM detections GROUP BY label ORDER BY cnt DESC")
        label_counts = {r[0]: r[1] for r in cur.fetchall()}
        cur.execute("SELECT COUNT(DISTINCT video_src) FROM detections WHERE video_src IS NOT NULL")
        unique_sources = cur.fetchone()[0] or 0
        cur.execute("SELECT AVG(confidence) FROM detections")
        avg_conf = round((cur.fetchone()[0] or 0) * 100, 1)
        cur.execute("SELECT COUNT(*) FROM detections WHERE date(timestamp) = date('now')")
        today_count = cur.fetchone()[0] or 0
        cur.execute("SELECT * FROM processing_sessions ORDER BY id DESC LIMIT 5")
        sessions = [dict(r) for r in cur.fetchall()]
        conn.close()
        return {
            "total_detections": total,
            "index_size": total,
            "label_counts": label_counts,
            "unique_sources": unique_sources,
            "avg_confidence_pct": avg_conf,
            "today_count": today_count,
            "recent_sessions": sessions,
            "processing_status": "idle (serverless)",
        }
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=503, content={"detail": str(e)})


@app.get("/heatmap")
def get_heatmap(source: Optional[str] = None, label: Optional[str] = None, grid: int = 20):
    try:
        conn = _get_conn()
        cur = conn.cursor()
        where, params = ["bbox IS NOT NULL AND bbox != ''"], []
        if source: where.append("video_src = ?"); params.append(source)
        if label:  where.append("label = ?");     params.append(label)
        sql = "SELECT bbox FROM detections WHERE " + " AND ".join(where)
        cur.execute(sql, params)
        rows = cur.fetchall()
        vid_w, vid_h = 1920, 1080
        if source:
            cur.execute(
                "SELECT video_width, video_height FROM processing_sessions "
                "WHERE source = ? ORDER BY id DESC LIMIT 1", (source,)
            )
            row = cur.fetchone()
            if row and row[0] and row[1]:
                vid_w, vid_h = row[0], row[1]
        conn.close()
        cells: dict = {}
        total = 0
        for (bbox_str,) in rows:
            try:
                x1, y1, x2, y2 = json.loads(bbox_str)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                nx = max(0.0, min(0.9999, cx / vid_w))
                ny = max(0.0, min(0.9999, cy / vid_h))
                col_, row_ = int(nx * grid), int(ny * grid)
                cells[(col_, row_)] = cells.get((col_, row_), 0) + 1
                total += 1
            except Exception:
                continue
        return {
            "grid": grid, "total": total,
            "cells": [{"col": c, "row": r, "count": cnt} for (c, r), cnt in cells.items()],
            "video_width": vid_w, "video_height": vid_h,
        }
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=503, content={"detail": str(e)})


# ── Search: disabled on Vercel (needs CLIP/torch) ─────────────────────────────
_SEARCH_DISABLED = JSONResponse(
    status_code=501,
    content={
        "detail": "Semantic search requires CLIP/torch which are not available in serverless mode. "
                  "Run the backend locally for full search functionality."
    }
)


@app.post("/search")
async def search_disabled():
    return _SEARCH_DISABLED


@app.get("/search/similar/{detection_id}")
async def similar_disabled(detection_id: int):
    return _SEARCH_DISABLED


@app.post("/search/image")
async def image_search_disabled():
    return _SEARCH_DISABLED


# ── Processing: not available serverless ──────────────────────────────────────
_PROC_DISABLED = JSONResponse(
    status_code=501,
    content={"detail": "Video processing is not available in serverless mode. Run the local backend."}
)


@app.post("/upload")
async def upload_disabled():   return _PROC_DISABLED

@app.post("/process")
async def process_disabled():  return _PROC_DISABLED

@app.post("/stop")
async def stop_disabled():     return _PROC_DISABLED

@app.get("/frame")
async def frame_disabled():    return _PROC_DISABLED

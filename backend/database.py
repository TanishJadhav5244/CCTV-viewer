"""
database.py — SQLite + FAISS backend (Fallback Mode).

This implementation provides a "zero-infrastructure" replacement for the 
PostgreSQL setup, using local files for both metadata and vector search.
"""

import sqlite3
import json
import os
import faiss
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional

from config import SQLITE_PATH, FAISS_INDEX_PATH, EMBEDDING_DIM

class DetectionDB:
    def __init__(self):
        self.sqlite_path = str(SQLITE_PATH)
        self.index_path = str(FAISS_INDEX_PATH)
        self.dim = EMBEDDING_DIM
        
        self._init_sqlite()
        self._init_faiss()

    def _init_sqlite(self):
        """Initialize the SQLite database schema."""
        conn = sqlite3.connect(self.sqlite_path)
        cur = conn.cursor()
        # Detections table (matches PG schema minus the vector column)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                frame_no INTEGER NOT NULL,
                video_src TEXT,
                video_time REAL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                crop_path TEXT,
                bbox TEXT,
                attributes TEXT
            )
        """)
        # Migration: Add columns if they don't exist
        cur.execute("PRAGMA table_info(detections)")
        columns = [info[1] for info in cur.fetchall()]
        if "video_src" not in columns:
            cur.execute("ALTER TABLE detections ADD COLUMN video_src TEXT")
        if "video_time" not in columns:
            cur.execute("ALTER TABLE detections ADD COLUMN video_time REAL")
        # Sessions table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS processing_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                started_at TEXT,
                finished_at TEXT,
                total_frames INTEGER DEFAULT 0,
                processed_frames INTEGER DEFAULT 0,
                total_detections INTEGER DEFAULT 0,
                status TEXT DEFAULT 'running'
            )
        """)
        conn.commit()
        conn.close()
        print(f"[DB] SQLite initialized at {self.sqlite_path}")

    def _init_faiss(self):
        """Initialize the FAISS index."""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            print(f"[DB] FAISS index loaded from {self.index_path} ({self.index.ntotal} vectors)")
        else:
            # Using IndexFlatIP for cosine similarity (with normalized vectors)
            self.index = faiss.IndexFlatIP(self.dim)
            print(f"[DB] New FAISS IndexFlatIP initialized (dim={self.dim})")

    # ── Session Management ──────────────────────────────────────────────────
    def start_session(self, source: str) -> int:
        conn = sqlite3.connect(self.sqlite_path)
        cur = conn.cursor()
        started_at = datetime.utcnow().isoformat()
        cur.execute(
            "INSERT INTO processing_sessions (source, started_at, status) VALUES (?, ?, ?)",
            (source, started_at, "running")
        )
        session_id = cur.lastrowid
        conn.commit()
        conn.close()
        return session_id

    def update_session(self, session_id: int, **kwargs):
        if not kwargs: return
        conn = sqlite3.connect(self.sqlite_path)
        cur = conn.cursor()
        fields = ", ".join([f"{k} = ?" for k in kwargs.keys()])
        values = list(kwargs.values()) + [session_id]
        cur.execute(f"UPDATE processing_sessions SET {fields} WHERE id = ?", values)
        conn.commit()
        conn.close()

    def get_session(self, session_id: int) -> Optional[Dict]:
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM processing_sessions WHERE id = ?", (session_id,))
        row = cur.fetchone()
        conn.close()
        return dict(row) if row else None

    # ── Detections ─────────────────────────────────────────────────────────
    def insert_detection(
        self,
        timestamp: str,
        video_src: str,
        frame_no: int,
        video_time: float,
        label: str,
        confidence: float,
        crop_path: str,
        bbox: list,
        embedding: np.ndarray,
        attributes: dict = None,
    ) -> int:
        # 1. Save to SQLite
        conn = sqlite3.connect(self.sqlite_path)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO detections (timestamp, frame_no, video_src, video_time, label, confidence, crop_path, bbox, attributes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, frame_no, video_src, round(float(video_time), 2),
            label, round(float(confidence), 4),
            crop_path, json.dumps(bbox), json.dumps(attributes or {})
        ))
        det_id = cur.lastrowid
        conn.commit()
        conn.close()

        # 2. Save to FAISS
        # Normalize for cosine similarity
        vec = embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)
        self.index.add(vec)
        
        return det_id

    def get_recent_detections(
        self,
        limit: int = 50,
        label: str = None,
        color: str = None,
        source: str = None,
        gender: str = None,
        has_hat: bool = None,
        has_bag: bool = None,
    ) -> List[Dict]:
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        query = "SELECT * FROM detections"
        params = []
        where_clauses = []

        if label:
            where_clauses.append("label = ?")
            params.append(label)

        if source:
            where_clauses.append("video_src = ?")
            params.append(source)

        if color:
            where_clauses.append(
                "(json_extract(attributes, '$.upper_color') = ? "
                "OR json_extract(attributes, '$.lower_color') = ?)"
            )
            params.extend([color, color])

        if gender:
            where_clauses.append("json_extract(attributes, '$.gender') = ?")
            params.append(gender)

        if has_hat is not None:
            where_clauses.append("json_extract(attributes, '$.has_hat') = ?")
            params.append(1 if has_hat else 0)

        if has_bag is not None:
            where_clauses.append("json_extract(attributes, '$.has_bag') = ?")
            params.append(1 if has_bag else 0)

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        cur.execute(query, params)
        rows = cur.fetchall()
        conn.close()
        return [self._row_to_dict(r) for r in rows]

    # ── Vector Search ─────────────────────────────────────────────────────
    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 12,
        label: str = None,
        color: str = None,
    ) -> List[Dict]:
        """Cosine similarity search using FAISS + SQLite filtering."""
        if self.index.ntotal == 0:
            return []

        # Generate broad search range
        vec = query_vec.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)
        
        # If we have filters, we might need a larger initial pool because we'll filter them manually
        search_k = min(self.index.ntotal, 500 if (label or color) else top_k)
        scores, indices = self.index.search(vec, search_k)
        
        # Map indices to SQLite IDs
        # NOTE: This implementation assumes order in FAISS matches row order in SQLite.
        # This is true for a freshly created index on a single processing run.
        # IDs in SQLite (1-indexed) map to indices (0-indexed) as det_id = idx + 1
        
        results = []
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        for i in range(len(indices[0])):
            idx = int(indices[0][i])
            if idx < 0: continue
            
            det_id = idx + 1
            cur.execute("SELECT * FROM detections WHERE id = ?", (det_id,))
            row = cur.fetchone()
            if not row: continue
            
            d = self._row_to_dict(row)
            
            # Apply hard filters
            if label and d['label'] != label: continue
            if color:
                attr = json.loads(row['attributes'] or '{}')
                if attr.get('upper_color') != color and attr.get('lower_color') != color:
                    continue
            
            d["score"] = round(float(scores[0][i]), 4)
            results.append(d)
            if len(results) >= top_k:
                break
                
        conn.close()
        return results

    # ── Stats ─────────────────────────────────────────────────────────────
    def get_stats(self) -> Dict:
        conn = sqlite3.connect(self.sqlite_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM detections")
        total = cur.fetchone()[0]
        
        cur.execute("SELECT label, COUNT(*) as cnt FROM detections GROUP BY label ORDER BY cnt DESC")
        counts = {r[0]: r[1] for r in cur.fetchall()}
        
        conn.close()
        return {
            "total_detections": total,
            "label_counts": counts,
            "index_size": self.index.ntotal,
        }

    def get_timeline(self, source: str) -> List[Dict]:
        """Return all detections for a video ordered by video_time."""
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
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
                "id": r["id"],
                "video_time": r["video_time"],
                "label": r["label"],
                "confidence": round(float(r["confidence"]), 4),
                "crop_path": r["crop_path"],
                "attributes": json.loads(r["attributes"] or "{}"),
            })
        return result

    def get_sources(self) -> List[Dict]:
        """Return distinct video_src values with detection counts."""
        conn = sqlite3.connect(self.sqlite_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT video_src, COUNT(*) as count FROM detections "
            "WHERE video_src IS NOT NULL GROUP BY video_src ORDER BY count DESC"
        )
        rows = cur.fetchall()
        conn.close()
        return [{"source": r[0], "count": r[1]} for r in rows]

    def get_all_detections_for_export(self, source: str = None) -> List[Dict]:
        """Return full detection rows for CSV export."""
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        if source:
            cur.execute("SELECT * FROM detections WHERE video_src = ? ORDER BY id", (source,))
        else:
            cur.execute("SELECT * FROM detections ORDER BY id")
        rows = cur.fetchall()
        conn.close()
        return [self._row_to_dict(r) for r in rows]

    def get_sessions(self) -> List[Dict]:
        """Return all processing sessions ordered by most recent first."""
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM processing_sessions ORDER BY id DESC")
        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_person_groups(self) -> List[Dict]:
        """
        Group person detections by their attribute signature
        (upper_color, lower_color, gender, has_hat, has_bag).
        Returns one representative crop per group plus appearance count.
        """
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        # Fetch all person detections with their attribute fields extracted
        cur.execute("""
            SELECT
                id,
                crop_path,
                video_src,
                video_time,
                confidence,
                attributes,
                json_extract(attributes, '$.upper_color') AS upper_color,
                json_extract(attributes, '$.lower_color') AS lower_color,
                json_extract(attributes, '$.gender')      AS gender,
                json_extract(attributes, '$.has_hat')     AS has_hat,
                json_extract(attributes, '$.has_bag')     AS has_bag
            FROM detections
            WHERE label = 'person'
            ORDER BY id ASC
        """)
        rows = cur.fetchall()
        conn.close()

        # Group in Python: key = (upper_color, lower_color, gender, has_hat, has_bag)
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

        # Sort by appearance count descending
        return sorted(groups.values(), key=lambda g: g["count"], reverse=True)

    def save_index(self):
        """Persist the FAISS index to disk."""
        faiss.write_index(self.index, self.index_path)
        print(f"[DB] FAISS index saved to {self.index_path}")

    def close(self):
        self.save_index()

    @staticmethod
    def _row_to_dict(r) -> Dict:
        d = dict(r)
        if 'bbox' in d and d['bbox']:
            d['bbox'] = json.loads(d['bbox'])
        if 'attributes' in d and d['attributes']:
            d['attributes'] = json.loads(d['attributes'])
        return d

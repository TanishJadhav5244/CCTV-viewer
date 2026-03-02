import sqlite3
import json
import numpy as np
import faiss
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from config import (
    SQLITE_DB_PATH, FAISS_INDEX_PATH, FAISS_META_PATH, EMBEDDING_DIM
)


class DetectionDB:
    def __init__(self):
        self.conn = sqlite3.connect(str(SQLITE_DB_PATH), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        self.index, self.id_map = self._load_or_create_index()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  TEXT NOT NULL,
                video_src  TEXT,
                frame_no   INTEGER,
                label      TEXT,
                confidence REAL,
                crop_path  TEXT,
                bbox       TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS processing_sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source      TEXT,
                started_at  TEXT,
                finished_at TEXT,
                total_frames INTEGER DEFAULT 0,
                processed_frames INTEGER DEFAULT 0,
                total_detections INTEGER DEFAULT 0,
                status      TEXT DEFAULT 'running'
            )
        """)
        self.conn.commit()

    def _load_or_create_index(self):
        """Load existing FAISS index or create a new one."""
        if FAISS_INDEX_PATH.exists() and FAISS_META_PATH.exists():
            index = faiss.read_index(str(FAISS_INDEX_PATH))
            with open(FAISS_META_PATH) as f:
                id_map = json.load(f)
            return index, id_map
        # Inner product on L2-normalized vectors = cosine similarity
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        return index, []

    def save_index(self):
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        with open(FAISS_META_PATH, "w") as f:
            json.dump(self.id_map, f)

    def insert_detection(
        self,
        timestamp: str,
        video_src: str,
        frame_no: int,
        label: str,
        confidence: float,
        crop_path: str,
        bbox: list,
        embedding: np.ndarray,
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO detections (timestamp, video_src, frame_no, label, confidence, crop_path, bbox)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (timestamp, video_src, frame_no, label, round(confidence, 4),
             crop_path, json.dumps(bbox)),
        )
        self.conn.commit()
        row_id = cur.lastrowid

        # Normalize and add to FAISS
        vec = embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)
        self.index.add(vec)
        self.id_map.append(row_id)

        # Persist periodically
        if len(self.id_map) % 100 == 0:
            self.save_index()

        return row_id

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Dict]:
        if self.index.ntotal == 0:
            return []

        vec = query_vec.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.id_map):
                continue
            row_id = self.id_map[idx]
            row = self.conn.execute(
                "SELECT * FROM detections WHERE id = ?", (row_id,)
            ).fetchone()
            if row:
                results.append({
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "label": row["label"],
                    "confidence": row["confidence"],
                    "crop_path": row["crop_path"],
                    "score": round(float(score), 4),
                })
        return results

    def get_recent_detections(self, limit: int = 50) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> Dict:
        total = self.conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        labels = self.conn.execute(
            "SELECT label, COUNT(*) as cnt FROM detections GROUP BY label ORDER BY cnt DESC"
        ).fetchall()
        return {
            "total_detections": total,
            "label_counts": {r["label"]: r["cnt"] for r in labels},
            "index_size": self.index.ntotal,
        }

    def start_session(self, source: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO processing_sessions (source, started_at, status) VALUES (?, ?, 'running')",
            (source, datetime.now().isoformat()),
        )
        self.conn.commit()
        return cur.lastrowid

    def update_session(self, session_id: int, **kwargs):
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [session_id]
        self.conn.execute(f"UPDATE processing_sessions SET {sets} WHERE id = ?", vals)
        self.conn.commit()

    def get_session(self, session_id: int) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT * FROM processing_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    def close(self):
        self.save_index()
        self.conn.close()

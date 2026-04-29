"""
database.py — PostgreSQL + pgvector backend.

Replaces the old SQLite + FAISS implementation.
Uses SQLAlchemy for ORM and pgvector for 512-dim cosine search.
"""

import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional

from sqlalchemy import (
    create_engine, text,
    Column, Integer, Float, String, Text, DateTime, JSON
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from pgvector.sqlalchemy import Vector

from config import DATABASE_URL, EMBEDDING_DIM

# ── Engine & Session ─────────────────────────────────────────────────────────
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


# ── ORM Models ───────────────────────────────────────────────────────────────
class VideoRecord(Base):
    __tablename__ = "videos"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    source        = Column(Text, nullable=False)
    status        = Column(String(32), default="pending")
    total_frames  = Column(Integer, default=0)
    created_at    = Column(DateTime, default=datetime.utcnow)
    finished_at   = Column(DateTime, nullable=True)


class DetectionRecord(Base):
    __tablename__ = "detections"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    video_id   = Column(Integer, nullable=True)
    timestamp  = Column(Text, nullable=False)
    frame_no   = Column(Integer, nullable=False)
    label      = Column(String(128), nullable=False)
    confidence = Column(Float, nullable=False)
    crop_path  = Column(Text, nullable=True)
    bbox       = Column(Text, nullable=True)   # JSON string
    attributes = Column(JSON, nullable=True)   # person attributes dict
    embedding  = Column(Vector(EMBEDDING_DIM), nullable=True)


class SessionRecord(Base):
    __tablename__ = "processing_sessions"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    source           = Column(Text, nullable=True)
    started_at       = Column(Text, nullable=True)
    finished_at      = Column(Text, nullable=True)
    total_frames     = Column(Integer, default=0)
    processed_frames = Column(Integer, default=0)
    total_detections = Column(Integer, default=0)
    status           = Column(String(32), default="running")


# ── DB init ───────────────────────────────────────────────────────────────────
def init_db():
    """Create pgvector extension and all tables."""
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(bind=engine)
    print("[DB] Tables created / verified.")


# ── DetectionDB class (same interface as old version) ────────────────────────
class DetectionDB:
    """Drop-in replacement for the old SQLite+FAISS DetectionDB."""

    def __init__(self):
        init_db()
        self._db: Session = SessionLocal()

    # ── Session management ────────────────────────────────────────────────
    def start_session(self, source: str) -> int:
        rec = SessionRecord(
            source=source,
            started_at=datetime.utcnow().isoformat(),
            status="running",
        )
        self._db.add(rec)
        self._db.commit()
        self._db.refresh(rec)
        return rec.id

    def update_session(self, session_id: int, **kwargs):
        rec = self._db.get(SessionRecord, session_id)
        if rec:
            for k, v in kwargs.items():
                setattr(rec, k, v)
            self._db.commit()

    def get_session(self, session_id: int) -> Optional[Dict]:
        rec = self._db.get(SessionRecord, session_id)
        return self._row_to_dict(rec) if rec else None

    # ── Detections ────────────────────────────────────────────────────────
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
        attributes: dict = None,
    ) -> int:
        vec = embedding.astype(np.float32).tolist()
        rec = DetectionRecord(
            timestamp=timestamp,
            frame_no=frame_no,
            label=label,
            confidence=round(float(confidence), 4),
            crop_path=crop_path,
            bbox=json.dumps(bbox),
            attributes=attributes or {},
            embedding=vec,
        )
        self._db.add(rec)
        self._db.commit()
        self._db.refresh(rec)
        return rec.id

    def get_recent_detections(
        self,
        limit: int = 50,
        label: str = None,
        color: str = None,
    ) -> List[Dict]:
        q = self._db.query(DetectionRecord)
        if label:
            q = q.filter(DetectionRecord.label == label)
        if color:
            # filter where upper_color OR lower_color matches
            q = q.filter(
                text(
                    "(attributes->>'upper_color' = :c OR "
                    " attributes->>'lower_color' = :c)"
                ).bindparams(c=color)
            )
        rows = q.order_by(DetectionRecord.id.desc()).limit(limit).all()
        return [self._detection_to_dict(r) for r in rows]

    # ── Vector search ─────────────────────────────────────────────────────
    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 12,
        label: str = None,
        color: str = None,
    ) -> List[Dict]:
        """Cosine similarity search using pgvector <=> operator.

        Optional hard filters:
          label — restrict to one object class e.g. 'person'
          color — restrict to a clothing colour e.g. 'red'
        """
        total = self._db.query(DetectionRecord).count()
        if total == 0:
            return []

        vec_list = query_vec.astype(np.float32).tolist()
        k = min(top_k * 4, total)   # fetch more, filter, then trim

        q = self._db.query(DetectionRecord)
        if label:
            q = q.filter(DetectionRecord.label == label)
        if color:
            q = q.filter(
                text(
                    "(attributes->>'upper_color' = :c OR "
                    " attributes->>'lower_color' = :c)"
                ).bindparams(c=color)
            )
        rows = (
            q.order_by(DetectionRecord.embedding.cosine_distance(vec_list))
            .limit(k)
            .all()
        )

        # Compute similarity scores
        results = []
        vec_np = query_vec.astype(np.float32)
        for r in rows[:top_k]:
            d = self._detection_to_dict(r)
            if r.embedding is not None:
                emb = np.array(r.embedding, dtype=np.float32)
                sim = float(np.dot(vec_np, emb) /
                            (np.linalg.norm(vec_np) * np.linalg.norm(emb) + 1e-9))
                d["score"] = round(sim, 4)
            else:
                d["score"] = 0.0
            results.append(d)
        return results

    # ── Stats ─────────────────────────────────────────────────────────────
    def get_stats(self) -> Dict:
        total = self._db.query(DetectionRecord).count()
        rows = self._db.execute(
            text(
                "SELECT label, COUNT(*) as cnt FROM detections "
                "GROUP BY label ORDER BY cnt DESC"
            )
        ).fetchall()
        return {
            "total_detections": total,
            "label_counts": {r[0]: r[1] for r in rows},
            "index_size": total,   # every detection is indexed in pgvector
        }

    # ── Compat stubs (FAISS used save_index — now a no-op) ────────────────
    def save_index(self):
        pass   # pgvector persists automatically

    def close(self):
        self._db.close()

    # ── Helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _detection_to_dict(r: DetectionRecord) -> Dict:
        return {
            "id":         r.id,
            "timestamp":  r.timestamp,
            "frame_no":   r.frame_no,
            "label":      r.label,
            "confidence": r.confidence,
            "crop_path":  r.crop_path,
            "bbox":       json.loads(r.bbox) if r.bbox else None,
            "attributes": r.attributes or {},
        }

    @staticmethod
    def _row_to_dict(r) -> Dict:
        return {c.name: getattr(r, c.name) for c in r.__table__.columns}

import cv2
import time
import uuid
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional
from PIL import Image
import numpy as np
from ultralytics import YOLO

from config import (
    YOLO_MODEL, CONFIDENCE_THRESHOLD, FRAME_SKIP, CROPS_DIR,
    YOLO_CLASSES_OF_INTEREST, MIN_CROP_SIZE, APPLY_CLAHE
)
from embedder import embedder
from attributes import attribute_extractor
from database import DetectionDB


# ── CLAHE engine (built once, reused) ────────────────────────────────────────
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the
    luminance channel of the frame.  Improves detection on dark / washed-out
    CCTV footage without altering hue.
    """
    if not APPLY_CLAHE:
        return frame
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = _clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    except Exception:
        return frame   # fall back to raw frame on any error


class VideoProcessor:
    """Processes a video source with YOLOv8 segmentation + CLIP embedding."""

    def __init__(self, db: DetectionDB):
        self.db = db
        self.model = None
        self._stop_event = threading.Event()
        self.session_id = None
        self._latest_frame_bytes: Optional[bytes] = None
        self._frame_lock = threading.Lock()
        self.progress = {
            "status": "idle",
            "total_frames": 0,
            "processed_frames": 0,
            "total_detections": 0,
            "current_fps": 0.0,
            "source": None,
            "video_width": 0,
            "video_height": 0,
            "eta_seconds": None,
            "detection_rate": 0.0,   # detections / second of wall-clock time
            "skipped_crops": 0,
        }

    def _load_yolo(self):
        if self.model is None:
            print(f"[YOLO] Loading {YOLO_MODEL}...")
            self.model = YOLO(YOLO_MODEL)
            print("[YOLO] Ready.")

    def stop(self):
        self._stop_event.set()

    def get_latest_frame(self) -> Optional[bytes]:
        """Return the latest frame bytes (JPEG) for live preview."""
        with self._frame_lock:
            return self._latest_frame_bytes

    def _store_frame(self, frame):
        """Encode a BGR frame to JPEG and cache it."""
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self._frame_lock:
                self._latest_frame_bytes = bytes(buf)

    def process(self, source: str):
        """Main processing loop. Runs in a background thread."""
        try:
            self._stop_event.clear()
            self._load_yolo()
            embedder.load()

            self.progress["status"] = "running"
            self.progress["source"] = source
            self.progress["eta_seconds"] = None
            self.progress["skipped_crops"] = 0
            self.session_id = self.db.start_session(source)

            cap = cv2.VideoCapture(source if not source.isdigit() else int(source))
            if not cap.isOpened():
                self.progress["status"] = "error: could not open source"
                return

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
            vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 0
            vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

            self.progress["total_frames"]  = total
            self.progress["video_width"]   = vid_w
            self.progress["video_height"]  = vid_h

            # Adaptive frame skip: aim to sample ~6 fps regardless of source FPS
            adaptive_skip  = max(1, int(fps / 6))
            effective_skip = min(FRAME_SKIP, adaptive_skip)
            print(f"[Processor] Video: {vid_w}x{vid_h}, {fps:.1f}fps  frame_skip={effective_skip}  clahe={APPLY_CLAHE}")

            self.db.update_session(self.session_id, total_frames=total,
                                   video_width=vid_w, video_height=vid_h)

            # Normalize video source for frontend
            norm_source = source
            try:
                source_path = Path(source)
                if source_path.exists() and "data" in source_path.parts:
                    data_idx   = source_path.parts.index("data")
                    norm_source = "/".join(source_path.parts[data_idx:])
            except Exception:
                pass

            frame_no  = 0
            det_count = 0
            skip_count = 0
            t0 = time.time()

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_no += 1
                if frame_no % effective_skip != 0:
                    continue

                # ── Pre-process frame (CLAHE contrast equalisation) ────────
                proc_frame = preprocess_frame(frame)

                # Store latest (pre-processed) frame for live preview
                self._store_frame(proc_frame)

                elapsed = time.time() - t0
                wall_fps = frame_no / elapsed if elapsed > 0 else 0
                self.progress["processed_frames"] = frame_no
                self.progress["current_fps"] = round(wall_fps, 2)

                # ETA
                if total > 0 and wall_fps > 0:
                    remaining_frames = total - frame_no
                    self.progress["eta_seconds"] = round(remaining_frames / (wall_fps * effective_skip))
                else:
                    self.progress["eta_seconds"] = None

                # Detection rate (dets / elapsed wall-clock second)
                self.progress["detection_rate"] = round(det_count / elapsed, 2) if elapsed > 0 else 0.0

                # ── YOLOv8 segmentation ────────────────────────────────────
                results = self.model(proc_frame, verbose=False,
                                     conf=CONFIDENCE_THRESHOLD,
                                     classes=YOLO_CLASSES_OF_INTEREST)

                timestamp = datetime.now().isoformat()

                for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        continue

                    for i, box in enumerate(boxes):
                        try:
                            cls_id = int(box.cls[0])
                            label  = self.model.names[cls_id]
                            conf   = float(box.conf[0])
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                            # Crop
                            h, w = proc_frame.shape[:2]
                            x1c, y1c = max(0, x1), max(0, y1)
                            x2c, y2c = min(w, x2), min(h, y2)
                            crop_bgr = proc_frame[y1c:y2c, x1c:x2c]

                            if crop_bgr.size == 0:
                                skip_count += 1
                                continue

                            # ── Minimum crop size filter ───────────────────
                            ch, cw = crop_bgr.shape[:2]
                            if cw < MIN_CROP_SIZE or ch < MIN_CROP_SIZE:
                                skip_count += 1
                                continue

                            # Save crop as JPEG
                            crop_name = f"{uuid.uuid4().hex}.jpg"
                            crop_path = CROPS_DIR / crop_name
                            cv2.imwrite(str(crop_path), crop_bgr)

                            # CLIP embedding
                            crop_pil  = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
                            embedding = embedder.embed_image_pil(crop_pil)

                            # Attribute extraction
                            attrs = attribute_extractor.extract(crop_pil, label)

                            # Store detection
                            self.db.insert_detection(
                                timestamp  = timestamp,
                                video_src  = norm_source,
                                frame_no   = frame_no,
                                video_time = frame_no / fps,
                                label      = label,
                                confidence = conf,
                                crop_path  = str(crop_path.relative_to(CROPS_DIR.parent.parent)),
                                bbox       = [x1, y1, x2, y2],
                                embedding  = embedding,
                                attributes = attrs,
                            )
                            det_count += 1

                        except Exception as det_exc:
                            # One bad detection should never abort the frame
                            print(f"[Processor] Detection error (frame {frame_no}): {det_exc}")
                            continue

                self.progress["total_detections"] = det_count
                self.progress["skipped_crops"]    = skip_count
                self.db.update_session(
                    self.session_id,
                    processed_frames  = frame_no,
                    total_detections  = det_count,
                )

            cap.release()
            elapsed_total = time.time() - t0
            self.db.update_session(
                self.session_id,
                status           = "finished",
                finished_at      = datetime.now().isoformat(),
                total_detections = det_count,
            )
            self.progress["status"]      = "finished"
            self.progress["eta_seconds"] = 0
            print(
                f"[Processor] Done. {det_count} detections | "
                f"{frame_no} frames | {skip_count} skipped crops | "
                f"{elapsed_total:.1f}s total"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.progress["status"] = f"error: {str(e)}"

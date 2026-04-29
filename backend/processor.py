import cv2
import time
import uuid
import threading
from datetime import datetime
from pathlib import Path
from PIL import Image
import numpy as np
from ultralytics import YOLO

from config import (
    YOLO_MODEL, CONFIDENCE_THRESHOLD, FRAME_SKIP, CROPS_DIR, YOLO_CLASSES_OF_INTEREST
)
from embedder import embedder
from attributes import attribute_extractor
from database import DetectionDB


class VideoProcessor:
    """Processes a video source with YOLOv8 segmentation + CLIP embedding."""

    def __init__(self, db: DetectionDB):
        self.db = db
        self.model = None
        self._stop_event = threading.Event()
        self.session_id = None
        self.progress = {
            "status": "idle",
            "total_frames": 0,
            "processed_frames": 0,
            "total_detections": 0,
            "current_fps": 0.0,
            "source": None,
        }

    def _load_yolo(self):
        if self.model is None:
            print(f"[YOLO] Loading {YOLO_MODEL}...")
            self.model = YOLO(YOLO_MODEL)
            print("[YOLO] Ready.")

    def stop(self):
        self._stop_event.set()

    def process(self, source: str):
        """Main processing loop. Runs in a background thread."""
        try:
            self._stop_event.clear()
            self._load_yolo()
            embedder.load()

            self.progress["status"] = "running"
            self.progress["source"] = source
            self.session_id = self.db.start_session(source)

            cap = cv2.VideoCapture(source if not source.isdigit() else int(source))
            if not cap.isOpened():
                self.progress["status"] = "error: could not open source"
                return

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.progress["total_frames"] = total
            self.db.update_session(self.session_id, total_frames=total)

            # Normalize video source for frontend (e.g. data/videos/test.mp4 -> videos/test.mp4)
            norm_source = source
            try:
                # If it's a file inside our project, make it relative to the 'data' parent
                source_path = Path(source)
                if source_path.exists() and "data" in source_path.parts:
                    # Find 'data' and take everything from there
                    data_idx = source_path.parts.index("data")
                    norm_source = "/".join(source_path.parts[data_idx:])
            except:
                pass

            frame_no = 0
            det_count = 0
            t0 = time.time()

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_no += 1
                if frame_no % FRAME_SKIP != 0:
                    continue

                self.progress["processed_frames"] = frame_no
                elapsed = time.time() - t0
                self.progress["current_fps"] = round(frame_no / elapsed, 2) if elapsed > 0 else 0

                # Run YOLOv8 segmentation
                results = self.model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD,
                                      classes=YOLO_CLASSES_OF_INTEREST)

                timestamp = datetime.now().isoformat()

                for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        continue

                    for i, box in enumerate(boxes):
                        cls_id = int(box.cls[0])
                        label = self.model.names[cls_id]
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                        # Crop object from frame
                        h, w = frame.shape[:2]
                        x1c, y1c = max(0, x1), max(0, y1)
                        x2c, y2c = min(w, x2), min(h, y2)
                        crop_bgr = frame[y1c:y2c, x1c:x2c]
                        if crop_bgr.size == 0:
                            continue

                        # Save crop as JPEG
                        crop_name = f"{uuid.uuid4().hex}.jpg"
                        crop_path = CROPS_DIR / crop_name
                        cv2.imwrite(str(crop_path), crop_bgr)

                        # Generate CLIP embedding
                        crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
                        embedding = embedder.embed_image_pil(crop_pil)

                        # Extract person attributes (colour, gender, accessories)
                        attrs = attribute_extractor.extract(crop_pil, label)

                        # Store in PostgreSQL + pgvector
                        self.db.insert_detection(
                            timestamp=timestamp,
                            video_src=norm_source,
                            frame_no=frame_no,
                            video_time=frame_no / fps,
                            label=label,
                            confidence=conf,
                            crop_path=str(crop_path.relative_to(CROPS_DIR.parent.parent)),
                            bbox=[x1, y1, x2, y2],
                            embedding=embedding,
                            attributes=attrs,
                        )
                        det_count += 1

                self.progress["total_detections"] = det_count
                self.db.update_session(
                    self.session_id,
                    processed_frames=frame_no,
                    total_detections=det_count,
                )

            cap.release()
            self.db.update_session(
                self.session_id,
                status="finished",
                finished_at=datetime.now().isoformat(),
                total_detections=det_count,
            )
            self.progress["status"] = "finished"
            print(f"[Processor] Done. {det_count} detections from {frame_no} frames.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.progress["status"] = f"error: {str(e)}"

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CROPS_DIR = DATA_DIR / "crops"
VIDEOS_DIR = DATA_DIR / "videos"
DEMO_DIR = BASE_DIR / "demo"

# Create directories
for d in [CROPS_DIR, VIDEOS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Database & Index ─────────────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/cctv_analytics"
)

DB_DIR = DATA_DIR / "db"
DB_DIR.mkdir(parents=True, exist_ok=True)

SQLITE_PATH = DB_DIR / "detections.db"
FAISS_INDEX_PATH = DB_DIR / "vector_index.faiss"

# Models
YOLO_MODEL    = os.getenv("YOLO_MODEL", "yolov8n-seg.pt")
CLIP_MODEL    = os.getenv("CLIP_MODEL", "ViT-B-32")
CLIP_PRETRAINED = os.getenv("CLIP_PRETRAINED", "openai")

# Processing
VIDEO_SOURCE         = os.getenv("VIDEO_SOURCE", str(DEMO_DIR / "sample.mp4"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))  # lowered for unseen videos
FRAME_SKIP           = int(os.getenv("FRAME_SKIP", "3"))  # adaptive: process more frames for short/low-FPS videos
EMBEDDING_DIM        = 512

# Preprocessing
MIN_CROP_SIZE  = int(os.getenv("MIN_CROP_SIZE", "30"))    # skip crops smaller than N×N px
APPLY_CLAHE    = os.getenv("APPLY_CLAHE", "true").lower() != "false"  # contrast normalisation

# YOLO — None means detect all classes
YOLO_CLASSES_OF_INTEREST = None

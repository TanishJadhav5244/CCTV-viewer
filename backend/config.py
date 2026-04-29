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

# ── PostgreSQL ───────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/cctv_analytics"
)

# Models
YOLO_MODEL    = os.getenv("YOLO_MODEL", "yolov8n-seg.pt")
CLIP_MODEL    = os.getenv("CLIP_MODEL", "ViT-B-32")
CLIP_PRETRAINED = os.getenv("CLIP_PRETRAINED", "openai")

# Processing
VIDEO_SOURCE         = os.getenv("VIDEO_SOURCE", str(DEMO_DIR / "sample.mp4"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.4"))
FRAME_SKIP           = int(os.getenv("FRAME_SKIP", "5"))
EMBEDDING_DIM        = 512

# YOLO — None means detect all classes
YOLO_CLASSES_OF_INTEREST = None

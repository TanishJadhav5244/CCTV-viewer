import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CROPS_DIR = DATA_DIR / "crops"
VIDEOS_DIR = DATA_DIR / "videos"
DB_DIR = DATA_DIR / "db"
DEMO_DIR = BASE_DIR / "demo"

# Create directories if they don't exist
for d in [CROPS_DIR, VIDEOS_DIR, DB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Database
SQLITE_DB_PATH = DB_DIR / "detections.db"
FAISS_INDEX_PATH = DB_DIR / "faiss.index"
FAISS_META_PATH = DB_DIR / "faiss_meta.json"

# Models
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n-seg.pt")
CLIP_MODEL = os.getenv("CLIP_MODEL", "ViT-B-32")
CLIP_PRETRAINED = os.getenv("CLIP_PRETRAINED", "openai")

# Processing
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", str(DEMO_DIR / "sample.mp4"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.4"))
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "5"))  # process every N frames
MAX_CROPS = int(os.getenv("MAX_CROPS", "50000"))
EMBEDDING_DIM = 512

# CLIP target labels for filtering detections
YOLO_CLASSES_OF_INTEREST = None  # None = all classes

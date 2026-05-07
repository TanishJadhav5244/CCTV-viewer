"""
Vercel / deployment entrypoint.
Vercel looks for a FastAPI 'app' object in app.py (or main.py) at the project root.
The real app lives in backend/main.py — we just re-export it from here.
"""
import sys
from pathlib import Path

# Ensure backend/ is importable before anything else
BACKEND_DIR = Path(__file__).parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from backend.main import app  # noqa: F401  (backend/main.py → FastAPI instance)

__all__ = ["app"]

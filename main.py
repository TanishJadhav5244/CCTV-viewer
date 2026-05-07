"""
Root-level entrypoint for deployment platforms (Railway, Render, Heroku, etc.)
that look for a FastAPI 'app' object in main.py / app.py at the project root.

The actual app is defined in backend/main.py — this file simply re-exports it.
"""
import sys
import os
from pathlib import Path

# Make sure the backend directory is on the Python path so all relative imports work
BACKEND_DIR = Path(__file__).parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# Re-export the FastAPI app so deployment platforms can find it
from main import app  # noqa: F401  (backend/main.py)

__all__ = ["app"]

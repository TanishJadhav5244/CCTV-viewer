"""
app.py — Root-level alias for deployment platforms that look for app.py.

This re-exports the same FastAPI app as api/index.py so that
platforms like Railway / Render also find an entrypoint.

Vercel uses api/index.py directly (configured in vercel.json).
"""
import sys
from pathlib import Path

# Ensure api/ is importable
sys.path.insert(0, str(Path(__file__).parent / "api"))

from index import app  # noqa: F401

__all__ = ["app"]

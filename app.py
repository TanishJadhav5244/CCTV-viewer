"""
Alias entrypoint — re-exports 'app' from main.py for platforms that look for app.py.
"""
from main import app  # noqa: F401

__all__ = ["app"]

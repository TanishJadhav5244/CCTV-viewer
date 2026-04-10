"""
One-command launcher for CCTV Deep Learning Analytics.
Run: python run.py
"""
import sys
import os
import subprocess
import webbrowser
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent
BACKEND_DIR = BASE_DIR / "backend"

def main():
    print("=" * 55)
    print("  CCTV Deep Learning Analytics")
    print("  YOLOv8 + CLIP | FastAPI Backend + Web Dashboard")
    print("=" * 55)

    # Copy .env.example to .env if .env doesn't exist
    env_file = BASE_DIR / ".env"
    env_example = BASE_DIR / ".env.example"
    if not env_file.exists() and env_example.exists():
        env_file.write_text(env_example.read_text())
        print("[Setup] Created .env from .env.example")

    # Ensure data directories exist
    for d in ["data/crops", "data/videos", "data/db", "demo"]:
        (BASE_DIR / d).mkdir(parents=True, exist_ok=True)

    print(f"\n[Info] Backend: http://localhost:8000")
    print(f"[Info] Dashboard: http://localhost:8000")
    print(f"[Info] API Docs:  http://localhost:8000/docs\n")

    # Open browser after short delay
    def open_browser():
        time.sleep(3)
        webbrowser.open("http://localhost:8000")

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # Start uvicorn
    try:
        subprocess.run(
            [
                sys.executable, "-m", "uvicorn",
                "main:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload",
            ],
            cwd=str(BACKEND_DIR),
            check=True
        )
    except KeyboardInterrupt:
        print("\n[Info] Server stopped.")

if __name__ == "__main__":
    main()

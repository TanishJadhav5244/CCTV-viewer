"""
init_db.py — Run this once to set up PostgreSQL schema.

Usage:
    python backend/init_db.py

Requires:
    - PostgreSQL running with pgvector extension available
    - DATABASE_URL set in .env or environment
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from database import init_db

if __name__ == "__main__":
    print("[init_db] Connecting to PostgreSQL...")
    init_db()
    print("[init_db] Done. Tables and pgvector extension are ready.")

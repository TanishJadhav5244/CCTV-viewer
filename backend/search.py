from typing import List, Dict
from embedder import embedder
from database import DetectionDB


def search_by_text(query: str, db: DetectionDB, top_k: int = 12) -> List[Dict]:
    """
    Encode a natural-language query with CLIP and retrieve the most
    visually similar object detections from the FAISS index.
    """
    embedder.load()
    query_vec = embedder.embed_text(query)
    results = db.search(query_vec, top_k=top_k)
    return results

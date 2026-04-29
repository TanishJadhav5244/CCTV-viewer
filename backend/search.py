from typing import List, Dict, Optional
from embedder import embedder
from database import DetectionDB


def search_by_text(
    query: str,
    db: DetectionDB,
    top_k: int = 12,
    label: Optional[str] = None,
    color: Optional[str] = None,
) -> List[Dict]:
    """
    Encode a natural-language query with CLIP and retrieve the most
    visually similar object detections using pgvector cosine search.

    Optional hard filters:
      label — e.g. 'person' to only return persons
      color — e.g. 'red' to only return detections where clothing is red
    """
    embedder.load()
    query_vec = embedder.embed_text(query)
    results = db.search(query_vec, top_k=top_k, label=label, color=color)
    return results

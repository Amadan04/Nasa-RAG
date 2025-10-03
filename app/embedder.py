# app/embedder.py
import os
from functools import lru_cache
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

@lru_cache(maxsize=1)
def _model():
    return SentenceTransformer(MODEL_NAME, device="cpu")  # CPU is fine

def embed_text(text: str) -> list[float]:
    """Return a single embedding vector (local, CPU)."""
    v = _model().encode([text], normalize_embeddings=True)[0]
    return v.tolist()

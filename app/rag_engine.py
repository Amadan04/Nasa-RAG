# app/rag_engine.py
import os, chromadb
from chromadb.config import Settings

CHROMA_DIR = os.path.join("app", "vector_store")
_client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=False))
_collection = _client.get_or_create_collection("nasa_bio")

def retrieve_chunks_by_query_embedding(query_embedding, k=5):
    res = _collection.query(query_embeddings=[query_embedding], n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return docs, metas

def retrieve_with_scores(query_embedding, k=5):
    """Return (docs, metas, distances). Chroma distances are cosine distances (0 = identical)."""
    res = _collection.query(query_embeddings=[query_embedding], n_results=k, include=["documents","metadatas","distances"])
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0] or []
    return docs, metas, distances

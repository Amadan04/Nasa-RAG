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

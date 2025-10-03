# app/test_rag.py (fast version)
import chromadb
from app.embedder import embed_text  # our local MiniLM embedder

client = chromadb.PersistentClient(path="app/vector_store")
col = client.get_collection("nasa_bio")

query = "How does microgravity affect plant growth?"
q_emb = embed_text(query)

res = col.query(query_embeddings=[q_emb], n_results=3)
print("\n🔎 Top 3 results:")
for i, doc in enumerate(res["documents"][0], 1):
    print(f"{i}. {doc[:300]}...\n")

import os, json
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from app.openrouter_client import embed_text
from app.embedder import embed_text   # <-- uses local embeddings now

load_dotenv()

DATA_JSON = os.path.join("data", "nasa_bio.json")
CHROMA_DIR = os.path.join("app", "vector_store")

def main():
    if not os.path.exists(DATA_JSON):
        raise FileNotFoundError(f"Missing {DATA_JSON}. Run ingest first.")

    with open(DATA_JSON, "r", encoding="utf-8") as f:
        docs = json.load(f)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "]
    )

    print("🚀 Starting to embed and index documents...")
    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=True))
    collection = client.get_or_create_collection("nasa_bio")

    ids, embeddings, texts, metas = [], [], [], []
    idx = 0

    for d in docs:
        text = (d.get("text") or "").strip()
        if not text:
            continue
        for chunk in splitter.split_text(text):
            e = embed_text(chunk)  # OpenRouter embeddings
            ids.append(str(idx))
            embeddings.append(e)
            texts.append(chunk)
            metas.append({"source": d.get("url",""), "title": d.get("title","Unknown")})
            idx += 1
            # Print a dot every 50 chunks so you see progress
            if idx % 50 == 0:
                print(".", end="", flush=True)

    print("\n✅ Embeddings ready. Upserting into Chroma...")
    BATCH = 128
    for i in range(0, len(ids), BATCH):
        collection.upsert(
            ids=ids[i:i+BATCH],
            embeddings=embeddings[i:i+BATCH],
            documents=texts[i:i+BATCH],
            metadatas=metas[i:i+BATCH]
        )
        print(f"  • Indexed {min(i+BATCH, len(ids))}/{len(ids)}")

    print("🎉 Done! Vector DB created in:", CHROMA_DIR)

if __name__ == "__main__":
    main()

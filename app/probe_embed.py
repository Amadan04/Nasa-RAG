# app/probe_embed.py
import os, requests
from dotenv import load_dotenv
load_dotenv()

BASE = "https://openrouter.ai/api/v1"
HEADERS = {
    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
    "Accept": "application/json",
    "Content-Type": "application/json",
    "User-Agent": "SpaceBio-RAG/1.0",
    "HTTP-Referer": "https://github.com/yourname/nasa-rag",
    "X-Title": "SpaceBio RAG",
}
EMBED_MODEL = os.getenv("EMBED_MODEL")

resp = requests.post(
    f"{BASE}/embeddings",
    headers=HEADERS,
    json={"model": EMBED_MODEL, "input": ["hello world"]},
    timeout=30,
)
print("status:", resp.status_code)
print("ct:", resp.headers.get("content-type"))
print("body:", (resp.text or "")[:300])

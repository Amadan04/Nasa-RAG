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

r = requests.get(f"{BASE}/models", headers=HEADERS, timeout=30)
print("status:", r.status_code)
print("content-type:", r.headers.get("content-type"))
print("first 300 chars:", r.text[:300])
# app/openrouter_client.py
import os, requests, json
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL")
CHAT_MODEL  = os.getenv("CHAT_MODEL")

if not OPENROUTER_API_KEY:
    raise ValueError("❌ OPENROUTER_API_KEY is missing in .env")
if not EMBED_MODEL:
    raise ValueError("❌ EMBED_MODEL is missing in .env")
if not CHAT_MODEL:
    raise ValueError("❌ CHAT_MODEL is missing in .env")

BASE_URL = "https://openrouter.ai/api/v1"
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "SpaceBio-RAG/1.0 (+https://example.local)",
    # optional but recommended by OpenRouter:
    "HTTP-Referer": "http://localhost",
    "X-Title": "SpaceBio RAG",
}

def _ensure_json(resp: requests.Response) -> dict:
    ct = resp.headers.get("content-type", "")
    txt = resp.text or ""
    # Raise for obvious HTTP errors first
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # Show server text for easier debugging
        raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {txt[:500]}") from e
    # Some proxies/CDNs can return 200 with HTML
    if "application/json" not in ct.lower():
        raise RuntimeError(f"OpenRouter non-JSON response (ct={ct}): {txt[:500]}")
    try:
        return resp.json()
    except json.JSONDecodeError:
        raise RuntimeError(f"OpenRouter JSON decode error. Body (first 500): {txt[:500]}")

# app/openrouter_client.py (only this function)
def embed_text(text: str) -> list[float]:
    payload = {"model": EMBED_MODEL, "input": [text]}  # LIST!
    r = requests.post(f"{BASE_URL}/embeddings", headers=HEADERS, json=payload, timeout=60)
    # Debug on any non-JSON
    if "application/json" not in (r.headers.get("content-type") or "").lower():
        print(">>> DEBUG embeddings",
              "status:", r.status_code,
              "ct:", r.headers.get("content-type"),
              "url:", r.url,
              "body:", (r.text or "")[:400])
        r.raise_for_status()
        raise RuntimeError("Embeddings endpoint did not return JSON")
    j = r.json()
    return j["data"][0]["embedding"]


def chat_messages(messages, model: str | None = None, temperature: float = 0.2) -> str:
    payload = {
        "model": model or CHAT_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    r = requests.post(f"{BASE_URL}/chat/completions", headers=HEADERS, json=payload, timeout=120)
    j = _ensure_json(r)
    return j["choices"][0]["message"]["content"]

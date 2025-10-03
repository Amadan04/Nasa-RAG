from typing import List, Literal
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.embedder import embed_text               # local embeddings
from app.openrouter_client import chat_messages   # OpenRouter for generation
from app.rag_engine import retrieve_with_scores

app = FastAPI(title="SpaceBio RAG API 🚀")

# Allow your frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # replace with your domain(s) for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def context_quality(docs: List[str], min_chars=30) -> bool:
    return len("\n\n".join(docs)) >= min_chars

# cosine distance: 0 = identical, 1 = unrelated.
# Lower is better. 0.32–0.40 is a good starting threshold for MiniLM.
RELEVANCE_DISTANCE_THRESHOLD = 0.8

def is_relevant(distances: List[float], threshold: float = RELEVANCE_DISTANCE_THRESHOLD) -> bool:
    return any(d is not None and d < threshold for d in distances)

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/ask")
def ask(query: str = Query(..., min_length=3), only_context: bool = False, k: int = 5):
    q_emb = embed_text(query)
    docs, metas, distances = retrieve_with_scores(q_emb, k=k)
    ctx = "\n\n".join(docs)
    citations = sorted({m.get("source", "") for m in metas if m.get("source")})

    relevant = is_relevant(distances)
    enough_text = context_quality(docs)

    if only_context or (relevant and enough_text):
        sys_prompt = ("You are a space biology expert. Answer ONLY using the provided context. "
                      "If the answer isn't there, say 'I don't know.' Keep it concise.")
        user_prompt = f"Context:\n{ctx}\n\nQuestion: {query}"
        mode = "RAG"
    else:
        sys_prompt = "You are a helpful science assistant. Use general knowledge. If uncertain, say you're unsure."
        user_prompt = query
        mode = "AI"

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    answer = chat_messages(messages)

    return {"answer": answer, "mode": mode, "citations": citations, "chunks_used": len(docs)}

class Msg(BaseModel):
    role: Literal["user","assistant","system"]
    content: str

class ChatReq(BaseModel):
    messages: List[Msg]
    only_context: bool = False
    k: int = 5

@app.post("/chat")
def chat(req: ChatReq):
    user_msgs = [m for m in req.messages if m.role == "user"]
    query = user_msgs[-1].content if user_msgs else ""

    q_emb = embed_text(query)
    docs, metas, distances = retrieve_with_scores(q_emb, k=req.k)
    ctx = "\n\n".join(docs)
    citations = sorted({m.get("source", "") for m in metas if m.get("source")})

    relevant = is_relevant(distances)
    enough_text = context_quality(docs)

    if req.only_context or (relevant and enough_text):
        sys_prompt = ("You are a space biology expert. Answer ONLY using the provided context. "
                      "If the answer isn't there, say 'I don't know.' Keep it concise.")
        user_prompt = f"Context:\n{ctx}\n\nQuestion: {query}"
        mode = "RAG"
    else:
        sys_prompt = "You are a helpful science assistant. Use general knowledge. If uncertain, say you're unsure."
        user_prompt = query
        mode = "AI"

    history = req.messages[-6:]
    messages = [{"role": "system", "content": sys_prompt}] + history[:-1] + [{"role": "user", "content": user_prompt}]
    answer = chat_messages(messages)

    return {"answer": answer, "mode": mode, "citations": citations, "chunks_used": len(docs)}

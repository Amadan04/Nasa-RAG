from typing import List, Literal
from fastapi import FastAPI, Query
from pydantic import BaseModel

#from app.openrouter_client import embed_text, chat_messages
from app.rag_engine import retrieve_chunks_by_query_embedding
# ❌ current (causes 500)
# from app.openrouter_client import embed_text, chat_messages

# ✅ correct (local embeddings + OpenRouter for chat)
from app.embedder import embed_text
from app.openrouter_client import chat_messages

app = FastAPI(title="SpaceBio RAG API 🚀")

def context_quality(docs: List[str], min_chars=800) -> bool:
    return len("\n\n".join(docs)) >= min_chars

@app.get("/ask")
def ask(query: str = Query(..., min_length=3), only_context: bool = False, k: int = 5):
    q_emb = embed_text(query)
    docs, metas = retrieve_chunks_by_query_embedding(q_emb, k=k)
    ctx = "\n\n".join(docs)
    citations = sorted({m.get("source","") for m in metas if m.get("source")})

    use_rag = context_quality(docs) or only_context
    if use_rag:
        sys_prompt = ("You are a space biology expert. Answer ONLY using the provided context. "
                      "If the answer isn't there, say 'I don't know.' Keep it concise.")
        user_prompt = f"Context:\n{ctx}\n\nQuestion: {query}"
        mode = "RAG"
    else:
        sys_prompt = "You are a helpful science assistant. Use general knowledge. If uncertain, say you're unsure."
        user_prompt = query
        mode = "Fallback"

    messages = [
        {"role":"system","content": sys_prompt},
        {"role":"user","content": user_prompt}
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
    docs, metas = retrieve_chunks_by_query_embedding(q_emb, k=req.k)
    ctx = "\n\n".join(docs)
    citations = sorted({m.get("source","") for m in metas if m.get("source")})

    use_rag = context_quality(docs) or req.only_context
    if use_rag:
        sys_prompt = ("You are a space biology expert. Answer ONLY using the provided context. "
                      "If the answer isn't there, say 'I don't know.' Keep it concise.")
        user_prompt = f"Context:\n{ctx}\n\nQuestion: {query}"
        mode = "RAG"
    else:
        sys_prompt = "You are a helpful science assistant. Use general knowledge. If uncertain, say you're unsure."
        user_prompt = query
        mode = "Fallback"

    history = req.messages[-6:]
    messages = [{"role":"system","content": sys_prompt}] + history[:-1] + [{"role":"user","content": user_prompt}]
    answer = chat_messages(messages)

    return {"answer": answer, "mode": mode, "citations": citations, "chunks_used": len(docs)}

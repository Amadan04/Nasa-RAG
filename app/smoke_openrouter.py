# app/smoke_openrouter.py
from app.openrouter_client import embed_text, chat_messages

print("Embedding test…")
vec = embed_text("hello world")
print("OK embedding length:", len(vec))

print("Chat test…")
ans = chat_messages([
    {"role":"system","content":"Reply with the word PONG only."},
    {"role":"user","content":"PING"}
])
print("Chat reply:", ans)

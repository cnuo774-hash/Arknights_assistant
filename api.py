"""
FastAPI backend — wraps existing RagService / KnowledgeBaseService as REST + SSE.
Minimal changes to the original Python modules (zero changes to rag.py, knowledge_base.py, etc.).
"""
import json
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import RagService
from knowledge_base import KnowledgeBaseService
from file_history_store import FileChatMessageHistory

app = FastAPI(title="Arknights Assistant API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- singleton services (lazy init) ----
_rag: RagService | None = None
_kb: KnowledgeBaseService | None = None


def get_rag() -> RagService:
    global _rag
    if _rag is None:
        _rag = RagService()
    return _rag


def get_kb() -> KnowledgeBaseService:
    global _kb
    if _kb is None:
        _kb = KnowledgeBaseService()
    return _kb


# ---- request models ----
class ChatRequest(BaseModel):
    prompt: str
    session_id: str = "user_001"


# ---- chat endpoint (SSE streaming) ----
@app.post("/api/chat")
async def chat(req: ChatRequest):
    rag = get_rag()

    async def event_stream():
        full_response = ""
        history = FileChatMessageHistory(req.session_id)
        try:
            chain_input = {"input": req.prompt}
            config = {"configurable": {"session_id": req.session_id}}
            for chunk in rag.chain.stream(chain_input, config):
                full_response += chunk
                yield f"data: {json.dumps({'token': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True, 'full': full_response})}\n\n"
        except Exception:
            if not full_response:
                try:
                    result = rag.chain.invoke(chain_input, config)
                    yield f"data: {json.dumps({'token': result, 'done': True})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---- knowledge upload ----
@app.post("/api/knowledge/upload")
async def upload_knowledge(file: UploadFile = File(...)):
    global _rag
    if not file.filename or not file.filename.endswith(".txt"):
        raise HTTPException(400, "Only .txt files are supported")
    try:
        text = (await file.read()).decode("utf-8")
    except Exception:
        raise HTTPException(400, "Cannot decode file as UTF-8")
    kb = get_kb()
    result = kb.upload_by_str(text, file.filename)
    # Rebuild the RAG service on the next chat request so the retriever sees
    # newly uploaded documents and a refreshed BM25 index.
    _rag = None
    return JSONResponse({"status": "ok", "message": result})


# ---- chat history ----
@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    history = FileChatMessageHistory(session_id)
    try:
        msgs = history.messages
        return [
            {
                "role": "user" if msg.type == "human" else "assistant",
                "content": msg.content,
            }
            for msg in msgs
        ]
    except Exception:
        return []


@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    history = FileChatMessageHistory(session_id)
    history.clear()
    return {"status": "ok"}


# ---- static frontend ----
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)


@app.get("/")
async def index():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Frontend file not found: static/index.html</h1>")


app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ---- entry point ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8090, reload=True)

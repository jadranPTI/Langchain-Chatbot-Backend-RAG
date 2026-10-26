from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.search import RAGSearch
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Bot Rag API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Har jagah se request allow karein
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



try:
    bot = RAGSearch()
    print("[INFO] RAG Search Engine loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load RAG engine: {e}")


# Model of chatbot
class QueryRequest(BaseModel):
    query: str
    device_id: str


@app.post("/chat")
def ask_bot(request: QueryRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        return StreamingResponse(
            bot.search_and_stream(request.query, request.device_id),
            media_type="text/plain"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
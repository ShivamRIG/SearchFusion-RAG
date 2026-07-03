from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.pipeline import HybridRAGPipeline


# Initialize the pipeline once when the application starts
pipeline = HybridRAGPipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events.
    """
    print(" Starting Hybrid RAG API...")
    yield
    print(" Shutting down Hybrid RAG API...")


app = FastAPI(
    title="Hybrid RAG API",
    description="Hybrid Retrieval-Augmented Generation using Qdrant, Elasticsearch, OpenAI, and Langfuse",
    version="1.0.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


@app.get("/")
def health():
    return {
        "status": "running",
        "service": "Hybrid RAG API"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Ask a question to the Hybrid RAG pipeline.
    """
    try:
        answer = pipeline.ask(request.question)
        return ChatResponse(answer=answer)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/chat/stream")
def stream(request: ChatRequest):

    return StreamingResponse(
        pipeline.stream(request.question),
        media_type="text/plain"
    )

@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }
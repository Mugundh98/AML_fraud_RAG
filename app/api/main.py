import logging
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI

from app.api.schemas import ChatRequest, ChatResponse, HealthResponse, IngestResponse
from app.config import get_settings
from app.llm import Message, build_router
from app.rag.ingest import ingest_directory
from app.rag.pipeline import RAGPipeline
from app.rag.store import DocStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

state: dict = {}
# Embedded Qdrant isn't safe for concurrent writes, and re-indexing mid-search gives odd results.
store_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    store = DocStore(settings)
    router = build_router(settings)
    state.update(settings=settings, store=store, router=router, pipeline=RAGPipeline(store, router, settings.top_k))
    yield
    store.client.close()


app = FastAPI(title="AML & Fraud Assistant", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    store: DocStore = state["store"]
    with store_lock:
        count = store.count()
    return HealthResponse(
        status="ok",
        llm_providers=state["router"].configured(),
        vector_store="qdrant-local" if store.is_local else "qdrant-cloud",
        indexed_chunks=count,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    history = [Message(turn.role, turn.content) for turn in request.history]
    with store_lock:
        result = state["pipeline"].answer(request.message, history)
    return ChatResponse(**asdict(result))


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    with store_lock:
        stats = ingest_directory(state["store"], state["settings"])
    return IngestResponse(**asdict(stats))

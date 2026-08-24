import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager

from api.v1.agent import engine as engine_module
from api.v1.api.router import router as api_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        engine_module.initialize_graph()
    except Exception as e:
        raise RuntimeError("Engine initialization failed") from e

    try:
        from api.v1.memory.memory_service import get_langmem_manager
        get_langmem_manager()
    except Exception as e:
        logger.warning("LangMem warmup failed; DB memory fallback will be used: %s", e)

    try:
        yield
    finally:
        from api.v1.memory.memory_service import shutdown_langmem
        shutdown_langmem()
        engine_module.close_graph()
        try:
            from api.v1.db.db import pool, engine
            pool.close()
            engine.dispose()
        except Exception as e:
            logger.warning("DB resource shutdown issue: %s", e)

app = FastAPI(title="Context-Engineered Agent API", version="1.0", lifespan=lifespan)

app.include_router(api_router)

@app.get("/", response_class=HTMLResponse, tags=["UI"])
def serve_chat_ui():
    html_path = Path(__file__).with_name("index.html")
    return html_path.read_text(encoding="utf-8")

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy"}

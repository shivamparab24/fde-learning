from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pathlib import Path
from contextlib import asynccontextmanager
from api.v1.agent import engine as engine_module
from api.v1.api.router import router as api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        engine_module.initialize_graph()
    except Exception as e:
        raise RuntimeError("Engine initialization failed") from e
    try:
        yield
    finally:
        # Optionally add cleanup here if engine exposes it
        pass

app = FastAPI(title="Context-Engineered Agent API", version="1.0", lifespan=lifespan)

app.include_router(api_router)

@app.get("/", response_class=HTMLResponse, tags=["UI"])
def serve_chat_ui():
    html_path = Path(__file__).with_name("index.html")
    return html_path.read_text(encoding="utf-8")

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy"}

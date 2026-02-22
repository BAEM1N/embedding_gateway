from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from embedding_gateway.backends.ollama import OllamaBackend
from embedding_gateway.backends.tei import TEIBackend
from embedding_gateway.backends.vllm import VLLMBackend
from embedding_gateway.registry import ModelRegistry
from embedding_gateway.health import health_router
from embedding_gateway.router import router
from embedding_gateway import health as health_module
from embedding_gateway import router as router_module


TEI_MODELS = [
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large-instruct",
    "nlpai-lab/KURE-v1",
]

VLLM_MODELS = [
    "jinaai/jina-embeddings-v3",
]


@pytest.fixture
async def client():
    # Set up registry directly (ASGITransport doesn't trigger lifespan)
    reg = ModelRegistry()

    ollama = OllamaBackend(base_url="http://localhost:11434", timeout=5.0)
    reg.register_backend("ollama", ollama)

    tei = TEIBackend(
        base_url="http://localhost:8080",
        default_model="intfloat/multilingual-e5-large-instruct",
        available_models=TEI_MODELS,
        docker_image="ghcr.io/huggingface/text-embeddings-inference:89-1.9",
        timeout=5.0,
    )
    tei.current_model = "intfloat/multilingual-e5-large-instruct"
    reg.register_backend("tei", tei)

    vllm = VLLMBackend(
        base_url="http://localhost:8081",
        default_model="jinaai/jina-embeddings-v3",
        available_models=VLLM_MODELS,
        docker_image="vllm/vllm-openai:latest",
        timeout=5.0,
    )
    vllm.current_model = "jinaai/jina-embeddings-v3"
    reg.register_backend("vllm", vllm)

    for m in ["bge-m3", "bge-large", "snowflake-arctic-embed2",
              "qwen3-embedding:0.6b", "qwen3-embedding:4b", "qwen3-embedding:8b",
              "nomic-embed-text", "embeddinggemma", "mxbai-embed-large"]:
        reg.register_model(m, ollama)
    for m in TEI_MODELS:
        reg.register_model(m, tei)
    for m in VLLM_MODELS:
        reg.register_model(m, vllm)

    router_module.registry = reg
    health_module.registry = reg

    app = FastAPI()
    app.include_router(router)
    app.include_router(health_router)

    # Playground
    static_dir = Path(__file__).parent.parent / "src" / "embedding_gateway" / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/playground")
    async def playground():
        return FileResponse(str(static_dir / "playground.html"))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    await ollama.close()
    await tei.close()
    await vllm.close()
    router_module.registry = None
    health_module.registry = None

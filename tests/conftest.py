import pytest
from httpx import ASGITransport, AsyncClient

from fastapi import FastAPI
from embedding_gateway.backends.ollama import OllamaBackend
from embedding_gateway.backends.tei import TEIBackend
from embedding_gateway.registry import ModelRegistry
from embedding_gateway.health import health_router
from embedding_gateway.router import router
from embedding_gateway import health as health_module
from embedding_gateway import router as router_module


@pytest.fixture
async def client():
    # Set up registry directly (ASGITransport doesn't trigger lifespan)
    reg = ModelRegistry()

    ollama = OllamaBackend(base_url="http://localhost:11434", timeout=5.0)
    reg.register_backend("ollama", ollama)

    tei = TEIBackend(
        base_url="http://localhost:8080",
        default_model="intfloat/multilingual-e5-large-instruct",
        timeout=5.0,
    )
    reg.register_backend("tei", tei)

    for m in ["bge-m3", "snowflake-arctic-embed2"]:
        reg.register_model(m, ollama)
    reg.register_model("intfloat/multilingual-e5-large-instruct", tei)

    router_module.registry = reg
    health_module.registry = reg

    app = FastAPI()
    app.include_router(router)
    app.include_router(health_router)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    await ollama.close()
    await tei.close()
    router_module.registry = None
    health_module.registry = None

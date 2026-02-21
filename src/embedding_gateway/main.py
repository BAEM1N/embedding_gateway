from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from embedding_gateway.backends.ollama import OllamaBackend
from embedding_gateway.backends.tei import TEIBackend
from embedding_gateway.config import settings
from embedding_gateway.health import health_router
from embedding_gateway.registry import ModelRegistry
from embedding_gateway.router import router
from embedding_gateway import health as health_module
from embedding_gateway import router as router_module


@asynccontextmanager
async def lifespan(app: FastAPI):
    reg = ModelRegistry()

    # Ollama backend
    ollama = OllamaBackend(
        base_url=settings.ollama_base_url,
        timeout=settings.backend_timeout,
    )
    reg.register_backend("ollama", ollama)

    # TEI backend
    tei = TEIBackend(
        base_url=settings.tei_base_url,
        default_model=settings.tei_default_model,
        timeout=settings.backend_timeout,
    )
    reg.register_backend("tei", tei)

    # Pre-register known model mappings
    for m in ["bge-m3", "snowflake-arctic-embed2", "qwen3-embedding:0.6b"]:
        reg.register_model(m, ollama)
    for m in [settings.tei_default_model]:
        reg.register_model(m, tei)

    # Auto-discover additional models from running backends
    await reg.discover_models()

    # Wire registry into routers
    router_module.registry = reg
    health_module.registry = reg

    yield

    # Cleanup
    await ollama.close()
    await tei.close()


app = FastAPI(
    title="Embedding Gateway",
    description="Unified OpenAI-compatible embedding API for Ollama and TEI backends",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
app.include_router(health_router)


def main():
    uvicorn.run(
        "embedding_gateway.main:app",
        host=settings.gateway_host,
        port=settings.gateway_port,
        reload=True,
    )


if __name__ == "__main__":
    main()

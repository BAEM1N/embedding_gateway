from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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

    # TEI backend (dynamic model swapping)
    tei_models = settings.get_tei_model_list()
    tei = TEIBackend(
        base_url=settings.tei_base_url,
        default_model=settings.tei_default_model,
        available_models=tei_models,
        docker_image=settings.tei_docker_image,
        container_name=settings.tei_container_name,
        wsl_distro=settings.tei_wsl_distro,
        swap_timeout=settings.tei_swap_timeout,
        timeout=settings.backend_timeout,
        hf_token=settings.hf_token,
    )
    await tei.initialize()
    reg.register_backend("tei", tei)

    # Pre-register known Ollama embedding models
    ollama_models = [
        # Qwen3 Embedding (GGUF 양자화)
        "qwen3-embedding:0.6b",
        "qwen3-embedding:4b",
        "qwen3-embedding:8b",
        # BAAI
        "bge-m3",
        "bge-large",
        # Nomic
        "nomic-embed-text",
        "nomic-embed-text-v2-moe",
        # Google / Mixedbread
        "embeddinggemma",
        "mxbai-embed-large",
        # Snowflake
        "snowflake-arctic-embed2",
        "snowflake-arctic-embed:335m",
        # Sentence Transformers
        "all-minilm:33m",
        "paraphrase-multilingual",
        # IBM Granite
        "granite-embedding:278m",
    ]
    for m in ollama_models:
        reg.register_model(m, ollama)

    # TEI models (safetensors fp16, dynamic swap)
    for m in tei_models:
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
    version="0.3.0",
    lifespan=lifespan,
)

app.include_router(router)
app.include_router(health_router)

# Static files & playground
_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/playground")
async def playground():
    return FileResponse(str(_static_dir / "playground.html"))


def main():
    uvicorn.run(
        "embedding_gateway.main:app",
        host=settings.gateway_host,
        port=settings.gateway_port,
        reload=True,
    )


if __name__ == "__main__":
    main()

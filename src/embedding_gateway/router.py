from fastapi import APIRouter, HTTPException

from embedding_gateway.models import (
    EmbeddingRequest,
    EmbeddingResponse,
    ModelInfo,
    ModelListResponse,
)
from embedding_gateway.registry import ModelRegistry

router = APIRouter()

# Set during app startup via lifespan
registry: ModelRegistry | None = None


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    if registry is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    backend = registry.get_backend(request.model)
    if backend is None:
        available = ", ".join(registry.all_model_names()) or "(none)"
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model}' not found. Available: {available}",
        )

    texts = request.input if isinstance(request.input, list) else [request.input]

    try:
        return await backend.embed(texts, request.model, request.dimensions)
    except Exception as e:
        msg = str(e) or f"{type(e).__name__} (no message)"
        raise HTTPException(status_code=502, detail=f"Backend error: {msg}")


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    if registry is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    models: list[ModelInfo] = []
    for backend_name, backend in registry.backends.items():
        try:
            backend_models = await backend.list_models()
            for m in backend_models:
                models.append(
                    ModelInfo(id=m, owned_by=backend_name, backend=backend_name)
                )
        except Exception:
            pass

    return ModelListResponse(data=models)

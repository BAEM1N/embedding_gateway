from fastapi import APIRouter

from embedding_gateway.registry import ModelRegistry

health_router = APIRouter(tags=["health"])

# Set during app startup via lifespan
registry: ModelRegistry | None = None


@health_router.get("/health")
async def health() -> dict:
    if registry is None:
        return {"status": "unhealthy", "detail": "Not initialized"}

    results = {}
    overall = "healthy"
    for name, backend in registry.backends.items():
        check = await backend.health_check()
        results[name] = check
        if check.get("status") != "healthy":
            overall = "degraded"

    return {"status": overall, "backends": results}


@health_router.get("/health/ready")
async def readiness() -> dict:
    """Returns ready=True if at least one backend is healthy."""
    if registry is None:
        return {"ready": False}

    for _, backend in registry.backends.items():
        check = await backend.health_check()
        if check.get("status") == "healthy":
            return {"ready": True}

    return {"ready": False}

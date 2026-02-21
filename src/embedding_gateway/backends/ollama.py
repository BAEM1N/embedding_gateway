import httpx

from embedding_gateway.backends.base import EmbeddingBackend
from embedding_gateway.models import EmbeddingData, EmbeddingResponse, UsageInfo


class OllamaBackend(EmbeddingBackend):
    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def embed(
        self,
        texts: list[str],
        model: str,
        dimensions: int | None = None,
    ) -> EmbeddingResponse:
        response = await self.client.post(
            "/api/embed",
            json={"model": model, "input": texts},
        )
        response.raise_for_status()
        data = response.json()

        embeddings = data["embeddings"]
        if dimensions:
            embeddings = [emb[:dimensions] for emb in embeddings]

        return EmbeddingResponse(
            data=[
                EmbeddingData(embedding=emb, index=i)
                for i, emb in enumerate(embeddings)
            ],
            model=model,
            usage=UsageInfo(
                prompt_tokens=data.get("prompt_eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0),
            ),
        )

    async def health_check(self) -> dict:
        try:
            r = await self.client.get("/", timeout=5.0)
            return {"status": "healthy" if r.status_code == 200 else "unhealthy"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def list_models(self) -> list[str]:
        r = await self.client.get("/api/tags")
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    async def close(self) -> None:
        await self.client.aclose()

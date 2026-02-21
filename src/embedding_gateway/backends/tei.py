import httpx

from embedding_gateway.backends.base import EmbeddingBackend
from embedding_gateway.models import EmbeddingData, EmbeddingResponse, UsageInfo


class TEIBackend(EmbeddingBackend):
    def __init__(self, base_url: str, default_model: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def embed(
        self,
        texts: list[str],
        model: str,
        dimensions: int | None = None,
    ) -> EmbeddingResponse:
        response = await self.client.post(
            "/v1/embeddings",
            json={"input": texts, "model": model},
        )
        response.raise_for_status()
        data = response.json()

        embeddings_data = []
        for d in data["data"]:
            emb = d["embedding"]
            if dimensions:
                emb = emb[:dimensions]
            embeddings_data.append(EmbeddingData(embedding=emb, index=d["index"]))

        return EmbeddingResponse(
            data=embeddings_data,
            model=model,
            usage=UsageInfo(
                prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
                total_tokens=data.get("usage", {}).get("total_tokens", 0),
            ),
        )

    async def health_check(self) -> dict:
        try:
            r = await self.client.get("/health", timeout=5.0)
            return {"status": "healthy" if r.status_code == 200 else "unhealthy"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def list_models(self) -> list[str]:
        try:
            r = await self.client.get("/info")
            r.raise_for_status()
            return [r.json().get("model_id", self.default_model)]
        except Exception:
            return [self.default_model]

    async def close(self) -> None:
        await self.client.aclose()

from abc import ABC, abstractmethod

from embedding_gateway.models import EmbeddingResponse


class EmbeddingBackend(ABC):
    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        model: str,
        dimensions: int | None = None,
    ) -> EmbeddingResponse: ...

    @abstractmethod
    async def health_check(self) -> dict: ...

    @abstractmethod
    async def list_models(self) -> list[str]: ...

    @abstractmethod
    async def close(self) -> None: ...

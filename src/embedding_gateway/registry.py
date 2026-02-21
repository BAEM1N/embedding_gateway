from embedding_gateway.backends.base import EmbeddingBackend


class ModelRegistry:
    def __init__(self) -> None:
        self._model_map: dict[str, EmbeddingBackend] = {}
        self.backends: dict[str, EmbeddingBackend] = {}

    def register_backend(self, name: str, backend: EmbeddingBackend) -> None:
        self.backends[name] = backend

    def register_model(self, model_name: str, backend: EmbeddingBackend) -> None:
        self._model_map[model_name] = backend

    def get_backend(self, model_name: str) -> EmbeddingBackend | None:
        # Exact match
        if model_name in self._model_map:
            return self._model_map[model_name]
        # Partial match (e.g. "bge-m3:latest" matches "bge-m3")
        for registered, backend in self._model_map.items():
            if model_name.startswith(registered) or registered.startswith(model_name):
                return backend
        return None

    async def discover_models(self) -> None:
        """Auto-discover models from all backends and register them."""
        for backend_name, backend in self.backends.items():
            try:
                models = await backend.list_models()
                for model in models:
                    if model not in self._model_map:
                        self.register_model(model, backend)
            except Exception:
                pass  # Backend might be offline

    def all_model_names(self) -> list[str]:
        return list(self._model_map.keys())

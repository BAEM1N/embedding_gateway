from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Server
    gateway_host: str = "0.0.0.0"
    gateway_port: int = 8000

    # Backend URLs
    ollama_base_url: str = "http://localhost:11434"
    tei_base_url: str = "http://localhost:8080"

    # Default models
    ollama_default_model: str = "bge-m3"
    tei_default_model: str = "intfloat/multilingual-e5-large-instruct"

    # TEI dynamic model swapping
    tei_models: str = "intfloat/multilingual-e5-large-instruct"
    tei_docker_image: str = (
        "ghcr.io/huggingface/text-embeddings-inference:89-1.9"
    )
    tei_container_name: str = "tei-embeddings"
    tei_swap_timeout: float = 120.0
    tei_wsl_distro: str = "Ubuntu-24.04"

    # HuggingFace token (gated 모델 접근용)
    hf_token: str = ""

    # Timeouts (seconds)
    backend_timeout: float = 120.0
    health_check_timeout: float = 5.0

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def get_tei_model_list(self) -> list[str]:
        return [m.strip() for m in self.tei_models.split(",") if m.strip()]


settings = Settings()

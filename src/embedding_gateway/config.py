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

    # Timeouts (seconds)
    backend_timeout: float = 120.0
    health_check_timeout: float = 5.0

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

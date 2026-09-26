import os
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _resolve_vector_db_path() -> str:
    """Resolve VECTOR_DB_PATH or the active versioned index."""
    configured = os.getenv("VECTOR_DB_PATH", "").strip()
    if configured:
        return configured
    current_file = Path("indexes/current.txt")
    if current_file.exists():
        version = current_file.read_text(encoding="utf-8").strip()
        if version and Path(version).name == version:
            return str(Path("indexes") / version)
    return "./regulatory_chroma_db"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ALLOWED_ORIGINS: str = "http://localhost:8000"
    AUTH_REQUIRED: bool = True
    AUTH_TOKEN: str = ""
    LOG_LEVEL: str = "INFO"
    CORPUS_VERSION: str = "2026-09-24"

    GEMINI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    DEEPSEEK_API_KEY: str = ""
    COHERE_API_KEY: str = ""
    MISTRAL_API_KEY: str = ""
    OLLAMA_HOST: str = "http://localhost:11434"

    DEFAULT_LLM_PROVIDER: str = "gemini"
    DEFAULT_LLM_MODEL: str = "gemini/gemini-2.5-flash"
    EMBEDDING_PROVIDER: str = "gemini"
    EMBEDDING_MODEL: str = "gemini/gemini-embedding-001"
    PLANNER_LLM_MODEL: str = ""
    SPECIALIST_LLM_MODEL: str = ""
    AGGREGATOR_LLM_MODEL: str = ""
    VERIFIER_LLM_MODEL: str = ""

    VECTOR_DB_TYPE: str = "chroma"
    VECTOR_DB_PATH: str = Field(default_factory=_resolve_vector_db_path)
    RAG_TOP_K: int = Field(default=4, ge=1, le=50)
    RAG_SIMILARITY_THRESHOLD: float = Field(default=0.75, ge=0.0, le=1.0)
    MAX_CONCURRENT_QUERIES: int = Field(default=5, ge=1, le=100)
    QUERY_TIMEOUT_SECONDS: float = Field(default=120.0, gt=0.0, le=600.0)
    QUEUE_TIMEOUT_SECONDS: float = Field(default=10.0, gt=0.0, le=120.0)

    @model_validator(mode="after")
    def validate_auth_configuration(self):
        if self.AUTH_REQUIRED:
            token = self.AUTH_TOKEN.strip()
            if not token or token == "change-me-in-production" or len(token) < 32:
                raise ValueError(
                    "AUTH_REQUIRED is enabled. Set AUTH_TOKEN to a unique secret "
                    "containing at least 32 characters."
                )
        return self

    @property
    def auth_required(self) -> bool:
        return self.AUTH_REQUIRED

    @property
    def auth_token(self) -> str:
        return self.AUTH_TOKEN

    @property
    def corpus_version(self) -> str:
        return self.CORPUS_VERSION

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]


settings = Settings()

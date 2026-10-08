import os
from pathlib import Path

from pydantic_settings import BaseSettings


def _resolve_vector_db_path() -> str:
    """
    Resolve the active vector DB path.
    Priority: explicit VECTOR_DB_PATH env var > indexes/current.txt pointer > default.
    """
    configured = os.getenv("VECTOR_DB_PATH", "").strip()
    if configured:
        return configured

    current_file = Path("indexes/current.txt")
    if current_file.exists():
        version = current_file.read_text(encoding="utf-8").strip()
        if version:
            return str(Path("indexes") / version)

    return "./regulatory_chroma_db"


class Settings(BaseSettings):
    # Server Settings
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000")
    AUTH_REQUIRED: bool = os.getenv("AUTH_REQUIRED", "false").lower() == "true"
    AUTH_TOKEN: str = os.getenv("AUTH_TOKEN", "")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    CORPUS_VERSION: str = os.getenv("CORPUS_VERSION", "2026-09-24")
    QUERY_TIMEOUT_SECONDS: float = float(os.getenv("QUERY_TIMEOUT_SECONDS", "120"))
    MAX_CONCURRENT_QUERIES: int = int(os.getenv("MAX_CONCURRENT_QUERIES", "5"))

    # Lowercase property accessors
    @property
    def auth_required(self) -> bool:
        return self.AUTH_REQUIRED

    @property
    def corpus_version(self) -> str:
        return self.CORPUS_VERSION

    @property
    def auth_token(self) -> str:
        return self.AUTH_TOKEN

    @property
    def allowed_origins(self) -> list[str]:
        if isinstance(self.ALLOWED_ORIGINS, str):
            return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]
        return self.ALLOWED_ORIGINS

    # Multi-LLM Provider API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")

    # Local Ollama Host
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # Model Settings - Dynamic LiteLLM Format
    DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "ollama")
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "ollama/llama3.1:8b")

    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "ollama")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "ollama/nomic-embed-text")
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))

    # Vector DB — now resolved via indexes/current.txt if VECTOR_DB_PATH isn't set explicitly
    VECTOR_DB_TYPE: str = os.getenv("VECTOR_DB_TYPE", "chroma")
    VECTOR_DB_PATH: str = _resolve_vector_db_path()
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "4"))
    RAG_SIMILARITY_THRESHOLD: float = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.75"))
    ENABLE_AUTH_TOKEN_VALIDATION: bool = os.getenv("ENABLE_AUTH_TOKEN_VALIDATION", "true").lower() == "true"


settings = Settings()


def validate_auth_settings() -> None:
    """Fail fast on insecure auth configuration when auth is enabled."""
    if not settings.ENABLE_AUTH_TOKEN_VALIDATION or not settings.auth_required:
        return

    token = (settings.auth_token or "").strip()
    weak_tokens = {"", "change-me-in-production", "changeme", "default", "password"}
    if token.lower() in weak_tokens or len(token) < 16:
        raise ValueError(
            "AUTH_REQUIRED is enabled but AUTH_TOKEN is missing or too weak. "
            "Set AUTH_TOKEN to a strong secret (16+ chars)."
        )
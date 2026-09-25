import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Server Settings
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000")
    AUTH_REQUIRED: bool = os.getenv("AUTH_REQUIRED", "true").lower() == "true"
    AUTH_TOKEN: str = os.getenv("AUTH_TOKEN", "change-me-in-production")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    CORPUS_VERSION: str = os.getenv("CORPUS_VERSION", "2026-09-24")

    # Lowercase property accessors
    @property
    def auth_required(self) -> bool:
        return self.AUTH_REQUIRED

    @property
    def corpus_version(self) -> str:
        return self.CORPUS_VERSION

    @property
    def allowed_origins(self) -> list[str]:
        if isinstance(self.ALLOWED_ORIGINS, str):
            return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]
        return self.ALLOWED_ORIGINS

    # Model Settings - Dynamic LiteLLM Format
    # Formats: "gemini/gemini-2.5-flash", "gpt-4o", "anthropic/claude-3-5-sonnet-20241022", "ollama/llama3"
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "gemini/gemini-2.5-flash")
    
    # Formats: "gemini/text-embedding-004", "text-embedding-3-small", "cohere/embed-english-v3.0"
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini/text-embedding-004")

    # Vector DB
    VECTOR_DB_TYPE: str = os.getenv("VECTOR_DB_TYPE", "chroma")
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./regulatory_chroma_db")
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "4"))

settings = Settings()
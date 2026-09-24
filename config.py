from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    allowed_origins: list[str]
    auth_required: bool
    ollama_host: str
    vector_db_path: str
    log_level: str
    corpus_version: str


def _split_csv(value: str | None, default: str) -> list[str]:
    raw = value if value else default
    return [item.strip() for item in raw.split(",") if item.strip()]


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


settings = Settings(
    allowed_origins=_split_csv(os.getenv("ALLOWED_ORIGINS"), "http://localhost:8000"),
    auth_required=_bool_env("AUTH_REQUIRED", False),
    ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    vector_db_path=os.getenv("VECTOR_DB_PATH", "./regulatory_chroma_db"),
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    corpus_version=os.getenv("CORPUS_VERSION", "2026-09-24"),
)

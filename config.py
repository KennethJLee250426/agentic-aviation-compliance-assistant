from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    allowed_origins: list[str]
    auth_required: bool
    auth_token: str
    ollama_host: str
    vector_db_path: str
    log_level: str
    corpus_version: str
    worker_model: str
    verifier_model: str


def _split_csv(value: str | None, default: str) -> list[str]:
    raw = value if value else default
    return [item.strip() for item in raw.split(",") if item.strip()]


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_vector_db_path() -> str:
    configured = os.getenv("VECTOR_DB_PATH", "").strip()
    if configured:
        return configured

    current_file = Path("indexes/current.txt")
    if current_file.exists():
        version = current_file.read_text(encoding="utf-8").strip()
        if version:
            return str(Path("indexes") / version)

    return "./regulatory_chroma_db"


settings = Settings(
    allowed_origins=_split_csv(os.getenv("ALLOWED_ORIGINS"), "http://localhost:8000"),
    auth_required=_bool_env("AUTH_REQUIRED", False),
    auth_token=os.getenv("AUTH_TOKEN", "change-me"),
    ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    vector_db_path=resolve_vector_db_path(),
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    corpus_version=os.getenv("CORPUS_VERSION", "2026-09-24"),
    worker_model=os.getenv("WORKER_MODEL", "llama3.1:8b"),
    verifier_model=os.getenv("VERIFIER_MODEL", "deepseek-r1:8b"),
)

import gc
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from litellm import embedding
from pypdf import PdfReader

from config import settings
from graph import resolve_embedding_model


def file_sha256(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as source:
        for chunk in iter(lambda: source.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_embedding(text: str) -> list[float]:
    model, provider = resolve_embedding_model()
    keys = {
        "gemini": settings.GEMINI_API_KEY,
        "openai": settings.OPENAI_API_KEY,
        "cohere": settings.COHERE_API_KEY,
        "mistral": settings.MISTRAL_API_KEY,
    }
    response = embedding(
        model=model,
        input=[text],
        api_key=keys.get(provider) or None,
        timeout=settings.QUERY_TIMEOUT_SECONDS,
    )
    return response.data[0]["embedding"]


def load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def load_docx(file_path: str) -> str:
    document = DocxDocument(file_path)
    return "\n\n".join(p.text for p in document.paragraphs if p.text.strip())


def load_xml(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as source:
        return BeautifulSoup(source.read(), "xml").get_text(separator="\n")


def load_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as source:
        return source.read()


LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".xml": load_xml,
    ".txt": load_text,
    ".md": load_text,
}


def _target_index_path() -> Path:
    version = settings.CORPUS_VERSION.strip()
    if not version or Path(version).name != version or version in {".", ".."}:
        raise ValueError("CORPUS_VERSION must be a simple directory name.")
    # Respect an explicit path override supplied through the environment or .env.
    if "VECTOR_DB_PATH" in settings.model_fields_set:
        return Path(settings.VECTOR_DB_PATH)
    return Path("indexes") / version


def _write_current_pointer(indexes_dir: Path, version: str) -> None:
    indexes_dir.mkdir(parents=True, exist_ok=True)
    temp_path = indexes_dir / ".current.txt.tmp"
    temp_path.write_text(version, encoding="utf-8")
    os.replace(temp_path, indexes_dir / "current.txt")


def run_ingestion():
    version = settings.CORPUS_VERSION
    final_dir = _target_index_path()
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        raise FileExistsError(
            f"Index path already exists: {final_dir}. Use a new CORPUS_VERSION "
            "or a new VECTOR_DB_PATH to avoid modifying the active index."
        )

    staging_dir = final_dir.with_name(f".{final_dir.name}.staging")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    generated_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "corpus_version": version,
        "generated_at": generated_at,
        "embedding_model": settings.EMBEDDING_MODEL,
        "documents": [],
        "failed_documents": [],
    }

    client = chromadb.PersistentClient(path=str(staging_dir))
    collection = client.get_or_create_collection(name="aviation_regulations")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunk_count = 0
    regulatory_dirs = {
        "EASA": Path("regulations/easa"),
        "CAAS": Path("regulations/caas"),
        "CAAC": Path("regulations/caac"),
    }

    for authority, directory in regulatory_dirs.items():
        if not directory.is_dir():
            continue
        for source_path in sorted(directory.iterdir()):
            if not source_path.is_file():
                continue
            loader = LOADERS.get(source_path.suffix.lower())
            if loader is None:
                continue

            relative_source = source_path.as_posix()
            source_hash = file_sha256(str(source_path))
            try:
                text = loader(str(source_path))
                if not text.strip():
                    raise ValueError("No extractable text found")
                chunks = splitter.split_text(text)
                for index, chunk in enumerate(chunks):
                    stable_key = f"{authority}\0{relative_source}\0{source_hash}\0{index}"
                    chunk_id = hashlib.sha256(stable_key.encode("utf-8")).hexdigest()
                    vector = get_embedding(chunk)
                    collection.upsert(
                        ids=[chunk_id],
                        embeddings=[vector],
                        documents=[chunk],
                        metadatas=[{
                            "authority": authority,
                            "source_file": source_path.name,
                            "source_hash": source_hash,
                            "chunk_id": index,
                            "corpus_version": version,
                            "ingested_at": generated_at,
                        }],
                    )
                    chunk_count += 1
                manifest["documents"].append({
                    "source": relative_source,
                    "authority": authority,
                    "sha256": source_hash,
                    "chunks": len(chunks),
                    "status": "ok",
                })
            except Exception as error:
                manifest["failed_documents"].append({
                    "source": relative_source,
                    "authority": authority,
                    "error": str(error),
                })

    if chunk_count == 0 or manifest["failed_documents"]:
        (staging_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        del collection, client
        gc.collect()
        raise RuntimeError(
            f"Index build failed: {chunk_count} chunks created and "
            f"{len(manifest['failed_documents'])} document failures. "
            f"Diagnostics: {staging_dir / 'manifest.json'}"
        )

    manifest["chunk_count"] = chunk_count
    (staging_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    del collection, client
    gc.collect()

    os.replace(staging_dir, final_dir)
    if "VECTOR_DB_PATH" not in settings.model_fields_set:
        _write_current_pointer(final_dir.parent, version)

    print(f"Ingestion complete: {chunk_count} chunks indexed at {final_dir}")


if __name__ == "__main__":
    run_ingestion()

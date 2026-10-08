import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

import chromadb
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from litellm import embedding
from pypdf import PdfReader

from config import settings
from graph import resolve_embedding_model

load_dotenv()


def file_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate vector embeddings using the configured embedding provider/model."""
    embed_model, embed_provider = resolve_embedding_model(
        provider=getattr(settings, "EMBEDDING_PROVIDER", None)
    )

    active_key = None
    if embed_provider == "gemini":
        active_key = getattr(settings, "GEMINI_API_KEY", None)
    elif embed_provider == "openai":
        active_key = getattr(settings, "OPENAI_API_KEY", None)
    elif embed_provider == "cohere":
        active_key = getattr(settings, "COHERE_API_KEY", None)
    elif embed_provider == "mistral":
        active_key = getattr(settings, "MISTRAL_API_KEY", None)

    response = embedding(
        model=embed_model,
        input=texts,
        api_key=active_key if active_key else None,
    )
    return [item["embedding"] for item in response.data]


def load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def load_docx(file_path: str) -> str:
    docx_file = DocxDocument(file_path)
    paragraphs = [p.text for p in docx_file.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def load_xml(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        xml_content = f.read()
    soup = BeautifulSoup(xml_content, "xml")
    return soup.get_text(separator="\n")


def load_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".xml": load_xml,
    ".txt": load_text,
    ".md": load_text,
}


def _existing_ids_for_file(collection: Any, authority: str, source_file: str) -> list[str]:
    existing = collection.get(
        where={"$and": [{"authority": authority}, {"source_file": source_file}]},
        include=[],
    )
    return existing.get("ids", []) if isinstance(existing, dict) else []


def _existing_ids_for_hash(collection: Any, authority: str, source_file: str, source_hash: str) -> list[str]:
    existing = collection.get(
        where={"$and": [{"authority": authority}, {"source_file": source_file}, {"source_hash": source_hash}]},
        include=[],
    )
    return existing.get("ids", []) if isinstance(existing, dict) else []


def run_ingestion() -> None:
    corpus_version = settings.CORPUS_VERSION
    persist_dir = os.path.join("indexes", corpus_version)
    os.makedirs(persist_dir, exist_ok=True)

    print(f"Initializing Chroma DB at {persist_dir}...")
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name="aviation_regulations")

    manifest: dict[str, Any] = {
        "corpus_version": corpus_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "documents": [],
        "failed_documents": [],
        "skipped_documents": [],
    }

    regulatory_dirs = {
        "easa": "regulations/easa",
        "caas": "regulations/caas",
        "caac": "regulations/caac",
    }
    doc_id = 0
    batch_size = max(1, int(getattr(settings, "EMBEDDING_BATCH_SIZE", 8)))

    for authority_key, reg_dir in regulatory_dirs.items():
        if not os.path.exists(reg_dir):
            print(f"Directory not found (skipping): {reg_dir}")
            continue

        authority = authority_key.upper()

        for file in sorted(os.listdir(reg_dir)):
            file_path = os.path.join(reg_dir, file)
            if not os.path.isfile(file_path):
                continue

            ext = os.path.splitext(file)[1].lower()
            loader = LOADERS.get(ext)
            if loader is None:
                print(f" -> Skipping unsupported file type: {file_path}")
                continue

            print(f"Processing ({authority}): {file_path}")
            source_hash = file_sha256(file_path)

            try:
                existing_same_hash = _existing_ids_for_hash(collection, authority, file, source_hash)
                if existing_same_hash:
                    print(f" -> Unchanged file detected; skipping: {file_path}")
                    manifest["skipped_documents"].append(
                        {
                            "source": file_path,
                            "authority": authority,
                            "sha256": source_hash,
                            "status": "unchanged",
                            "chunks": len(existing_same_hash),
                        }
                    )
                    continue

                stale_ids = _existing_ids_for_file(collection, authority, file)
                if stale_ids:
                    collection.delete(ids=stale_ids)

                content = loader(file_path)
                if not content.strip():
                    raise ValueError("No extractable text found in file")

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = splitter.split_text(content)

                chunk_embeddings: list[list[float]] = []
                for start in range(0, len(chunks), batch_size):
                    chunk_embeddings.extend(get_embeddings(chunks[start : start + batch_size]))

                ids: list[str] = []
                metadatas: list[dict[str, Any]] = []
                for idx, _chunk in enumerate(chunks):
                    doc_id += 1
                    ids.append(f"{authority}_{doc_id}")
                    metadatas.append(
                        {
                            "authority": authority,
                            "source_file": file,
                            "source_hash": source_hash,
                            "chunk_id": idx,
                            "corpus_version": corpus_version,
                            "ingested_at": manifest["generated_at"],
                        }
                    )

                collection.add(
                    ids=ids,
                    embeddings=chunk_embeddings,
                    documents=chunks,
                    metadatas=metadatas,
                )

                manifest["documents"].append(
                    {
                        "source": file_path,
                        "authority": authority,
                        "sha256": source_hash,
                        "chunks": len(chunks),
                        "status": "ok",
                    }
                )

            except Exception as e:  # noqa: BLE001
                print(f" -> FAILED to load {file_path}: {e}")
                manifest["failed_documents"].append(
                    {
                        "source": file_path,
                        "authority": authority,
                        "error": str(e),
                    }
                )

    total_chunks = collection.count()
    if total_chunks == 0:
        print("CRITICAL: No chunks were created — check regulations/ folders and file types.")
        return

    manifest_path = os.path.join("indexes", f"{corpus_version}-manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    current_path = os.path.join("indexes", "current.txt")
    with open(current_path, "w", encoding="utf-8") as f:
        f.write(corpus_version)

    print(f"Ingestion complete! Total regulatory chunks stored: {total_chunks}")
    print(f"Successful files: {len(manifest['documents'])}")
    print(f"Skipped unchanged files: {len(manifest['skipped_documents'])}")
    print(f"Failed files: {len(manifest['failed_documents'])}")
    print(f"Manifest saved to: {manifest_path}")
    print(f"Active index pointer updated: {current_path} -> {corpus_version}")


if __name__ == "__main__":
    run_ingestion()

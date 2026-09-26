import hashlib
import json
import os
from datetime import datetime, timezone

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


def get_embedding(text: str) -> list[float]:
    """Generate vector embedding using the configured embedding provider/model."""
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
        input=[text],
        api_key=active_key if active_key else None,
    )
    return response.data[0]["embedding"]


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


def run_ingestion():
    corpus_version = settings.CORPUS_VERSION
    persist_dir = os.path.join("indexes", corpus_version)
    os.makedirs(persist_dir, exist_ok=True)

    print(f"Initializing Chroma DB at {persist_dir}...")
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name="aviation_regulations")

    manifest = {
        "corpus_version": corpus_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "documents": [],
        "failed_documents": [],
    }

    regulatory_dirs = {
        "easa": "regulations/easa",
        "caas": "regulations/caas",
        "caac": "regulations/caac",
    }
    doc_id = 0

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
                content = loader(file_path)
                if not content.strip():
                    raise ValueError("No extractable text found in file")

                # Smarter chunking via LangChain's splitter (still no full LangChain/LangGraph dependency)
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = splitter.split_text(content)

                for idx, chunk in enumerate(chunks):
                    doc_id += 1
                    vec = get_embedding(chunk)

                    metadata = {
                        "authority": authority,
                        "source_file": file,
                        "source_hash": source_hash,
                        "chunk_id": idx,
                        "corpus_version": corpus_version,
                        "ingested_at": manifest["generated_at"],
                    }

                    collection.add(
                        ids=[f"{authority}_{doc_id}"],
                        embeddings=[vec],
                        documents=[chunk],
                        metadatas=[metadata],
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

            except Exception as e: # noqa: BLE001
                print(f" -> FAILED to load {file_path}: {e}")
                manifest["failed_documents"].append(
                    {
                        "source": file_path,
                        "authority": authority,
                        "error": str(e),
                    }
                )

    if doc_id == 0:
        print("CRITICAL: No chunks were created — check regulations/ folders and file types.")
        return

    manifest_path = os.path.join("indexes", f"{corpus_version}-manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    current_path = os.path.join("indexes", "current.txt")
    with open(current_path, "w", encoding="utf-8") as f:
        f.write(corpus_version)

    print(f"Ingestion complete! Total regulatory chunks stored: {doc_id}")
    print(f"Manifest saved to: {manifest_path}")
    print(f"Active index pointer updated: {current_path} -> {corpus_version}")


if __name__ == "__main__":
    run_ingestion()
import hashlib
import json
import os
from datetime import datetime, timezone

from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings


def file_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_vector_db():
    docs = []
    authorities = ["easa", "caas", "caac"]
    base_dir = "regulations"
    manifest = {
        "corpus_version": settings.corpus_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "documents": [],
        "failed_documents": [],
    }

    print("=== STARTING INGESTION ===")

    for authority in authorities:
        folder_path = os.path.join(base_dir, authority)
        if not os.path.exists(folder_path):
            continue

        for file in sorted(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, file)
            if not os.path.isfile(file_path):
                continue

            file_lower = file.lower()
            source_hash = file_sha256(file_path)

            try:
                if file_lower.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    pdf_docs = loader.load()
                    for doc in pdf_docs:
                        doc.metadata = dict(doc.metadata or {})
                        doc.metadata["source"] = file_path
                        doc.metadata["authority"] = authority.upper()
                        doc.metadata["source_hash"] = source_hash
                        doc.metadata["ingested_at"] = manifest["generated_at"]
                        doc.metadata["corpus_version"] = settings.corpus_version
                        docs.append(doc)
                        manifest["documents"].append(
                            {
                                "source": file_path,
                                "authority": authority.upper(),
                                "sha256": source_hash,
                                "status": "ok",
                            }
                        )
                elif file_lower.endswith(".xml"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        xml_content = f.read()
                    soup = BeautifulSoup(xml_content, "xml")
                    xml_text = soup.get_text(separator="\n")
                    docs.append(
                        Document(
                            page_content=xml_text,
                            metadata={
                                "source": file_path,
                                "authority": authority.upper(),
                                "source_hash": source_hash,
                                "ingested_at": manifest["generated_at"],
                                "corpus_version": settings.corpus_version,
                            },
                        )
                    )
                    manifest["documents"].append(
                        {
                            "source": file_path,
                            "authority": authority.upper(),
                            "sha256": source_hash,
                            "status": "ok",
                        }
                    )
            except Exception as e:
                print(f" -> FAILED to load {file_path}: {e}")
                manifest["failed_documents"].append(
                    {
                        "source": file_path,
                        "authority": authority.upper(),
                        "error": str(e),
                    }
                )

    print(f"Total documents loaded: {len(docs)}")
    if not docs:
        print("CRITICAL: No documents were successfully loaded!")
        return

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(docs)
    if not chunks:
        print("CRITICAL: No chunks were created from the loaded documents!")
        return

    print(f"Split into {len(chunks)} chunks. Initializing embeddings...")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    batch_size = 2000
    persist_dir = os.path.join("indexes", settings.corpus_version)

    os.makedirs(os.path.dirname(persist_dir), exist_ok=True)

    vectorstore = None
    print(f"Embedding and saving chunks in batches of {batch_size}...")
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        print(f" -> Processing batch {i} to {i + len(batch)} of {len(chunks)}...")
        if i == 0:
            vectorstore = Chroma.from_documents(batch, embeddings, persist_directory=persist_dir)
        else:
            if vectorstore is None:
                raise RuntimeError("Vectorstore was not initialized before adding documents.")
            vectorstore.add_documents(batch)

    manifest_path = os.path.join("indexes", f"{settings.corpus_version}-manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    current_path = os.path.join("indexes", "current.txt")
    with open(current_path, "w", encoding="utf-8") as f:
        f.write(settings.corpus_version)

    print("Vector database successfully built and saved locally!")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    build_vector_db()

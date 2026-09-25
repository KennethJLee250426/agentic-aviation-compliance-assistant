import glob
import os

import chromadb
from dotenv import load_dotenv
from litellm import embedding

from config import settings

load_dotenv()

def get_embedding(text: str) -> list[float]:
    """Generate vector embedding using standard Gemini embedding model."""
    response = embedding(
        model=f"gemini/{settings.EMBEDDING_MODEL}",
        input=[text],
        api_key=settings.GEMINI_API_KEY
    )
    return response.data[0]["embedding"]

def run_ingestion():
    print(f"Initializing Chroma DB at {settings.VECTOR_DB_PATH}...")
    client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
    collection = client.get_or_create_collection(name="aviation_regulations")

    regulatory_dirs = ["regulations/easa", "regulations/caas", "regulations/caac"]
    doc_id = 0

    for reg_dir in regulatory_dirs:
        if not os.path.exists(reg_dir):
            print(f"Directory not found (skipping): {reg_dir}")
            continue
            
        files = glob.glob(os.path.join(reg_dir, "*.txt")) + glob.glob(os.path.join(reg_dir, "*.md"))
        
        # Cross-platform authority extraction (e.g., "EASA", "CAAS", "CAAC")
        authority = os.path.basename(os.path.normpath(reg_dir)).upper()

        for file_path in files:
            print(f"Processing ({authority}): {file_path}")
            
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Simple chunking by paragraph/section break
            chunks = [c.strip() for c in content.split("\n\n") if c.strip()]

            for idx, chunk in enumerate(chunks):
                doc_id += 1
                vec = get_embedding(chunk)
                
                metadata = {
                    "authority": authority,
                    "source_file": os.path.basename(file_path),
                    "chunk_id": idx,
                    "corpus_version": settings.CORPUS_VERSION
                }
                
                collection.add(
                    ids=[f"{authority}_{doc_id}"],
                    embeddings=[vec],
                    documents=[chunk],
                    metadatas=[metadata]
                )

    print(f"Ingestion complete! Total regulatory chunks stored: {doc_id}")

if __name__ == "__main__":
    run_ingestion()
import os
from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_vector_db():
  docs = []
  authorities = ["easa", "caas", "caac"]
  base_dir = "regulations"

  print("=== STARTING INGESTION ===")

  for authority in authorities:
    folder_path = os.path.join(base_dir, authority)
    if not os.path.exists(folder_path):
      continue

    files = os.listdir(folder_path)
    for file in files:
      file_path = os.path.join(folder_path, file)
      if not os.path.isfile(file_path):
        continue

      file_lower = file.lower()

      try:
        if file_lower.endswith(".pdf"):
          loader = PyPDFLoader(file_path)
          pdf_docs = loader.load()
          for doc in pdf_docs:
            doc.metadata = dict(doc.metadata or {})
            doc.metadata["source"] = file_path
            doc.metadata["authority"] = authority.upper()
            docs.append(doc)
        elif file_lower.endswith(".xml"):
          with open(file_path, "r", encoding="utf-8") as f:
            xml_content = f.read()
            soup = BeautifulSoup(xml_content, "xml")
            xml_text = soup.get_text(separator="\n")
            docs.append(
                Document(
                    page_content=xml_text,
                    metadata={"source": file_path, "authority": authority.upper()},
                )
            )
      except Exception as e:
        print(f" -> FAILED to load {file_path}: {e}")

  print(f"Total documents loaded: {len(docs)}")
  if not docs:
    print("CRITICAL: No documents were successfully loaded!")
    return

  print("Splitting documents into chunks...")
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000, chunk_overlap=150
  )
  chunks = text_splitter.split_documents(docs)
  if not chunks:
    print("CRITICAL: No chunks were created from the loaded documents!")
    return

  print(f"Split into {len(chunks)} chunks. Initializing embeddings...")

  embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

  # Process in batches to prevent hanging/memory locks
  batch_size = 2000
  persist_dir = "./regulatory_chroma_db"

  vectorstore = None
  print(f"Embedding and saving chunks in batches of {batch_size}...")
  for i in range(0, len(chunks), batch_size):
    batch = chunks[i : i + batch_size]
    print(f" -> Processing batch {i} to {i + len(batch)} of {len(chunks)}...")
    if i == 0:
      vectorstore = Chroma.from_documents(
          batch, embeddings, persist_directory=persist_dir
      )
    else:
      if vectorstore is None:
        raise RuntimeError("Vectorstore was not initialized before adding documents.")
      vectorstore.add_documents(batch)

  print("Vector database successfully built and saved locally!")


if __name__ == "__main__":
  build_vector_db()

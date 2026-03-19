"""
Document loaders for PDF, DOCX, and TXT files.
Returns LangChain Document objects with source metadata.
"""
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.schema import Document


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def load_document(file_path: str) -> list[Document]:
    """Load a single document and return a list of LangChain Documents."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}")

    if ext == ".pdf":
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        # PyPDFLoader already adds 'page' metadata; add 'source' too
        for doc in docs:
            doc.metadata["source"] = path.name

    elif ext == ".docx":
        loader = Docx2txtLoader(str(path))
        docs = loader.load()
        for i, doc in enumerate(docs):
            doc.metadata["source"] = path.name
            doc.metadata["page"] = i  # DOCX has no native page numbers

    elif ext in {".txt", ".md"}:
        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = path.name
            doc.metadata["page"] = 0

    return docs


def load_documents_from_dir(directory: str) -> list[Document]:
    """Load all supported documents from a directory."""
    dir_path = Path(directory)
    all_docs = []

    for file_path in sorted(dir_path.iterdir()):
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                docs = load_document(str(file_path))
                all_docs.extend(docs)
                print(f"  Loaded: {file_path.name} ({len(docs)} pages/sections)")
            except Exception as e:
                print(f"  Warning: Could not load {file_path.name}: {e}")

    print(f"\nTotal documents loaded: {len(all_docs)}")
    return all_docs

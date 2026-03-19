"""
FAISS vector store: build from chunks, persist to disk, and retrieve.
"""
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from app.embeddings import get_embeddings


VECTOR_STORE_PATH = "vector_store"


def build_vector_store(chunks: list[Document], persist_path: str = VECTOR_STORE_PATH) -> FAISS:
    """
    Embed chunks and save a FAISS index to disk.

    Args:
        chunks: Chunked Document objects with metadata.
        persist_path: Directory where the FAISS index is saved.

    Returns:
        FAISS vector store instance.
    """
    embeddings = get_embeddings()
    print("Building FAISS index (this may take a minute for large document sets)...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(persist_path)
    print(f"Vector store saved to '{persist_path}/'")
    return db


def load_vector_store(persist_path: str = VECTOR_STORE_PATH) -> FAISS:
    """
    Load an existing FAISS index from disk.

    Args:
        persist_path: Directory containing saved FAISS files.

    Returns:
        FAISS vector store instance.

    Raises:
        FileNotFoundError: If the index directory doesn't exist.
    """
    if not Path(persist_path).exists():
        raise FileNotFoundError(
            f"No vector store found at '{persist_path}'. "
            "Upload documents first to build the index."
        )
    embeddings = get_embeddings()
    db = FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)
    print(f"Vector store loaded from '{persist_path}/'")
    return db


def retrieve_chunks(
    query: str,
    db: FAISS,
    k: int = 4,
    score_threshold: float = 0.0,
) -> list[Document]:
    """
    Semantic search: return the top-k most relevant chunks.

    Args:
        query: User's natural language question.
        db: Loaded FAISS vector store.
        k: Number of chunks to retrieve.
        score_threshold: Minimum similarity score (0–1). 0 = no filter.

    Returns:
        List of the most relevant Document chunks.
    """
    results = db.similarity_search_with_score(query, k=k)

    # Filter by score if threshold set
    if score_threshold > 0:
        results = [(doc, score) for doc, score in results if score >= score_threshold]

    return [doc for doc, _ in results]

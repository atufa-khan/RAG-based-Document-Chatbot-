"""
Text chunking with metadata preservation.
Chunks inherit page number and source filename from parent documents.
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Document]:
    """
    Split documents into overlapping chunks.

    Args:
        documents: List of LangChain Document objects.
        chunk_size: Max tokens per chunk (characters here; ~1.3 chars per token).
        chunk_overlap: Overlap between consecutive chunks to preserve context.

    Returns:
        List of chunked Document objects with preserved metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Ensure every chunk carries a human-readable citation label
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        # Pages are 0-indexed from PyPDFLoader; display as 1-indexed
        chunk.metadata["citation"] = f"{source}, page {int(page) + 1}"
        chunk.metadata["chunk_id"] = i

    print(f"Created {len(chunks)} chunks from {len(documents)} document sections.")
    return chunks

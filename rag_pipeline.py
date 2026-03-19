"""
RAG pipeline: orchestrates document ingestion and question answering.
Used by both the Streamlit UI and any CLI/API layer.
"""
from pathlib import Path
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from utils.loaders import load_document
from utils.chunking import chunk_documents
from app.retriever import build_vector_store, load_vector_store, retrieve_chunks
from app.generator import generate_answer, list_available_models


class RAGPipeline:
    """
    Encapsulates the full RAG workflow.

    Usage:
        rag = RAGPipeline()
        rag.ingest_file("report.pdf")
        for token in rag.ask("What is the main finding?"):
            print(token, end="", flush=True)
    """

    def __init__(self, vector_store_path: str = "vector_store", model: str = "llama3.2"):
        self.vector_store_path = vector_store_path
        self.model = model
        self.db: FAISS | None = None
        self.chat_history: list[dict] = []
        self._try_load_existing_store()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_file(self, file_path: str) -> int:
        """
        Load, chunk, embed, and index a single file.

        Args:
            file_path: Absolute or relative path to document.

        Returns:
            Number of chunks added to the index.
        """
        print(f"\nIngesting: {file_path}")
        docs = load_document(file_path)
        chunks = chunk_documents(docs)

        if self.db is None:
            self.db = build_vector_store(chunks, self.vector_store_path)
        else:
            # Add to existing index without rebuilding from scratch
            self.db.add_documents(chunks)
            self.db.save_local(self.vector_store_path)
            print(f"Added {len(chunks)} chunks to existing index.")

        return len(chunks)

    def ingest_uploaded_bytes(self, file_bytes: bytes, filename: str, tmp_dir: str = "/tmp") -> int:
        """
        Ingest a file uploaded via Streamlit's st.file_uploader.

        Args:
            file_bytes: Raw bytes from st.file_uploader.read().
            filename: Original filename (used for extension detection and citation).
            tmp_dir: Temp directory to write the file before loading.

        Returns:
            Number of chunks added.
        """
        tmp_path = Path(tmp_dir) / filename
        tmp_path.write_bytes(file_bytes)
        return self.ingest_file(str(tmp_path))

    # ------------------------------------------------------------------
    # Retrieval + Generation
    # ------------------------------------------------------------------

    def ask(self, question: str, k: int = 4, stream: bool = True):
        """
        Answer a question using retrieved context.

        Args:
            question: Natural language question.
            k: Number of context chunks to retrieve.
            stream: Stream tokens (True) or return full string (False).

        Yields/Returns:
            Answer string with inline citations.
        """
        if self.db is None:
            msg = "No documents indexed yet. Please upload documents first."
            if stream:
                yield msg
            else:
                return msg
            return

        chunks = retrieve_chunks(question, self.db, k=k)

        if not chunks:
            msg = "No relevant content found in the uploaded documents."
            if stream:
                yield msg
            else:
                return msg
            return

        answer_tokens = []

        if stream:
            for token in generate_answer(question, chunks, self.chat_history, self.model, stream=True):
                answer_tokens.append(token)
                yield token
        else:
            full = generate_answer(question, chunks, self.chat_history, self.model, stream=False)
            answer_tokens = [full]
            return full

        # Persist turn to history after streaming completes
        full_answer = "".join(answer_tokens)
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": full_answer})

    def get_source_chunks(self, question: str, k: int = 4) -> list[Document]:
        """Return raw chunks used for a question (for source inspection UI)."""
        if self.db is None:
            return []
        return retrieve_chunks(question, self.db, k=k)

    def clear_history(self):
        """Reset the conversation history."""
        self.chat_history = []

    def clear_index(self):
        """Delete the vector store from disk and memory."""
        import shutil
        if Path(self.vector_store_path).exists():
            shutil.rmtree(self.vector_store_path)
        self.db = None
        self.chat_history = []
        print("Vector store cleared.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _try_load_existing_store(self):
        """Load existing FAISS index on startup if available."""
        try:
            self.db = load_vector_store(self.vector_store_path)
        except FileNotFoundError:
            pass  # Fresh start — user will upload docs

    @property
    def is_ready(self) -> bool:
        """True if documents have been indexed and the pipeline can answer questions."""
        return self.db is not None

    @staticmethod
    def available_models() -> list[str]:
        return list_available_models()

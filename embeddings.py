"""
Local embeddings using sentence-transformers.
No API key required. Model is downloaded once and cached by HuggingFace.
"""
from langchain_community.embeddings import HuggingFaceEmbeddings


# Best balance of speed and quality for RAG; 384-dim vectors, ~22M params
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings(model_name: str = DEFAULT_MODEL) -> HuggingFaceEmbeddings:
    """
    Return a HuggingFace embeddings instance.
    Model is downloaded on first call and cached in ~/.cache/huggingface.

    Args:
        model_name: HuggingFace model ID.

    Returns:
        HuggingFaceEmbeddings instance ready for use with FAISS.
    """
    print(f"Loading embedding model: {model_name}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},       # switch to "cuda" if GPU available
        encode_kwargs={"normalize_embeddings": True},  # cosine similarity
    )
    return embeddings

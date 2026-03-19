"""
Answer generation using a local Ollama model.
Builds a citation-aware prompt and streams the response.

Requires Ollama running locally:
  Install: https://ollama.com/download
  Pull a model: ollama pull llama3.2
  Start server: ollama serve  (auto-starts on most installs)
"""
from langchain.schema import Document
import ollama


# Change this to any model you've pulled with `ollama pull <model>`
DEFAULT_MODEL = "llama3.2"

SYSTEM_PROMPT = """You are a precise research assistant that answers questions ONLY from the provided context.

Rules:
1. Answer using ONLY information present in the context below.
2. After each factual statement, add a citation in the format: (Source: <citation>)
3. If the answer is not found in the context, respond exactly: "I don't have enough information in the provided documents to answer this question."
4. Be concise and accurate. Do not make up or infer information not in the context.
5. If multiple sources support a point, cite all of them."""


def build_context(chunks: list[Document]) -> str:
    """
    Format retrieved chunks into a numbered context block with citations.

    Args:
        chunks: Retrieved Document chunks with metadata.

    Returns:
        Formatted context string ready for injection into the prompt.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        citation = chunk.metadata.get("citation", f"chunk {i}")
        parts.append(f"[{i}] {chunk.page_content.strip()}\n(Citation: {citation})")
    return "\n\n".join(parts)


def build_prompt(query: str, context: str, chat_history: list[dict] = None) -> list[dict]:
    """
    Construct the message list for the Ollama chat API.

    Args:
        query: The user's current question.
        context: Formatted retrieved context.
        chat_history: Previous turns as list of {"role": ..., "content": ...} dicts.

    Returns:
        Messages list ready for ollama.chat().
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject past turns (last 6 messages to keep context window manageable)
    if chat_history:
        messages.extend(chat_history[-6:])

    user_message = f"""Context from uploaded documents:
{context}

Question: {query}"""

    messages.append({"role": "user", "content": user_message})
    return messages


def generate_answer(
    query: str,
    chunks: list[Document],
    chat_history: list[dict] = None,
    model: str = DEFAULT_MODEL,
    stream: bool = True,
):
    """
    Generate an answer from retrieved chunks using the local Ollama model.

    Args:
        query: User's question.
        chunks: Retrieved context chunks.
        chat_history: Conversation history for multi-turn support.
        model: Ollama model name (must be pulled first).
        stream: If True, yields tokens as they arrive (for Streamlit streaming).

    Yields (stream=True) or Returns (stream=False):
        Answer string with inline citations.
    """
    context = build_context(chunks)
    messages = build_prompt(query, context, chat_history)

    if stream:
        response = ollama.chat(model=model, messages=messages, stream=True)
        for chunk in response:
            yield chunk["message"]["content"]
    else:
        response = ollama.chat(model=model, messages=messages)
        return response["message"]["content"]


def list_available_models() -> list[str]:
    """Return names of Ollama models currently pulled on this machine."""
    try:
        models = ollama.list()
        return [m["name"] for m in models.get("models", [])]
    except Exception:
        return []

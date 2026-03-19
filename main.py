"""
Streamlit UI for the RAG Document Chatbot.

Run with:
    streamlit run app/main.py
"""
import streamlit as st
from pathlib import Path
import sys

# Ensure project root is on the path when run as `streamlit run app/main.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag_pipeline import RAGPipeline

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .source-card {
    background: var(--secondary-background-color);
    border-left: 3px solid #7F77DD;
    border-radius: 4px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.85rem;
    color: var(--text-color);
  }
  .source-label {
    font-weight: 600;
    color: #7F77DD;
    margin-bottom: 4px;
  }
  .chunk-text {
    opacity: 0.8;
    line-height: 1.5;
  }
  .status-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 600;
  }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline(model=st.session_state.get("model", "llama3.2"))

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

rag: RAGPipeline = st.session_state.rag

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 RAG Chatbot")
    st.caption("Answers grounded in your documents only.")
    st.divider()

    # Model selector
    available_models = RAGPipeline.available_models()
    if available_models:
        selected_model = st.selectbox(
            "Ollama model",
            available_models,
            index=0,
            help="Pull models with: ollama pull llama3.2",
        )
        if selected_model != rag.model:
            rag.model = selected_model
    else:
        st.warning("No Ollama models found.\n\nRun:\n```\nollama pull llama3.2\n```")
        selected_model = st.text_input("Model name", value="llama3.2")
        rag.model = selected_model

    st.divider()

    # Document upload
    st.subheader("Upload documents")
    uploaded_files = st.file_uploader(
        "PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("Index documents", type="primary", use_container_width=True):
            progress = st.progress(0, text="Indexing...")
            for i, f in enumerate(uploaded_files):
                progress.progress((i + 1) / len(uploaded_files), text=f"Indexing {f.name}…")
                rag.ingest_uploaded_bytes(f.read(), f.name)
            progress.empty()
            st.success(f"✅ Indexed {len(uploaded_files)} file(s)")
            st.rerun()

    # Index status
    st.divider()
    if rag.is_ready:
        st.markdown('<span class="status-pill" style="background:#E1F5EE;color:#085041">● Index ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill" style="background:#FAECE7;color:#712B13">● No documents indexed</span>', unsafe_allow_html=True)
        st.caption("Upload and index files to start chatting.")

    # Controls
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Clear chat", use_container_width=True):
            st.session_state.messages = []
            rag.clear_history()
            st.session_state.last_sources = []
            st.rerun()
    with col2:
        if st.button("🗂 Clear index", use_container_width=True):
            rag.clear_index()
            st.session_state.messages = []
            st.session_state.last_sources = []
            st.rerun()

    # Retrieval settings
    with st.expander("⚙ Retrieval settings"):
        k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=4)
        show_sources = st.toggle("Show source chunks", value=True)

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("Document Chat")

if not rag.is_ready:
    st.info("👈 Upload documents in the sidebar to get started.")
    st.stop()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if question := st.chat_input("Ask a question about your documents…"):
    # Append and show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Fetch sources (before streaming, so we can show them after)
    source_chunks = rag.get_source_chunks(question, k=k)
    st.session_state.last_sources = source_chunks

    # Stream assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        for token in rag.ask(question, k=k, stream=True):
            full_response += token
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Source inspector (shown below the last answer)
if show_sources and st.session_state.last_sources:
    with st.expander(f"📎 Source chunks used ({len(st.session_state.last_sources)})", expanded=False):
        for i, chunk in enumerate(st.session_state.last_sources, start=1):
            citation = chunk.metadata.get("citation", "unknown")
            snippet = chunk.page_content[:400].replace("\n", " ")
            st.markdown(
                f"""<div class="source-card">
                    <div class="source-label">[{i}] {citation}</div>
                    <div class="chunk-text">{snippet}{'…' if len(chunk.page_content) > 400 else ''}</div>
                </div>""",
                unsafe_allow_html=True,
            )

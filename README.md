# RAG Document Chatbot

A fully local RAG pipeline: no OpenAI API key, no cloud services.
Every component runs on your machine.

## Stack

| Component      | Tool                              |
|----------------|-----------------------------------|
| Embeddings     | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Vector store   | FAISS (local, persisted to disk)  |
| LLM            | Ollama (llama3.2 or any local model) |
| UI             | Streamlit                         |
| Doc loaders    | LangChain + pypdf + python-docx   |

---

## Quickstart

### 1. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install and start Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download from https://ollama.com/download

# Pull a model (llama3.2 is fast and accurate; ~2 GB)
ollama pull llama3.2

# Other good options:
# ollama pull mistral        # 7B, good for reasoning
# ollama pull phi3           # 3.8B, very fast
# ollama pull gemma2         # 9B, strong comprehension
```

Ollama starts automatically on most systems. If not: `ollama serve`

### 3. Run the app

```bash
streamlit run app/main.py
```

Open http://localhost:8501 in your browser.

---

## Project structure

```
rag-chatbot/
├── app/
│   ├── main.py            # Streamlit UI
│   ├── rag_pipeline.py    # Orchestrator (ingest + ask)
│   ├── embeddings.py      # HuggingFace embeddings
│   ├── retriever.py       # FAISS build + query
│   └── generator.py       # Ollama prompt + generation
├── utils/
│   ├── loaders.py         # PDF / DOCX / TXT loaders
│   └── chunking.py        # Recursive text splitter
├── data/documents/        # (Optional) drop files here for CLI ingest
├── vector_store/          # FAISS index (auto-created)
└── requirements.txt
```

---

## How to use

1. Open the sidebar → upload one or more PDF / DOCX / TXT files
2. Click **Index documents**
3. Ask questions in the chat box
4. Expand **Source chunks** below each answer to see exactly which passages were used

---

## Changing the model

In the sidebar, the model selector lists every model you've pulled with Ollama.
To add a new model:

```bash
ollama pull mistral
```

Then select it in the UI — no restart needed.

---

## Advanced: CLI ingestion

To pre-index documents without the UI:

```python
from app.rag_pipeline import RAGPipeline

rag = RAGPipeline(model="llama3.2")
rag.ingest_file("data/documents/my_report.pdf")

for token in rag.ask("What is the main conclusion?"):
    print(token, end="", flush=True)
```

---

## Tips for better results

| Goal | Setting |
|------|---------|
| More context per answer | Increase `k` (retrieval slider) |
| Faster responses | Use `phi3` or `gemma2:2b` model |
| Better accuracy on technical docs | Use `mistral` or `llama3.2:70b` |
| Larger documents | Increase `chunk_size` in `utils/chunking.py` |
| Less hallucination | Decrease `k`, keep context focused |

---

## Troubleshooting

**"No Ollama models found"** → Run `ollama pull llama3.2` and restart the app.

**Slow first response** → The embedding model (~90 MB) downloads on first run. Subsequent runs are instant.

**Out of memory** → Use a smaller model (`phi3`, `gemma2:2b`) or reduce `k`.

**FAISS `allow_dangerous_deserialization` warning** → Expected; safe since we wrote the index ourselves.

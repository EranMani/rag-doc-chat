# RAG-Doc-Chat

A production-style **Retrieval-Augmented Generation (RAG)** app: upload documents (PDF, CSV, TXT, MD), get a streamed summary, then ask questions and get answers grounded in your data with source citations. Built for clarity, maintainability, and correct RAG behavior.

---

## What It Does

1. **Ingest** — Load a file → chunk with overlap → embed with a single embedding model → store in Chroma → stream an LLM-generated summary.
2. **Query** — Embed the question → retrieve only chunks above a similarity threshold → build a system prompt with context → call the LLM → return answer + cited sources.

No retrieval means no LLM call: off-topic questions short-circuit at the vector store.

---

## Architecture (High Level)

- **Two separate paths:** **Ingestion** (write) and **Query** (read) are different packages (`ingestion/`, `query/`). They share only `src/config.py` — so you can change retrieval or chunking without tangling the two pipelines.
- **Single config:** All constants and prompts live in `src/config.py`: embedding model, chunk size/overlap, retrieval `k`, score threshold, and both system prompts. One place to tune behavior and avoid ingest/query drift.
- **UI layer:** `app.py` is Gradio-only: it imports from `ingestion` and `query` and wires events; no business logic lives there.

```
rag-doc-chat/
├── app.py              # Gradio UI entry point
├── main.py             # CLI ingest (e.g. for testing)
├── pyproject.toml      # Dependencies
├── src/
│   └── config.py       # Single source of truth (models, prompts, RAG params)
├── ingestion/          # Load → chunk → embed → store → summarize
│   ├── loaders.py      # PDF, CSV, TXT, MD loaders → LangChain Documents
│   ├── ingest.py       # RecursiveCharacterTextSplitter, Chroma, summary
│   └── summary.py      # LLM summary (blocking + streaming)
├── query/
│   └── answer.py       # Chroma retriever → RAG prompt → LLM → answer + sources
└── chroma_db/          # Persisted vector store (runtime; set via CHROMA_PERSIST_DIR)
```

---

## How It’s Built

| Layer | Tech |
|-------|------|
| **Embeddings** | HuggingFace `sentence-transformers` (e.g. `all-MiniLM-L6-v2`) via `langchain-huggingface`; one model for both ingest and query, instantiated once and cached with `lru_cache`. |
| **Vector store** | Chroma with SQLite persistence; `CHROMA_PERSIST_DIR` from env. |
| **Chunking** | LangChain `RecursiveCharacterTextSplitter` (configurable `CHUNK_SIZE`, `CHUNK_OVERLAP`). |
| **LLM** | OpenAI (e.g. GPT) for summary and for RAG answers; model names and prompts in `src/config.py`. |
| **UI** | Gradio: file upload, progress, streamed summary, chat with source citations. |

**Ingest pipeline:** `load_document` (by extension) → split into chunks → enrich metadata (`source`, `type`, `embedding_model`, `parent_document`, `date`) → remove existing chunks for same filename (re-upload = replace) → embed and store in Chroma → generate (and optionally stream) summary from first N chunks.

**Query pipeline:** Load Chroma with same embedding function → retriever with `similarity_score_threshold` and `k` → if no docs above threshold, return “no relevant information” (no LLM call) → else build `RAG_SYSTEM_PROMPT` with `{document_summary}` → LLM answer + formatted source list.

---

## RAG Rules Implemented

- **One embedding model everywhere** — Ingest and query both use `EMBEDDING_MODEL` from `src/config.py`. Changing it is a data-layer change: re-ingest after switching.
- **Score threshold** — Retrieval uses `similarity_score_threshold` (e.g. `0.75`). Below threshold → no chunks returned → no context, no LLM call; user gets a clear “no relevant information” message.
- **Re-upload = replace** — Re-uploading the same file deletes its existing chunks in Chroma then adds the new ones, avoiding duplicate chunks and wasted retrieval slots.
- **Metadata** — Chunks carry `source`, `type`, `embedding_model`, `parent_document`, `date` for filtering, citations, and debugging.
- **Prompts in config** — `RAG_SYSTEM_PROMPT` and `SUMMARY_SYSTEM_PROMPT` live in `src/config.py`; RAG prompt includes “If the question is not related to the document, say so.”
- **Source citations** — Answers append a “Sources” section listing distinct document names used in the retrieved context.

---

## Quick Start

**Requirements:** Python 3.10+, `uv` or `pip`.

1. **Clone and install**
   ```bash
   cd rag-doc-chat
   uv sync   # or: pip install -e .
   ```

2. **Environment**
   - Copy `.env.example` to `.env`.
   - Set `OPENAI_API_KEY`, `HUGGINGFACEHUB_API_TOKEN`, and `CHROMA_PERSIST_DIR` (path to directory for Chroma, e.g. `./chroma_db`).

3. **Run**
   ```bash
   python app.py
   ```
   Open the Gradio URL, upload a document, process it, then ask questions in the chat.

**CLI ingest (no UI):** `python main.py` runs a small ingest example (see `main.py` for paths).

---

## For Developers

- **Deep dive:** [ARCHITECTURE.md](ARCHITECTURE.md) — design, data flow, Gradio wiring, Chroma, and the “same embedding model” rule.
- **Hardening and correctness:** [RAG_HARDENING_GUIDE.md](RAG_HARDENING_GUIDE.md) — embedding sync, score threshold, deduplication, and production considerations.
- **Code review notes:** [TECHNICAL_REVIEW.md](TECHNICAL_REVIEW.md) — rationale for patterns used in the codebase.

If you’re extending or forking: keep ingest and query in sync on `EMBEDDING_MODEL`, use the score threshold to avoid useless LLM calls, and keep prompts and tunables in `src/config.py` so the project stays predictable and easy to tune.

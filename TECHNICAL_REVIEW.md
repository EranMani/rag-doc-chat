# 🏗️ RAG-Doc-Chat: Senior Technical Review & Architecture Audit

**Core Theme:** *Production Paranoia* — transitioning from code that works locally to code that survives in production.

---

## 📋 Executive Summary

This document captures the findings of a technical code review of the `rag-doc-chat` application: a RAG pipeline built on LangChain, Chroma, and Gradio. The review examined five areas:

| Area | Files |
|------|-------|
| Architectural separation of concerns | All modules |
| I/O resource management | `ingestion/loaders.py` |
| Embedding model performance | `ingestion/ingest.py`, `query/answer.py` |
| Vector store database integrity | `ingestion/ingest.py`, `query/answer.py` |
| UI state and event wiring | `app.py` |

---

## Phase 1: Ingestion & Loader Dispatch (`loaders.py`)

### 1. Keyword-Only Arguments

**The question:** What is the purpose of the bare `*` in:

```python
def load_document(*, file_path=None, file_bytes=None, filename: str):
```

**Analysis:** The `*` forces every caller to use named arguments. Without it, calling `load_document(raw_bytes, "report.pdf")` would silently bind the byte string to `file_path`, causing a cryptic crash deep inside LangChain's file loaders — not at the call site where the mistake was made. With `*`, Python raises `TypeError` immediately at the call site if a positional argument is passed. This is a defensive API contract, not style.

---

### 2. Temporary File Leak

**The question:** The loader uses `tempfile.NamedTemporaryFile(delete=False, ...)`. What is the operational risk and how should it be fixed?

**Current code:**

```python
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="rag_upload_") as tmp:
    tmp.write(file_bytes)
    file_path = tmp.name
# tmp is closed here but NOT deleted — file_path stays on disk indefinitely
```

**Risk:** Every document upload from file bytes creates a temp file that is never deleted. Over time this silently consumes disk space. In a long-running server environment this is a resource exhaustion bug.

**Fix:** Track whether a temp file was created with an `is_temp_file` boolean flag and delete it in a `try...finally` block, so the file is always cleaned up — including when the loader raises an exception:

```python
is_temp_file = False
if file_bytes is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        file_path = tmp.name
        is_temp_file = True

try:
    loader = _get_loader(file_path=file_path, ext=ext)
    raw_docs = loader.load()
finally:
    if is_temp_file:
        Path(file_path).unlink(missing_ok=True)
```

The `is_temp_file` guard is critical: a plain `os.remove(file_path)` would be destructive if the caller passed a legitimate `file_path` (not bytes), deleting the user's actual file.

---

### 3. Metadata: Separation of Concerns

**The question:** Why is `source`/`type` attached in `loaders.py`, but `embedding_model`/`parent_document`/`date` are attached in `ingest.py`?

**Analysis:** This is an intentional application of the **Single Responsibility Principle**. Each module stamps only what it knows:

| Module | What it knows | Metadata it stamps |
|--------|--------------|-------------------|
| `loaders.py` | File path and extension | `source` (filename), `type` (extension) |
| `ingest.py` | Embedding model, pipeline timing | `embedding_model`, `parent_document`, `date` |

`loaders.py` handles file I/O dispatch. It has no knowledge of the embedding model or pipeline context. Stamping `embedding_model` in the loader would couple a file-loading module to an AI configuration detail — a violation of separation of concerns that would make the loader harder to reuse or test independently.

---

## Phase 2: Processing & Vector Storage (`ingest.py`)

### 1. Embedding Model Initialization: Performance Constraint

**The question:** `_embed_and_store_chunks` runs `HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)` on every call. Why is this a problem?

**Current code:**

```python
def _embed_and_store_chunks(document_chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)  # ← cold-loads every call
    ...
```

The same pattern exists in `query/answer.py`:

```python
def _load_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)  # ← called on every answer_question()
```

**Risk:** `HuggingFaceEmbeddings` loads `sentence-transformers/all-MiniLM-L6-v2` from disk into memory on every invocation. For a model of this size, this adds hundreds of milliseconds to every ingest and query call. In a memory-constrained environment, repeated instantiation without releasing the previous instance can cause OOM (out-of-memory) crashes.

**Fix:** Lazy-load using `@lru_cache` so the model is loaded exactly once for the lifetime of the process:

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def _load_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
```

This is preferable to a module-level instantiation (which would load the model at import time, slowing startup) and to placing it in `config.py` (which would make `config.py` responsible for ML model lifecycle — a responsibility it should not have).

---

### 2. Database Integrity & Deduplication

**The question:** What happens if a user re-uploads `Q3_Report.pdf`?

**Current code:**

```python
chroma_db.add_documents(document_chunks)  # no existence check
```

**Risk:** Every re-upload appends all chunks from that document again. Chroma does not deduplicate. With `RETRIEVAL_K = 3`, a document uploaded twice occupies all three retrieval slots with identical content. The LLM receives three copies of the same text, wasting the entire context budget on redundancy and starving retrieval of diverse information.

**Fix:** Use Chroma's `get()` API to find existing chunks by `source` metadata, delete them first, then insert the new ones. This supports both deduplication and document updates:

```python
existing = chroma_db.get(where={"source": filename})
if existing and existing["ids"]:
    chroma_db.delete(ids=existing["ids"])
chroma_db.add_documents(document_chunks)
```

This requires `source` to be stamped in chunk metadata — which it already is, via `loaders.py`. The metadata design decision made in Phase 1 directly enables this fix.

---

## Phase 3: Query Pipeline & UI Logic (`answer.py` & `app.py`)

### 1. Relevance Filtering: The Declared-but-Unenforced Threshold

**The question:** `SCORE_THRESHOLD = 0.75` is defined in config, but `_load_retriever` only passes `{"k": RETRIEVAL_K}`. What happens when the user asks an off-topic question?

**Current code:**

```python
def _load_retriever(chroma_db: Chroma):
    return chroma_db.as_retriever(search_kwargs={"k": RETRIEVAL_K})
```

**Risk:** The retriever always returns the top-k chunks regardless of semantic distance. For a completely off-topic question, these chunks are irrelevant, but they are still injected into the system prompt and sent to the LLM. `RAG_SYSTEM_PROMPT` does tell the model to say "If the question is not related to the document, say so" — so the answer quality is acceptable. But the compute cost is not: an embedding call and a paid LLM API call are made just to produce a "not relevant" response that the vector store could have short-circuited immediately.

**Fix:** Use Chroma's `similarity_score_threshold` search type and add an early return if no documents meet the threshold:

```python
# In answer.py — add SCORE_THRESHOLD to the import
from src.config import CHAT_MODEL, RAG_SYSTEM_PROMPT, CHROMA_PERSIST_DIR, EMBEDDING_MODEL, RETRIEVAL_K, SCORE_THRESHOLD

def _load_retriever(chroma_db: Chroma):
    return chroma_db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": RETRIEVAL_K, "score_threshold": SCORE_THRESHOLD}
    )

# In answer_question(), after retrieval:
if not documents:
    return (
        "I couldn't find relevant information in the uploaded documents to answer that question.",
        ""
    )
```

> **Note on distance metrics:** Chroma supports both L2 distance (lower = more similar) and cosine similarity (higher = more similar). Verify the collection's metric before setting a numeric threshold. The interpretation of `0.75` and the comparison direction depend on which metric is active.

---

### 2. UI Logic: Answer vs. Sources in the Chatbot

**The question:** Look at `get_model_answer` in `app.py`. How should the answer and sources be combined in the assistant message?

**Current code (fixed):**

```python
answer_text, sources_display = answer_question(question=question)

user_msg = {"role": "user", "content": [{"type": "text", "text": question}]}
assistant_content = answer_text
if sources_display:
    assistant_content += sources_display
assistant_msg = {"role": "assistant", "content": [{"type": "text", "text": assistant_content}]}
```

**What to watch for:** `sources_display` in `answer.py` is built as:

```python
sources_display = "\n\n---\n**Sources:**\n" + "\n".join(unique_sources_names)
```

It already contains the markdown "Sources:" header. The UI must concatenate `answer_text + sources_display` directly — not add another "Sources:" label — or the header appears twice. The current code does this correctly by appending `sources_display` as-is to `assistant_content`.

---

## 🚀 Prioritized Hardening Roadmap

| Priority | Fix | File | Risk if skipped |
|----------|-----|------|-----------------|
| 🔴 Critical | Fix temp file cleanup (`try/finally` + `is_temp_file` flag) | `ingestion/loaders.py` | Disk exhaustion in any long-running deployment |
| 🔴 Critical | Add deduplication before `add_documents` | `ingestion/ingest.py` | Retrieval slots consumed by identical chunks; silent quality degradation |
| 🟠 High | Cache `HuggingFaceEmbeddings` with `@lru_cache` | `ingestion/ingest.py`, `query/answer.py` | Latency spike on every call; OOM risk under load |
| 🟠 High | Enforce `SCORE_THRESHOLD` with `similarity_score_threshold` + early return | `query/answer.py` | Wasted API costs; irrelevant context injected into every off-topic query |
| 🟡 Medium | Implement conversational memory with standalone question reformulation | `query/answer.py`, `app.py` | Follow-up questions with pronouns retrieve wrong chunks silently |
| 🟡 Medium | Multi-tenant isolation: `gr.Request` → metadata stamping → retriever filter | `app.py`, `ingestion/ingest.py`, `query/answer.py` | Any user can retrieve any other user's document chunks |
| 🟢 Low | Remove unused `import time` | `app.py` | Code hygiene only |
| 🟢 Low | Remove duplicate `from langchain_chroma import Chroma` | `ingestion/ingest.py` | Code hygiene only |

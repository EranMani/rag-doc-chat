# RAG-Doc-Chat — Architecture & Engineering Guide

A technical deep-dive into how this project is built: its concepts, design decisions, code patterns, and the reasoning behind each choice. Written for a senior software engineer reading the codebase for the first time.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Project Structure](#2-project-structure)
3. [Core Concepts: RAG in Plain Terms](#3-core-concepts-rag-in-plain-terms)
4. [Two Separate Paths: Ingest vs Query](#4-two-separate-paths-ingest-vs-query)
5. [Configuration Design (`src/config.py`)](#5-configuration-design-srcconfigpy)
6. [The Ingest Module (`ingestion/`)](#6-the-ingest-module-ingestion)
7. [The Query Module (`query/`)](#7-the-query-module-query)
8. [The UI Layer (`app.py`)](#8-the-ui-layer-apppy)
9. [Public Entrypoints and `__init__.py` Contracts](#9-public-entrypoints-and-__init__py-contracts)
10. [Streaming: Generator Pattern](#10-streaming-generator-pattern)
11. [Gradio: Blocks, Components, and Wiring](#11-gradio-blocks-components-and-wiring)
12. [The Chatbot Element and Conversation History](#12-the-chatbot-element-and-conversation-history)
13. [The Golden Rule: Same Embedding Model Everywhere](#13-the-golden-rule-same-embedding-model-everywhere)
14. [Chroma: Persistent Vector Store](#14-chroma-persistent-vector-store)
15. [Metadata: Why and What](#15-metadata-why-and-what)
16. [Dependency Management (`pyproject.toml`)](#16-dependency-management-pyprojecttoml)
17. [Known Limitations and Open Decisions](#17-known-limitations-and-open-decisions)

---

## 1. What This Project Does

The user uploads a document (PDF, CSV, TXT, or Markdown). The app:

1. **Loads** it via the right parser.
2. **Chunks** it into overlapping segments.
3. **Embeds** each chunk into a vector and stores it in a local Chroma database.
4. **Summarizes** the document using an LLM and streams the result to the UI.
5. Allows the user to **ask questions** about the document. Each question is embedded, the closest chunks are retrieved from Chroma, and an LLM produces an answer grounded in those chunks — with source citations.

This is a minimal, end-to-end Retrieval-Augmented Generation (RAG) pipeline.

---

## 2. Project Structure

```
rag-doc-chat/
├── app.py                  # Gradio UI + event wiring. Entry point.
├── main.py                 # CLI/dev runner for testing ingest without the UI.
├── pyproject.toml          # Dependency manifest (uv/pip).
├── .env                    # API keys and runtime paths (not committed).
├── .env.example            # Template showing what keys are required.
├── PROJECT_PLAN.md         # Build order and feature plan.
├── ARCHITECTURE.md         # This file.
│
├── src/
│   ├── __init__.py
│   └── config.py           # Single source of truth for all constants and prompts.
│
├── ingestion/
│   ├── __init__.py         # Public API: ingest_document, ingest_document_stream.
│   ├── loaders.py          # File-type-aware document loaders.
│   ├── ingest.py           # Pipeline: load → chunk → embed → store → summarize.
│   └── summary.py          # LLM-based summary, blocking and streaming variants.
│
├── query/
│   ├── __init__.py         # Public API: answer_question.
│   └── answer.py           # Pipeline: retrieve → build prompt → call LLM → return answer + sources.
│
└── chroma_db/              # Persisted Chroma vector store (runtime artifact, not source).
```

**Why this layout?**

- `ingestion/` and `query/` are **separate packages** because they have different responsibilities, different dependencies, and different calling patterns. Ingest *writes* to the vector store; query *reads* from it. Keeping them separate means you can change retrieval strategy without touching ingestion code (and vice versa).
- `src/config.py` is isolated in its own package so both `ingestion/` and `query/` can import from it without either depending on the other.
- `app.py` is a pure **UI layer** — it imports from both `ingestion` and `query` but contains no business logic of its own.

---

## 3. Core Concepts: RAG in Plain Terms

**The problem RAG solves:** LLMs have a training cut-off date and no knowledge of your private documents. You cannot just paste a 50-page PDF into a prompt — context windows are finite and expensive.

**The solution:**
- Pre-process the document: split it into chunks, turn each chunk into a vector (embedding), and store those vectors in a searchable database.
- At query time: turn the user's question into a vector too (using the *same* model), find the most similar stored chunks, and send only those chunks to the LLM along with the question.

The LLM never sees the whole document — only the relevant parts. This is called **retrieval-augmented generation**.

**Key insight:** The quality of the system depends entirely on how good the chunks and embeddings are. If similar concepts don't land near each other in vector space, retrieval will fail and the LLM will hallucinate or say it doesn't know.

---

## 4. Two Separate Paths: Ingest vs Query

```
INGEST PATH (write)
User uploads file
  → load_document()         [loaders.py]
  → _split_document_to_chunks()  [ingest.py]
  → _embed_and_store_chunks()    [ingest.py]
  → _build_chunks_content_for_summary() + generate_summary_stream()  [summary.py]
  → yields (progress, summary_text) to UI

QUERY PATH (read)
User asks question
  → _load_chroma_db()       [answer.py]
  → _load_retriever()       [answer.py]
  → retriever.invoke(question)
  → build context string
  → RAG_SYSTEM_PROMPT.format(context)
  → _get_client_model_response()  [answer.py]
  → returns (answer_text, sources_display) to UI
```

These two paths share only three things:
- `CHROMA_PERSIST_DIR` — where Chroma lives on disk.
- `EMBEDDING_MODEL` — the model used to create vectors (must be identical in both paths).
- `src/config.py` — the single place both import from.

Nothing else is shared. This is an intentional separation of concerns.

---

## 5. Configuration Design (`src/config.py`)

All configuration lives in one file:

```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "gpt-4o-mini"
SUMMARY_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
RETRIEVAL_K = 5
SCORE_THRESHOLD = 0.75
SUMMARY_SYSTEM_PROMPT = "..."
RAG_SYSTEM_PROMPT = "..."
```

**Why these specific constants, and why a config file?**

- **`EMBEDDING_MODEL`** — The exact model name is a string that must be identical in `ingestion` (when chunks are embedded and stored) and in `query` (when the question is embedded and searched). A single typo between those two would silently break retrieval because the vectors would live in different spaces. Centralising it prevents that.

- **`CHUNK_SIZE` and `CHUNK_OVERLAP`** — These are tuning parameters. 512 characters (roughly 128 tokens) is a common starting point for semantic chunks. Overlap of 100 characters ensures that meaning at a chunk boundary isn't lost — adjacent chunks share context so retrieval doesn't miss a sentence split across a boundary.

- **`RETRIEVAL_K`** — How many chunks to retrieve per query. More chunks = more context for the LLM = higher chance of finding the answer, but also higher cost and risk of diluting relevance. 5 is a reasonable starting default.

- **`SCORE_THRESHOLD`** — A similarity filter. Chunks below this threshold are too semantically distant from the question to be useful. Not yet enforced in the retriever call (it uses `k` only), but it's declared here so enforcement can be added without touching multiple files.

- **`SUMMARY_SYSTEM_PROMPT` / `RAG_SYSTEM_PROMPT`** — Prompts are configuration, not code. Moving them to config means you can tune the model's behavior (tone, format, language) without changing any function logic. The RAG prompt includes `{document_summary}` as a placeholder, which is filled at runtime with retrieved context:
  ```python
  system_content = RAG_SYSTEM_PROMPT.format(document_summary=context)
  ```

- **Fail-fast validation** — API keys use `if not os.getenv(...): raise ValueError(...)` rather than `os.getenv(..., default)`. This means the app crashes immediately at startup with a clear error if an API key is missing, instead of silently returning `None` and failing later with a cryptic message deep inside a library call.

---

## 6. The Ingest Module (`ingestion/`)

### `loaders.py` — file-type dispatch

```python
def load_document(*, file_path=None, file_bytes=None, filename: str) -> list[Document]:
```

**Keyword-only arguments (`*`):** The `*` before parameters forces callers to always use named arguments: `load_document(file_path="...", filename="...")`. This prevents silent argument-order bugs — passing a path as `file_bytes` by accident would be a silent `None` without the `*`.

**Dual input modes:** Accepts either a file path (from disk or `main.py`) or raw bytes (from Gradio, which provides a temp path). When bytes are passed, a `tempfile` is created so every loader downstream can work with a path — no loader needs to understand bytes vs path.

**Metadata at load time:** After `loader.load()`, two fields are immediately attached to every `Document.metadata`:
- `source` — the original filename, used later for citations.
- `type` — the extension (e.g. `"pdf"`), useful for filtering or debugging.

### `ingest.py` — the pipeline

Four private functions with a single public entrypoint:

```
_split_document_to_chunks()
_embed_and_store_chunks()
_build_chunks_content_for_summary()
ingest_document()           ← public, blocking
ingest_document_stream()    ← public, streaming generator
```

**Private functions (underscore prefix):** Functions prefixed with `_` are implementation details. They exist to make the pipeline readable and testable in isolation, but they are not part of the module's contract. Callers (including `app.py`) should never call them directly.

**`_split_document_to_chunks`** uses `RecursiveCharacterTextSplitter`, which tries to split on `\n\n`, then `\n`, then `. `, then ` `, then individual characters — in that priority order. This preserves natural language boundaries as much as possible. After splitting, three metadata fields are added to each chunk:
- `embedding_model` — which model was used (for debugging and future migration).
- `parent_document` — the filename, so you can trace any chunk back to its source.
- `date` — ingestion date, useful for freshness filtering.

**`_embed_and_store_chunks`** handles both first-run and subsequent ingestion: if `CHROMA_PERSIST_DIR` exists, it loads the existing collection and calls `add_documents`; otherwise it creates a fresh collection with `from_documents`. This means uploading multiple documents accumulates them all in the same store.

**`_build_chunks_content_for_summary`** takes the first 5 chunks (a bounded excerpt), not the full document. This is intentional — sending the entire document to the LLM for summarization would be expensive and likely exceed context limits. Five chunks give the model enough to understand what the document is about.

---

## 7. The Query Module (`query/`)

```
_load_chroma_db()
_load_embedding_model()
_load_retriever()
_get_client_model_response()
answer_question()           ← public entrypoint
```

### Retriever pattern

```python
retriever = chroma_db.as_retriever(search_kwargs={"k": RETRIEVAL_K})
documents = retriever.invoke(question)
```

Calling `retriever.invoke(question)` with a **plain string** is correct. The `Chroma` retriever automatically embeds the question using the `embedding_function` it was initialized with. You do not embed the question manually. The retriever: embeds the question → runs cosine similarity search → returns the top-k `Document` objects sorted by similarity.

### Empty-store guard

```python
documents_amount = chroma_db._collection.count()
if documents_amount == 0:
    return ("Upload and process a document first.", "")
```

`chroma_db._collection` accesses the underlying Chroma collection object (not the LangChain wrapper). The `_` prefix is a convention meaning "private/internal," but it is the practical way to get the collection's document count at this time. If `k` were greater than the number of documents in the collection, the retriever would return all documents — an empty-store check avoids presenting the user with a meaningless response.

### Context building and prompt injection

```python
context = ""
for i, doc in enumerate(documents):
    context += f"Document {i+1}:\n{doc.page_content}\n\n"

system_content = RAG_SYSTEM_PROMPT.format(document_summary=context)
```

The retrieved chunks are formatted as numbered "documents" so the LLM can reference them clearly. They are injected into the system prompt (not the user message) so the LLM treats them as ground truth context rather than part of the question.

### Source deduplication

```python
unique_sources_names = set()
source_name = doc.metadata.get("source", "Unknown")
if source_name != "Unknown":
    unique_sources_names.add(source_name)
```

Multiple chunks may come from the same file. A `set` automatically deduplicates so the sources list doesn't repeat the same filename five times.

---

## 8. The UI Layer (`app.py`)

`app.py` is intentionally thin. It contains **no business logic** — no chunking, no LLM calls, no prompt building. Its only responsibilities are:

1. Defining the layout (Gradio `Blocks`).
2. Wiring UI events to backend functions.
3. Translating backend return values into the format Gradio expects.

### `process_document_upload` — a generator function

```python
def process_document_upload(file):
    yield 10, "⏳ Loading and processing document..."
    for progress, text in ingest_document_stream(...):
        yield progress, text
```

This function is a **generator** (it uses `yield`). Gradio detects that the wired function is a generator and calls `next()` on it repeatedly, updating the outputs with each yielded value. The outputs are `[progress_bar, summary_out]`, so each `yield` must be a tuple of `(int, str)` — first value for the Slider, second for the Markdown. Swapping these (e.g. yielding `(str, int)`) would cause the Markdown component to receive an integer and raise an `AttributeError`. The `demo.queue()` call before `demo.launch()` is required to enable generator-based streaming.

### `get_model_answer` — building conversation history

```python
def get_model_answer(history, question):
    answer_text, sources_display = answer_question(question=question)
    user_msg = {"role": "user", "content": [{"type": "text", "text": question}]}
    assistant_msg = {"role": "assistant", "content": [{"type": "text", "text": sources_display}]}
    new_history = history + [user_msg, assistant_msg]
    return new_history, ""
```

This function takes **two inputs**: `history` (the Chatbot's current value) and `question` (the Textbox value). It returns **two outputs**: the updated history for the Chatbot and `""` to clear the Textbox. Gradio maps these directly to `outputs=[chatbot, answer_textbox]` — returning one value would raise `"didn't return enough output values"`.

---

## 9. Public Entrypoints and `__init__.py` Contracts

Each package exposes only what it intends to be used by others:

**`ingestion/__init__.py`:**
```python
from .ingest import ingest_document, ingest_document_stream
__all__ = ["ingest_document", "ingest_document_stream"]
```

**`query/__init__.py`:**
```python
from .answer import answer_question
__all__ = ["answer_question"]
```

**Why this matters:**

- `__all__` defines the **public API contract** of a package. If another module does `from ingestion import *`, only the names in `__all__` are imported. Private helpers like `_split_document_to_chunks` are invisible to the outside world.
- `app.py` imports `from ingestion import ingest_document_stream` — it doesn't need to know which file inside `ingestion/` that function lives in. If `ingest.py` is later split into two files, `app.py` doesn't change.
- This is the same principle as public/private in object-oriented languages: internals can be refactored freely as long as the public contract (the `__init__.py` exports) stays the same.

---

## 10. Streaming: Generator Pattern

Two generators exist in the codebase, both following the same pattern:

**`generate_summary_stream` (summary.py):**
```python
def generate_summary_stream(document_chunks: str):
    summary = ""
    for chunk in client.stream(messages):
        if getattr(chunk, "content", None):
            summary += chunk.content
            yield summary  # cumulative, not just the new token
```

**`ingest_document_stream` (ingest.py):**
```python
def ingest_document_stream(...):
    # ... pipeline ...
    for summary in generate_summary_stream(summary_content):
        yield 70, summary
    yield 100, summary
```

**Key design decisions:**

- **Cumulative yield, not delta yield.** Each `yield` sends the *full text so far*, not just the new token. Gradio's Markdown component replaces its entire content with each update, so sending only deltas would overwrite the display with just the new piece. Cumulative yield means the UI always shows the complete summary up to that point.

- **`getattr(chunk, "content", None)`.** LangChain's streaming response can emit non-text chunks (e.g. role-only chunks at the start, function-call chunks). These don't have a `.content` attribute or have it as `None`. Using `getattr` with a default avoids `AttributeError` and silently skips non-text chunks.

- **Progress signaling via tuples.** `yield (70, summary)` bundles the progress bar value and the text in one yield. This avoids having a separate channel for progress — the generator drives everything. `70` during streaming, `100` on completion is a deliberate UX choice: the bar moves to 70% when streaming starts, giving feedback that the heavy work (load/chunk/embed) is done.

- **`demo.queue()` is required.** Without `queue()`, Gradio runs handlers synchronously and cannot handle generator functions. `queue()` runs each event in an async queue, making streaming possible.

---

## 11. Gradio: Blocks, Components, and Wiring

```python
with gr.Blocks(title="RAG Doc Chat") as demo:
    with gr.Row():
        with gr.Column():
            file_input = gr.File(...)
            process_btn = gr.Button("Process")
            progress_bar = gr.Slider(0, 100, ...)
            summary_out = gr.Markdown(...)
        with gr.Column():
            chatbot = gr.Chatbot(...)
            answer_textbox = gr.Textbox(...)
            ask_btn = gr.Button("Ask")

    process_btn.click(fn=..., inputs=file_input, outputs=[progress_bar, summary_out])
    ask_btn.click(fn=..., inputs=[chatbot, answer_textbox], outputs=[chatbot, answer_textbox])
```

**`gr.Blocks` vs `gr.Interface`:** `gr.Interface` is a one-function, one-input, one-output convenience wrapper. `gr.Blocks` gives full layout control — rows, columns, multiple components, multiple event handlers. This project uses `Blocks` because it needs two independent panels (upload + chat) and two independent event flows (Process button and Ask button).

**Component as variable:** Every interactive component that participates in inputs or outputs is assigned to a Python variable (`file_input`, `progress_bar`, `summary_out`, etc.). Static display components like `gr.Markdown("# RAG-DOC-CHAT")` don't need a variable because nothing reads or writes them programmatically.

**`gr.Slider` for progress, not `gr.Progress()`:** Gradio's `gr.Progress()` is a special context-variable for built-in progress reporting inside `gr.Interface`-style callbacks. Inside `gr.Blocks` with a generator, the simpler and more explicit pattern is a `gr.Slider(interactive=False)` updated by the generator's yields. Note that `gr.Progress()` is declared in the code but overwritten immediately by `gr.Slider` — the `gr.Progress()` line is vestigial and has no effect.

**`gr.File(type="filepath")`:** Gradio by default can pass file content as bytes or as a path. Setting `type="filepath"` tells Gradio to pass the uploaded file's temporary path as a string. This is simpler because all LangChain loaders accept a file path.

---

## 12. The Chatbot Element and Conversation History

```python
chatbot = gr.Chatbot(label="Chat", value=[])
```

**History format:** Gradio's `Chatbot` component uses the **messages format** — a list of message dictionaries:

```python
[
    {"role": "user",      "content": [{"type": "text", "text": "What is this?"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "This is a receipt."}]}
]
```

Each message has:
- `role`: `"user"` or `"assistant"`.
- `content`: a list of content blocks (supporting text, images, etc.). For plain text, this is always a one-element list with `{"type": "text", "text": "..."}`.

**Why a list, not a dict?** The history is ordered and append-only. A list naturally represents a sequence of messages. A dict would require managing integer keys or UUIDs and is semantically wrong for a conversation.

**How history flows through Gradio:** The Chatbot component's value at any point in time is the full conversation history. When `ask_btn.click` fires, Gradio passes the Chatbot's current value as the first argument to `get_model_answer`. The function appends the new user/assistant pair and returns the new list. Gradio sets the Chatbot's value to that new list. The next click passes the updated history again. There is no external state — the component itself is the state.

**Why the Chatbot is both input and output:**
```python
ask_btn.click(fn=get_model_answer, inputs=[chatbot, answer_textbox], outputs=[chatbot, answer_textbox])
```
The same `chatbot` component appears in both `inputs` and `outputs`. This is the standard Gradio pattern for stateful components: read the current state, compute the next state, write it back. Gradio handles the binding.

---

## 13. The Golden Rule: Same Embedding Model Everywhere

Embeddings are vectors in a high-dimensional space. The geometry of that space — which concepts are "close" to which — is entirely determined by the model that created the vectors.

If you embed chunks with model A and embed queries with model B, the vectors live in different spaces. Similarity search becomes meaningless — not wrong in an obvious way, just wrong in a silent, hard-to-debug way (retrieval returns irrelevant chunks, the LLM sees unrelated context, answers seem off).

**In this project:**

- Ingest: `HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)` in `_embed_and_store_chunks`.
- Query: `HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)` in `_load_embedding_model`.
- Both import `EMBEDDING_MODEL` from `src.config`. This is the only enforcement — a constant that both sides read.

**The consequence:** If you want to change the embedding model (e.g. upgrade to a more capable model), you must:
1. Change `EMBEDDING_MODEL` in `config.py`.
2. **Delete and rebuild the Chroma database** (`chroma_db/`). Existing vectors were made with the old model and are incompatible with the new one.

---

## 14. Chroma: Persistent Vector Store

Chroma is a local, file-backed vector database. No server, no Docker, no cloud — it reads and writes a SQLite file at `CHROMA_PERSIST_DIR`.

**First upload:**
```python
Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_PERSIST_DIR)
```
Creates the directory and collection.

**Subsequent uploads:**
```python
chroma_db = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
chroma_db.add_documents(document_chunks)
```
Loads the existing collection and appends. Multiple documents accumulate in the same store.

**Retrieval:**
```python
retriever = chroma_db.as_retriever(search_kwargs={"k": RETRIEVAL_K})
docs = retriever.invoke(question)
```
`as_retriever` wraps Chroma in LangChain's `BaseRetriever` interface. `invoke(question)` embeds the question internally, runs similarity search, and returns `Document` objects.

**Important:** Chroma does not deduplicate. If the same file is uploaded twice, its chunks are stored twice. This is a known limitation listed as a future hardening task.

---

## 15. Metadata: Why and What

Every `Document` object in LangChain has a `metadata` dict. This project uses it for two purposes:

**1. Traceability (set at load time in `loaders.py`):**
- `source` — original filename. Used in query for source citations.
- `type` — file extension (e.g. `"pdf"`). Useful for filtering or debugging.

**2. Observability (set at chunk time in `ingest.py`):**
- `embedding_model` — which model version embedded this chunk. Critical when migrating models or debugging retrieval quality.
- `parent_document` — filename, redundant with `source` but explicitly named for easy access in logs.
- `date` — ingestion date. Enables time-based filtering in future iterations.

The metadata travels with each chunk through Chroma. When the retriever returns documents, their metadata comes with them. That's how `answer.py` builds the source citations: read `doc.metadata.get("source")` from each returned document.

---

## 16. Dependency Management (`pyproject.toml`)

```toml
[project]
dependencies = [
    "python-dotenv>=1.0.0",
    "langchain==1.2.10",
    "langchain-chroma==1.1.0",
    "langchain-openai==1.1.10",
    "langchain-huggingface==1.2.0",
    "langchain-community==0.4.0",
    "langchain-text-splitters==1.1.1",
    "pypdf==4.0.0",
    "sentence-transformers>=5.2.3",
    "gradio==6.6.0",
]
```

**Why `pyproject.toml` over `requirements.txt`?** `pyproject.toml` is the modern Python standard (PEP 517/518). It supports metadata, build configuration, and optional dependency groups in one file. `uv` (used here as the package manager) reads it natively. `requirements.txt` is a flat install list with no metadata and no build semantics.

**Why LangChain is split into many packages?** LangChain was split from one monolithic package into modular packages (`langchain-core`, `langchain-community`, `langchain-openai`, etc.) so you only install what you need. This avoids pulling in all provider SDKs (Anthropic, Google, Cohere, etc.) just to use OpenAI. Each provider has its own package and release cadence.

**`sentence-transformers`** is the underlying library that `langchain-huggingface` uses to run the local embedding model. It runs entirely on CPU/GPU locally — no API key required for embeddings.

---

## 17. Known Limitations and Open Decisions

| Issue | Current State | Mitigation Path |
|-------|---------------|-----------------|
| **No deduplication on re-upload** | Uploading the same file twice doubles its chunks in Chroma | Check by filename in metadata before adding; delete old chunks first |
| **Score threshold unused** | `SCORE_THRESHOLD = 0.75` is declared but not applied to the retriever | Switch to `similarity_search_with_score` and filter manually, or use `search_type="similarity_score_threshold"` in `as_retriever` |
| **Summary uses only first 5 chunks** | Long documents are summarized from their opening content only | Increase the excerpt, or use a map-reduce summarization chain |
| **No conversation memory** | Each question is answered independently; the LLM has no memory of previous turns | Pass last N `(user, assistant)` pairs as additional context into the prompt |
| **Chroma `_collection` is internal** | Empty-store check uses `chroma_db._collection.count()`, a private attribute | Replace with a retrieve-and-check pattern or use Chroma's public `get()` API |
| **Embedding model loaded on every call** | `HuggingFaceEmbeddings` is instantiated on every `ingest_document_stream` and `answer_question` call, which downloads/loads the model each time | Cache the model in a module-level variable or use `lru_cache` |
| **No multi-user isolation** | All users share the same Chroma collection | Namespace collections by session ID or add a user/session filter to metadata |
| **Single-turn answers only** | The chatbot displays history but the LLM only sees the current question | Include recent history in the system or user message when calling the LLM |

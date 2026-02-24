# rag-doc-chat — Roadmap / Application Plan

**In a nutshell:** Upload documents (PDF, CSV, TXT, MD) → get a short summary → chat with the model about your docs and see source citations. Built with LangChain, Chroma, and Gradio.

---

## Table of contents

| Section | Contents |
|---------|----------|
| [1. Goal & scope](#1-goal--scope) | What we're building, definition of done, out of scope, audience |
| [2. User flow](#2-user-flow) | Steps the user takes and what the system does |
| [3. Tech stack & constraints](#3-tech-stack--constraints) | Framework, infrastructure, and non-negotiables |
| [4. Architecture](#4-architecture-high-level) | Ingest path, query path, document summary, shared concerns |
| [5. Build order / phases](#5-build-order--phases) | Implementation order and outcomes per phase |
| [6. Open questions / risks](#6-open-questions--risks) | Decisions to make and risks with mitigations |

---

## 1. Goal & scope

- **What we're building:** A RAG demo where the user uploads files, gets a summary, and asks questions about them.
- **What "done" looks like:** Upload → short summary + chat + Gradio UI with source citations.
- **Out of scope:** Multi-user or multi-tenant isolation; LangSmith / full eval harness (RAGAS, TruLens).
- **Audience:** Me (during build) and future viewers (developers, interviewers).

## 2. User flow

1. Open application.
2. **Upload a document:** load (by type: PDF, CSV, TXT, MD) → chunk → embed → store in persistent Chroma + metadata (date, embedding model version, parent document name).
3. Display short summary on the left view.
4. Ask a question → get a response from the model.
5. Show source citations on the right view.

## 3. Tech stack & constraints

### Tech stack

- **Framework:** LangChain (langchain-chroma, langchain-text-splitters, langchain-community, etc.) for loaders, embeddings, vector store, and LLM.
- **Vector store:** Chroma (persisted on disk).
- **Embedding model:** all-MiniLM-L6-v2 (via langchain-huggingface).
- **Chat model:** GPT-4o-mini or equivalent (cloud).

### Constraints

- Gradio for UI so we can demo without a separate frontend.
- Same embedding model for ingestion and query (Golden Rule).
- API keys, model names, and chunking/tuning params only in config or env — no hardcoding.

## 4. Architecture (high-level)

### Ingest path (write)

`load document → chunk → embed → store in Chroma` (+ metadata). After storing, call the LLM with a bounded excerpt of chunks to produce a short summary; show it in the UI.

### Query path (read)

`question → embed question → search Chroma (retriever) → get chunks → build prompt (context + question) → call model → return answer + source citations`.

### Why separate ingest and query?

Different responsibilities; separate dependencies for easier debugging and scaling. Ingest writes; query only reads. Same embedding model for both paths is required and makes swapping models easier.

## 5. Build order / phases

**Approach:** `env → config → ingest → UI → query → harden`

| Phase | What to do | By the end |
|-------|------------|------------|
| **1** | Create code environment (uv/venv). Use `.env` for API keys. | Environment ready to run the application. |
| **2** | Create config file: model names (local + cloud), chunk tuning. | One config used by both ingest and query. |
| **3** | Create ingest module: load → chunk → embed → store in Chroma → short summary. | Function that takes an uploaded document, chunks it, embeds it, stores in Chroma, returns summary. |
| **4** | Create UI: Gradio app, upload document, processing labels, summary on left view. | User can upload and see summary on the left. |
| **5** | Create answer module: embed query → retrieve from Chroma → build prompt (context + question) → call model → update UI. | User can ask a question and see answer + source citations. |
| **6** | Edge cases: empty retrieval, document type validation, no empty chunks, basic tests. | Tests and handling for edge cases; better maintainability. |

## 6. Open questions / risks

### Open questions

- **Context window:** Check the model's context limit to decide how much text to send for the document summary.
- **Chunk size and overlap:** Decide via testing retrieval quality.
- **Multiple file types:** Prepare a loader per type (PDF, CSV, TXT, MD); dispatch by extension.
- **Local vs cloud:** Use local model for summary and GPT-4o-mini for answers, or same model for both?
- **Token budget:** Understand tokens per process to stay within a fixed budget.

### Risks and mitigations

| Risk | Mitigation |
|------|------------|
| User asks a question with an empty vector store | Check if Chroma has documents; if not, show upload prompt or "Upload a document first." |
| Long conversation blows up context window | Config: keep last N turns in the prompt; when limit reached, summarize older messages. |
| Rate limits; very large files or long questions | Validate file type and max size on upload. Truncate long questions. Show clear API/rate-limit errors. |

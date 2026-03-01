# 🏗️ RAG System Architecture & Production Hardening Guide

This document covers five production-grade concerns for any RAG system. Each section explains the underlying principle, why skipping it causes real failures, and a concrete implementation path aligned with this codebase.

---

## 1. 🧬 The Golden Rule of Embeddings

### Principle

The `EMBEDDING_MODEL` must be perfectly synchronized across both the ingestion and query paths. This is not a best practice — it is a hard correctness requirement.

### Why It Matters

Embeddings are vectors in a high-dimensional geometric space. The *shape* of that space — which concepts are considered near or far — is entirely determined by the model that created the vectors. If ingest uses model A and the query uses model B, the vectors live in structurally different spaces. Cosine similarity between them becomes meaningless. Retrieval will not obviously fail: it will silently return irrelevant chunks, the LLM will receive unrelated context, and the answers will appear plausible but be grounded in the wrong content. This class of bug is particularly dangerous because it produces no errors — only degraded quality.

### Current Implementation

Both paths import from the same constant:

```python
# src/config.py
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

```python
# ingestion/ingest.py
HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# query/answer.py
HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
```

`src/config.py` is the single source of truth. Neither side hardcodes the model string.

### Upgrading the Embedding Model

Changing `EMBEDDING_MODEL` is a **breaking change to the data layer**, not a configuration update. The procedure is:

1. Update `EMBEDDING_MODEL` in `src/config.py`.
2. **Delete the entire `chroma_db/` directory.** Existing vectors are incompatible with the new model and cannot be incrementally migrated.
3. Re-ingest all documents. Vectors computed by the new model are only comparable to vectors computed by the same new model.

Mixing old and new vectors in the same collection by skipping step 2 will produce the exact silent failure described above.

---

## 2. 🗄️ Database State & Deduplication

### Principle

The vector store must prevent duplicate document chunks. Re-uploading a file without deduplication silently degrades retrieval quality and wastes LLM context budget.

### Why It Matters

The current `_embed_and_store_chunks` function appends chunks to the existing Chroma collection without checking if they already exist:

```python
# ingestion/ingest.py
chroma_db.add_documents(document_chunks)
```

If a user uploads the same PDF twice:
- Every chunk from that file is stored twice.
- `RETRIEVAL_K = 3` means 3 slots are returned per query. If the document has duplicates, those slots are partly or entirely filled with identical content.
- The LLM receives repeated context, wasting tokens and reducing the surface area of information it can reason over.
- More critically, a duplicate-heavy collection will surface the same few chunks regardless of the query, breaking the semantic diversity that makes retrieval useful.

### Implementation

Before ingesting a new file, check whether chunks from the same `source` already exist in the collection and delete them first:

```python
# In ingestion/ingest.py, inside _embed_and_store_chunks()
existing = chroma_db.get(where={"source": filename})
if existing and existing["ids"]:
    chroma_db.delete(ids=existing["ids"])
```

This check uses the `source` metadata field already stamped on every chunk at load time in `loaders.py`. The pattern is **delete-then-insert**, not **insert-if-absent**. This approach also naturally handles document *updates* — if the user uploads a revised version of a file, the old chunks are removed and replaced with the new ones, keeping retrieval grounded in the current content.

The `where={"source": filename}` argument is a Chroma metadata filter. It is only possible because `source` is stored in the metadata of every chunk. This is one of the reasons metadata is added at ingest time rather than only when displaying citations.

---

## 3. 💬 Conversational Memory & Query Reformulation

### Principle

Vector databases require a precise, self-contained semantic target. LLMs require full conversational context to produce natural multi-turn responses. These two requirements are in tension and must be resolved with a two-stage architecture.

### Why It Matters

The current `answer_question` function treats every question as independent:

```python
# query/answer.py
documents = retriever.invoke(question)
```

This works for isolated questions but breaks in conversation. Consider:

> **Turn 1:** "What was the total amount on the invoice?"
> **Turn 2:** "Who was it issued to?"

On turn 2, the retriever receives the bare string `"Who was it issued to?"` — a pronoun with no referent. The vector search has no way to resolve "it" and will retrieve semantically unrelated chunks, or nothing useful at all.

### Implementation: Two-Stage Query Pipeline

**Stage 1 — Standalone Question Reformulation:**

Before hitting the retriever, use a lightweight LLM call to combine the conversation history with the new question into a single self-contained query:

```python
def _reformulate_question(history: list[dict], question: str) -> str:
    """
    Given conversation history and a follow-up question,
    produce a standalone question that a vector search can understand without context.
    """
    history_text = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content'][0]['text']}"
        for msg in history
    )
    prompt = (
        f"Given the following conversation:\n{history_text}\n\n"
        f"Rephrase this follow-up question as a standalone question "
        f"that can be understood without the conversation:\n{question}"
    )
    client = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    response = client.invoke([HumanMessage(content=prompt)])
    return response.content
```

**Stage 2 — Generation with full history:**

Pass the full conversation history to the final generation call so the LLM can produce a contextually aware, natural response:

```python
messages = [
    SystemMessage(content=system_content),  # RAG context
    *[HumanMessage(content=m["content"][0]["text"]) if m["role"] == "user"
      else AIMessage(content=m["content"][0]["text"])
      for m in history],
    HumanMessage(content=standalone_question)
]
```

**Wire it from the app:**

```python
# app.py
def get_model_answer(history, question):
    answer_text, sources = answer_question(question=question, history=history)
```

The function signature of `answer_question` needs to accept an optional `history` parameter. When `history` is empty (the first turn), reformulation is skipped and the original question is used directly.

---

## 4. 🎯 Relevance Filtering

### Principle

Blindly returning the top-k chunks regardless of semantic distance will inject irrelevant context into the prompt. An LLM receiving irrelevant context will either ignore it (wasting tokens) or, worse, hallucinate a plausible-sounding answer grounded in the wrong information.

### Why It Matters

The current retriever uses only `k`:

```python
# query/answer.py
chroma_db.as_retriever(search_kwargs={"k": RETRIEVAL_K})
```

`SCORE_THRESHOLD = 0.75` is declared in `src/config.py` but is not enforced. This means a user asking "What is the weather in Tokyo?" against an invoice document will still receive the top-3 invoice chunks, and the LLM will be forced to reason about irrelevance instead of being told upfront there is no relevant context.

### Implementation

Switch from the default retriever to `similarity_search_with_score`, which returns `(Document, score)` pairs alongside their similarity distances, then apply the threshold as a filter:

```python
# query/answer.py
from src.config import SCORE_THRESHOLD

def _retrieve_with_threshold(chroma_db: Chroma, question: str) -> list:
    results = chroma_db.similarity_search_with_score(question, k=RETRIEVAL_K)
    # Chroma returns L2 distance: lower = more similar. Convert if needed.
    # For cosine similarity (inner product space): higher = more similar.
    # Filter: keep only results that meet the threshold.
    relevant = [doc for doc, score in results if score >= SCORE_THRESHOLD]
    return relevant
```

If `relevant` is empty after filtering, halt the pipeline and return a safe fallback:

```python
if not relevant:
    return (
        "I couldn't find relevant information in the uploaded documents to answer that question.",
        ""
    )
```

This is a deliberate short-circuit: it is better to tell the user that their question is outside the scope of the documents than to generate a confident but hallucinated answer.

> **Note on Chroma distance metrics:** Chroma supports both L2 distance (where lower means more similar) and cosine similarity (where higher means more similar), depending on how the collection was created. Verify which metric your collection uses before applying a threshold. The `SCORE_THRESHOLD` value and comparison direction must match the metric. Default Chroma collections use L2 distance.

---

## 5. 🛡️ Multi-Tenant Security & Isolation

### Principle

A shared vector database without access controls is a data security vulnerability. Without isolation, every user can retrieve and read chunks from every other user's documents.

### Why It Matters

The current system stores all document chunks in a single Chroma collection with no user-scoping. Any query retrieves from the full corpus. In a single-user demo this is acceptable; in any shared or production deployment this constitutes a data leakage risk.

### Implementation: Metadata-Stamped Isolation

The architecture for multi-tenant isolation has four parts:

**1 — Authenticate at the UI layer using Gradio's built-in auth:**

```python
# app.py
demo.launch(auth=[("alice", "pass1"), ("bob", "pass2")])
```

For production this would be backed by a proper user store or OAuth, not a hardcoded list.

**2 — Capture the authenticated user identity via `gr.Request`:**

Gradio injects the active user's identity into any handler that declares a `gr.Request` parameter:

```python
# app.py
def process_document_upload(file, request: gr.Request):
    username = request.username
    for progress, text in ingest_document_stream(file_path=path, filename=filename, user=username):
        yield progress, text
```

**3 — Stamp every chunk's metadata with the user at ingest time:**

```python
# ingestion/ingest.py
def _split_document_to_chunks(filename: str, document, user: str):
    chunks = RecursiveCharacterTextSplitter(...).split_documents(document)
    for chunk in chunks:
        chunk.metadata["user"] = user          # ownership stamp
        chunk.metadata["parent_document"] = filename
        chunk.metadata["date"] = datetime.now().strftime("%Y-%m-%d")
    return chunks
```

**4 — Enforce a metadata filter on every retrieval:**

```python
# query/answer.py
def answer_question(question: str, user: str) -> tuple[str, str]:
    retriever = chroma_db.as_retriever(
        search_kwargs={
            "k": RETRIEVAL_K,
            "filter": {"user": user}   # hard namespace boundary
        }
    )
    documents = retriever.invoke(question)
```

The `filter` argument is passed directly to Chroma's `where` clause. It is enforced at the database layer, not in application code, which means it cannot be bypassed by manipulating the question. Each user only ever searches their own chunks.

**Security boundary summary:**

| Layer | Enforcement |
|-------|-------------|
| UI | Gradio `auth=` prevents unauthenticated access |
| Identity | `gr.Request.username` is set by Gradio's session, not user-supplied input |
| Storage | `user` field stamped in chunk metadata at ingest; cannot be altered after the fact |
| Retrieval | `filter={"user": username}` applied at the Chroma query layer |

No single layer is sufficient on its own. All four must be present for genuine isolation.

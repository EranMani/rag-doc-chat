# Retrieval Score Issue & Mitigation Guide

This document explains the retrieval score problem we encountered, why it happened, and the mitigations applied—in priority order.

---

## 1. The Problem

### Symptoms

- After uploading a document (e.g. a CV), asking "who is Eran Mani?" returned: *"I couldn't find relevant information in the uploaded documents."*
- The retriever was returning an **empty list** even though the document was in Chroma and contained the answer.
- A LangChain warning appeared: *"Relevance scores must be between 0 and 1."*

### Root Cause

Two issues combined:

| Issue | What was wrong |
|-------|----------------|
| **Metric mismatch** | We used `chroma_db.as_retriever(search_type="similarity_score_threshold", score_threshold=0.75)`. LangChain's `similarity_score_threshold` assumes **similarity** scores in the range **0–1** (higher = more relevant). Chroma's default is **L2 (Euclidean) distance**, where **lower = more similar** and values are not in 0–1. |
| **Wrong comparison** | We filtered with `score >= SCORE_THRESHOLD`. For L2 distance, that means "keep documents that are **far away**" and "drop documents that are **close**"—the opposite of what we want. |

Observed scores looked like: `0.105` (best match), `-0.31`, `-0.36`. With a threshold of `0.75` and `>=`, **no** document passed, so retrieval was always empty.

When we switched to `similarity_search_with_score`, Chroma returned **L2 distances** in a different scale (e.g. `1.26`, `1.67`, `1.74`). The **most relevant** document had the **lowest** score (1.26), confirming: **for L2, lower = better**.

---

## 2. Mitigations (in priority order)

Apply these in order. Earlier items fix correctness; later items improve coverage and quality.

---

### Priority 1: Use direct score API and correct filter direction

**Goal:** Retrieve with Chroma’s real scores and filter in a way that matches L2 distance.

**What we did:**

- Stopped using `as_retriever(search_type="similarity_score_threshold", ...)`.
- Called `chroma_db.similarity_search_with_score(question, k=RETRIEVAL_K)` to get `(Document, score)` pairs.
- Filtered with **`score <= SCORE_THRESHOLD`** so we **keep** close (relevant) documents and **drop** far (irrelevant) ones.

**Code (in `query/answer.py`):**

```python
results = chroma_db.similarity_search_with_score(retrieval_question, k=RETRIEVAL_K)
documents = [doc for doc, score in results if score <= SCORE_THRESHOLD]
```

**Config:** Set `SCORE_THRESHOLD` to a value that matches your **observed L2 distances**. For example, if good matches are around 1.2–1.4 and bad ones around 1.6–1.8, use something like `1.4` or `1.5`. Tune by printing scores for a few queries.

**Why first:** Without this, retrieval can return nothing or the wrong documents. Everything else builds on top of correct scoring.

---

### Priority 2: Tune the threshold value

**Goal:** Choose a numeric threshold that separates relevant from irrelevant chunks for your data.

**What we did:**

- Printed scores for several queries: `for doc, score in results: print(f"Score: {score:.4f} | {doc.page_content[:60]}...")`.
- Identified a gap: e.g. relevant chunks ~1.26, irrelevant ~1.67+.
- Set `SCORE_THRESHOLD` in `src/config.py` inside that gap (e.g. `1.4` or `1.5`).

**Config (`src/config.py`):**

```python
SCORE_THRESHOLD = 1.5   # Keep docs with L2 distance <= 1.5; tune from your printed scores
```

**Why second:** Correct direction (`<=`) plus a sensible threshold gives reliable filtering. Too low → miss relevant chunks; too high → let irrelevant chunks in.

---

### Priority 3: Increase retrieval count (`RETRIEVAL_K`)

**Goal:** Retrieve more candidates per query so we don’t miss relevant chunks that rank 4th or 5th.

**What we did:**

- Increased `RETRIEVAL_K` in config (e.g. from 3 to 5 or 6).
- Still filter with `score <= SCORE_THRESHOLD`; we just consider more candidates before filtering.

**Config (`src/config.py`):**

```python
RETRIEVAL_K = 6   # Fetch more candidates; threshold still filters
```

**Why third:** Cheap change, no new code. Reduces the risk of dropping a relevant chunk that’s just outside the top 3.

---

### Priority 4: Relax threshold when coverage matters

**Goal:** When a single document has chunks with a wide range of scores, a strict threshold can keep only the “best” chunk and drop other relevant chunks from the same document.

**What we did:**

- Slightly increased `SCORE_THRESHOLD` (e.g. from 1.4 to 1.5 or 1.6) so more chunks from the same document pass.
- Trade-off: better coverage of one document, but slightly higher chance of including a weaker match from another document.

**Config (`src/config.py`):**

```python
SCORE_THRESHOLD = 1.6   # More permissive for better document coverage
```

**Why fourth:** Only after 1–3 are correct and stable. Use when you see “right document, but the model missed a chunk that was just over the threshold.”

---

### Priority 5: Parent-document / context expansion

**Goal:** When we retrieve one chunk from a document, also include neighbouring chunks (same `source`) so the LLM has full context.

**What we did:**

- After filtering by score, for each matched document we fetch **all chunks** with the same `source` metadata from Chroma.
- Add those chunks to the list (deduplicating so we don’t duplicate already-retrieved chunks).
- The LLM then sees the full document (or all chunks from that file) for any hit.

**Code idea (in `query/answer.py`, after building `documents`):**

```python
expanded_docs = list(documents)
for doc in documents:
    source = doc.metadata.get("source")
    if not source:
        continue
    # Fetch all chunks from the same source (implementation depends on your Chroma/LangChain API)
    # Add chunks not already in expanded_docs, then:
    # expanded_docs = deduplicated union of documents + neighbour chunks
return expanded_docs
```

**Why fifth:** Improves answer quality when the answer spans several chunks. Requires correct use of Chroma’s `get`/metadata filter and deduplication; implement after 1–4.

---

### Priority 6: Multi-query retrieval

**Goal:** One phrasing of the question might miss some relevant chunks; multiple phrasings increase coverage.

**What we did:**

- Generate 2–3 alternative phrasings of the user question (e.g. via a small LLM call).
- Run `similarity_search_with_score` for the original question and each variant.
- Merge results and deduplicate by document/chunk ID.
- Apply the same `score <= SCORE_THRESHOLD` filter to each result set.

**Code idea:**

```python
query_variants = _generate_query_variants(retrieval_question)  # e.g. [original, variant1, variant2]
seen_ids = set()
documents = []
for variant in query_variants:
    results = chroma_db.similarity_search_with_score(variant, k=RETRIEVAL_K)
    for doc, score in results:
        if score <= SCORE_THRESHOLD and doc.id not in seen_ids:
            documents.append(doc)
            seen_ids.add(doc.id)
```

**Why sixth:** Extra LLM call and more retrieval; use when you’ve already applied 1–5 and still miss relevant chunks for some queries.

---

### Priority 7: Smaller chunks and overlap (re-ingest required)

**Goal:** Smaller chunks give more precise embeddings; overlap reduces the chance that important text is split across chunk boundaries.

**What we did:**

- In `src/config.py`: decrease `CHUNK_SIZE` (e.g. 512 → 256), set `CHUNK_OVERLAP` (e.g. 50 or 100).
- **Delete the existing Chroma DB** and re-ingest all documents so every chunk uses the new settings.
- Re-tune `SCORE_THRESHOLD` after re-ingest, because score distribution will change.

**Config (`src/config.py`):**

```python
CHUNK_SIZE = 256
CHUNK_OVERLAP = 50
```

**Why last:** Requires a full re-index and re-tuning. Do when you’re still missing information after 1–6.

---

## 3. Quick reference

| Priority | Mitigation | Where | Re-ingest? |
|----------|------------|--------|------------|
| 1 | Use `similarity_search_with_score` + `score <= SCORE_THRESHOLD` | `query/answer.py` | No |
| 2 | Tune `SCORE_THRESHOLD` from printed scores | `src/config.py` | No |
| 3 | Increase `RETRIEVAL_K` | `src/config.py` | No |
| 4 | Slightly relax `SCORE_THRESHOLD` | `src/config.py` | No |
| 5 | Parent-document / context expansion | `query/answer.py` | No |
| 6 | Multi-query retrieval | `query/answer.py` | No |
| 7 | Smaller `CHUNK_SIZE` and `CHUNK_OVERLAP` | `src/config.py` + ingest | **Yes** (delete Chroma, re-ingest) |

---

## 4. Summary

- **Issue:** LangChain’s `similarity_score_threshold` retriever assumed 0–1 similarity scores; Chroma gave L2 distances (lower = better). We also used `>=` instead of `<=`.
- **Fix:** Use `similarity_search_with_score`, filter with `score <= SCORE_THRESHOLD`, and set the threshold from observed scores.
- **Then:** Increase `RETRIEVAL_K`, tune/relax threshold, add context expansion and/or multi-query if needed, and only then change chunk size (with full re-ingest).

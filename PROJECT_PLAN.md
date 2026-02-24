# rag-doc-chat — Roadmap / Application Plan

## 1. Goal & scope
- What we're building: A demo application of RAG system where the user can upload files, get summary and ask questions about them
- What "done" looks like: Upload -> short summary + chat + gradio UI with source citations
- Out of scope: Multi-user or multi-tenant isolation, LangSmith / full eval harness (RAGAS, TruLens)
- Audience: Me (during build) and future viewers (developers, interviewrs)

## 2. User flow
- Open application
- Upload a document: load document (by type: PDF, CSV, TXT, MD) -> create chunks from document -> embed chunks (create vectors for them) -> store them in a persistent Chroma -> add metadata (date, embedding model version, parent document name)
- Display short summary on the left view
- Show the source citations on the right view
- Ask a question
- Get a response from the model

## 3. Tech stack & constraints
### TECH STACK
- Langchain framework, using abstraction layers such as langchain chroma, text splitters, community tools for faster iterations 
- Chroma for persisted vector store
- all-MiniLM-L6-v2 as the embedding model using the langchain huggingface library
- GPT-5 mini as the chat model

### CONSTRAINTS
- Gradio for UI so we can demo without a separate frontend
- Same embedding model for ingestion and query
- API keys, model names and chunking/tuning params only in config or env (no hardcoding!)

## 4. Architecture (high-level)
- Ingest path: load -> chunk -> embed -> store in chroma + metadata
- Query path: question -> embed question -> search chroma (retriever) -> get chunks -> build prompt (context +question) -> call model -> return answer + source citations
- Why use different paths for ingest and query? each has different responsibilities, separate dependencies for easier debugging and scaling
- Same embedding model for both chunks and query is more maintainable and easier to swap
- After chunks are created and stored, we call the LLM with a bounded excerpt of chunks to produce a short summary. This runs in the ingest path and is shown in the UI
- Response to user will happen after compiling chunk context + question into single system prompt for the model

## 5. Build order / phases
- general approach: env -> config -> ingest -> ui -> query 
- Phase 1: Create the code environment using uv and venv. Use .env file for the API keys.
By the end: establish the required environment to run the application
- Phase 2: Create the config file that includes constants for model names (both local and from cloud), chunks tuning
By the end: A config file that will be used by both ingest and query / answer
- Phase 3: Create the ingest file, which will handle the document loading -> chunking -> embedding -> store in Chroma -> short summary
By the end: A file with main function that will receive the uploaded document -> chunk it -> embed it -> stores in chroma -> return a summary
- Phase 4: Create the UI file which will run the gradio UI -> upload document -> show processing labels -> inject summary into the left view
By the end: A file with gradio app where the user can upload and see the summary on the left side view
- Phase 5: Create the answer file, which will embed the user query -> retrieve from Chroma -> find relevant chunks -> compile system prompt with context + question -> update UI with response
By the end: An option for the user to ask a question and see answer + source citations
- Phase 6: Handling edge cases, such as empty retrieval, document type validation on upload, no empty chunks, basic tests
By the end: A file that will run few tests to find edge cases and improve maintainability

## 6. Open questions / risks
### OPEN QUESTIONS
- I need to check the context window limit of the model, to understand the amount of text I can send him for the document summary process
- I need to figure out the values for chunk size and overlap. this will require testing the retrievel quality before I decide
- What happens when a user upload multiple files that are different types? I need to prepare a loader that can handle each type of file
- Should I use local model for the summary and gpt-5-nano/mini for answering the user?
- I need to understand the amount of tokens per process according to a fixed budget

### RISKS
- User may ask a question while having an empty vector store
Mitigation: Check if Chroma has documents. if not, show the upload button or tell the user that he needs to first upload a document in order to continue
- Optimize system prompt through a long conversation. I need to decide when to perform a history summary in order to reduce the history window
Mitigation: Create a variable in config file to keep the last N turns in the prompt. When the limit reached, perform a summary on the older messages
- Rate limits issues. User can upload very large files, or ask very long questions. How can I handle that?
Mitigation: Validate file types on upload against a predefined max size. When user provides a long question, truncate it. Show clear API and rate limit errors to the user for good UX



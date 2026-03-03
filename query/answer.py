from mpmath.calculus.optimization import str2solver
from src.config import CHAT_MODEL, RAG_SYSTEM_PROMPT, CHROMA_PERSIST_DIR, EMBEDDING_MODEL, RETRIEVAL_K, SCORE_THRESHOLD, MESSAGE_FORMAT_PROMPT
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pathlib import Path
from functools import lru_cache

def _get_client_model_response(system_content: str, question: str, lines: list | None = None) -> str:
    """
    Return the response from the client model.
    Use system message that includes the RAG retrieval context and the user question
    """
    
    # Create client model
    client = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=question),
    ]
    if lines:
        messages.append(AIMessage(content="\n".join(lines)))

    # Send messages to client model and return the response
    response = client.invoke(messages)
    return response.content

def _reformulate_question(history: list, question: str) -> tuple[str, list]:
    """
    Reformat the follow-up question to be a standalone question that can be understood without the conversation history.
    """

    # If history is empty, return the original question
    if not history:
        return question, []

    # NOTE: example for message in history
    # {'role': 'user', 'metadata': None, 'content': [{'text': 'who is eran mani', 'type': 'text'}], 'options': None}

    # Build conversation text from gradio history format
    lines = []
    for msg in history:
        role = msg.get("role", "").capitalize()
        content = msg.get("content", [])
        text = content[0].get("text", "") if content else ""
        if text:
            # Append the generated role and message content to the lines list
            lines.append(f"{role}: {text}")

    # Call the model with the history as context and the original question
    client = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    # No system message is needed because the prompt is designed to handle the conversation history
    response = client.invoke([HumanMessage(content=MESSAGE_FORMAT_PROMPT.format(history_text="\n".join(lines), question=question))])
    return response.content, lines

def answer_question(username: str, question: str = "", history: list | None = None) -> tuple[str, str]:
    """
    Answer a question using the RAG system.
    Load the Chroma DB -> retrieve relevant documents to user question -> build system prompt message -> get model response
    """

    unique_sources_names = set()

    # Load existing Chroma DB
    chroma_db = _load_chroma_db()

    # Check if there are any documents in the Chroma DB
    documents_amount = chroma_db._collection.count()
    if documents_amount == 0:
        return ("Upload and process a document first.", "")

    retrieval_question = question
    # Reformat the retrieval question to include the conversation history + current question
    retrieval_question, lines = _reformulate_question(history, question)
    # print(retrieval_question)
    # print(lines)

    # Load the retriever object from the Chroma DB
    retrieved_documents = _retrieve_documents_based_on_score(username, chroma_db, retrieval_question)
        
    # When no documents are found, return a general response to the user. Avoid unnecessary api calls
    if not retrieved_documents:
        return (
            "I couldn't find relevant information in the uploaded documents to answer your question.",
            ""
        )
    
    # Organize the documents into a single string with new lines between each document
    context = ""
    for i, doc in enumerate(retrieved_documents):
        context += f"Document {i+1}:\n{doc.page_content}\n\n"
        source_name = doc.metadata.get("source", "Unknown")
        if source_name != "Unknown":
            unique_sources_names.add(source_name)

    sources_display = "\n\n---\n**Sources:**\n" + "\n".join(unique_sources_names)

    # Build the system prompt message that includes the RAG retrieval context
    system_content = RAG_SYSTEM_PROMPT.format(document_summary=context)

    # Get model response using the system prompt message and the user question
    model_response = _get_client_model_response(system_content, question, lines)

    return model_response, sources_display


def _load_chroma_db() -> Chroma:
    if Path(CHROMA_PERSIST_DIR).exists():
        return Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=_get_embeddings())
    
    raise ValueError(f"Chroma DB not found at {CHROMA_PERSIST_DIR}")

# NOTE: LRU = Least Recently Used. when the cache is full, it drops the least recently used entry
# maxsize = maximum number of entries to cache.
# No arguments -> one possible call -> cache once
# With arguments that change, each distinct argument tuple is one cache entry. set maxsize to how many such entries you want to keep at most
@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    """
    HuggingFaceEmbeddings loads a model from disk (weights, config) into memory
    This function runs only once, then the return value is cached and reused instead of running this function again
    The embedding model is stateless, so one instance can be reused
    Use lru_cache to avoid repeated model loading.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def _retrieve_documents_based_on_score(username: str, chroma_db: Chroma, retrieval_question: str) -> list[Document]:
    # NOTE: Instead of using as_retriever based approach, which handles the search score automatically, use similarity_search_with_score directly on the chroma db
    # NOTE: Chroma uses L2 distance which is a metric. it measures how far apart two vectors are in space. Distance = 0 -> most similar. Distance = 1 -> least similar.
    # NOTE: L2 distance -> lowest score -> best match. Cosine similarity -> highest score -> best match.
    # OLD WAY: return chroma_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": RETRIEVAL_K, "score_threshold": SCORE_THRESHOLD})
    
    # Get similiarity scores straight from the chroma db
    results = chroma_db.similarity_search_with_score(retrieval_question, k=RETRIEVAL_K)

    # DEBUGGING: print the documents found
    # for doc, score in results:
    #     print(f"Score: {score:.4f} | {doc.page_content[:60]}...")

    # Get the documents with score below the threshold and from the same signed user
    documents = [
        doc for doc, score in results if score <= SCORE_THRESHOLD and doc.metadata.get("username") == username
    ]
    
    # Gather they passed documents into a list
    expanded_docs = list(documents)

    # Run on all found documents and find chunks that includes the same source
    for doc in documents:
        source = doc.metadata.get("source", "Unknown")
        if not source:
            continue
        
        # Fetch the chunks using source and user name filters as query
        all_chunks = chroma_db._collection.get(where={"$and": [{"source": source}, {"username": username}]}, include=["documents", "metadatas"])

        # Get content and metadata for each chunk
        for content, meta in zip(all_chunks["documents"], all_chunks["metadatas"]):
            # Append chunk only if its source is the same as the document parent name
            if meta.get("source") == source:
                if content not in {d.page_content for d in expanded_docs}:
                    # Add the new chunks content to the existing expanded docs list
                    expanded_docs.append(Document(page_content=content, metadata=meta))
    
    return expanded_docs

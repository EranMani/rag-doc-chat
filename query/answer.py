from src.config import CHAT_MODEL, RAG_SYSTEM_PROMPT, CHROMA_PERSIST_DIR, EMBEDDING_MODEL, RETRIEVAL_K
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from pathlib import Path

def _get_client_model_response(system_content: str, question: str):
    """
    Return the response from the client model.
    Use system message that includes the RAG retrieval context and the user question
    """
    # Create client model
    client = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=question)
    ]

    # Send messages to client model and return the response
    response = client.invoke(messages)
    return response.content

def answer_question(question: str = "") -> tuple[str, str]:
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

    # Load the retriever object from the Chroma DB
    retriever = _load_retriever(chroma_db)
    # Get the documents found by the retriever
    documents = retriever.invoke(question)
    
    # Organize the documents into a single string with new lines between each document
    context = ""
    for i, doc in enumerate(documents):
        context += f"Document {i+1}:\n{doc.page_content}\n\n"
        source_name = doc.metadata.get("source", "Unknown")
        if source_name != "Unknown":
            unique_sources_names.add(source_name)

    sources_display = "\n\n---\n**Sources:**\n" + "\n".join(unique_sources_names)

    # Build the system prompt message that includes the RAG retrieval context
    system_content = RAG_SYSTEM_PROMPT.format(document_summary=context)

    # Get model response using the system prompt message and the user question
    model_response = _get_client_model_response(system_content, question)

    return model_response, sources_display


def _load_chroma_db() -> Chroma:
    if Path(CHROMA_PERSIST_DIR).exists():
        return Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=_load_embedding_model())
    
    raise ValueError(f"Chroma DB not found at {CHROMA_PERSIST_DIR}")

def _load_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def _load_retriever(chroma_db: Chroma):
    return chroma_db.as_retriever(search_kwargs={"k": RETRIEVAL_K})

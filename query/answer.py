from src.config import CHAT_MODEL, RAG_SYSTEM_PROMPT, CHROMA_PERSIST_DIR, EMBEDDING_MODEL, RETRIEVAL_K
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from pathlib import Path


def answer_question(question: str = "") -> tuple[str, str]:
    chroma_db = _load_chroma_db()
    documents_amount = chroma_db._collection.count()
    if documents_amount == 0:
        return ("Upload and process a document first.", "")

    retriever = _load_retriever(chroma_db)
    documents = retriever.invoke(question)
    for doc in documents:
        print(doc.metadata)

    return ("Not implemented", "")

def _load_chroma_db() -> Chroma:
    if Path(CHROMA_PERSIST_DIR).exists():
        return Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=_load_embedding_model())
    
    raise ValueError(f"Chroma DB not found at {CHROMA_PERSIST_DIR}")

def _load_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def _load_retriever(chroma_db: Chroma):
    return chroma_db.as_retriever(search_kwargs={"k": RETRIEVAL_K})

answer_question(question="who is eran mani?")




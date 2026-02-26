"""
This file contains the ingestion logic for the RAG system.
It will call the load_document function from the loaders.py file to load the documents from the user's uploaded files.
It will chunk them using langchain RecursiveCharacterTextSplitter
It will enrich the metadata with embedding model version, parent document name, date
It will embed the chunks into vectors and store them in ChromaDB
It will create a short summary on amount of chunks and return it to the user

load -> chunk -> embed & store -> summarize -> return summary
"""

import chromadb
from .loaders import load_document
from src.config import (
    EMBEDDING_MODEL, CHAT_MODEL, SUMMARY_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_PERSIST_DIR
)
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from datetime import datetime
from pathlib import Path

def split_document_to_chunks(filename: str, document):
    """
    Chunk a list of langchain documents into smaller chunks.
    """

    # RecursiveCharacterTextSplitter splits text by trying separators in order (\n\n, \n, ". ", " ", then char)
    # split_documents takes a list of LangChain Documents, splits each page_content, keeps metadata on every resulting chunk, and returns one flat list of chunk Documents.
    chunk_document = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).split_documents(document)

    print(f"Chunked {len(chunk_document)} documents")
    for chunk in chunk_document:
        chunk.metadata["embedding_model"] = EMBEDDING_MODEL
        chunk.metadata["parent_document"] = filename
        chunk.metadata["date"] = datetime.now().strftime("%Y-%m-%d")

    return chunk_document

def embed_and_store_chunks(document_chunks):
    """
    Embed and store a list of document chunks in ChromaDB.
    """

    # Use HuggingFaceEmbeddings to embed the document chunks
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if not Path(CHROMA_PERSIST_DIR).exists():
        # Create a new ChromaDB database
        chroma_db = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    else:
        # Load the existing ChromaDB database
        chroma_db = Chroma.from_documents(documents=document_chunks, embedding=embeddings, persist_directory=CHROMA_PERSIST_DIR)

    # Add the generated chunks to the ChromaDB database
    chroma_db.add_documents(document_chunks)
    print(f"Added {len(document_chunks)} chunks to ChromaDB")

    
def ingest_document(file_path=None, file_bytes=None, filename=None):
    """
    Ingest a document into the RAG system.
    """

    # Turn user document into a list of langchain documents
    document = load_document(file_path=file_path, file_bytes=file_bytes, filename=filename)
    print(f"Loaded {len(document)} documents")

    # Chunk the document into smaller chunks
    document_chunks = split_document_to_chunks(filename, document)
    
    # Embed and store the chunks in ChromaDB
    embed_and_store_chunks(document_chunks)
    
    return document
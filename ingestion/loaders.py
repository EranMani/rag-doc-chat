"""
This file contains the loaders for the different file types
Load a single document (PDF, CSV, TXT, MD) into LangChain Documents with metadata.

Get user document -> pick loader by extension -> load() -> attach metadata -> return list of Document objects
"""

import tempfile
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader
)

# Supported extensions and their loader type for validation and metadata
SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".txt", ".md"}

def load_document(*, file_path: str | None = None, file_bytes: bytes | None = None, filename: str) -> list[Document]:
    """
    Load one document into a list of LangChain Documents.

    Call with either:
      - file_path="path/to/file.pdf", filename="file.pdf"
      - file_bytes=b"...", filename="file.pdf"  (e.g. from Gradio upload)

    File name is required to set metadata.

    Returns list of Document with page_content and metadata (source, type).
    """

    ext = _get_extension(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {ext}")

    # If getting bytes (from gradio for example), write to a temp file and load from path
    if file_bytes is not None:
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, prefix="rag_upload_"
        ) as tmp:
            tmp.write(file_bytes)
            file_path = tmp.name

    elif file_path is None:
        raise ValueError("Either file_bytes or file_path must be provided")

    # The loader turns a user-uploaded file into a list of langchain Document objects
    loader = _get_loader(file_path=file_path, ext=ext)
    # Trigger the loader to read the file and return a list of Document objects
    raw_docs = loader.load()

    # Attach metadata so ingest and query can use it as the source type for citations / filtering
    doc_type = ext.lstrip(".")
    for doc in raw_docs:
        doc.metadata["source"] = filename
        doc.metadata["type"] = doc_type

    return raw_docs

def _get_extension(filename: str) -> str:
    """Get the extension of the file"""
    return Path(filename).suffix.lower()

def _get_loader(file_path: str, ext: str):
    """Get the appropriate loader for the file extension"""
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    elif ext == ".csv":
        return CSVLoader(file_path)
    
    elif ext == ".txt":
        return TextLoader(file_path)
    
    elif ext == ".md":
        return TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

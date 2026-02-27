"""
This file contains the summary logic for the RAG system.
It will read an N amount of chunks from ChromaDB and generate a summary of the document.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import (
    SUMMARY_MODEL, SUMMARY_SYSTEM_PROMPT
)

def generate_summary(document_chunks):
    """
    Generate a summary from a list of given document chunks.
    """

    client = ChatOpenAI(model=SUMMARY_MODEL, temperature=0)
    messages = [
        SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
        HumanMessage(content=document_chunks)
    ]

    response = client.invoke(messages)
    return response.content

def generate_summary_stream(document_chunks: str):
    """
    Stream the summary token-by-token; yields cumulative text so the UI can update as it arrives.
    """
    client = ChatOpenAI(model=SUMMARY_MODEL, temperature=0)
    messages = [
        SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
        HumanMessage(content=document_chunks)
    ]
    summary = ""
    for chunk in client.stream(messages):
        # Stream chunks may not always have .content (e.g. role-only or tool-call chunks);
        # getattr avoids AttributeError and skips non-content chunks.
        if getattr(chunk, "content", None):
            # Yield cumulative text so far; Gradio treats this generator as "streaming"
            # and redraws the output with each yield, showing the summary as it grows.
            summary += chunk.content
            yield summary

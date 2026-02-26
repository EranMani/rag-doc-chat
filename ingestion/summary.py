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

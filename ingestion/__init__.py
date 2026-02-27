from .ingest import ingest_document, ingest_document_stream

# Make ingest_document the only public entrypoint from the ingest module
__all__ = ["ingest_document", "ingest_document_stream"]
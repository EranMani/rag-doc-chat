from .ingest import ingest_document

# Make ingest_document the only public entrypoint from the ingest module
__all__ = ["ingest_document"]
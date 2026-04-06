"""TrifectaRAG — Unified multi-modal RAG engine."""

from trifecta.client import TrifectaClient, _normalize  # noqa: F401
from trifecta import trifecta_py  # noqa: F401
from trifecta.pdf_ingest import PDFIngestor  # noqa: F401

__all__ = ["TrifectaClient", "_normalize", "trifecta_py", "PDFIngestor"]

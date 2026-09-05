"""
TrifectaRAG — multi-modal RAG engine (HNSW + BM25 + knowledge graph).

Public SDK::

    from trifecta import TrifectaClient, PDFIngestor
    from trifecta import trifecta_py as tr

    client = TrifectaClient(device="cpu")
    client.add_document("Lagrange interpolation uses basis polynomials.")
    hits = client.get_results(client.query("interpolation", top_k=5))
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from trifecta._version import __version__

__all__ = [
    "__version__",
    "TrifectaClient",
    "PDFIngestor",
    "trifecta_py",
    "_normalize",
]


def _windows_dll_search_paths() -> None:
    """Allow the extension to find bundled MinGW / MSVC runtime DLLs."""
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return
    seen: set[str] = set()
    candidates = [Path(__file__).resolve().parent]
    for entry in sys.path:
        candidates.append(Path(entry) / "trifecta")
    for folder in candidates:
        key = str(folder.resolve()) if folder.exists() else ""
        if not key or key in seen:
            continue
        seen.add(key)
        try:
            os.add_dll_directory(key)
        except OSError:
            pass


_windows_dll_search_paths()


def _load_extension() -> Any:
    import importlib

    try:
        return importlib.import_module("trifecta.trifecta_py")
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The C++ extension 'trifecta_py' failed to load. "
            "Install a wheel with `pip install trifectarag` or build from "
            "source (`pip install -e .`) with CMake and a C++17 compiler."
        ) from exc


def __getattr__(name: str) -> Any:
    if name == "trifecta_py":
        return _load_extension()
    if name in {"TrifectaClient", "_normalize"}:
        from trifecta.client import TrifectaClient, _normalize

        return TrifectaClient if name == "TrifectaClient" else _normalize
    if name == "PDFIngestor":
        from trifecta.pdf_ingest import PDFIngestor

        return PDFIngestor
    if name in {"hybrid_search", "format_passages", "serialize_sources", "mmr_rerank"}:
        from trifecta import retrieve as _retrieve

        return getattr(_retrieve, name)
    if name == "run_agent":
        from trifecta.agent import run_agent

        return run_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

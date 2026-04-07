"""
Shared settings for textbook PDF examples.

Place your PDF under examples/data/ (default: Numerical_Analysis.pdf) or set
TRIFECTA_TEXTBOOK_PDF to an absolute path.

Page window env vars (0-based, end exclusive):
  TRIFECTA_PAGE_START=<int>        explicit start page index
  TRIFECTA_PAGE_END=<int>          explicit end page index (exclusive)
  TRIFECTA_FRONT_MATTER_PAGES=<int> pages to skip from the front
    If none are set, a sensible default is chosen per known PDF filename.
    Known defaults:
      Numerical_Analysis   -> 11  (cover, title, publisher, 4x ToC, 4x preface)

Ollama:
  TRIFECTA_OLLAMA_MODEL=<model>    default: qwen2.5:7b
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_PDF_NAME = "Numerical_Analysis.pdf"

# Front-matter page counts for known PDFs (0-based, i.e. skip first N pages).
_FRONT_MATTER: dict[str, int] = {
    "Numerical_Analysis": 11,
}


def resolve_pdf_path() -> Path:
    """Return the primary PDF to use for ingestion / query examples."""
    env = os.environ.get("TRIFECTA_TEXTBOOK_PDF", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(
                f"TRIFECTA_TEXTBOOK_PDF does not exist: {p}"
            )
        return p

    preferred = DATA_DIR / DEFAULT_PDF_NAME
    if preferred.is_file():
        return preferred.resolve()

    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDF found in {DATA_DIR}. Add {DEFAULT_PDF_NAME} or set "
            "TRIFECTA_TEXTBOOK_PDF to a .pdf path."
        )
    return pdfs[0].resolve()


def resolve_all_pdfs() -> list[Path]:
    """Return all PDFs in the data directory, sorted alphabetically.

    Falls back to the single PDF from resolve_pdf_path() if the data dir
    is empty or missing.
    """
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        return [resolve_pdf_path()]
    return [p.resolve() for p in pdfs]


def page_range_for_document(num_pages: int, pdf_path: Optional[Path] = None) -> range:
    """
    Return range(start, end) for PDFIngestor based on env vars and known defaults.

    Priority:
      1. TRIFECTA_PAGE_START / TRIFECTA_PAGE_END (explicit window)
      2. TRIFECTA_FRONT_MATTER_PAGES (skip N front pages, go to end)
      3. Known per-filename front-matter table
      4. Full document (range(0, num_pages))
    """
    start_s = os.environ.get("TRIFECTA_PAGE_START", "").strip()
    end_s = os.environ.get("TRIFECTA_PAGE_END", "").strip()
    fm_s = os.environ.get("TRIFECTA_FRONT_MATTER_PAGES", "").strip()

    if start_s:
        start = max(0, int(start_s))
    elif fm_s:
        start = max(0, int(fm_s))
    elif pdf_path is not None:
        start = _FRONT_MATTER.get(pdf_path.stem, 0)
    else:
        start = 0

    end = min(int(end_s), num_pages) if end_s else num_pages

    if end <= start:
        raise ValueError(
            f"Content page range is empty: start={start}, end={end}. "
            "Check TRIFECTA_PAGE_START / TRIFECTA_PAGE_END / TRIFECTA_FRONT_MATTER_PAGES."
        )
    return range(start, end)


def snapshot_path(pdf_path: Path, mode: str, page_range: range) -> Path:
    """
    Return the canonical *base* path for snapshot files.

    The new v3 format creates two files:
      ``<stem>.trifecta`` (binary engine) and ``<stem>.meta.gz`` (metadata).
    Legacy code that checked ``snap.exists()`` should use ``snapshot_exists()``
    instead, or check for the ``.trifecta`` file.
    """
    stem = f"{pdf_path.stem}_{mode}_p{page_range.start}-{page_range.stop}"
    return DATA_DIR / stem


def snapshot_exists(base: Path) -> bool:
    """Return True if the v3 snapshot files exist (or legacy .snap.gz)."""
    return (Path(str(base) + ".trifecta").exists()
            or Path(str(base) + ".snap.gz").exists())


def ollama_model() -> str:
    """Return the Ollama model name to use (env override or default)."""
    return os.environ.get("TRIFECTA_OLLAMA_MODEL", "qwen2.5:7b").strip()


def print_retrieval_result(r: dict, *, max_text_chars: Optional[int] = None) -> None:
    """
    Print one enriched query hit: full text for TEXT chunks, paths + caption
    for IMAGE chunks (figures / rasterized tables from the PDF).
    """
    meta = r["metadata"]
    gid = r["global_id"]
    score = r["score"]
    mod = r["modality"]
    page = meta.get("page", "?")
    src = meta.get("source", "?")
    chunk_type = meta.get("type", "?")

    print(f"  -- hit global_id={gid}  [{mod}]  score={score:.6f}  "
          f"page={page}  source={src}  type={chunk_type}")

    if mod == "IMAGE":
        ip = meta.get("image_path", "")
        cap = meta.get("caption", "")
        print(f"     image_path: {ip}")
        if cap:
            print(f"     caption: {cap}")
        return

    body = meta.get("full_text") or meta.get("text_preview") or ""
    if max_text_chars is not None and len(body) > max_text_chars:
        body = body[:max_text_chars] + "\n     ... [truncated]"
    for line in body.splitlines():
        print(f"     {line}")
    if not body.strip():
        print("     (no stored text in metadata)")

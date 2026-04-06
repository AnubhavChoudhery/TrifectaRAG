"""
Shared settings for textbook PDF examples.

Place your PDF under examples/data/ (default: Numerical_Analysis.pdf) or set
TRIFECTA_TEXTBOOK_PDF to an absolute path.

Optional page window (0-based indices, end exclusive):
  TRIFECTA_PAGE_START=0
  TRIFECTA_PAGE_END=50
If both are unset, all pages are processed. If only TRIFECTA_PAGE_START is set,
pages from that index through the end of the document are used.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_PDF_NAME = "Numerical_Analysis.pdf"


def resolve_pdf_path() -> Path:
    """Return the PDF to use for ingestion / query examples."""
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
    if len(pdfs) > 1:
        names = ", ".join(p.name for p in pdfs)
        raise FileNotFoundError(
            f"Multiple PDFs in {DATA_DIR}: {names}. "
            f"Rename one to {DEFAULT_PDF_NAME} or set TRIFECTA_TEXTBOOK_PDF."
        )
    return pdfs[0].resolve()


def page_range_for_document(num_pages: int) -> Optional[range]:
    """
    Return None to mean \"all pages\" for PDFIngestor.ingest_pdf, or a bounded
    range after reading *num_pages* from the opened PDF.
    """
    start_s = os.environ.get("TRIFECTA_PAGE_START", "").strip()
    end_s = os.environ.get("TRIFECTA_PAGE_END", "").strip()
    if not start_s and not end_s:
        return None
    start = max(0, int(start_s or "0"))
    if not end_s:
        return range(start, num_pages)
    end = min(int(end_s), num_pages)
    if end <= start:
        raise ValueError("TRIFECTA_PAGE_END must be greater than TRIFECTA_PAGE_START")
    return range(start, end)


def print_retrieval_result(r: dict, *, max_text_chars: Optional[int] = None) -> None:
    """
    Print one enriched query hit: full text for TEXT chunks, paths + caption
    for IMAGE chunks (tables/figures are stored as raster images from the PDF).
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

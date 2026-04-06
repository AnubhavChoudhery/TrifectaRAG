"""
01_ingest_textbook.py — Ingest a local PDF from examples/data/.

Default file: Numerical_Analysis.pdf (or the only *.pdf in that folder).
Override with TRIFECTA_TEXTBOOK_PDF. Optional page window: TRIFECTA_PAGE_START,
TRIFECTA_PAGE_END (0-based, end exclusive).

Usage (from repo root):
    python examples/01_ingest_textbook.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_EX = Path(__file__).resolve().parent
sys.path.insert(0, str(_EX.parent))
sys.path.insert(0, str(_EX))

import fitz  # PyMuPDF

from textbook_config import DATA_DIR, page_range_for_document, resolve_pdf_path
from trifecta import TrifectaClient, PDFIngestor


def run_ingest(pdf: Path, page_range, mode: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Ingestion mode: {mode.upper()}")
    print(f"{'=' * 60}")

    client = TrifectaClient(device="cpu")
    ingestor = PDFIngestor(
        client, mode=mode, chunk_size=256, overlap=64, min_img_px=60
    )

    t0 = time.perf_counter()
    stats = ingestor.ingest_pdf(
        str(pdf),
        output_dir=str(DATA_DIR / f"extracted_{mode}"),
        page_range=page_range,
    )
    elapsed = time.perf_counter() - t0

    print(f"\n  Results ({mode}):")
    print(f"    Pages processed : {stats['pages']}")
    print(f"    Text chunks     : {stats['text_chunks']}")
    print(f"    Images extracted: {stats['images']}")
    print(f"    KG edges created: {stats['kg_edges']}")
    print(f"    Engine size     : {client.size} nodes")
    print(f"    Time            : {elapsed:.1f}s")


def main() -> None:
    print("TrifectaRAG — Textbook PDF ingestion\n")

    pdf = resolve_pdf_path()
    doc = fitz.open(pdf)
    try:
        n = len(doc)
    finally:
        doc.close()

    pr = page_range_for_document(n, pdf)
    print(f"  PDF: {pdf}")
    print(
        f"  Size: {pdf.stat().st_size / 1024 ** 2:.1f} MB  |  "
        f"pages: {n} total, ingesting pages {pr.start + 1}..{pr.stop} "
        f"(0-based {pr.start}..{pr.stop - 1})"
    )

    run_ingest(pdf, pr, "page")
    run_ingest(pdf, pr, "classical")

    print(
        "\nDone. Extracted images: examples/data/extracted_page/ and "
        "examples/data/extracted_classical/"
    )


if __name__ == "__main__":
    main()

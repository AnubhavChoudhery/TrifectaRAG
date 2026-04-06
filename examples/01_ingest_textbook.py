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

from textbook_config import DATA_DIR, page_range_for_document, resolve_pdf_path, snapshot_path
from trifecta import TrifectaClient, PDFIngestor


def run_ingest(pdf: Path, page_range, mode: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Ingestion mode: {mode.upper()}")
    print(f"{'=' * 60}")

    snap = snapshot_path(pdf, mode, page_range)
    if snap.exists():
        print(f"\n  Snapshot already exists: {snap.name}")
        print("  Delete it and re-run to force re-ingestion.")
        return

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
    print(f"    Images extracted: {stats['images']}  (raster + vector figures)")
    print(f"    KG edges created: {stats['kg_edges']}")
    print(f"    Engine size     : {client.size} nodes")
    print(f"    Time            : {elapsed:.1f}s")

    # Persist: saves all embeddings + metadata + KG edges to a compressed file
    # so 02/03 examples can reload the engine in seconds without re-ingesting.
    client.save_snapshot(str(snap))
    snap_kb = snap.stat().st_size / 1024
    print(f"    Snapshot        : {snap.name} ({snap_kb:.0f} KB)")


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
        "\nDone.\n"
        "  Images (raster + rendered vector figures):\n"
        "    examples/data/extracted_page/\n"
        "    examples/data/extracted_classical/\n"
        "  Snapshots (fast reload for 02/03):\n"
        f"    {snapshot_path(pdf, 'page', pr)}\n"
        f"    {snapshot_path(pdf, 'classical', pr)}"
    )


if __name__ == "__main__":
    main()

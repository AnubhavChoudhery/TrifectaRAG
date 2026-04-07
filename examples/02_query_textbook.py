"""
02_query_textbook.py — Query the ingested textbook PDF.

Prints full stored text for each TEXT hit and paths + captions for IMAGE hits
(figures and rasterized tables from the PDF).

Usage:
    python examples/02_query_textbook.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_EX = Path(__file__).resolve().parent
sys.path.insert(0, str(_EX.parent))
sys.path.insert(0, str(_EX))

import fitz

from textbook_config import (
    DATA_DIR,
    page_range_for_document,
    print_retrieval_result,
    resolve_pdf_path,
    snapshot_exists,
    snapshot_path,
)
from trifecta import TrifectaClient, PDFIngestor

# Numerical analysis–style queries (BM25 + CLIP + KG)
QUERIES = [
    "Newton method nonlinear equation",
    "bisection root finding",
    "LU factorization linear system",
    "polynomial interpolation",
    "trapezoidal rule numerical integration",
    "floating point rounding error",
]


def build_engine(mode: str = "page") -> TrifectaClient:
    pdf = resolve_pdf_path()
    doc = fitz.open(pdf)
    try:
        n = len(doc)
    finally:
        doc.close()
    pr = page_range_for_document(n, pdf)

    snap = snapshot_path(pdf, mode, pr)
    if snapshot_exists(snap):
        print(f"  Loading from snapshot: {snap.name}  (fast reload -- run 01 to re-ingest)")
        client = TrifectaClient.from_snapshot(str(snap), device="cpu")
        print(f"  Engine: {client.size} nodes, {client.page_count} pages indexed\n")
        return client

    print(f"  No snapshot found. Ingesting pages {pr.start + 1}..{pr.stop} ({len(pr)} pages)...")
    print("  Tip: run 01_ingest_textbook.py first to save a snapshot for fast reloads.\n")
    client = TrifectaClient(device="cpu")
    ingestor = PDFIngestor(client, mode=mode, min_img_px=60)
    stats = ingestor.ingest_pdf(
        str(pdf),
        output_dir=str(DATA_DIR / f"extracted_{mode}"),
        page_range=pr,
    )
    print(
        f"  Engine: {stats['text_chunks']} text chunks, "
        f"{stats['images']} images, {stats['kg_edges']} KG edges\n"
    )
    return client


def run_queries(client: TrifectaClient) -> None:
    for query_text in QUERIES:
        print(f"{'=' * 72}")
        print(f"Query: {query_text!r}")
        print(f"{'=' * 72}")
        t0 = time.perf_counter()
        raw = client.query(text=query_text, top_k=5)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        results = client.get_results(raw)

        if not results:
            print("  (no results)\n")
            continue

        for rank, r in enumerate(results, start=1):
            print(f"\n  ### Rank {rank}")
            print_retrieval_result(r)
        print(f"\n  ({elapsed_ms:.0f} ms)\n")


def main() -> None:
    print("TrifectaRAG — Textbook query demo (full text + image paths)\n")

    client = build_engine("page")
    run_queries(client)

    print("Done.")


if __name__ == "__main__":
    main()

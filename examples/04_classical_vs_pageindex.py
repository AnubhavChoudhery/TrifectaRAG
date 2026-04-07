"""
04_classical_vs_pageindex.py — Compare page-indexed vs classical chunk
retrieval on the same local textbook PDF.

Usage:
    python examples/04_classical_vs_pageindex.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_EX = Path(__file__).resolve().parent
sys.path.insert(0, str(_EX.parent))
sys.path.insert(0, str(_EX))

import fitz

from textbook_config import DATA_DIR, page_range_for_document, resolve_pdf_path
from trifecta import TrifectaClient, PDFIngestor

QUERIES = [
    "Newton method",
    "interpolation polynomial",
    "numerical integration",
    "LU decomposition",
]


def build(mode: str) -> TrifectaClient:
    pdf = resolve_pdf_path()
    doc = fitz.open(pdf)
    try:
        n = len(doc)
    finally:
        doc.close()
    pr = page_range_for_document(n, pdf)

    client = TrifectaClient(device="cpu")
    ingestor = PDFIngestor(
        client, mode=mode, chunk_size=256, overlap=64, min_img_px=60
    )
    ingestor.ingest_pdf(
        str(pdf),
        output_dir=str(DATA_DIR / f"extracted_{mode}"),
        page_range=pr,
    )
    return client


def query_mode(client: TrifectaClient):
    results_by_query = {}
    for q in QUERIES:
        t0 = time.perf_counter()
        raw = client.query(text=q, top_k=3)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        enriched = client.get_results(raw)
        results_by_query[q] = {"results": enriched, "ms": elapsed_ms}
    return results_by_query


def format_result(r: dict) -> str:
    meta = r["metadata"]
    page = meta.get("page", "?")
    tag = r["modality"]
    score = r["score"]
    if tag == "IMAGE":
        return f"[IMG p{page}] {score:.4f}"
    preview = (meta.get("full_text") or meta.get("text_preview", ""))[:50].replace(
        "\n", " "
    )
    return f"[p{page}] {score:.4f} {preview}..."


def main() -> None:
    resolve_pdf_path()

    print("TrifectaRAG — Classical vs page-index (Numerical Analysis PDF)")
    print("=" * 70)

    print("\nBuilding page-indexed engine...", end=" ", flush=True)
    t0 = time.perf_counter()
    page_client = build("page")
    page_time = time.perf_counter() - t0
    print(f"({page_time:.1f}s, {page_client.size} nodes, {page_client.page_count} pages)")

    print("Building classical engine...", end=" ", flush=True)
    t0 = time.perf_counter()
    chunk_client = build("classical")
    chunk_time = time.perf_counter() - t0
    print(f"({chunk_time:.1f}s, {chunk_client.size} nodes, {chunk_client.page_count} pages)")

    page_results = query_mode(page_client)
    chunk_results = query_mode(chunk_client)

    print(
        f"\n{'Query':<35s} | {'Page-indexed (top-1)':<40s} | {'Classical (top-1)':<40s}"
    )
    print("-" * 120)

    for q in QUERIES:
        pr = page_results[q]
        cr = chunk_results[q]

        page_top = format_result(pr["results"][0]) if pr["results"] else "(none)"
        chunk_top = format_result(cr["results"][0]) if cr["results"] else "(none)"

        print(f"{q:<35s} | {page_top:<40s} | {chunk_top:<40s}")

    print()
    print("For full chunk text and image paths, run: python examples/02_query_textbook.py")
    print("\nDone.")


if __name__ == "__main__":
    main()

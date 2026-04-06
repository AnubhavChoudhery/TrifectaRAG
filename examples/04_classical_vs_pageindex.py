"""
04_classical_vs_pageindex.py — Compare classical chunk-based RAG vs
page-indexed RAG on the same PDF and queries.

Usage:
    python examples/04_classical_vs_pageindex.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trifecta import TrifectaClient, PDFIngestor

DATA_DIR = Path(__file__).resolve().parent / "data"
PDF_PATH = DATA_DIR / "openstax_calculus_v1.pdf"

PAGE_START = 249
PAGE_END = 318

QUERIES = [
    "What is the chain rule?",
    "derivative of sin(x)",
    "power rule for differentiation",
    "implicit differentiation",
]


def build(mode: str) -> TrifectaClient:
    client = TrifectaClient(device="cpu")
    ingestor = PDFIngestor(client, mode=mode, chunk_size=256, overlap=64, min_img_px=60)
    stats = ingestor.ingest_pdf(
        str(PDF_PATH),
        output_dir=str(DATA_DIR / f"extracted_{mode}"),
        page_range=range(PAGE_START, PAGE_END),
    )
    return client


def query_mode(client: TrifectaClient, label: str) -> dict:
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
    preview = meta.get("text_preview", "")[:50].replace("\n", " ")
    return f"[p{page}] {score:.4f} {preview}..."


def main() -> None:
    if not PDF_PATH.exists():
        print("ERROR: PDF not found. Run 01_ingest_math_book.py first.")
        sys.exit(1)

    print("TrifectaRAG — Classical vs Page-Index Comparison")
    print("=" * 70)

    print("\nBuilding page-indexed engine...", end=" ", flush=True)
    t0 = time.perf_counter()
    page_client = build("page")
    page_time = time.perf_counter() - t0
    print(f"({page_time:.1f}s, {page_client.size} nodes)")

    print("Building classical engine...", end=" ", flush=True)
    t0 = time.perf_counter()
    chunk_client = build("classical")
    chunk_time = time.perf_counter() - t0
    print(f"({chunk_time:.1f}s, {chunk_client.size} nodes)")

    page_results = query_mode(page_client, "page")
    chunk_results = query_mode(chunk_client, "classical")

    print(f"\n{'Query':<35s} | {'Page-indexed':<40s} | {'Classical (chunked)':<40s}")
    print("-" * 120)

    for q in QUERIES:
        pr = page_results[q]
        cr = chunk_results[q]

        page_top = format_result(pr["results"][0]) if pr["results"] else "(none)"
        chunk_top = format_result(cr["results"][0]) if cr["results"] else "(none)"

        print(f"{q:<35s} | {page_top:<40s} | {chunk_top:<40s}")

    print()
    print("Observations:")
    print("  - Page-indexed mode preserves full page context (better for long answers).")
    print("  - Classical mode finds more specific chunk matches (better for precise facts).")
    print("  - KG expansion (RELATES_TO between consecutive pages) gives page-indexed")
    print("    mode access to surrounding pages automatically.")
    print("\nDone.")


if __name__ == "__main__":
    main()

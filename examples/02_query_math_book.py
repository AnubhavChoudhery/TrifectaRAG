"""
02_query_math_book.py — Query the ingested OpenStax Calculus chapter.

Demonstrates text queries, formula queries, image-aware retrieval, and
KG expansion. Requires 01_ingest_math_book.py to have been run first
(to download the PDF).

Usage:
    python examples/02_query_math_book.py
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
    "What is the chain rule for derivatives?",
    "derivative of sin(x)",
    "graph of a parabola",
    "power rule differentiation",
    "implicit differentiation example",
    "table of derivative formulas",
]


def build_engine(mode: str = "page") -> TrifectaClient:
    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found at {PDF_PATH}")
        print("Run 01_ingest_math_book.py first to download it.")
        sys.exit(1)

    client = TrifectaClient(device="cpu")
    ingestor = PDFIngestor(client, mode=mode, min_img_px=60)
    stats = ingestor.ingest_pdf(
        str(PDF_PATH),
        output_dir=str(DATA_DIR / f"extracted_{mode}"),
        page_range=range(PAGE_START, PAGE_END),
    )
    print(f"  Engine loaded: {stats['text_chunks']} text chunks, "
          f"{stats['images']} images, {stats['kg_edges']} KG edges\n")
    return client


def run_queries(client: TrifectaClient) -> None:
    for query_text in QUERIES:
        print(f"--- Query: \"{query_text}\" ---")
        t0 = time.perf_counter()
        raw = client.query(text=query_text, top_k=5)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        results = client.get_results(raw)

        if not results:
            print("  (no results)\n")
            continue

        for r in results:
            meta = r["metadata"]
            tag = r["modality"]
            page = meta.get("page", "?")
            score = r["score"]

            if tag == "IMAGE":
                img_path = meta.get("image_path", "")
                print(f"  [{tag:5s}] page {page:>3s}  score={score:.5f}  -> {img_path}")
            else:
                preview = meta.get("text_preview", "")[:80].replace("\n", " ")
                print(f"  [{tag:5s}] page {page:>3s}  score={score:.5f}  {preview}...")

        print(f"  ({elapsed_ms:.0f} ms)\n")


def main() -> None:
    print("TrifectaRAG — Math Book Query Demo")
    print("=" * 60)
    print("\nBuilding page-indexed engine from PDF...\n")

    client = build_engine("page")
    run_queries(client)

    print("Done.")


if __name__ == "__main__":
    main()

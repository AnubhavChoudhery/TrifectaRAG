"""
01_ingest_math_book.py — Download and ingest an OpenStax Calculus chapter.

Downloads Chapter 3 (Derivatives) of OpenStax Calculus Volume 1 (CC-BY-4.0),
extracts text and images per page, and ingests into TrifectaRAG in both
page-indexed and classical chunk modes for comparison.

Usage:
    python examples/01_ingest_math_book.py
"""

import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trifecta import TrifectaClient, PDFIngestor

DATA_DIR = Path(__file__).resolve().parent / "data"
PDF_URL = (
    "https://assets.openstax.org/oscms-prodcms/media/documents/"
    "Calculus_Volume_1_-_WEB_aaWYqJq.pdf"
)
PDF_PATH = DATA_DIR / "openstax_calculus_v1.pdf"

PAGE_START = 249
PAGE_END = 318


def download_pdf() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if PDF_PATH.exists():
        print(f"  PDF already cached at {PDF_PATH}")
        return PDF_PATH
    print(f"  Downloading OpenStax Calculus Vol 1 ({PDF_URL}) ...")
    urllib.request.urlretrieve(PDF_URL, PDF_PATH)
    print(f"  Saved to {PDF_PATH}")
    return PDF_PATH


def run_ingest(mode: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Ingestion mode: {mode.upper()}")
    print(f"{'='*60}")

    client = TrifectaClient(device="cpu")
    ingestor = PDFIngestor(client, mode=mode, chunk_size=256, overlap=64, min_img_px=60)

    t0 = time.perf_counter()
    stats = ingestor.ingest_pdf(
        str(PDF_PATH),
        output_dir=str(DATA_DIR / f"extracted_{mode}"),
        page_range=range(PAGE_START, PAGE_END),
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
    print("TrifectaRAG — Math Book Ingestion Demo")
    print("OpenStax Calculus Vol 1, Chapter 3 (Derivatives)")
    print(f"Pages {PAGE_START+1}–{PAGE_END}\n")

    pdf = download_pdf()
    print(f"  PDF: {pdf} ({pdf.stat().st_size / 1024**2:.1f} MB)")

    run_ingest("page")
    run_ingest("classical")

    print("\nDone. Extracted images are in examples/data/extracted_page/ and "
          "examples/data/extracted_classical/")


if __name__ == "__main__":
    main()

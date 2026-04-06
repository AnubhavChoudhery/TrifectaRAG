"""
01_ingest_math_book.py — Download and ingest an OpenStax Calculus chapter.

Downloads Chapter 3 (Derivatives) of OpenStax Calculus Volume 1 (CC-BY-4.0),
extracts text and images per page, and ingests into TrifectaRAG in both
page-indexed and classical chunk modes for comparison.

Usage:
    python examples/01_ingest_math_book.py
"""

import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trifecta import TrifectaClient, PDFIngestor

DATA_DIR = Path(__file__).resolve().parent / "data"
# Alternate filenames are used when OpenStax rotates CDN asset names.
PDF_URLS = (
    "https://assets.openstax.org/oscms-prodcms/media/documents/"
    "Calculus_Volume_1_-_WEB_aaWYqJq.pdf",
    "https://assets.openstax.org/oscms-prodcms/media/documents/"
    "Calculus_Volume_1_-_WEB_68M1Z5W.pdf",
)
PDF_PATH = DATA_DIR / "openstax_calculus_v1.pdf"

PAGE_START = 249
PAGE_END = 318


def download_pdf() -> Path:
    env_path = os.environ.get("TRIFECTA_OPENSTAX_PDF", "").strip()
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if not p.is_file():
            print(f"ERROR: TRIFECTA_OPENSTAX_PDF does not exist: {p}", file=sys.stderr)
            sys.exit(1)
        print(f"  Using PDF from TRIFECTA_OPENSTAX_PDF: {p}")
        return p

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if PDF_PATH.exists():
        print(f"  PDF already cached at {PDF_PATH}")
        return PDF_PATH

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf,*/*;q=0.8",
        "Referer": "https://openstax.org/",
    }
    last_err: Exception | None = None
    for url in PDF_URLS:
        print(f"  Trying download: {url} ...")
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=180) as resp, open(
                PDF_PATH, "wb"
            ) as out:
                out.write(resp.read())
            print(f"  Saved to {PDF_PATH}")
            return PDF_PATH
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
            last_err = e
            print(f"    ({e})")

    print(
        "\n  Automatic download failed (OpenStax may block scripted requests).\n"
        "  Download Calculus Volume 1 PDF from https://openstax.org/details/books/calculus-volume-1\n"
        "  Save it as:\n"
        f"    {PDF_PATH}\n"
        "  Or set TRIFECTA_OPENSTAX_PDF to the full path of your copy.\n",
        file=sys.stderr,
    )
    if last_err:
        raise last_err
    sys.exit(1)


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

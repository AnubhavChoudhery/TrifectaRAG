"""
demo_math_quick.py — Offline math-style RAG demo (no OpenStax download).

Builds a tiny synthetic PDF (chain rule text, a trig-derivatives “table”,
and a raster figure), ingests it with PDFIngestor, then runs queries so you
can see TEXT vs IMAGE hits and file paths for figures.

Usage (from repo root):
    python examples/demo_math_quick.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz  # PyMuPDF
from PIL import Image, ImageDraw

from trifecta import TrifectaClient, PDFIngestor

DATA_DIR = Path(__file__).resolve().parent / "data"
DEMO_PDF = DATA_DIR / "demo_calculus_snippet.pdf"


def _make_demo_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()

    p0 = doc.new_page(width=612, height=792)
    p0.insert_text(
        (72, 100),
        "The Chain Rule\n\n"
        "If h(x) = f(g(x)), then h'(x) = f'(g(x)) g'(x).\n"
        "This lets us differentiate compositions of functions.",
        fontsize=11,
    )

    p1 = doc.new_page(width=612, height=792)
    p1.insert_text(
        (72, 72),
        "Common derivatives (table)\n\n"
        "d/dx sin(x) = cos(x)\n"
        "d/dx cos(x) = -sin(x)\n"
        "d/dx tan(x) = sec^2(x)\n"
        "d/dx x^n = n x^(n-1)   (power rule)\n",
        fontsize=11,
    )

    # Raster “figure”: simple parabola sketch
    img = Image.new("RGB", (320, 200), color=(255, 255, 255))
    dr = ImageDraw.Draw(img)
    dr.rectangle([0, 0, 319, 199], outline=(80, 80, 80))
    for x in range(0, 300, 4):
        px = 20 + x
        py = 180 - int((x - 150) ** 2 / 200)
        dr.ellipse([px - 1, py - 1, px + 1, py + 1], fill=(0, 100, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    p2 = doc.new_page(width=612, height=792)
    p2.insert_text(
        (72, 72),
        "Figure: graph of a parabola opening upward (y ~ x^2).",
        fontsize=11,
    )
    rect = fitz.Rect(72, 100, 72 + 320, 100 + 200)
    p2.insert_image(rect, stream=png_bytes)

    doc.save(path)
    doc.close()


def main() -> None:
    print("TrifectaRAG — Quick math demo (synthetic PDF)\n")
    print(f"  Writing demo PDF -> {DEMO_PDF}")
    _make_demo_pdf(DEMO_PDF)

    client = TrifectaClient(device="cpu")
    ingestor = PDFIngestor(client, mode="page", min_img_px=40)
    stats = ingestor.ingest_pdf(
        str(DEMO_PDF),
        output_dir=str(DATA_DIR / "extracted_demo"),
        page_range=None,  # all pages
    )
    print(
        f"  Ingested: {stats['pages']} pages, {stats['text_chunks']} text nodes, "
        f"{stats['images']} images, {stats['kg_edges']} KG edges\n"
    )

    queries = [
        "What is the chain rule?",
        "derivative of sin(x)",
        "table of derivative formulas",
        "graph of a parabola",
    ]

    for q in queries:
        print(f"--- Query: {q!r} ---")
        raw = client.query(text=q, top_k=5)
        rows = client.get_results(raw)
        if not rows:
            print("  (no results)\n")
            continue
        for r in rows:
            m = r["metadata"]
            page = m.get("page", "?")
            mod = r["modality"]
            sc = r["score"]
            if mod == "IMAGE":
                ip = m.get("image_path", "")
                print(f"  [IMAGE] page {page}  score={sc:.5f}  {ip}")
            else:
                prev = m.get("text_preview", "")[:100].replace("\n", " ")
                print(f"  [TEXT]  page {page}  score={sc:.5f}  {prev}...")
        print()

    print("Open extracted images under:")
    print(f"  {DATA_DIR / 'extracted_demo'}")
    print("\nFor the full OpenStax chapter, run examples/01_ingest_math_book.py")
    print("after placing the PDF manually if the automated download is blocked.")


if __name__ == "__main__":
    main()

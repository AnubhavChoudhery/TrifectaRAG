"""
05_page_index_demo.py — Demonstrate the TrifectaClient page index.

The page index is a Python-level map that answers two questions:
  - "Which global_ids live on page N of source X?"  -> get_page_chunks()
  - "Which (source, page) does global_id G come from?" -> get_chunk_page()

This lets you:
  1. Jump directly to every chunk (text + image) from a specific page
     without doing a semantic query at all.
  2. Enrich query results with page provenance at zero cost.
  3. Build multi-source disambiguation: e.g. ask which pages from
     *each* ingested PDF contain a particular term.

Run this after 01_ingest_textbook.py so the snapshot is ready.

Usage:
    python examples/05_page_index_demo.py

Env vars:
    TRIFECTA_TEXTBOOK_PDF   path to your PDF (default: data/Numerical_Analysis.pdf)
    TRIFECTA_RAG_MODE       page | classical  (default: page)
"""

from __future__ import annotations

import sys
from pathlib import Path

_EX = Path(__file__).resolve().parent
sys.path.insert(0, str(_EX.parent))
sys.path.insert(0, str(_EX))

import fitz

from textbook_config import (
    DATA_DIR,
    ensure_utf8_stdout,
    page_range_for_document,
    print_retrieval_result,
    resolve_pdf_path,
    snapshot_exists,
    snapshot_path,
)
from trifecta import PDFIngestor, TrifectaClient

_DEMO_PAGE_OFFSET = 10   # explore this many pages in from the content start


def build_engine(mode: str) -> TrifectaClient:
    pdf = resolve_pdf_path()
    doc = fitz.open(pdf)
    try:
        n = len(doc)
    finally:
        doc.close()
    pr = page_range_for_document(n, pdf)
    snap = snapshot_path(pdf, mode, pr)

    if snapshot_exists(snap):
        print(f"  Loading {mode!r} snapshot: {snap.name}")
        client = TrifectaClient.from_snapshot(str(snap), device="cpu")
        print(f"  Engine: {client.size} nodes, {client.page_count} pages indexed")
        return client

    print(f"  No snapshot for mode={mode!r}. Ingesting now (this may take a while)...")
    print("  Tip: run 01_ingest_textbook.py first to save a snapshot.\n")
    client = TrifectaClient(device="cpu")
    ingestor = PDFIngestor(client, mode=mode, min_img_px=60)
    ingestor.ingest_pdf(
        str(pdf),
        output_dir=str(DATA_DIR / f"extracted_{mode}"),
        page_range=pr,
    )
    return client


def demo_page_index(client: TrifectaClient) -> None:
    print("\n" + "=" * 72)
    print("PAGE INDEX SUMMARY")
    print("=" * 72)

    sources = client.list_sources()
    print(f"\n  Indexed sources  : {len(sources)}")
    for src in sources:
        pages = client.list_pages(src)
        print(f"    {src!r}: {len(pages)} pages  "
              f"(page {pages[0]}..{pages[-1]})")

    if not sources:
        print("  (no sources indexed)")
        return

    src = sources[0]
    pages = client.list_pages(src)

    # Pick a demo page roughly in the middle of the content
    demo_page = pages[min(_DEMO_PAGE_OFFSET, len(pages) - 1)]

    print(f"\n  Demo: all chunks on page {demo_page} of {src!r}")
    print("  " + "-" * 68)

    gids = client.get_page_chunks(src, demo_page)
    print(f"  global_ids on page {demo_page}: {gids}")

    for gid in gids:
        info = client.get_node(gid)
        # Verify round-trip via get_chunk_page
        provenance = client.get_chunk_page(gid)
        assert provenance is not None, f"gid {gid} not in gid_to_page!"
        assert provenance == (src, demo_page), (
            f"gid {gid}: expected ({src!r}, {demo_page}), got {provenance!r}"
        )
        print(f"\n    gid={gid}  [{info['modality']}]  page={demo_page}")
        meta = info["metadata"]
        if info["modality"] == "IMAGE":
            print(f"      image_path : {meta.get('image_path', '(none)')}")
            print(f"      caption    : {meta.get('caption', '(none)')}")
        else:
            preview = (meta.get("full_text") or meta.get("text_preview") or "").strip()
            preview = preview[:200].replace("\n", " ")
            print(f"      text       : {preview}...")

    print("\n  [PASS] get_page_chunks + get_chunk_page round-trip verified")


def demo_query_enrichment(client: TrifectaClient, query: str = "Newton method") -> None:
    """Run a normal query and show page provenance for every result."""
    print("\n" + "=" * 72)
    print(f"QUERY ENRICHMENT DEMO  ('{query}')")
    print("=" * 72)

    raw = client.query(text=query, top_k=5)
    results = client.get_results(raw)

    for rank, r in enumerate(results, 1):
        provenance = client.get_chunk_page(r["global_id"])
        prov_str = f"({provenance[0]!r}, page {provenance[1]})" if provenance else "(no page)"
        print(f"\n  ### Rank {rank}  [page index: {prov_str}]")
        print_retrieval_result(r, max_text_chars=200)

    print("\n  Every result has page-level provenance from the page index.")


def demo_multi_source_potential(client: TrifectaClient) -> None:
    """Show that each chunk knows which source it belongs to."""
    print("\n" + "=" * 72)
    print("MULTI-SOURCE AWARENESS")
    print("=" * 72)
    sources = client.list_sources()
    print(f"\n  {len(sources)} source(s) in this engine:")
    for src in sources:
        pages = client.list_pages(src)
        total_gids = sum(len(client.get_page_chunks(src, p)) for p in pages)
        print(f"    {src!r}: {len(pages)} pages, {total_gids} chunks total")
        if len(pages) > 3:
            sample = pages[:3]
            print(f"    First 3 pages: {sample}")

    print(
        "\n  To add a second PDF: run 01_ingest_textbook.py with a different "
        "TRIFECTA_TEXTBOOK_PDF, or use PDFIngestor.ingest_pdfs([...]) to batch-ingest."
    )
    print("  After that, query results will show provenance from each PDF separately.")


def main() -> None:
    ensure_utf8_stdout()

    import os
    mode = os.environ.get("TRIFECTA_RAG_MODE", "page").strip()
    if mode not in ("page", "classical"):
        print(f"  Warning: unknown TRIFECTA_RAG_MODE={mode!r}, defaulting to 'page'")
        mode = "page"

    print(f"TrifectaRAG — Page Index Demo  (mode={mode!r})\n")

    client = build_engine(mode)

    demo_page_index(client)
    demo_query_enrichment(client)
    demo_multi_source_potential(client)

    print("\n" + "=" * 72)
    print("Page index demo complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()

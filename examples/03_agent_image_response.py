"""
03_agent_image_response.py — Simulated agent loop: search the textbook and
assemble context with full text chunks plus figure paths (same data an LLM
would receive from MCP tools).

Usage:
    python examples/03_agent_image_response.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_EX = Path(__file__).resolve().parent
sys.path.insert(0, str(_EX.parent))
sys.path.insert(0, str(_EX))

import fitz

from textbook_config import DATA_DIR, page_range_for_document, print_retrieval_result, resolve_pdf_path
from trifecta import TrifectaClient, PDFIngestor


def build_engine() -> TrifectaClient:
    pdf = resolve_pdf_path()
    doc = fitz.open(pdf)
    try:
        n = len(doc)
    finally:
        doc.close()
    pr = page_range_for_document(n)

    client = TrifectaClient(device="cpu")
    ingestor = PDFIngestor(client, mode="page", min_img_px=60)
    stats = ingestor.ingest_pdf(
        str(pdf),
        output_dir=str(DATA_DIR / "extracted_page"),
        page_range=pr,
    )
    print(
        f"  Engine: {stats['text_chunks']} text, {stats['images']} images, "
        f"{stats['kg_edges']} KG edges\n"
    )
    return client


def agent_tool_search(client: TrifectaClient, query: str):
    raw = client.query(text=query, top_k=5)
    return client.get_results(raw)


def main() -> None:
    print("TrifectaRAG — Agent-style context (full chunks + figures)\n")
    print("=" * 60)

    client = build_engine()

    questions = [
        "Newton iteration for solving equations",
        "numerical integration error",
        "Show me a figure or diagram from the book",
    ]

    for q in questions:
        print(f"\nUser: {q}")
        print("-" * 60)

        results = agent_tool_search(client, q)
        if not results:
            print("  (no results)")
            continue
        for rank, r in enumerate(results, start=1):
            print(f"\n  ### Context block {rank}")
            print_retrieval_result(r)

    print("\n" + "=" * 60)
    print(
        "A real agent would send this context to an LLM API; MCP tools mirror "
        "the same fields (see trifecta.mcp_server)."
    )
    print("Done.")


if __name__ == "__main__":
    main()

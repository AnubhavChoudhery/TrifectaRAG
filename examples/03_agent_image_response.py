"""
03_agent_image_response.py — Demonstrates an AI agent answering questions
about a math textbook and responding with relevant images via the
TrifectaRAG MCP tools.

This example uses the MCP tools *directly* (in-process) rather than over
stdio, so it does not require spinning up a subprocess. It shows the exact
flow an LLM agent would follow via tool calls.

Usage:
    python examples/03_agent_image_response.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trifecta import TrifectaClient, PDFIngestor

DATA_DIR = Path(__file__).resolve().parent / "data"
PDF_PATH = DATA_DIR / "openstax_calculus_v1.pdf"

PAGE_START = 249
PAGE_END = 318


def build_engine() -> TrifectaClient:
    if not PDF_PATH.exists():
        print("ERROR: PDF not found. Run 01_ingest_math_book.py first.")
        sys.exit(1)

    client = TrifectaClient(device="cpu")
    ingestor = PDFIngestor(client, mode="page", min_img_px=60)
    stats = ingestor.ingest_pdf(
        str(PDF_PATH),
        output_dir=str(DATA_DIR / "extracted_page"),
        page_range=range(PAGE_START, PAGE_END),
    )
    print(f"  Engine: {stats['text_chunks']} text, {stats['images']} images\n")
    return client


def agent_tool_search(client: TrifectaClient, query: str) -> list:
    """Simulate an agent calling the trifecta_search MCP tool."""
    raw = client.query(text=query, top_k=5)
    return client.get_results(raw)


def agent_answer(query: str, results: list) -> str:
    """
    Simulate the LLM assembling an answer from retrieved chunks.
    In production, this context would be sent to an LLM via API.
    """
    text_context = []
    image_refs = []

    for r in results:
        meta = r["metadata"]
        page = meta.get("page", "?")
        if r["modality"] == "IMAGE":
            img_path = meta.get("image_path", "unknown")
            image_refs.append(f"  [Figure, page {page}]: {img_path}")
        else:
            preview = meta.get("text_preview", "")[:200].replace("\n", " ")
            text_context.append(f"  [Page {page}]: {preview}")

    lines = [f"Agent answer for: \"{query}\"", ""]
    if text_context:
        lines.append("Relevant text:")
        lines.extend(text_context[:3])
    if image_refs:
        lines.append("\nRelevant figures (agent would render these):")
        lines.extend(image_refs[:3])
    if not text_context and not image_refs:
        lines.append("  (no relevant results found)")
    return "\n".join(lines)


def main() -> None:
    print("TrifectaRAG — Agent Image Response Demo")
    print("=" * 60)
    print("\nLoading math book into engine...\n")

    client = build_engine()

    questions = [
        "Show me the graph of a derivative function",
        "What is the power rule?",
        "Can you show me a table of common derivatives?",
    ]

    for q in questions:
        print(f"\nUser: {q}")
        print("-" * 50)

        results = agent_tool_search(client, q)
        answer = agent_answer(q, results)
        print(answer)
        print()

    print("=" * 60)
    print("In a real deployment, the agent would use the MCP server over stdio.")
    print("The LLM would call trifecta_search and trifecta_get_chunk as tools,")
    print("then format the text + images into its response to the user.")
    print("Done.")


if __name__ == "__main__":
    main()

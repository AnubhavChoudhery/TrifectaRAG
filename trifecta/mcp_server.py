"""
trifecta.mcp_server — Model Context Protocol server for TrifectaRAG.

Exposes the TrifectaClient as a set of MCP tools over stdio transport so that
LLM agents (Claude Desktop, Cursor, custom loops) can search, ingest, and
inspect the multi-modal RAG engine.

Usage:
    python -m trifecta.mcp_server                      # empty engine
    python -m trifecta.mcp_server --ingest doc.pdf      # pre-load a PDF
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "TrifectaRAG",
    instructions=(
        "Multi-modal RAG engine. Use trifecta_search to find documents and "
        "images, trifecta_get_chunk to inspect a result, and the ingest tools "
        "to add new content."
    ),
)

_client: Any = None


def _get_client() -> Any:
    global _client
    if _client is None:
        raise RuntimeError(
            "TrifectaClient has not been initialised. "
            "Start the server via `python -m trifecta.mcp_server`."
        )
    return _client


# ── Tools ─────────────────────────────────────────────────────────────────────


@mcp.tool()
def trifecta_search(
    query_text: str = "",
    query_image_path: str = "",
    top_k: int = 10,
) -> str:
    """
    Search the TrifectaRAG engine with text and/or an image query.

    Returns a JSON array of results, each with global_id, score, modality,
    and metadata. Image results include image_path.
    """
    client = _get_client()

    image_arg = query_image_path if query_image_path else None
    text_arg = query_text if query_text else None

    raw = client.query(text=text_arg, image=image_arg, top_k=top_k)
    enriched = client.get_results(raw)
    return json.dumps(enriched, indent=2, default=str)


@mcp.tool()
def trifecta_get_chunk(global_id: int) -> str:
    """
    Retrieve full metadata and modality for a chunk by its global_id.

    If the chunk is an IMAGE and the image_path exists, the response includes
    a base64-encoded thumbnail for inline display.
    """
    client = _get_client()
    info = client.get_node(global_id)

    img_path = info["metadata"].get("image_path")
    if info["modality"] == "IMAGE" and img_path and Path(img_path).is_file():
        data = Path(img_path).read_bytes()
        info["image_base64"] = base64.b64encode(data).decode("ascii")

    return json.dumps(info, indent=2, default=str)


@mcp.tool()
def trifecta_ingest_text(
    text: str,
    metadata_json: str = "{}",
) -> str:
    """
    Ingest a text document into the engine.

    Args:
        text:          Raw text content.
        metadata_json: Optional JSON string with additional metadata.

    Returns:
        JSON with the assigned global_id.
    """
    client = _get_client()
    meta = json.loads(metadata_json) if metadata_json else {}
    gid = client.add_document(text=text, metadata=meta)
    return json.dumps({"global_id": gid})


@mcp.tool()
def trifecta_ingest_image(
    image_path: str,
    caption: str = "",
    metadata_json: str = "{}",
) -> str:
    """
    Ingest an image into the engine.

    Args:
        image_path:    Path to the image file.
        caption:       Optional text caption (indexed by BM25).
        metadata_json: Optional JSON string with additional metadata.

    Returns:
        JSON with the assigned global_id.
    """
    client = _get_client()
    meta = json.loads(metadata_json) if metadata_json else {}
    gid = client.add_image(image=image_path, caption=caption, metadata=meta)
    return json.dumps({"global_id": gid})


@mcp.tool()
def trifecta_add_edge(
    source_id: int,
    target_id: int,
    edge_type: str = "RELATES_TO",
) -> str:
    """
    Add a directed edge in the knowledge graph.

    Args:
        source_id: Source global_id.
        target_id: Target global_id.
        edge_type: One of RELATES_TO, EXPLAINS, DEPICTS.

    Returns:
        Confirmation JSON.
    """
    from . import trifecta_py as tr

    type_map = {
        "RELATES_TO": tr.EdgeType.RELATES_TO,
        "EXPLAINS": tr.EdgeType.EXPLAINS,
        "DEPICTS": tr.EdgeType.DEPICTS,
    }
    et = type_map.get(edge_type.upper())
    if et is None:
        return json.dumps({"error": f"Unknown edge_type: {edge_type}"})

    client = _get_client()
    client.add_edge(source_id, target_id, et)
    return json.dumps({"status": "ok", "source_id": source_id, "target_id": target_id})


@mcp.tool()
def trifecta_stats() -> str:
    """Return basic engine statistics: size, dim, device."""
    client = _get_client()
    return json.dumps({
        "size": client.size,
        "dim": client.dim,
        "device": client.device,
    })


# ── Entry point ───────────────────────────────────────────────────────────────


def run_server(client: Any) -> None:
    """Serve MCP tools against an already-constructed TrifectaClient."""
    global _client
    _client = client
    logger.info("TrifectaClient ready for MCP: %s", client)
    mcp.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="TrifectaRAG MCP Server")
    parser.add_argument(
        "--ingest", type=str, default=None,
        help="Path to a PDF to ingest on startup.",
    )
    parser.add_argument(
        "--mode", type=str, default="page", choices=["page", "classical"],
        help="PDF ingestion mode (default: page).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="examples/data",
        help="Directory for extracted images.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device (default: cpu).",
    )
    args = parser.parse_args()

    from .client import TrifectaClient
    client = TrifectaClient(device=args.device)
    logger.info("TrifectaClient initialised: %s", client)

    if args.ingest:
        from .pdf_ingest import PDFIngestor
        ingestor = PDFIngestor(client, mode=args.mode)
        stats = ingestor.ingest_pdf(args.ingest, output_dir=args.output_dir)
        logger.info("Pre-ingested PDF: %s", stats)

    run_server(client)


if __name__ == "__main__":
    main()

"""
trifecta.pdf_ingest — Extract text and images from a PDF and ingest into the
TrifectaClient, with optional KnowledgeGraph wiring.

Supports two indexing modes:
  - "page"      : one document per page; images linked via DEPICTS edges.
  - "classical" : overlapping ~chunk_size token chunks; images still per-page.
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image as PILImage

from . import trifecta_py as tr

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")


def _rough_token_count(text: str) -> int:
    return len(_WHITESPACE.split(text.strip()))


def _chunk_text(text: str, chunk_size: int = 256, overlap: int = 64) -> List[str]:
    """Split *text* into overlapping word-level chunks."""
    words = _WHITESPACE.split(text.strip())
    if not words:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks


class PDFIngestor:
    """
    Extract text blocks and images from a PDF and ingest them into
    a TrifectaClient instance.

    Args:
        client:      An initialised TrifectaClient.
        mode:        "page" (one doc per page) or "classical" (chunked text).
        chunk_size:  Target words per chunk (classical mode only).
        overlap:     Overlap words between consecutive chunks.
        min_img_px:  Skip images smaller than this in either dimension.
    """

    def __init__(
        self,
        client: Any,
        mode: str = "page",
        chunk_size: int = 256,
        overlap: int = 64,
        min_img_px: int = 50,
    ) -> None:
        if mode not in ("page", "classical"):
            raise ValueError(f"mode must be 'page' or 'classical', got {mode!r}")
        self._client = client
        self._mode = mode
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._min_img_px = min_img_px

    def ingest_pdf(
        self,
        pdf_path: str,
        output_dir: str = "examples/data",
        page_range: Optional[range] = None,
    ) -> Dict[str, int]:
        """
        Extract and ingest all pages (or a sub-range) of a PDF.

        Args:
            pdf_path:   Path to the PDF file.
            output_dir: Directory where extracted images are saved.
            page_range: Optional range of 0-based page indices to process.

        Returns:
            Stats dict: {pages, text_chunks, images, kg_edges}.
        """
        import fitz  # PyMuPDF — imported here to keep the dependency optional

        pdf_path = str(Path(pdf_path).resolve())
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(pdf_path)
        src_name = Path(pdf_path).stem
        pages_to_process = page_range if page_range is not None else range(len(doc))

        stats: Dict[str, int] = {
            "pages": 0,
            "text_chunks": 0,
            "images": 0,
            "kg_edges": 0,
        }

        prev_page_gid: Optional[int] = None
        # Deduplicate images by PDF xref so the same asset (e.g. a logo or a
        # figure reused across pages) is only embedded and stored once.
        seen_xrefs: set = set()

        for page_idx in pages_to_process:
            if page_idx >= len(doc):
                break
            page = doc[page_idx]
            page_num = page_idx + 1
            stats["pages"] += 1

            # ── Text ──────────────────────────────────────────────────────
            raw_text = page.get_text("text").strip()
            text_gids: List[int] = []

            if raw_text:
                if self._mode == "page":
                    gid = self._client.add_document(
                        text=raw_text,
                        metadata={
                            "source": src_name,
                            "page": page_num,
                            "type": "text",
                        },
                    )
                    text_gids.append(gid)
                    stats["text_chunks"] += 1
                else:
                    for ci, chunk in enumerate(_chunk_text(
                        raw_text, self._chunk_size, self._overlap
                    )):
                        gid = self._client.add_document(
                            text=chunk,
                            metadata={
                                "source": src_name,
                                "page": page_num,
                                "chunk_idx": ci,
                                "type": "text",
                            },
                        )
                        text_gids.append(gid)
                        stats["text_chunks"] += 1

            # ── Images ────────────────────────────────────────────────────
            img_list = page.get_images(full=True)
            for img_idx, img_info in enumerate(img_list):
                xref = img_info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                try:
                    base_image = doc.extract_image(xref)
                except Exception:
                    continue
                if base_image is None:
                    continue

                width = base_image.get("width", 0)
                height = base_image.get("height", 0)
                if width < self._min_img_px or height < self._min_img_px:
                    continue

                ext = base_image.get("ext", "png")
                img_filename = f"{src_name}_p{page_num}_img{img_idx}.{ext}"
                img_path = out / img_filename

                img_path.write_bytes(base_image["image"])

                try:
                    pil_img = PILImage.open(io.BytesIO(base_image["image"])).convert("RGB")
                except Exception:
                    continue

                cap = f"Figure from {src_name}, page {page_num}"
                img_gid = self._client.add_image(
                    image=pil_img,
                    caption=cap,
                    metadata={
                        "source": src_name,
                        "page": page_num,
                        "type": "figure",
                        "image_path": str(img_path.resolve()),
                        "caption": cap,
                    },
                )
                stats["images"] += 1

                for tgid in text_gids:
                    self._client.add_edge(img_gid, tgid, tr.EdgeType.DEPICTS)
                    stats["kg_edges"] += 1

            # ── Inter-page KG edges (page mode) ─────────────────────────
            if self._mode == "page" and text_gids and prev_page_gid is not None:
                self._client.add_edge(prev_page_gid, text_gids[0], tr.EdgeType.RELATES_TO)
                self._client.add_edge(text_gids[0], prev_page_gid, tr.EdgeType.RELATES_TO)
                stats["kg_edges"] += 2

            if text_gids:
                prev_page_gid = text_gids[0]

            if stats["pages"] % 10 == 0:
                logger.info("Processed %d pages...", stats["pages"])

        doc.close()
        logger.info(
            "PDF ingestion complete: %d pages, %d text chunks, %d images, %d KG edges",
            stats["pages"], stats["text_chunks"], stats["images"], stats["kg_edges"],
        )
        return stats

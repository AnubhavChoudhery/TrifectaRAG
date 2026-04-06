"""
trifecta.pdf_ingest — Extract text and images from a PDF and ingest into the
TrifectaClient, with optional KnowledgeGraph wiring.

Supports two indexing modes:
  - "page"      : one document per page; images linked via DEPICTS edges.
  - "classical" : overlapping ~chunk_size token chunks; images still per-page.

Vector figure extraction:
  Most math textbooks render graphs, plots, and tables as PDF vector drawings
  (not raster images).  ``page.get_images()`` returns nothing for these; they
  are invisible to a naive extractor.  We detect pages with a significant
  cluster of vector drawing paths, render just that region to a raster PNG,
  and ingest it via CLIP — making diagrams searchable.
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image as PILImage

from . import trifecta_py as tr

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")

# ── Vector-figure extraction tuning ──────────────────────────────────────────
# A page needs this many PDF drawing paths before we consider it a figure page.
_VECTOR_DRAWING_THRESHOLD = 20
# The union bounding-box of all drawings must cover at least this fraction of
# the page area to be treated as a real figure (filters out scattered symbols).
_MIN_FIGURE_AREA_FRAC = 0.04
# But if the bbox covers MORE than this fraction it is probably just math
# symbols scattered over a text page — skip rendering.
_MAX_FIGURE_AREA_FRAC = 0.85
# Render resolution multiplier (2.0 → ~144 DPI for a standard 72 DPI PDF).
_RENDER_SCALE = 2.0
# Minimum rendered pixel dimension (w or h) to accept the figure image.
_MIN_RENDER_PX = 80


def _rough_token_count(text: str) -> int:
    return len(_WHITESPACE.split(text.strip()))


# ── Caption extraction ────────────────────────────────────────────────────────

_FIG_CAPTION_RE = re.compile(
    r"(?:Fig(?:ure)?|Table)[\s.\u00a0]*[\d.]+[^\n]{0,200}", re.IGNORECASE
)


def _find_caption(page_text: str, page_num: int, src_name: str) -> str:
    """Return the first 'Fig.' / 'Table' caption found in *page_text*, or a default."""
    m = _FIG_CAPTION_RE.search(page_text)
    if m:
        return m.group(0).strip()[:200]
    return f"Figure/diagram from {src_name}, page {page_num}"


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

    # ── Vector figure helper ─────────────────────────────────────────────────

    def _render_vector_figure(
        self,
        page: Any,
        page_num: int,
        src_name: str,
        out: Path,
        fig_idx: int,
    ) -> Optional[Tuple[PILImage.Image, str, str]]:
        """
        If *page* has a significant cluster of vector drawings (graphs, tables,
        plots drawn as PDF paths rather than raster images), render that region
        to a PNG and return (pil_image, abs_path, caption).

        Returns ``None`` when the page has no qualifying figure cluster.
        """
        import fitz  # noqa: PLC0415 — local import keeps fitz optional

        drawings = page.get_drawings()
        if len(drawings) < _VECTOR_DRAWING_THRESHOLD:
            return None

        # Compute the union bounding-box of all drawing paths.
        rects = [fitz.Rect(d["rect"]) for d in drawings if d.get("rect")]
        if not rects:
            return None

        union_rect: Any = rects[0]
        for r in rects[1:]:
            union_rect = union_rect | r

        page_rect = page.rect
        page_area = page_rect.width * page_rect.height
        if page_area < 1:
            return None

        fig_area = union_rect.width * union_rect.height
        frac = fig_area / page_area

        # Skip: too small (scattered inline symbols) or too large (full-page text).
        if not (_MIN_FIGURE_AREA_FRAC <= frac <= _MAX_FIGURE_AREA_FRAC):
            logger.debug(
                "p%d: skipping vector figure (area frac=%.2f, drawings=%d)",
                page_num, frac, len(drawings),
            )
            return None

        # Expand by a small margin so axis labels are not clipped.
        margin = 12
        clip = fitz.Rect(
            max(page_rect.x0, union_rect.x0 - margin),
            max(page_rect.y0, union_rect.y0 - margin),
            min(page_rect.x1, union_rect.x1 + margin),
            min(page_rect.y1, union_rect.y1 + margin),
        )

        mat = fitz.Matrix(_RENDER_SCALE, _RENDER_SCALE)
        pixmap = page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csRGB)

        if pixmap.width < _MIN_RENDER_PX or pixmap.height < _MIN_RENDER_PX:
            return None

        img_filename = f"{src_name}_p{page_num}_fig{fig_idx:02d}.png"
        img_path = out / img_filename
        pixmap.save(str(img_path))

        raw_text = page.get_text("text")
        caption = _find_caption(raw_text, page_num, src_name)

        pil_img = PILImage.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
        logger.debug(
            "p%d: rendered vector figure %s (%.0f%%)", page_num, img_filename, frac * 100
        )
        return pil_img, str(img_path.resolve()), caption

    # ── Main ingestion entry-point ───────────────────────────────────────────

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

            # ── Raster images (JPEG/PNG embedded in PDF) ──────────────────
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

                cap = _find_caption(page.get_text("text"), page_num, src_name)
                img_gid = self._client.add_image(
                    image=pil_img,
                    caption=cap,
                    metadata={
                        "source": src_name,
                        "page": page_num,
                        "type": "figure_raster",
                        "image_path": str(img_path.resolve()),
                        "caption": cap,
                    },
                )
                stats["images"] += 1

                for tgid in text_gids:
                    self._client.add_edge(img_gid, tgid, tr.EdgeType.DEPICTS)
                    stats["kg_edges"] += 1

            # ── Vector figures (graphs/tables drawn as PDF paths) ─────────
            # Most math textbooks render figures as vector paths, not raster
            # images.  We detect and rasterise these so they are searchable.
            vec_result = self._render_vector_figure(
                page, page_num, src_name, out, stats["images"]
            )
            if vec_result is not None:
                pil_img, img_path_str, cap = vec_result
                img_gid = self._client.add_image(
                    image=pil_img,
                    caption=cap,
                    metadata={
                        "source": src_name,
                        "page": page_num,
                        "type": "figure_vector",
                        "image_path": img_path_str,
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

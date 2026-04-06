"""
trifecta_client.py — Phase 6: Python front-end for the Trifecta multi-modal
RAG pipeline.

Wraps the C++ TrifectaEngine (exposed via pybind11) and provides high-level
methods for ingesting text/images and performing multi-modal retrieval with
late-fusion query embedding.

Models:
  Text  -> sentence-transformers  (clip-ViT-B-32, 512-dim CLIP text encoder)
  Image -> transformers CLIPModel (openai/clip-vit-base-patch32, 512-dim)

Both models embed into the same CLIP latent space, enabling meaningful
cosine similarity across modalities and mathematically sound late fusion.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

import trifecta_py as tr

logger = logging.getLogger(__name__)

ImageInput = Union[str, Path, "Image.Image"]


def _normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector. Returns the zero vector unchanged."""
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return vec
    return vec / norm


class TrifectaClient:
    """
    High-level Python interface to the Trifecta multi-modal RAG engine.

    Integrates:
      - sentence-transformers for text embeddings (CLIP text encoder)
      - HuggingFace transformers CLIPModel for image embeddings
      - C++ TrifectaEngine for HNSW + BM25 + KnowledgeGraph retrieval
    """

    DEFAULT_TEXT_MODEL = "clip-ViT-B-32"
    DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"

    def __init__(
        self,
        text_model: str = DEFAULT_TEXT_MODEL,
        clip_model: str = DEFAULT_CLIP_MODEL,
        device: Optional[str] = None,
        hnsw_M: int = 16,
        ef_construction: int = 200,
        max_elements: int = 1_000_000,
    ) -> None:
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("TrifectaClient: device=%s", self._device)

        self._text_model = SentenceTransformer(text_model, device=self._device)
        self._dim: int = self._text_model.get_sentence_embedding_dimension()
        logger.info("Text model '%s' loaded, dim=%d", text_model, self._dim)

        self._clip_model = CLIPModel.from_pretrained(clip_model).to(self._device)
        self._clip_model.eval()
        self._clip_processor = CLIPProcessor.from_pretrained(clip_model)
        logger.info("CLIP model '%s' loaded", clip_model)

        self._engine = tr.TrifectaEngine(
            dim=self._dim,
            hnsw_M=hnsw_M,
            ef_construction=ef_construction,
            max_elements=max_elements,
        )

    # ── Ingestion ────────────────────────────────────────────────────────────

    def add_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Ingest a text document.

        Generates a text embedding via sentence-transformers, registers the
        chunk in all C++ indexes (GlobalRegistry, HNSW, BM25).

        Args:
            text:     Raw text content (must be non-empty).
            metadata: Optional metadata dictionary (serialized to JSON).

        Returns:
            The assigned global_id.
        """
        if not text or not text.strip():
            raise ValueError("add_document: text must be non-empty")

        embedding = self._embed_text(text)
        gid = self._engine.ingest(
            text=text,
            embedding=embedding.tolist(),
            metadata=json.dumps(metadata or {}),
            modality=tr.Modality.TEXT,
        )
        logger.debug("Ingested document gid=%d len(text)=%d", gid, len(text))
        return gid

    # ── Private: text embedding ──────────────────────────────────────────────

    def _embed_text(self, text: str) -> np.ndarray:
        """Encode text via sentence-transformers -> float32 numpy vector."""
        vec = self._text_model.encode(
            text, convert_to_numpy=True, show_progress_bar=False
        )
        return np.asarray(vec, dtype=np.float32).flatten()

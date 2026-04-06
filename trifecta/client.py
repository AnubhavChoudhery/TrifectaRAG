"""
trifecta.client — Python front-end for the Trifecta multi-modal RAG pipeline.

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

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment,misc]

try:
    from transformers import CLIPModel, CLIPProcessor
except Exception:  # pragma: no cover
    CLIPModel = None      # type: ignore[assignment,misc]
    CLIPProcessor = None  # type: ignore[assignment,misc]

from . import trifecta_py as tr

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

        meta = dict(metadata or {})
        meta.setdefault("text_preview", text[:500])

        embedding = self._embed_text(text)
        gid = self._engine.ingest(
            text=text,
            embedding=embedding.tolist(),
            metadata=json.dumps(meta),
            modality=tr.Modality.TEXT,
        )
        logger.debug("Ingested document gid=%d len(text)=%d", gid, len(text))
        return gid

    def add_image(
        self,
        image: ImageInput,
        caption: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Ingest an image.

        Generates a CLIP image embedding via transformers, optionally indexes
        the caption text via BM25, and registers in all C++ indexes.

        Args:
            image:    File path (str/Path) or PIL Image.
            caption:  Optional text caption (indexed by BM25).
            metadata: Optional metadata dictionary.

        Returns:
            The assigned global_id.
        """
        meta = dict(metadata or {})
        if isinstance(image, (str, Path)):
            meta.setdefault("image_path", str(Path(image).resolve()))

        pil_img = self._load_image(image)
        embedding = self._embed_image(pil_img)
        gid = self._engine.ingest(
            text=caption,
            embedding=embedding.tolist(),
            metadata=json.dumps(meta),
            modality=tr.Modality.IMAGE,
        )
        logger.debug("Ingested image gid=%d", gid)
        return gid

    # ── Knowledge Graph ──────────────────────────────────────────────────────

    def add_edge(
        self, source_id: int, target_id: int, edge_type: tr.EdgeType
    ) -> None:
        """Add a directed edge in the knowledge graph between ingested chunks."""
        self._engine.add_edge(source_id, target_id, edge_type)

    # ── Retrieval ────────────────────────────────────────────────────────────

    def query(
        self,
        text: Optional[str] = None,
        image: Optional[ImageInput] = None,
        top_k: int = 10,
        search_ef: int = 50,
    ) -> List[Tuple[int, float]]:
        """
        Multi-modal query with optional late fusion.

        Late fusion (both *text* and *image* provided):
          1. Embed text  -> v_t;  embed image -> v_i
          2. Normalize:   v_t' = v_t / ||v_t||,  v_i' = v_i / ||v_i||
          3. Fuse:        v_f  = normalize(v_t' + v_i')
          4. Send v_f + raw text to the C++ query engine.

        Args:
            text:      Query text (drives BM25 + optional HNSW).
            image:     Query image path or PIL Image (drives HNSW).
            top_k:     Maximum results to return.
            search_ef: HNSW search ef parameter.

        Returns:
            List of (global_id, rrf_score) tuples, descending by score.
        """
        if text is None and image is None:
            return []

        query_text = text or ""
        query_vec: List[float] = []

        if text is not None and image is not None:
            text_vec = _normalize(self._embed_text(text))
            img_vec = _normalize(self._embed_image(self._load_image(image)))
            fused = _normalize(text_vec + img_vec)
            query_vec = fused.tolist()
        elif image is not None:
            query_vec = self._embed_image(self._load_image(image)).tolist()
        elif text is not None:
            query_vec = self._embed_text(text).tolist()

        return self._engine.query(
            query_vec=query_vec,
            query_text=query_text,
            top_k=top_k,
            search_ef=search_ef,
        )

    def get_node(self, global_id: int) -> Dict[str, Any]:
        """
        Retrieve stored metadata and modality for a chunk by its global_id.

        Returns:
            dict with keys: global_id, modality ("TEXT" or "IMAGE"), metadata (parsed dict).
        """
        node = self._engine.get_node(global_id)
        try:
            meta = json.loads(node.metadata)
        except (json.JSONDecodeError, TypeError):
            meta = {"raw": node.metadata}
        return {
            "global_id": node.global_id,
            "modality": "IMAGE" if node.modality == tr.Modality.IMAGE else "TEXT",
            "metadata": meta,
        }

    def get_results(
        self, results: List[Tuple[int, float]]
    ) -> List[Dict[str, Any]]:
        """
        Enrich raw query results with metadata, modality, and score.

        Args:
            results: Output of query() — list of (global_id, score) tuples.

        Returns:
            List of dicts: {global_id, score, modality, metadata}.
        """
        enriched = []
        for gid, score in results:
            info = self.get_node(gid)
            info["score"] = score
            enriched.append(info)
        return enriched

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def engine(self) -> tr.TrifectaEngine:
        """Direct access to the underlying C++ engine."""
        return self._engine

    @property
    def size(self) -> int:
        """Number of ingested chunks."""
        return self._engine.size

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        return self._dim

    @property
    def device(self) -> str:
        """Torch device used for model inference."""
        return self._device

    def __len__(self) -> int:
        return self._engine.size

    def __repr__(self) -> str:
        return f"<TrifectaClient size={self.size} dim={self._dim} device={self._device!r}>"

    # ── Private ──────────────────────────────────────────────────────────────

    def _embed_text(self, text: str) -> np.ndarray:
        """Encode text via sentence-transformers -> float32 numpy vector."""
        vec = self._text_model.encode(
            text, convert_to_numpy=True, show_progress_bar=False
        )
        return np.asarray(vec, dtype=np.float32).flatten()

    def _embed_image(self, image: Image.Image) -> np.ndarray:
        """Encode a PIL image via CLIP -> float32 numpy vector."""
        inputs = self._clip_processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device)
        with torch.no_grad():
            features = self._clip_model.get_image_features(pixel_values=pixel_values)
        return features.cpu().numpy().astype(np.float32).flatten()

    @staticmethod
    def _load_image(image: ImageInput) -> Image.Image:
        """Resolve an image input to a PIL Image in RGB mode."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        return Image.open(path).convert("RGB")

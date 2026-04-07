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

import gzip
import hashlib
import json
import logging
import pickle
from collections import OrderedDict
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

        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers failed to import (often Keras 3 vs transformers). "
                "Try: pip install tf-keras"
            )
        if CLIPModel is None or CLIPProcessor is None:
            raise ImportError(
                "transformers CLIP failed to import. Try: pip install tf-keras"
            )

        self._text_model = SentenceTransformer(text_model, device=self._device)
        _dim = self._text_model.get_sentence_embedding_dimension()
        if _dim is None:
            probe = self._text_model.encode(
                "dim", convert_to_numpy=True, show_progress_bar=False
            )
            _dim = int(np.asarray(probe).reshape(-1).shape[0])
        self._dim: int = _dim
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

        # Snapshot tracking — populated by add_document / add_image / add_edge
        # so the full engine state can be serialised and restored without re-
        # running any PDF extraction or ML inference.
        self._records: List[Dict[str, Any]] = []
        self._edge_log: List[Tuple[int, int, str]] = []

        # LRU embedding cache — avoids redundant ML inference for repeated
        # queries.  Text keys are the string itself; image keys are a sha256
        # digest of the raw pixel bytes.
        self._embed_cache_max = 128
        self._text_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._image_cache: OrderedDict[str, np.ndarray] = OrderedDict()

        # Page index: (source, page_num) -> list of global_ids on that page.
        # Enables page-level retrieval and answers the "page-index equivalency"
        # question — every chunk is traceable back to its source page.
        self._page_index: Dict[Tuple[str, int], List[int]] = {}
        # Reverse map: global_id -> (source, page)
        self._gid_to_page: Dict[int, Tuple[str, int]] = {}

    # ── Page index helpers ─────────────────────────────────────────────────

    def _register_page(self, gid: int, metadata: Optional[Dict[str, Any]]) -> None:
        """If metadata contains source+page, register in the page index."""
        if metadata is None:
            return
        source = metadata.get("source")
        page = metadata.get("page")
        if source is not None and page is not None:
            key = (str(source), int(page))
            self._page_index.setdefault(key, []).append(gid)
            self._gid_to_page[gid] = key

    def get_page_chunks(self, source: str, page: int) -> List[int]:
        """Return all global_ids ingested from the given source and page number."""
        return list(self._page_index.get((source, page), []))

    def get_chunk_page(self, global_id: int) -> Optional[Tuple[str, int]]:
        """Return (source, page) for a global_id, or None if not page-indexed."""
        return self._gid_to_page.get(global_id)

    def list_sources(self) -> List[str]:
        """Return sorted list of unique source names in the page index."""
        return sorted({k[0] for k in self._page_index})

    def list_pages(self, source: str) -> List[int]:
        """Return sorted list of page numbers indexed for a given source."""
        return sorted({k[1] for k in self._page_index if k[0] == source})

    @property
    def page_count(self) -> int:
        """Total number of distinct (source, page) pairs indexed."""
        return len(self._page_index)

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
        meta.setdefault("full_text", text)
        meta_json = json.dumps(meta)
        emb_list = self._embed_text(text).tolist()

        gid = self._engine.ingest(
            text=text,
            embedding=emb_list,
            metadata=meta_json,
            modality=tr.Modality.TEXT,
        )
        self._records.append(
            {"gid": gid, "text": text, "embedding": emb_list,
             "metadata": meta_json, "modality": "TEXT"}
        )
        self._register_page(gid, meta)
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
        meta_json = json.dumps(meta)

        pil_img = self._load_image(image)
        emb_list = self._embed_image(pil_img).tolist()

        gid = self._engine.ingest(
            text=caption,
            embedding=emb_list,
            metadata=meta_json,
            modality=tr.Modality.IMAGE,
        )
        self._records.append(
            {"gid": gid, "text": caption, "embedding": emb_list,
             "metadata": meta_json, "modality": "IMAGE"}
        )
        self._register_page(gid, meta)
        logger.debug("Ingested image gid=%d", gid)
        return gid

    # ── Knowledge Graph ──────────────────────────────────────────────────────

    def add_edge(
        self, source_id: int, target_id: int, edge_type: tr.EdgeType
    ) -> None:
        """Add a directed edge in the knowledge graph between ingested chunks."""
        self._engine.add_edge(source_id, target_id, edge_type)
        if edge_type == tr.EdgeType.DEPICTS:
            etype_str = "DEPICTS"
        elif edge_type == tr.EdgeType.EXPLAINS:
            etype_str = "EXPLAINS"
        else:
            etype_str = "RELATES_TO"
        self._edge_log.append((source_id, target_id, etype_str))

    # ── Snapshot persistence ─────────────────────────────────────────────────

    def save_snapshot(self, path: str) -> None:
        """
        Serialise the full engine state (records + KG edges) to a gzip-
        compressed pickle file.  A subsequent call to ``from_snapshot`` can
        restore the engine in seconds without re-reading the PDF or re-running
        any ML inference.

        Args:
            path: Destination path.  A ``.snap.gz`` extension is appended if
                  the path does not already end with ``.gz``.
        """
        p = Path(path)
        if not str(p).endswith(".gz"):
            p = Path(str(p) + ".snap.gz") if not str(p).endswith(".snap.gz") else p
        data = {
            "version": 2,
            "dim": self._dim,
            "records": self._records,
            "edges": self._edge_log,
            "page_index": {f"{s}\x00{p}": gids for (s, p), gids in self._page_index.items()},
            "gid_to_page": {gid: [s, p] for gid, (s, p) in self._gid_to_page.items()},
        }
        with gzip.open(str(p), "wb", compresslevel=5) as f:
            pickle.dump(data, f, protocol=4)
        size_kb = p.stat().st_size / 1024
        logger.info(
            "Snapshot saved: %d records, %d edges -> %s (%.1f KB)",
            len(self._records), len(self._edge_log), p, size_kb,
        )

    @classmethod
    def from_snapshot(
        cls,
        snapshot_path: str,
        text_model: str = DEFAULT_TEXT_MODEL,
        clip_model: str = DEFAULT_CLIP_MODEL,
        device: Optional[str] = None,
        hnsw_M: int = 16,
        ef_construction: int = 200,
        max_elements: int = 1_000_000,
    ) -> "TrifectaClient":
        """
        Restore a ``TrifectaClient`` from a snapshot file.

        The ML models are loaded normally (needed for future queries), but
        the stored text/image embeddings are injected directly into the C++
        engine — no PDF re-reading, no CLIP/ST inference on the corpus.

        Args:
            snapshot_path: Path to a ``.snap.gz`` file created by
                           :meth:`save_snapshot`.
            text_model:    sentence-transformers model (for query embedding).
            clip_model:    HuggingFace CLIP model (for query image embedding).
            device:        Torch device (default: auto).
            hnsw_M / ef_construction / max_elements: HNSW parameters.

        Returns:
            A fully-populated ``TrifectaClient``.
        """
        sp = Path(snapshot_path)
        if not sp.exists():
            raise FileNotFoundError(f"Snapshot not found: {sp}")

        with gzip.open(str(sp), "rb") as f:
            data = pickle.load(f)

        records: List[Dict[str, Any]] = data["records"]
        edges: List[Tuple[int, int, str]] = data["edges"]
        snap_dim: int = data["dim"]

        logger.info(
            "Loading snapshot %s: %d records, %d edges",
            sp.name, len(records), len(edges),
        )

        # Create the client normally (loads ML models, creates empty engine).
        client = cls(
            text_model=text_model,
            clip_model=clip_model,
            device=device,
            hnsw_M=hnsw_M,
            ef_construction=ef_construction,
            max_elements=max_elements,
        )

        if client._dim != snap_dim:
            raise ValueError(
                f"Model embedding dim ({client._dim}) does not match snapshot "
                f"dim ({snap_dim}).  Use the same model that created the snapshot."
            )

        _mod_map = {"TEXT": tr.Modality.TEXT, "IMAGE": tr.Modality.IMAGE}
        _edge_map = {
            "RELATES_TO": tr.EdgeType.RELATES_TO,
            "EXPLAINS": tr.EdgeType.EXPLAINS,
            "DEPICTS": tr.EdgeType.DEPICTS,
        }

        for rec in records:
            client._engine.ingest(
                text=rec["text"],
                embedding=rec["embedding"],
                metadata=rec["metadata"],
                modality=_mod_map[rec["modality"]],
            )
            client._records.append(rec)

        for src, tgt, etype_str in edges:
            client._engine.add_edge(src, tgt, _edge_map[etype_str])
            client._edge_log.append((src, tgt, etype_str))

        # Restore page index (v2+); older snapshots rebuild from metadata.
        raw_pi = data.get("page_index")
        raw_g2p = data.get("gid_to_page")
        if raw_pi and raw_g2p:
            for key_str, gids in raw_pi.items():
                s, p_str = key_str.split("\x00", 1)
                client._page_index[(s, int(p_str))] = gids
            for gid_str, sp in raw_g2p.items():
                client._gid_to_page[int(gid_str)] = (sp[0], int(sp[1]))
        else:
            for rec in records:
                try:
                    meta = json.loads(rec["metadata"])
                except Exception:
                    continue
                client._register_page(rec["gid"], meta)

        logger.info("Snapshot loaded: engine has %d nodes, %d page entries",
                     client.size, client.page_count)
        return client

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
        """Encode text via sentence-transformers -> float32 numpy vector (LRU cached)."""
        cached = self._text_cache.get(text)
        if cached is not None:
            self._text_cache.move_to_end(text)
            return cached
        vec = self._text_model.encode(
            text, convert_to_numpy=True, show_progress_bar=False
        )
        result = np.asarray(vec, dtype=np.float32).flatten()
        self._text_cache[text] = result
        if len(self._text_cache) > self._embed_cache_max:
            self._text_cache.popitem(last=False)
        return result

    def _embed_image(self, image: Image.Image) -> np.ndarray:
        """Encode a PIL image via CLIP -> float32 numpy vector (LRU cached)."""
        img_hash = hashlib.sha256(image.tobytes()).hexdigest()
        cached = self._image_cache.get(img_hash)
        if cached is not None:
            self._image_cache.move_to_end(img_hash)
            return cached
        inputs = self._clip_processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device)
        with torch.no_grad():
            features = self._clip_model.get_image_features(pixel_values=pixel_values)
        result = features.cpu().numpy().astype(np.float32).flatten()
        self._image_cache[img_hash] = result
        if len(self._image_cache) > self._embed_cache_max:
            self._image_cache.popitem(last=False)
        return result

    @staticmethod
    def _load_image(image: ImageInput) -> Image.Image:
        """Resolve an image input to a PIL Image in RGB mode."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        return Image.open(path).convert("RGB")

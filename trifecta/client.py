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
import re
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
_FALLBACK_DIM = 512
_TOKEN_RE = re.compile(r"[A-Za-z0-9_+\-./]+")


def _normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector. Returns the zero vector unchanged."""
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return vec
    return vec / norm


def _hash_text_embedding(text: str, dim: int = _FALLBACK_DIM) -> np.ndarray:
    """Deterministic offline text embedding used when CLIP is unavailable."""
    vec = np.zeros(dim, dtype=np.float32)
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return vec

    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[bucket] += sign

    return _normalize(vec.astype(np.float32))


def _hash_image_embedding(image: Image.Image, dim: int = _FALLBACK_DIM) -> np.ndarray:
    """Small deterministic image signature for offline figure indexing."""
    thumb = image.convert("RGB").resize((32, 32))
    arr = np.asarray(thumb, dtype=np.float32) / 255.0
    vec = np.zeros(dim, dtype=np.float32)

    means = arr.mean(axis=(0, 1))
    stds = arr.std(axis=(0, 1))
    vec[:3] = means
    vec[3:6] = stds

    digest = hashlib.blake2b(arr.tobytes(), digest_size=32).digest()
    for i, b in enumerate(digest):
        vec[6 + i] = (float(b) / 255.0) - 0.5

    return _normalize(vec)


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
        self._clip_model = None
        self._clip_processor = None
        self._using_fallback_embeddings = True
        self._dim = _FALLBACK_DIM

        if CLIPModel is not None and CLIPProcessor is not None:
            try:
                self._clip_model = CLIPModel.from_pretrained(
                    clip_model,
                    use_safetensors=True,
                    local_files_only=True,
                ).to(self._device)
                self._clip_model.eval()
                self._clip_processor = CLIPProcessor.from_pretrained(
                    clip_model,
                    local_files_only=True,
                )
                self._dim = int(getattr(self._clip_model.config, "projection_dim", _FALLBACK_DIM))
                self._using_fallback_embeddings = False
                logger.info("CLIP model '%s' loaded locally, dim=%d", clip_model, self._dim)
            except Exception as exc:
                logger.warning(
                    "CLIP model not available locally (%s). Using offline hash embeddings.",
                    exc,
                )
        else:
            logger.warning("transformers CLIP is unavailable. Using offline hash embeddings.")

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
        Persist the full engine state to disk.

        This writes **two files** next to each other:
          1. A binary engine file (``<stem>.trifecta``) containing the full
             HNSW graph, BM25 inverted index, KG, and registry — no rebuild
             needed on load.
          2. A gzip-compressed pickle sidecar (``<stem>.meta.gz``) containing
             the Python-level records, edge log, and page index.

        A subsequent ``from_snapshot`` restores the engine via a direct
        binary load — orders of magnitude faster than re-inserting all
        embeddings.

        Args:
            path: Base destination path (extensions are added automatically).
        """
        p = Path(path)
        stem = str(p).removesuffix(".snap.gz").removesuffix(".gz")
        engine_path = Path(stem + ".trifecta")
        meta_path = Path(stem + ".meta.gz")

        self._engine.save_to_file(str(engine_path))

        sidecar = {
            "version": 3,
            "dim": self._dim,
            "records": self._records,
            "edges": self._edge_log,
            "page_index": {f"{s}\x00{pg}": gids for (s, pg), gids in self._page_index.items()},
            "gid_to_page": {gid: [s, pg] for gid, (s, pg) in self._gid_to_page.items()},
        }
        with gzip.open(str(meta_path), "wb", compresslevel=5) as f:
            pickle.dump(sidecar, f, protocol=4)

        eng_kb = engine_path.stat().st_size / 1024
        meta_kb = meta_path.stat().st_size / 1024
        logger.info(
            "Snapshot saved: %d nodes -> %s (%.0f KB engine + %.0f KB meta)",
            self.size, engine_path.name, eng_kb, meta_kb,
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
        Restore a ``TrifectaClient`` from a snapshot.

        Supports **three** snapshot formats for backward compatibility:
          - v3 (current): binary engine file + sidecar metadata.
          - v2: single gzip pickle with page index.
          - v1: single gzip pickle without page index.

        Args:
            snapshot_path: Path to a ``.trifecta``, ``.meta.gz``, or legacy
                           ``.snap.gz`` file.
        """
        sp = Path(snapshot_path)

        # Detect v3 format: look for the .trifecta engine file.
        stem = str(sp).removesuffix(".snap.gz").removesuffix(".gz") \
                       .removesuffix(".meta").removesuffix(".trifecta")
        engine_path = Path(stem + ".trifecta")
        meta_path = Path(stem + ".meta.gz")

        if engine_path.exists() and meta_path.exists():
            return cls._load_v3(engine_path, meta_path,
                                text_model, clip_model, device,
                                hnsw_M, ef_construction, max_elements)

        # Legacy fallback: single gzip pickle (.snap.gz).
        legacy = sp if sp.exists() else None
        if legacy is None:
            legacy_gz = Path(stem + ".snap.gz")
            if legacy_gz.exists():
                legacy = legacy_gz
        if legacy is None:
            raise FileNotFoundError(
                f"Snapshot not found: tried {engine_path}, {meta_path}, {sp}")

        return cls._load_legacy(legacy, text_model, clip_model, device,
                                hnsw_M, ef_construction, max_elements)

    @classmethod
    def _load_v3(
        cls,
        engine_path: Path,
        meta_path: Path,
        text_model: str,
        clip_model: str,
        device: Optional[str],
        hnsw_M: int,
        ef_construction: int,
        max_elements: int,
    ) -> "TrifectaClient":
        """Load v3 snapshot: binary engine + sidecar metadata."""
        with gzip.open(str(meta_path), "rb") as f:
            sidecar = pickle.load(f)

        snap_dim: int = sidecar["dim"]
        records: List[Dict[str, Any]] = sidecar["records"]
        edges: List[Tuple[int, int, str]] = sidecar["edges"]

        logger.info("Loading v3 snapshot: %s + %s (%d records)",
                     engine_path.name, meta_path.name, len(records))

        client = cls(text_model=text_model, clip_model=clip_model,
                      device=device, hnsw_M=hnsw_M,
                      ef_construction=ef_construction,
                      max_elements=max_elements)

        if client._dim != snap_dim:
            raise ValueError(
                f"Model dim ({client._dim}) != snapshot dim ({snap_dim})")

        # Replace the C++ engine state directly from the binary file.
        client._engine.load_from_file(str(engine_path))

        client._records = records
        client._edge_log = edges

        raw_pi = sidecar.get("page_index", {})
        raw_g2p = sidecar.get("gid_to_page", {})
        for key_str, gids in raw_pi.items():
            s, p_str = key_str.split("\x00", 1)
            client._page_index[(s, int(p_str))] = gids
        for gid_str, sp_pair in raw_g2p.items():
            client._gid_to_page[int(gid_str)] = (sp_pair[0], int(sp_pair[1]))

        logger.info("Snapshot loaded: engine has %d nodes, %d pages",
                     client.size, client.page_count)
        return client

    @classmethod
    def _load_legacy(
        cls,
        legacy_path: Path,
        text_model: str,
        clip_model: str,
        device: Optional[str],
        hnsw_M: int,
        ef_construction: int,
        max_elements: int,
    ) -> "TrifectaClient":
        """Load v1/v2 snapshot: single gzip pickle, re-inserts into engine."""
        with gzip.open(str(legacy_path), "rb") as f:
            data = pickle.load(f)

        records: List[Dict[str, Any]] = data["records"]
        edges: List[Tuple[int, int, str]] = data["edges"]
        snap_dim: int = data["dim"]

        logger.info("Loading legacy snapshot %s: %d records, %d edges",
                     legacy_path.name, len(records), len(edges))

        client = cls(text_model=text_model, clip_model=clip_model,
                      device=device, hnsw_M=hnsw_M,
                      ef_construction=ef_construction,
                      max_elements=max_elements)

        if client._dim != snap_dim:
            raise ValueError(
                f"Model dim ({client._dim}) != snapshot dim ({snap_dim})")

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

        logger.info("Legacy snapshot loaded: engine has %d nodes, %d pages",
                     client.size, client.page_count)
        return client

    # ── Retrieval ────────────────────────────────────────────────────────────

    def query(
        self,
        text: Optional[str] = None,
        image: Optional[ImageInput] = None,
        top_k: int = 10,
        search_ef: int = 50,
        use_hnsw: bool = True,
        use_bm25: bool = True,
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
            use_hnsw:  If False, skip vector search (empty query_vec).
            use_bm25:  If False, skip lexical search (empty query_text).

        Returns:
            List of (global_id, rrf_score) tuples, descending by score.
        """
        if text is None and image is None:
            return []

        query_text = (text or "") if use_bm25 else ""
        query_vec: List[float] = []

        if use_hnsw:
            if text is not None and image is not None:
                text_vec = _normalize(self._embed_text(text))
                img_vec = _normalize(self._embed_image(self._load_image(image)))
                fused = _normalize(text_vec + img_vec)
                query_vec = fused.tolist()
            elif image is not None:
                query_vec = self._embed_image(self._load_image(image)).tolist()
            elif text is not None:
                query_vec = self._embed_text(text).tolist()

        if not query_vec and not query_text:
            return []

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
        """Encode text via HF CLIP text encoder -> float32 numpy vector (LRU cached).

        We bypass sentence-transformers' CLIP wrapper here because it does not
        honour `max_seq_length` and crashes on inputs > 77 tokens. Calling the
        HF CLIP text model directly with `truncation=True, max_length=77`
        guarantees compliance with the hard position-embedding limit.
        """
        cached = self._text_cache.get(text)
        if cached is not None:
            self._text_cache.move_to_end(text)
            return cached

        if self._using_fallback_embeddings or self._clip_model is None or self._clip_processor is None:
            result = _hash_text_embedding(text, self._dim)
            self._text_cache[text] = result
            if len(self._text_cache) > self._embed_cache_max:
                self._text_cache.popitem(last=False)
            return result

        inputs = self._clip_processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs["attention_mask"].to(self._device)
        with torch.no_grad():
            text_out = self._clip_model.text_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            pooled = text_out.pooler_output
            features = self._clip_model.text_projection(pooled)
        vec = features.cpu().numpy().astype(np.float32).flatten()

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

        if self._using_fallback_embeddings or self._clip_model is None or self._clip_processor is None:
            result = _hash_image_embedding(image, self._dim)
            self._image_cache[img_hash] = result
            if len(self._image_cache) > self._embed_cache_max:
                self._image_cache.popitem(last=False)
            return result

        inputs = self._clip_processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device)
        with torch.no_grad():
            vision_out = self._clip_model.vision_model(pixel_values=pixel_values)
            pooled = vision_out.pooler_output
            features = self._clip_model.visual_projection(pooled)
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

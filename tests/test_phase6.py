"""
test_phase6.py — Comprehensive Phase 6 tests for TrifectaClient.

Tests cover:
  Section 1: _normalize helper (pure math, no external deps)
  Section 2: Late-fusion vector math
  Section 3: TrifectaClient with mocked ML models + real C++ engine
  Section 4: Error handling and edge cases
"""

import sys
import os
import json
import math

import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trifecta.client import _normalize, TrifectaClient
from trifecta import trifecta_py as tr

# ── Test harness ─────────────────────────────────────────────────────────

passed = 0
failed = 0


def run(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  [PASS] {name}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {name} -- {e}")
        failed += 1


def assert_true(cond, msg="assertion failed"):
    if not cond:
        raise AssertionError(msg)


def assert_near(a, b, eps=1e-5, msg=""):
    if abs(a - b) > eps:
        raise AssertionError(f"{msg}: got {a}, expected {b}")


# =========================================================================
# Section 1: _normalize
# =========================================================================

def test_normalize_basic():
    v = np.array([3.0, 4.0], dtype=np.float32)
    n = _normalize(v)
    assert_near(np.linalg.norm(n), 1.0, msg="unit norm")
    assert_near(n[0], 0.6, msg="x")
    assert_near(n[1], 0.8, msg="y")

run("normalize_basic", test_normalize_basic)


def test_normalize_zero_vector():
    v = np.zeros(5, dtype=np.float32)
    n = _normalize(v)
    assert_true(np.allclose(n, 0.0), "zero vector should stay zero")

run("normalize_zero_vector", test_normalize_zero_vector)


def test_normalize_already_unit():
    v = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    n = _normalize(v)
    assert_true(np.allclose(n, v, atol=1e-6), "already-unit unchanged")

run("normalize_already_unit", test_normalize_already_unit)


def test_normalize_negative_components():
    v = np.array([-3.0, 4.0], dtype=np.float32)
    n = _normalize(v)
    assert_near(np.linalg.norm(n), 1.0, msg="norm")
    assert_near(n[0], -0.6, msg="x")
    assert_near(n[1], 0.8, msg="y")

run("normalize_negative_components", test_normalize_negative_components)


def test_normalize_high_dim():
    rng = np.random.RandomState(42)
    v = rng.randn(512).astype(np.float32)
    n = _normalize(v)
    assert_near(np.linalg.norm(n), 1.0, eps=1e-4, msg="512-d unit norm")

run("normalize_high_dim", test_normalize_high_dim)


def test_normalize_tiny_vector():
    v = np.array([1e-14, 0.0], dtype=np.float32)
    n = _normalize(v)
    assert_true(np.allclose(n, v), "sub-threshold vector returned as-is")

run("normalize_tiny_vector", test_normalize_tiny_vector)


# =========================================================================
# Section 2: Late-fusion math
# =========================================================================

def test_fusion_orthogonal():
    a = _normalize(np.array([1.0, 0.0], dtype=np.float32))
    b = _normalize(np.array([0.0, 1.0], dtype=np.float32))
    fused = _normalize(a + b)
    s = 1.0 / math.sqrt(2)
    assert_near(fused[0], s, msg="x")
    assert_near(fused[1], s, msg="y")

run("fusion_orthogonal", test_fusion_orthogonal)


def test_fusion_same_direction():
    a = _normalize(np.array([3.0, 4.0], dtype=np.float32))
    b = _normalize(np.array([6.0, 8.0], dtype=np.float32))
    fused = _normalize(a + b)
    assert_near(fused[0], 0.6, msg="x")
    assert_near(fused[1], 0.8, msg="y")

run("fusion_same_direction", test_fusion_same_direction)


def test_fusion_opposite_cancels():
    a = _normalize(np.array([1.0, 0.0], dtype=np.float32))
    b = _normalize(np.array([-1.0, 0.0], dtype=np.float32))
    fused = _normalize(a + b)
    assert_true(np.allclose(fused, 0.0, atol=1e-6), "opposite vectors cancel")

run("fusion_opposite_cancels", test_fusion_opposite_cancels)


def test_fusion_preserves_unit_norm():
    rng = np.random.RandomState(123)
    a = _normalize(rng.randn(512).astype(np.float32))
    b = _normalize(rng.randn(512).astype(np.float32))
    fused = _normalize(a + b)
    norm = float(np.linalg.norm(fused))
    assert_true(
        norm < 1e-6 or abs(norm - 1.0) < 1e-5,
        f"fused norm should be ~0 or ~1, got {norm}",
    )

run("fusion_preserves_unit_norm", test_fusion_preserves_unit_norm)


def test_fusion_asymmetric_magnitudes():
    a = np.array([10.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 0.001], dtype=np.float32)
    na, nb = _normalize(a), _normalize(b)
    fused = _normalize(na + nb)
    assert_near(np.linalg.norm(fused), 1.0, msg="still unit after fusion")

run("fusion_asymmetric_magnitudes", test_fusion_asymmetric_magnitudes)


# =========================================================================
# Section 3: TrifectaClient with mocked models (real C++ engine)
# =========================================================================

DIM = 4


def _make_client():
    """Create a TrifectaClient with mocked embedding models but real C++ engine."""
    with patch("trifecta.client.SentenceTransformer") as MockST, \
         patch("trifecta.client.CLIPModel") as MockCLIP, \
         patch("trifecta.client.CLIPProcessor") as MockProc:

        # --- Text model: deterministic embeddings keyed by text hash ---
        mock_text = MockST.return_value
        mock_text.get_sentence_embedding_dimension.return_value = DIM

        def text_encode(text, **kwargs):
            seed = abs(hash(text)) % (2 ** 31)
            rng = np.random.RandomState(seed)
            return rng.randn(DIM).astype(np.float32)

        mock_text.encode.side_effect = text_encode

        # --- CLIP model: deterministic image + text embeddings ---
        mock_clip = MagicMock()
        MockCLIP.from_pretrained.return_value = mock_clip
        mock_clip.to.return_value = mock_clip
        mock_clip.eval.return_value = mock_clip
        mock_clip.config.projection_dim = DIM

        _img_counter = [0]

        def visual_proj_call(pooled):
            rng = np.random.RandomState(7777 + _img_counter[0])
            _img_counter[0] += 1
            feat = MagicMock()
            feat.cpu.return_value.numpy.return_value = (
                rng.randn(1, DIM).astype(np.float32)
            )
            return feat

        mock_clip.visual_projection = MagicMock(side_effect=visual_proj_call)

        # CLIP text encoder mock: stable embeddings keyed on the input_ids tensor.
        def text_model_call(input_ids=None, attention_mask=None):
            try:
                seed = int(abs(int(input_ids.sum().item()))) % (2 ** 31)
            except Exception:
                seed = 0
            rng = np.random.RandomState(seed)
            pooled = MagicMock()
            pooled._rng_state = rng  # carry through to text_projection
            out = MagicMock()
            out.pooler_output = pooled
            return out

        def text_proj_call(pooled):
            rng = getattr(pooled, "_rng_state", np.random.RandomState(0))
            feat = MagicMock()
            feat.cpu.return_value.numpy.return_value = (
                rng.randn(1, DIM).astype(np.float32)
            )
            return feat

        mock_clip.text_model = MagicMock(side_effect=text_model_call)
        mock_clip.text_projection = MagicMock(side_effect=text_proj_call)

        # --- Processor mock: dispatches on whether text= or images= is given ---
        mock_proc = MagicMock()
        MockProc.from_pretrained.return_value = mock_proc

        def process(**kwargs):
            if "text" in kwargs and kwargs["text"] is not None:
                text_arg = kwargs["text"]
                if isinstance(text_arg, list):
                    text_arg = text_arg[0] if text_arg else ""
                # Build a deterministic "input_ids" so the embedding is stable.
                seed = abs(hash(text_arg)) % (2 ** 31)
                rng = np.random.RandomState(seed)
                fake_ids = rng.randint(0, 1000, size=(1, 8))
                import torch as _torch
                ids = _torch.tensor(fake_ids, dtype=_torch.long)
                mask = _torch.ones_like(ids)
                ids_wrap = MagicMock(wraps=ids)
                ids_wrap.to.return_value = ids
                mask_wrap = MagicMock(wraps=mask)
                mask_wrap.to.return_value = mask
                return {"input_ids": ids_wrap, "attention_mask": mask_wrap}
            pv = MagicMock()
            pv.to.return_value = pv
            return {"pixel_values": pv}

        mock_proc.side_effect = process
        # Tokenizer attribute for any legacy code paths that still touch it.
        mock_proc.tokenizer = MagicMock()

        client = TrifectaClient(device="cpu")

    return client


# ── 3a: Ingestion ────────────────────────────────────────────────────────

def test_add_document_returns_sequential_ids():
    c = _make_client()
    id0 = c.add_document("hello world", metadata={"k": "v"})
    id1 = c.add_document("foo bar baz")
    assert_true(id0 == 0, f"expected 0, got {id0}")
    assert_true(id1 == 1, f"expected 1, got {id1}")
    assert_true(len(c) == 2, f"expected size 2, got {len(c)}")

run("add_document_returns_sequential_ids", test_add_document_returns_sequential_ids)


def test_add_document_queryable_by_text():
    c = _make_client()
    c.add_document("alpha beta gamma")
    c.add_document("delta epsilon zeta")
    results = c._engine.query(query_vec=[], query_text="alpha", top_k=5)
    ids = [r[0] for r in results]
    assert_true(0 in ids, f"doc 0 should match 'alpha': {ids}")

run("add_document_queryable_by_text", test_add_document_queryable_by_text)


def test_add_image_returns_valid_id():
    c = _make_client()
    img = Image.new("RGB", (32, 32), color=(128, 128, 128))
    gid = c.add_image(image=img, caption="test caption", metadata={"type": "test"})
    assert_true(gid == 0, f"expected 0, got {gid}")
    assert_true(c.size == 1)

run("add_image_returns_valid_id", test_add_image_returns_valid_id)


def test_add_image_caption_searchable():
    c = _make_client()
    img = Image.new("RGB", (32, 32), color=(255, 0, 0))
    c.add_image(image=img, caption="red sunset over mountains")
    results = c._engine.query(query_vec=[], query_text="sunset mountains", top_k=5)
    assert_true(len(results) > 0, "caption should be BM25 searchable")
    assert_true(results[0][0] == 0)

run("add_image_caption_searchable", test_add_image_caption_searchable)


def test_add_image_pil_object():
    c = _make_client()
    img = Image.new("RGBA", (64, 64), color=(0, 255, 0, 128))
    gid = c.add_image(image=img)
    assert_true(gid == 0)

run("add_image_pil_object", test_add_image_pil_object)


# ── 3b: Knowledge Graph ──────────────────────────────────────────────────

def test_add_edge_wires_through():
    c = _make_client()
    c.add_document("node a")
    c.add_document("node b")
    c.add_edge(0, 1, tr.EdgeType.RELATES_TO)
    c.add_edge(0, 1, tr.EdgeType.EXPLAINS)
    c.add_edge(0, 1, tr.EdgeType.DEPICTS)
    assert_true(c.size == 2)

run("add_edge_wires_through", test_add_edge_wires_through)


def test_add_edge_invalid_id_raises():
    c = _make_client()
    c.add_document("only one node")
    try:
        c.add_edge(0, 99, tr.EdgeType.RELATES_TO)
        raise AssertionError("expected error for invalid target_id")
    except (IndexError, RuntimeError):
        pass

run("add_edge_invalid_id_raises", test_add_edge_invalid_id_raises)


# ── 3c: Query ────────────────────────────────────────────────────────────

def test_query_text_only():
    c = _make_client()
    c.add_document("the quick brown fox")
    c.add_document("the lazy dog")
    results = c.query(text="quick fox", top_k=5)
    assert_true(len(results) > 0, "text-only query should return results")
    ids = [r[0] for r in results]
    assert_true(0 in ids, f"expected doc 0 in results: {ids}")

run("query_text_only", test_query_text_only)


def test_query_image_only():
    c = _make_client()
    img1 = Image.new("RGB", (32, 32), color=(255, 0, 0))
    img2 = Image.new("RGB", (32, 32), color=(0, 0, 255))
    c.add_image(image=img1, caption="red")
    c.add_image(image=img2, caption="blue")
    results = c.query(image=Image.new("RGB", (32, 32), color=(200, 0, 0)), top_k=5)
    assert_true(len(results) > 0, "image-only query should return results")

run("query_image_only", test_query_image_only)


def test_query_fused_multimodal():
    c = _make_client()
    c.add_document("quick brown fox jumps")
    img = Image.new("RGB", (32, 32), color=(100, 100, 100))
    c.add_image(image=img, caption="a fox in the wild")
    results = c.query(text="fox", image=Image.new("RGB", (32, 32)), top_k=5)
    assert_true(len(results) > 0, "fused query should return results")

run("query_fused_multimodal", test_query_fused_multimodal)


def test_query_none_returns_empty():
    c = _make_client()
    c.add_document("some data")
    results = c.query(text=None, image=None)
    assert_true(results == [], f"expected empty, got {results}")

run("query_none_returns_empty", test_query_none_returns_empty)


def test_query_top_k_respected():
    c = _make_client()
    for i in range(10):
        c.add_document(f"document number {i} with word{i}")
    results = c.query(text="document", top_k=3)
    assert_true(len(results) <= 3, f"expected <=3, got {len(results)}")

run("query_top_k_respected", test_query_top_k_respected)


def test_query_kg_expansion_surfaces_context():
    c = _make_client()
    c.add_document("primary target document")
    c.add_document("unrelated noise")
    c.add_document("contextual neighbor only reachable via KG")
    c.add_edge(0, 2, tr.EdgeType.EXPLAINS)
    results = c.query(text="primary target", top_k=5)
    ids = [r[0] for r in results]
    assert_true(0 in ids, f"primary doc missing: {ids}")
    assert_true(2 in ids, f"KG neighbor should appear via expansion: {ids}")

run("query_kg_expansion_surfaces_context", test_query_kg_expansion_surfaces_context)


# ── 3d: Properties ───────────────────────────────────────────────────────

def test_size_and_len():
    c = _make_client()
    assert_true(c.size == 0)
    assert_true(len(c) == 0)
    c.add_document("hello")
    assert_true(c.size == 1)
    assert_true(len(c) == 1)

run("size_and_len", test_size_and_len)


def test_dim_property():
    c = _make_client()
    assert_true(c.dim == DIM, f"expected dim={DIM}, got {c.dim}")

run("dim_property", test_dim_property)


def test_device_property():
    c = _make_client()
    assert_true(c.device == "cpu", f"expected 'cpu', got {c.device!r}")

run("device_property", test_device_property)


def test_repr_contains_info():
    c = _make_client()
    c.add_document("x")
    r = repr(c)
    assert_true("TrifectaClient" in r, f"repr missing class name: {r}")
    assert_true("size=1" in r, f"repr missing size: {r}")
    assert_true(str(DIM) in r, f"repr missing dim: {r}")

run("repr_contains_info", test_repr_contains_info)


# =========================================================================
# Section 4: Error handling
# =========================================================================

def test_add_document_empty_raises():
    c = _make_client()
    for bad in ["", "   ", "\t\n"]:
        try:
            c.add_document(bad)
            raise AssertionError(f"expected ValueError for text={bad!r}")
        except ValueError:
            pass

run("add_document_empty_raises", test_add_document_empty_raises)


def test_add_image_missing_file_raises():
    c = _make_client()
    try:
        c.add_image(image="nonexistent_file_12345.jpg")
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError:
        pass

run("add_image_missing_file_raises", test_add_image_missing_file_raises)


def test_load_image_converts_rgba_to_rgb():
    c = _make_client()
    rgba = Image.new("RGBA", (16, 16), (255, 0, 0, 128))
    gid = c.add_image(image=rgba)
    assert_true(gid == 0, "RGBA image should be accepted (converted to RGB)")

run("load_image_converts_rgba_to_rgb", test_load_image_converts_rgba_to_rgb)


def test_metadata_round_trip():
    c = _make_client()
    meta = {"author": "test", "page": 7, "tags": ["a", "b"]}
    gid = c.add_document("test text", metadata=meta)
    assert_true(gid == 0)

run("metadata_round_trip", test_metadata_round_trip)


# =========================================================================
print(f"\nPhase 6 test suite: {passed}/{passed + failed} passed")
sys.exit(0 if failed == 0 else 1)

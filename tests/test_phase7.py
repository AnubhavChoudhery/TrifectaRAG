"""
test_phase7.py — Comprehensive tests for Phase 7 features.

Covers:
  - C++ engine binary save/load (round-trip fidelity)
  - BM25 max_results parameter
  - TrifectaClient page index (page tracking, get_page_chunks, etc.)
  - TrifectaClient v3 snapshot (binary engine + sidecar metadata)
  - LRU embedding cache behaviour
  - Multi-PDF PDFIngestor.ingest_pdfs convenience

Uses mocked ML models (SentenceTransformer, CLIPModel, CLIPProcessor) to
avoid model downloads.  All C++ engine calls are real.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trifecta import trifecta_py as tr

DIM = 8

# ── Helpers ──────────────────────────────────────────────────────────────────

passed = 0
failed = 0


def run(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  [PASS] {name}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        failed += 1


def assert_eq(a, b, msg=""):
    if a != b:
        raise AssertionError(f"{msg}: {a!r} != {b!r}")


def assert_close(a, b, tol=1e-5, msg=""):
    if abs(a - b) > tol:
        raise AssertionError(f"{msg}: {a} not close to {b}")


def assert_true(cond, msg="expected True"):
    if not cond:
        raise AssertionError(msg)


def rand_vec(d=DIM, seed=None):
    rng = np.random.RandomState(seed)
    return rng.randn(d).astype(np.float32).tolist()


# ── 1. C++ engine binary save/load ──────────────────────────────────────────

def test_engine_save_load_roundtrip():
    e1 = tr.TrifectaEngine(dim=DIM)
    v0 = rand_vec(seed=0)
    v1 = rand_vec(seed=1)
    v2 = rand_vec(seed=2)
    g0 = e1.ingest(text="alpha beta gamma", embedding=v0, metadata="m0")
    g1 = e1.ingest(text="delta epsilon", embedding=v1, metadata="m1")
    g2 = e1.ingest(text="zeta eta theta", embedding=v2, metadata="m2", modality=tr.Modality.IMAGE)
    e1.add_edge(g0, g1, tr.EdgeType.RELATES_TO)
    e1.add_edge(g2, g0, tr.EdgeType.DEPICTS)

    with tempfile.NamedTemporaryFile(suffix=".trifecta", delete=False) as f:
        tmp = f.name
    try:
        e1.save_to_file(tmp)
        assert_true(os.path.getsize(tmp) > 0, "engine file should be non-empty")

        e2 = tr.TrifectaEngine(dim=1)
        e2.load_from_file(tmp)
        assert_eq(e2.size, 3)

        n0 = e2.get_node(0)
        assert_eq(n0.metadata, "m0")
        assert_eq(n0.modality, tr.Modality.TEXT)

        n2 = e2.get_node(2)
        assert_eq(n2.modality, tr.Modality.IMAGE)

        r = e2.query(query_vec=v0, query_text="alpha", top_k=5)
        gids = [gid for gid, _ in r]
        assert_true(g0 in gids, f"expected gid {g0} in query results")
        assert_true(g1 in gids, "KG neighbor g1 should surface via RELATES_TO")
    finally:
        os.unlink(tmp)


def test_engine_load_bad_file():
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(b"NOT_TRIFECTA_HEADER_DATA")
        tmp = f.name
    try:
        e = tr.TrifectaEngine(dim=DIM)
        try:
            e.load_from_file(tmp)
            raise AssertionError("Should have raised on bad magic")
        except RuntimeError:
            pass
    finally:
        os.unlink(tmp)


def test_engine_save_load_empty():
    """Save/load an empty engine."""
    e1 = tr.TrifectaEngine(dim=DIM)
    with tempfile.NamedTemporaryFile(suffix=".trifecta", delete=False) as f:
        tmp = f.name
    try:
        e1.save_to_file(tmp)
        e2 = tr.TrifectaEngine(dim=1)
        e2.load_from_file(tmp)
        assert_eq(e2.size, 0)
    finally:
        os.unlink(tmp)


# ── 2. BM25 max_results ─────────────────────────────────────────────────────

def test_bm25_top_k_truncation():
    e = tr.TrifectaEngine(dim=DIM)
    for i in range(20):
        e.ingest(text=f"document about topic{i} and general keyword",
                 embedding=rand_vec(seed=i), metadata=f"doc{i}")
    r_all = e.query(query_vec=[], query_text="keyword", top_k=5)
    assert_true(len(r_all) <= 5, f"expected <=5, got {len(r_all)}")
    r_big = e.query(query_vec=[], query_text="keyword", top_k=100)
    assert_true(len(r_big) <= 20, f"expected <=20, got {len(r_big)}")


# ── 3. Page index tracking ──────────────────────────────────────────────────

_MOCK_DIM = DIM
_COUNTER = 0


def _install_clip_mocks(MockST, MockCLIP, MockProc):
    """
    Wire up mocks compatible with TrifectaClient's HF-CLIP-direct text path.

    Returns (mock_st_inst, mock_clip_inst, mock_proc_inst).
    """
    mock_st_inst = MagicMock()
    mock_st_inst.get_sentence_embedding_dimension.return_value = _MOCK_DIM
    mock_st_inst.encode.side_effect = (
        lambda text, **kw: np.random.randn(_MOCK_DIM).astype(np.float32)
    )
    MockST.return_value = mock_st_inst

    mock_clip_inst = MagicMock()
    mock_clip_inst.to.return_value = mock_clip_inst
    mock_clip_inst.eval.return_value = mock_clip_inst
    mock_clip_inst.config.projection_dim = _MOCK_DIM
    mock_clip_inst.visual_projection.return_value = MagicMock(
        cpu=lambda: MagicMock(
            numpy=lambda: np.random.randn(1, _MOCK_DIM).astype(np.float32)
        )
    )

    def _text_model_call(input_ids=None, attention_mask=None):
        out = MagicMock()
        out.pooler_output = MagicMock()
        return out

    def _text_proj_call(pooled):
        return MagicMock(
            cpu=lambda: MagicMock(
                numpy=lambda: np.random.randn(1, _MOCK_DIM).astype(np.float32)
            )
        )

    mock_clip_inst.text_model = MagicMock(side_effect=_text_model_call)
    mock_clip_inst.text_projection = MagicMock(side_effect=_text_proj_call)
    MockCLIP.from_pretrained.return_value = mock_clip_inst

    mock_proc_inst = MagicMock()

    def _process(**kwargs):
        if "text" in kwargs and kwargs["text"] is not None:
            import torch as _torch
            ids = _torch.tensor([[1, 2, 3]], dtype=_torch.long)
            mask = _torch.ones_like(ids)
            ids_wrap = MagicMock()
            ids_wrap.to.return_value = ids
            mask_wrap = MagicMock()
            mask_wrap.to.return_value = mask
            return {"input_ids": ids_wrap, "attention_mask": mask_wrap}
        return {"pixel_values": MagicMock(to=lambda d: MagicMock())}

    mock_proc_inst.side_effect = _process
    mock_proc_inst.tokenizer = MagicMock()
    MockProc.from_pretrained.return_value = mock_proc_inst

    return mock_st_inst, mock_clip_inst, mock_proc_inst


def _make_mock_client():
    with patch("trifecta.client.SentenceTransformer") as MockST, \
         patch("trifecta.client.CLIPModel") as MockCLIP, \
         patch("trifecta.client.CLIPProcessor") as MockProc:

        _install_clip_mocks(MockST, MockCLIP, MockProc)

        from trifecta.client import TrifectaClient
        client = TrifectaClient(device="cpu")
        return client


def test_page_index_populated():
    client = _make_mock_client()
    g0 = client.add_document("Hello world", metadata={"source": "docA", "page": 1})
    g1 = client.add_document("Foo bar", metadata={"source": "docA", "page": 1})
    g2 = client.add_document("Baz qux", metadata={"source": "docA", "page": 2})
    g3 = client.add_document("No page info", metadata={"source": "docA"})

    chunks_p1 = client.get_page_chunks("docA", 1)
    assert_eq(sorted(chunks_p1), sorted([g0, g1]), "page 1 should have g0, g1")

    chunks_p2 = client.get_page_chunks("docA", 2)
    assert_eq(chunks_p2, [g2], "page 2 should have g2")

    assert_eq(client.get_page_chunks("docA", 99), [], "nonexistent page returns []")


def test_page_index_reverse_lookup():
    client = _make_mock_client()
    g0 = client.add_document("Test", metadata={"source": "src1", "page": 5})
    assert_eq(client.get_chunk_page(g0), ("src1", 5))
    g1 = client.add_document("No page", metadata={"source": "src1"})
    assert_eq(client.get_chunk_page(g1), None)


def test_list_sources_and_pages():
    client = _make_mock_client()
    client.add_document("A", metadata={"source": "pdf1", "page": 1})
    client.add_document("B", metadata={"source": "pdf2", "page": 3})
    client.add_document("C", metadata={"source": "pdf1", "page": 2})
    assert_eq(client.list_sources(), ["pdf1", "pdf2"])
    assert_eq(client.list_pages("pdf1"), [1, 2])
    assert_eq(client.list_pages("pdf2"), [3])
    assert_eq(client.page_count, 3)


# ── 4. V3 snapshot round-trip ────────────────────────────────────────────────

def test_v3_snapshot_roundtrip():
    client = _make_mock_client()
    client.add_document("Doc A page 1", metadata={"source": "test", "page": 1})
    client.add_document("Doc A page 2", metadata={"source": "test", "page": 2})
    client.add_edge(0, 1, tr.EdgeType.RELATES_TO)

    assert_eq(client.page_count, 2)

    with tempfile.TemporaryDirectory() as td:
        base = os.path.join(td, "snap_test")
        client.save_snapshot(base)

        eng_file = Path(base + ".trifecta")
        meta_file = Path(base + ".meta.gz")
        assert_true(eng_file.exists(), "engine file should exist")
        assert_true(meta_file.exists(), "meta file should exist")

        # from_snapshot creates a new TrifectaClient internally, so we
        # must keep the ML model patches active during the call.
        with patch("trifecta.client.SentenceTransformer") as MockST, \
             patch("trifecta.client.CLIPModel") as MockCLIP, \
             patch("trifecta.client.CLIPProcessor") as MockProc:

            _install_clip_mocks(MockST, MockCLIP, MockProc)

            from trifecta.client import TrifectaClient
            loaded = TrifectaClient.from_snapshot(base, device="cpu")

        assert_eq(loaded.size, 2)
        assert_eq(loaded.page_count, 2)
        assert_eq(sorted(loaded.get_page_chunks("test", 1)), [0])

        r = loaded.query(text="Doc A page 1", top_k=5)
        assert_true(len(r) > 0, "query should return results after load")


# ── 5. LRU embedding cache ──────────────────────────────────────────────────

def test_text_embedding_cache_hit():
    client = _make_mock_client()
    v1 = client._embed_text("cached query")
    call_count_before = client._clip_model.text_model.call_count

    v2 = client._embed_text("cached query")
    call_count_after = client._clip_model.text_model.call_count

    assert_eq(call_count_before, call_count_after,
              "second call should not invoke CLIP text model (cache hit)")
    np.testing.assert_array_equal(v1, v2)


def test_text_embedding_cache_eviction():
    client = _make_mock_client()
    client._embed_cache_max = 3
    client._text_cache.clear()

    client._embed_text("q1")
    client._embed_text("q2")
    client._embed_text("q3")
    assert_eq(len(client._text_cache), 3)

    client._embed_text("q4")
    assert_eq(len(client._text_cache), 3)
    assert_true("q1" not in client._text_cache, "q1 should be evicted")
    assert_true("q4" in client._text_cache, "q4 should be in cache")


# ── 6. Multi-PDF ingest_pdfs ────────────────────────────────────────────────

def test_ingest_pdfs_aggregation():
    """Verify ingest_pdfs aggregates stats from multiple calls."""
    client = _make_mock_client()
    from trifecta.pdf_ingest import PDFIngestor
    ingestor = PDFIngestor(client, mode="page")

    call_count = 0
    orig_ingest = ingestor.ingest_pdf

    def _fake_ingest(path, output_dir="", page_range=None):
        nonlocal call_count
        call_count += 1
        return {"pages": 10, "text_chunks": 20, "images": 2, "kg_edges": 5}

    ingestor.ingest_pdf = _fake_ingest

    totals = ingestor.ingest_pdfs(["a.pdf", "b.pdf", "c.pdf"])
    assert_eq(call_count, 3, "should call ingest_pdf three times")
    assert_eq(totals["pages"], 30)
    assert_eq(totals["text_chunks"], 60)
    assert_eq(totals["images"], 6)
    assert_eq(totals["kg_edges"], 15)


# ── Run all ──────────────────────────────────────────────────────────────────

run("engine_save_load_roundtrip", test_engine_save_load_roundtrip)
run("engine_load_bad_file", test_engine_load_bad_file)
run("engine_save_load_empty", test_engine_save_load_empty)
run("bm25_top_k_truncation", test_bm25_top_k_truncation)
run("page_index_populated", test_page_index_populated)
run("page_index_reverse_lookup", test_page_index_reverse_lookup)
run("list_sources_and_pages", test_list_sources_and_pages)
run("v3_snapshot_roundtrip", test_v3_snapshot_roundtrip)
run("text_embedding_cache_hit", test_text_embedding_cache_hit)
run("text_embedding_cache_eviction", test_text_embedding_cache_eviction)
run("ingest_pdfs_aggregation", test_ingest_pdfs_aggregation)

print(f"\nPhase 7 test suite: {passed}/{passed + failed} passed")
sys.exit(0 if failed == 0 else 1)

"""
Phase 5 Python smoke-test.

Validates the pybind11 bindings for TrifectaEngine:
  - ingest (text-only, embedding-only, combined)
  - add_edge / Knowledge Graph integration
  - query (vector-only, text-only, fused)
  - RRF score ordering invariants
  - error paths (wrong dim, bad ids)
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trifecta import trifecta_py as tr

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
        print(f"  [FAIL] {name} — {e}")
        failed += 1

def assert_true(cond, msg="assertion failed"):
    if not cond:
        raise AssertionError(msg)

def assert_near(a, b, eps=1e-4, msg=""):
    if abs(a - b) > eps:
        raise AssertionError(f"{msg} got {a} expected {b}")

# ── Tests ─────────────────────────────────────────────────────────────────────

def test_repr_and_len():
    e = tr.TrifectaEngine(dim=4)
    assert_true("TrifectaEngine" in repr(e))
    assert_true(len(e) == 0)
    assert_true(e.size == 0)

run("repr_and_len", test_repr_and_len)

# ─────────────────────────────────────────────────────────────────────────────
def test_ingest_text_only():
    e = tr.TrifectaEngine(dim=4)
    gid = e.ingest(text="hello world", embedding=[], metadata="doc0")
    assert_true(gid == 0)
    assert_true(len(e) == 1)

run("ingest_text_only", test_ingest_text_only)

# ─────────────────────────────────────────────────────────────────────────────
def test_ingest_embedding_only():
    e = tr.TrifectaEngine(dim=4)
    gid = e.ingest(text="", embedding=[1.0, 0.0, 0.0, 0.0], metadata="img0",
                   modality=tr.Modality.IMAGE)
    assert_true(gid == 0)
    assert_true(len(e) == 1)

run("ingest_embedding_only", test_ingest_embedding_only)

# ─────────────────────────────────────────────────────────────────────────────
def test_ingest_combined():
    e = tr.TrifectaEngine(dim=4)
    gid = e.ingest(text="quick brown fox", embedding=[0.1, 0.2, 0.3, 0.4],
                   metadata="doc0")
    assert_true(gid == 0)
    gid2 = e.ingest(text="lazy dog", embedding=[0.5, 0.6, 0.7, 0.8],
                    metadata="doc1")
    assert_true(gid2 == 1)
    assert_true(len(e) == 2)

run("ingest_combined", test_ingest_combined)

# ─────────────────────────────────────────────────────────────────────────────
def test_ingest_wrong_dim_raises():
    e = tr.TrifectaEngine(dim=4)
    try:
        e.ingest(text="x", embedding=[1.0, 2.0], metadata="bad")
        raise AssertionError("Expected ValueError / RuntimeError")
    except (ValueError, RuntimeError):
        pass

run("ingest_wrong_dim_raises", test_ingest_wrong_dim_raises)

# ─────────────────────────────────────────────────────────────────────────────
def test_add_edge_and_query_kg_expansion():
    e = tr.TrifectaEngine(dim=4)
    id0 = e.ingest(text="the quick brown fox", embedding=[1.0, 0.0, 0.0, 0.0], metadata="d0")
    id1 = e.ingest(text="lazy dog jumps high", embedding=[0.0, 1.0, 0.0, 0.0], metadata="d1")
    id2 = e.ingest(text="distant context node", embedding=[0.0, 0.0, 1.0, 0.0], metadata="d2")

    # id2 is not in the top HNSW/BM25 results for this query — but IS a KG neighbour
    e.add_edge(id0, id2, tr.EdgeType.RELATES_TO)

    results = e.query(query_vec=[1.0, 0.0, 0.0, 0.0], query_text="quick brown fox", top_k=5)
    ids_returned = [r[0] for r in results]
    assert_true(id0 in ids_returned, f"id0 not in results: {ids_returned}")
    # id2 should appear via KG expansion
    assert_true(id2 in ids_returned, f"id2 not in results via KG: {ids_returned}")

run("add_edge_and_kg_expansion", test_add_edge_and_query_kg_expansion)

# ─────────────────────────────────────────────────────────────────────────────
def test_add_edge_bad_id_raises():
    e = tr.TrifectaEngine(dim=4)
    e.ingest(text="hello", embedding=[], metadata="x")
    try:
        e.add_edge(0, 99, tr.EdgeType.EXPLAINS)
        raise AssertionError("Expected out-of-range error")
    except (IndexError, RuntimeError):
        pass

run("add_edge_bad_id_raises", test_add_edge_bad_id_raises)

# ─────────────────────────────────────────────────────────────────────────────
def test_query_text_only():
    e = tr.TrifectaEngine(dim=4)
    e.ingest(text="alpha beta gamma", embedding=[], metadata="d0")
    e.ingest(text="alpha alpha", embedding=[], metadata="d1")
    e.ingest(text="delta epsilon", embedding=[], metadata="d2")

    results = e.query(query_vec=[], query_text="alpha")
    assert_true(len(results) >= 2, f"expected >=2 results, got {len(results)}")
    # both docs containing 'alpha' should be present
    ids = [r[0] for r in results]
    assert_true(0 in ids and 1 in ids, f"missing alpha docs: {ids}")

run("query_text_only", test_query_text_only)

# ─────────────────────────────────────────────────────────────────────────────
def test_query_vector_only():
    e = tr.TrifectaEngine(dim=4)
    e.ingest(text="", embedding=[1.0, 0.0, 0.0, 0.0], metadata="d0")
    e.ingest(text="", embedding=[0.0, 1.0, 0.0, 0.0], metadata="d1")

    results = e.query(query_vec=[1.0, 0.0, 0.0, 0.0], query_text="", top_k=2)
    assert_true(len(results) > 0)
    assert_true(results[0][0] == 0, f"expected id=0 first, got {results[0][0]}")
    assert_true(results[0][1] > 0.0)

run("query_vector_only", test_query_vector_only)

# ─────────────────────────────────────────────────────────────────────────────
def test_rrf_ordering_direct_beats_context():
    """
    A directly-retrieved result (in HNSW/BM25) must score above a pure
    KG-context node that has no direct match.
    """
    e = tr.TrifectaEngine(dim=4)
    id_direct = e.ingest(text="target query word", embedding=[1.0, 0.0, 0.0, 0.0], metadata="direct")
    id_context = e.ingest(text="unrelated fringe node", embedding=[0.0, 0.0, 0.0, 1.0], metadata="context")

    e.add_edge(id_direct, id_context, tr.EdgeType.RELATES_TO)

    results = e.query(query_vec=[1.0, 0.0, 0.0, 0.0], query_text="target query word", top_k=5)
    scores = {r[0]: r[1] for r in results}

    assert_true(id_direct in scores, "direct result missing")
    assert_true(id_context in scores, "context node missing after KG expansion")
    assert_true(
        scores[id_direct] > scores[id_context],
        f"direct({scores[id_direct]:.4f}) should beat context({scores[id_context]:.4f})"
    )

run("rrf_ordering_direct_beats_context", test_rrf_ordering_direct_beats_context)

# ─────────────────────────────────────────────────────────────────────────────
def test_empty_query_returns_empty():
    e = tr.TrifectaEngine(dim=4)
    e.ingest(text="hello", embedding=[1.0, 0.0, 0.0, 0.0], metadata="x")
    results = e.query(query_vec=[], query_text="")
    assert_true(results == [], f"expected empty, got {results}")

run("empty_query_returns_empty", test_empty_query_returns_empty)

# ─────────────────────────────────────────────────────────────────────────────
def test_query_wrong_dim_raises():
    e = tr.TrifectaEngine(dim=4)
    e.ingest(text="x", embedding=[1.0, 0.0, 0.0, 0.0], metadata="d0")
    try:
        e.query(query_vec=[1.0, 2.0], query_text="")
        raise AssertionError("Expected error")
    except (ValueError, RuntimeError):
        pass

run("query_wrong_dim_raises", test_query_wrong_dim_raises)

# ─────────────────────────────────────────────────────────────────────────────
def test_top_k_respected():
    e = tr.TrifectaEngine(dim=4)
    for i in range(10):
        v = [1.0 if j == i % 4 else 0.0 for j in range(4)]
        e.ingest(text=f"doc {i} word{i}", embedding=v, metadata=f"d{i}")
    results = e.query(query_vec=[1.0, 0.0, 0.0, 0.0], query_text="doc", top_k=3)
    assert_true(len(results) <= 3, f"expected <=3 results, got {len(results)}")

run("top_k_respected", test_top_k_respected)

# ─────────────────────────────────────────────────────────────────────────────
def test_edge_types_all_accepted():
    e = tr.TrifectaEngine(dim=4)
    a = e.ingest(text="a", embedding=[], metadata="a")
    b = e.ingest(text="b", embedding=[], metadata="b")
    e.add_edge(a, b, tr.EdgeType.RELATES_TO)
    e.add_edge(a, b, tr.EdgeType.EXPLAINS)
    e.add_edge(a, b, tr.EdgeType.DEPICTS)

run("edge_types_all_accepted", test_edge_types_all_accepted)

# ─────────────────────────────────────────────────────────────────────────────
print(f"\nPhase 5 Python smoke-test: {passed}/{passed+failed} passed")
sys.exit(0 if failed == 0 else 1)

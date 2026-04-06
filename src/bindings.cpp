/**
 * =============================================================================
 * bindings.cpp — Phase 5: pybind11 Python bindings for TrifectaEngine
 * =============================================================================
 *
 * Exposes:
 *   trifecta_py.Modality    — TEXT / IMAGE enum
 *   trifecta_py.EdgeType    — RELATES_TO / EXPLAINS / DEPICTS enum
 *   trifecta_py.TrifectaEngine  — the unified retrieval engine
 *
 * Usage (Python):
 *   import trifecta_py as tr
 *
 *   engine = tr.TrifectaEngine(dim=4)
 *   id0 = engine.ingest(text="the quick brown fox", embedding=[0.1, 0.2, 0.3, 0.4],
 *                        metadata="doc0")
 *   id1 = engine.ingest(text="lazy dog jumps high", embedding=[0.5, 0.6, 0.7, 0.8],
 *                        metadata="doc1")
 *   engine.add_edge(id0, id1, tr.EdgeType.RELATES_TO)
 *
 *   results = engine.query(query_vec=[0.1, 0.2, 0.3, 0.4], query_text="quick fox")
 *   for gid, score in results:
 *       print(gid, score)
 * =============================================================================
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "trifecta_engine.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

using trifecta::EdgeType;
using trifecta::Modality;
using trifecta::TrifectaEngine;

PYBIND11_MODULE(trifecta_py, m) {
    m.doc() =
        "TrifectaRAG — Phase 5 Python bindings.\n"
        "Unified multi-modal RAG engine: HNSW vector search, BM25 keyword "
        "retrieval, and Knowledge-Graph context expansion fused via RRF.";

    // ── Modality enum ─────────────────────────────────────────────────────
    py::enum_<Modality>(m, "Modality",
        "Chunk modality. TEXT for text documents, IMAGE for image embeddings.")
        .value("TEXT",  Modality::TEXT)
        .value("IMAGE", Modality::IMAGE)
        .export_values();

    // ── EdgeType enum ─────────────────────────────────────────────────────
    py::enum_<EdgeType>(m, "EdgeType",
        "Directed edge semantics in the KnowledgeGraph.")
        .value("RELATES_TO", EdgeType::RELATES_TO)
        .value("EXPLAINS",   EdgeType::EXPLAINS)
        .value("DEPICTS",    EdgeType::DEPICTS)
        .export_values();

    // ── TrifectaEngine ────────────────────────────────────────────────────
    py::class_<TrifectaEngine>(m, "TrifectaEngine",
        R"doc(
Unified Trifecta retrieval engine.

Owns a GlobalRegistry, HNSWIndex (vector ANN), LexicalIndex (BM25), and
KnowledgeGraph. Provides `ingest` for populating all indexes simultaneously
and `query` for fused multi-modal retrieval with Reciprocal Rank Fusion.

Args:
    dim             (int):  Embedding dimensionality (must be > 0).
    hnsw_M          (int):  HNSW bi-directional links per node. Default 16.
    ef_construction (int):  HNSW construction ef. Default 200.
    max_elements    (int):  HNSW pre-allocated node capacity. Default 1_000_000.
        )doc")

        .def(py::init<std::size_t, std::size_t, std::size_t, std::size_t>(),
             "dim"_a,
             "hnsw_M"_a          = TrifectaEngine::kDefaultM,
             "ef_construction"_a = TrifectaEngine::kDefaultEfConstruct,
             "max_elements"_a    = TrifectaEngine::kDefaultMaxElements)

        // ── ingest ──────────────────────────────────────────────────────
        .def("ingest",
             &TrifectaEngine::ingest,
             R"doc(
Register one chunk across all relevant indexes.

Args:
    text      (str):       Raw text content. Pass "" for embedding-only chunks.
    embedding (list[float]): Dense embedding. Pass [] for text-only chunks.
    metadata  (str):       Arbitrary metadata stored in the registry.
    modality  (Modality):  Modality.TEXT (default) or Modality.IMAGE.

Returns:
    int: The newly assigned global_id.

Raises:
    ValueError: If embedding is non-empty but its size != dim.
             )doc",
             "text"_a,
             "embedding"_a,
             "metadata"_a  = std::string{},
             "modality"_a  = Modality::TEXT)

        // ── add_edge ─────────────────────────────────────────────────────
        .def("add_edge",
             &TrifectaEngine::add_edge,
             R"doc(
Add a directed edge in the KnowledgeGraph.

Args:
    source_id (int):      Source global_id (must already be ingested).
    target_id (int):      Target global_id (must already be ingested).
    edge_type (EdgeType): Semantic type of the relationship.

Raises:
    IndexError: If either id has not been ingested yet.
             )doc",
             "source_id"_a, "target_id"_a, "edge_type"_a)

        // ── query ────────────────────────────────────────────────────────
        .def("query",
             &TrifectaEngine::query,
             R"doc(
Fused multi-modal query using Reciprocal Rank Fusion (RRF).

At least one of `query_vec` / `query_text` must be non-empty.

Pipeline:
  1. HNSW ANN search on query_vec.
  2. BM25 keyword search on query_text.
  3. KnowledgeGraph 1-hop expansion from all retrieved IDs.
  4. RRF fusion of all three signals.

Args:
    query_vec  (list[float]): Query embedding. Pass [] to skip HNSW.
    query_text (str):         Query text.    Pass "" to skip BM25.
    top_k      (int):         Maximum results to return. Default 10.
    search_ef  (int):         HNSW search ef (candidate pool). Default 50.

Returns:
    list[tuple[int, float]]: (global_id, rrf_score) pairs, descending by score.

Raises:
    ValueError: If query_vec is non-empty but size != dim.
             )doc",
             "query_vec"_a,
             "query_text"_a,
             "top_k"_a     = TrifectaEngine::kDefaultTopK,
             "search_ef"_a = TrifectaEngine::kDefaultSearchEf)

        // ── read-only properties ─────────────────────────────────────────
        .def_property_readonly("size",
            &TrifectaEngine::size,
            "Number of chunks currently ingested.")

        .def("__len__", &TrifectaEngine::size)

        .def("__repr__", [](const TrifectaEngine& e) {
            return "<TrifectaEngine size=" + std::to_string(e.size()) + ">";
        });
}

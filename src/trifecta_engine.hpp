/**
 * =============================================================================
 * trifecta_engine.hpp — Phase 5: Unified Orchestration Layer
 * =============================================================================
 *
 * PURPOSE:
 *   TrifectaEngine is the single entry-point that owns and coordinates all
 *   four Phase 1-4 components:
 *
 *     GlobalRegistry  — authoritative source of truth for all ingested chunks
 *     HNSWIndex       — approximate nearest-neighbour vector search
 *     LexicalIndex    — BM25 keyword retrieval over text chunks
 *     KnowledgeGraph  — directed adjacency graph for relational context
 *
 * QUERY STRATEGY — Reciprocal Rank Fusion (RRF)
 *   1. Vector search:   HNSW  →  ranked list R_v  (by cosine distance)
 *   2. Keyword search:  BM25  →  ranked list R_t  (by BM25 score)
 *   3. Context expand:  KG BFS 1-hop from union(R_v, R_t) → set N_kg
 *   4. Fuse:            score(d) = Σ_i 1/(k + rank_i(d))  + KG_BOOST  if d ∈ N_kg
 *   5. Return:          top_k (global_id, rrf_score) pairs, descending.
 *
 * THREAD SAFETY:
 *   Not thread-safe for concurrent writes. External synchronisation required
 *   when calling ingest() or add_edge() from multiple threads.
 *
 * =============================================================================
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

#include "hnsw_index.hpp"
#include "knowledge_graph.hpp"
#include "lexical_index.hpp"
#include "trifecta_core.hpp"

namespace trifecta {

/**
 * TrifectaEngine
 * --------------
 * Owns all four indexes. Drive the complete multi-modal RAG retrieval pipeline
 * through two public methods: ingest() and query().
 */
class TrifectaEngine {
public:
    // ── RRF constant k (default 60, a standard choice from the literature) ──
    static constexpr int   kRrfK          = 60;
    // Multiplier for the flat KG-context contribution (down-weights pure
    // context neighbours relative to directly-ranked results).
    static constexpr float kKgContextMult = 0.5f;

    // ── HNSW defaults ───────────────────────────────────────────────────────
    static constexpr std::size_t kDefaultM            = 16;
    static constexpr std::size_t kDefaultEfConstruct  = 200;
    static constexpr std::size_t kDefaultMaxElements  = 1'000'000;
    static constexpr std::size_t kDefaultSearchEf     = 50;
    static constexpr std::size_t kDefaultTopK         = 10;

    /**
     * Construct the engine.
     *
     * @param dim              Embedding dimensionality (must be > 0).
     * @param hnsw_M           HNSW number of bi-directional links per node.
     * @param ef_construction  HNSW construction-time ef parameter.
     * @param max_elements     HNSW pre-allocated node capacity.
     */
    explicit TrifectaEngine(std::size_t dim,
                            std::size_t hnsw_M          = kDefaultM,
                            std::size_t ef_construction = kDefaultEfConstruct,
                            std::size_t max_elements    = kDefaultMaxElements);

    // ── Ingestion ────────────────────────────────────────────────────────────

    /**
     * ingest — register one chunk across all relevant indexes.
     *
     * Rules:
     *   • Always: registers the chunk in GlobalRegistry → assigns global_id.
     *   • If embedding is non-empty AND embedding.size() == dim_: adds to HNSW.
     *   • If text is non-empty: adds to LexicalIndex (BM25).
     *   • KnowledgeGraph edges are NOT added here; use add_edge() separately.
     *
     * @param text      Raw text content (may be empty for image-only chunks).
     * @param embedding Dense float embedding (may be empty for text-only).
     * @param metadata  Arbitrary metadata string stored in the registry.
     * @param modality  Modality::TEXT (default) or Modality::IMAGE.
     * @return          The newly assigned global_id.
     * @throws std::invalid_argument if a non-empty embedding has wrong dimension.
     */
    [[nodiscard]] uint32_t ingest(const std::string&        text,
                                  const std::vector<float>& embedding,
                                  const std::string&        metadata,
                                  Modality                  modality = Modality::TEXT);

    // ── Knowledge-Graph construction ─────────────────────────────────────────

    /**
     * Add a directed edge between two previously ingested global_ids.
     *
     * @throws std::out_of_range if either id has not been ingested yet.
     */
    void add_edge(uint32_t source_id, uint32_t target_id, EdgeType type);

    // ── Retrieval ────────────────────────────────────────────────────────────

    /**
     * query — fused multi-modal retrieval with RRF.
     *
     * At least one of query_vec / query_text must be non-empty, otherwise an
     * empty result set is returned immediately.
     *
     * @param query_vec   Query embedding (may be empty → skip HNSW).
     * @param query_text  Query string    (may be empty → skip BM25).
     * @param top_k       Maximum number of results to return.
     * @param search_ef   HNSW ef parameter (candidate pool size).
     * @return            Vector of (global_id, rrf_score) pairs, descending.
     * @throws std::invalid_argument if query_vec is non-empty but wrong dim.
     */
    [[nodiscard]] std::vector<std::pair<uint32_t, float>>
    query(const std::vector<float>& query_vec,
          const std::string&        query_text,
          std::size_t               top_k     = kDefaultTopK,
          std::size_t               search_ef = kDefaultSearchEf);

    // ── Binary Persistence ────────────────────────────────────────────────────

    /** Serialize the full engine state (registry, HNSW graph, BM25, KG) to a binary stream. */
    void save(std::ostream& os) const;
    /** Replace engine state from a binary stream written by save(). */
    void load(std::istream& is);

    /** Convenience: serialize to / deserialize from a file path. */
    void save_to_file(const std::string& path) const;
    void load_from_file(const std::string& path);

    // ── Accessors ────────────────────────────────────────────────────────────

    [[nodiscard]] std::size_t          size()       const noexcept { return registry_.size(); }
    [[nodiscard]] const GlobalRegistry& registry()  const noexcept { return registry_; }
    [[nodiscard]] const LexicalIndex&  lexical()    const noexcept { return lexical_;  }
    [[nodiscard]] const KnowledgeGraph& graph()     const noexcept { return kg_;       }

    /** O(1) lookup — returns the NodeData stored in the registry for a given id. */
    [[nodiscard]] const NodeData& get_node(uint32_t id) const {
        return registry_.get_node(id);
    }

private:
    std::size_t    dim_;
    GlobalRegistry registry_;
    HNSWIndex      hnsw_;
    LexicalIndex   lexical_;
    KnowledgeGraph kg_;
};

}  // namespace trifecta

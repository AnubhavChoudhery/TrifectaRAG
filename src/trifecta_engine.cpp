/**
 * =============================================================================
 * trifecta_engine.cpp — Phase 5: Unified Orchestration Layer
 * =============================================================================
 */

#include "trifecta_engine.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace trifecta {

TrifectaEngine::TrifectaEngine(std::size_t dim,
                               std::size_t hnsw_M,
                               std::size_t ef_construction,
                               std::size_t max_elements)
    : dim_(dim),
      registry_(),
      hnsw_(dim, hnsw_M, ef_construction, max_elements),
      lexical_(),
      kg_()
{
    if (dim == 0) {
        throw std::invalid_argument("TrifectaEngine: dim must be > 0");
    }
}

uint32_t TrifectaEngine::ingest(const std::string&        text,
                                const std::vector<float>& embedding,
                                const std::string&        metadata,
                                Modality                  modality)
{
    if (!embedding.empty() && embedding.size() != dim_) {
        throw std::invalid_argument(
            "TrifectaEngine::ingest — embedding size " +
            std::to_string(embedding.size()) +
            " does not match engine dim " + std::to_string(dim_));
    }

    const uint32_t gid = registry_.add_node(modality, metadata);

    if (!embedding.empty()) {
        hnsw_.add_point(gid, embedding);
    }
    if (!text.empty()) {
        lexical_.add_document(gid, text);
    }

    return gid;
}

void TrifectaEngine::add_edge(uint32_t source_id, uint32_t target_id, EdgeType type) {
    // validate both ids exist in the registry
    if (source_id >= registry_.size()) {
        throw std::out_of_range(
            "TrifectaEngine::add_edge — source_id " +
            std::to_string(source_id) + " not yet ingested");
    }
    if (target_id >= registry_.size()) {
        throw std::out_of_range(
            "TrifectaEngine::add_edge — target_id " +
            std::to_string(target_id) + " not yet ingested");
    }
    kg_.add_edge(source_id, target_id, type);
}

std::vector<std::pair<uint32_t, float>>
TrifectaEngine::query(const std::vector<float>& query_vec,
                      const std::string&        query_text,
                      std::size_t               top_k,
                      std::size_t               search_ef) const
{
    if (query_vec.empty() && query_text.empty()) {
        return {};
    }
    if (registry_.empty()) {
        return {};
    }
    if (!query_vec.empty() && query_vec.size() != dim_) {
        throw std::invalid_argument(
            "TrifectaEngine::query — query_vec size " +
            std::to_string(query_vec.size()) +
            " does not match engine dim " + std::to_string(dim_));
    }

    const std::size_t hnsw_k = top_k * 2;

    // ── 1. Vector search ───────────────────────────────────────────────────
    std::vector<std::pair<uint32_t, float>> hnsw_results;
    if (!query_vec.empty()) {
        hnsw_results = hnsw_.search(query_vec, hnsw_k, search_ef);
    }

    // ── 2. BM25 keyword search ─────────────────────────────────────────────
    std::vector<std::pair<uint32_t, float>> bm25_results;
    if (!query_text.empty()) {
        bm25_results = lexical_.score_query(query_text);
        if (bm25_results.size() > top_k * 2) {
            bm25_results.resize(top_k * 2);
        }
    }

    // ── 3. Build seed set for KG expansion ────────────────────────────────
    std::vector<uint32_t> seed_ids;
    seed_ids.reserve(hnsw_results.size() + bm25_results.size());
    for (const auto& r : hnsw_results) seed_ids.push_back(r.first);
    for (const auto& r : bm25_results) seed_ids.push_back(r.first);
    std::sort(seed_ids.begin(), seed_ids.end());
    seed_ids.erase(std::unique(seed_ids.begin(), seed_ids.end()), seed_ids.end());

    // ── 4. KG 1-hop context expansion ─────────────────────────────────────
    const std::vector<uint32_t> kg_neighbors = kg_.bfs_one_hop_neighbors(seed_ids);

    // ── 5. Reciprocal Rank Fusion ──────────────────────────────────────────
    //   score(d) = Σ_i  1 / (kRrfK + rank_i(d))    for each list i
    //            + kKgContextMult / (kRrfK + 1)      if d ∈ KG expansion
    //
    //   rank_i is 1-indexed (first result = rank 1).
    std::unordered_map<uint32_t, float> acc;
    acc.reserve(seed_ids.size() + kg_neighbors.size());

    for (std::size_t i = 0; i < hnsw_results.size(); ++i) {
        acc[hnsw_results[i].first] +=
            1.0f / (static_cast<float>(kRrfK) + static_cast<float>(i) + 1.0f);
    }
    for (std::size_t i = 0; i < bm25_results.size(); ++i) {
        acc[bm25_results[i].first] +=
            1.0f / (static_cast<float>(kRrfK) + static_cast<float>(i) + 1.0f);
    }

    // Flat KG context boost: down-weighted so pure-context nodes don't
    // outrank directly-retrieved results but still surface in top_k.
    static constexpr float kKgScore =
        kKgContextMult / (static_cast<float>(kRrfK) + 1.0f);
    for (const uint32_t nbr : kg_neighbors) {
        acc[nbr] += kKgScore;
    }

    // ── 6. Sort and truncate ───────────────────────────────────────────────
    std::vector<std::pair<uint32_t, float>> result;
    result.reserve(acc.size());
    for (auto& p : acc) {
        result.push_back(std::move(p));
    }
    std::sort(result.begin(), result.end(),
              [](const std::pair<uint32_t, float>& a,
                 const std::pair<uint32_t, float>& b) {
                  if (a.second != b.second) return a.second > b.second;
                  return a.first < b.first;
              });

    if (result.size() > top_k) {
        result.resize(top_k);
    }
    return result;
}

}  // namespace trifecta

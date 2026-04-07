/**
 * =============================================================================
 * knowledge_graph.hpp — Phase 4: Knowledge Graph (Adjacency + BFS)
 * =============================================================================
 */

#pragma once

#include <cstdint>
#include <iosfwd>
#include <unordered_map>
#include <vector>

namespace trifecta {

enum class EdgeType : std::uint8_t {
    RELATES_TO = 0,
    EXPLAINS,
    DEPICTS,
};

struct Edge {
    uint32_t target_id;
    EdgeType type;
};

/**
 * Directed adjacency list for relational context between GlobalRegistry chunks.
 */
class KnowledgeGraph {
public:
    KnowledgeGraph() = default;

    void add_edge(uint32_t source_id, uint32_t target_id, EdgeType type);

    [[nodiscard]] const std::unordered_map<uint32_t, std::vector<Edge>>& adjacency() const noexcept {
        return adj_;
    }

    /**
     * Breadth-first expansion limited to depth 1: collect unique target global_ids
     * reachable from any seed in one hop (outgoing edges only). Result sorted
     * ascending for stable tests and cache-friendly iteration.
     */
    [[nodiscard]] std::vector<uint32_t> bfs_one_hop_neighbors(
        const std::vector<uint32_t>& seed_ids) const;

    void save(std::ostream& os) const;
    void load(std::istream& is);

private:
    std::unordered_map<uint32_t, std::vector<Edge>> adj_;
};

}  // namespace trifecta

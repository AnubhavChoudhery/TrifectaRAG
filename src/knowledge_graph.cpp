/**
 * =============================================================================
 * knowledge_graph.cpp — Phase 4: Knowledge Graph (Adjacency + BFS)
 * =============================================================================
 */

#include "knowledge_graph.hpp"
#include "serialize_utils.hpp"

#include <algorithm>
#include <queue>
#include <unordered_set>

namespace trifecta {

void KnowledgeGraph::add_edge(uint32_t source_id, uint32_t target_id, EdgeType type) {
    adj_[source_id].push_back(Edge{target_id, type});
}

std::vector<uint32_t> KnowledgeGraph::bfs_one_hop_neighbors(
    const std::vector<uint32_t>& seed_ids) const {
    std::queue<uint32_t> q;
    for (uint32_t s : seed_ids) {
        q.push(s);
    }

    std::unordered_set<uint32_t> collected;
    std::vector<uint32_t> out;

    while (!q.empty()) {
        const uint32_t u = q.front();
        q.pop();
        auto it = adj_.find(u);
        if (it == adj_.end()) {
            continue;
        }
        for (const Edge& e : it->second) {
            if (collected.insert(e.target_id).second) {
                out.push_back(e.target_id);
            }
        }
    }

    std::sort(out.begin(), out.end());
    return out;
}

// =============================================================================
// Binary persistence
// =============================================================================

void KnowledgeGraph::save(std::ostream& os) const {
    io::write_u64(os, adj_.size());
    for (const auto& entry : adj_) {
        io::write_u32(os, entry.first);
        io::write_u64(os, entry.second.size());
        for (const Edge& e : entry.second) {
            io::write_u32(os, e.target_id);
            io::write_u8(os, static_cast<uint8_t>(e.type));
        }
    }
}

void KnowledgeGraph::load(std::istream& is) {
    adj_.clear();
    const uint64_t n = io::read_u64(is);
    for (uint64_t i = 0; i < n; ++i) {
        uint32_t src = io::read_u32(is);
        const uint64_t n_edges = io::read_u64(is);
        auto& edges = adj_[src];
        edges.reserve(static_cast<std::size_t>(n_edges));
        for (uint64_t j = 0; j < n_edges; ++j) {
            uint32_t tgt = io::read_u32(is);
            auto et = static_cast<EdgeType>(io::read_u8(is));
            edges.push_back(Edge{tgt, et});
        }
    }
}

}  // namespace trifecta

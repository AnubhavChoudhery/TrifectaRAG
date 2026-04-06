/**
 * =============================================================================
 * knowledge_graph.cpp — Phase 4: Knowledge Graph (Adjacency + BFS)
 * =============================================================================
 */

#include "knowledge_graph.hpp"

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

}  // namespace trifecta

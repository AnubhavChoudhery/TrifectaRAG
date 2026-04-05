#include "knowledge_graph.hpp"

#include <queue>


void KnowledgeGraph::add_edge(uint32_t source_id,
                             uint32_t target_id,
                             EdgeType type) {
    adjacency_list_[source_id].push_back(Edge{target_id, type});
}

const std::vector<Edge>& KnowledgeGraph::get_neighbors(uint32_t node_id) const {
    static const std::vector<Edge> empty;

    auto it = adjacency_list_.find(node_id);
    if (it == adjacency_list_.end()) {
        return empty;
    }

    return it->second;
}


std::vector<uint32_t>
KnowledgeGraph::expand_1hop(const std::vector<uint32_t>& seed_ids) const {
    std::unordered_set<uint32_t> visited;
    std::vector<uint32_t> result;


    for (uint32_t id : seed_ids) {
        visited.insert(id);
    }

   
    for (uint32_t source_id : seed_ids) {
        auto it = adjacency_list_.find(source_id);
        if (it == adjacency_list_.end()) continue;

        for (const Edge& edge : it->second) {
            if (visited.insert(edge.target_id).second) {
                result.push_back(edge.target_id);
            }
        }
    }

    return result;
}
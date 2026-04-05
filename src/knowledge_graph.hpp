#ifndef KNOWLEDGE_GRAPH_HPP
#define KNOWLEDGE_GRAPH_HPP

#include <cstdint>
#include <unordered_map>
#include <vector>
#include <unordered_set>


enum class EdgeType {
    RELATES_TO,
    EXPLAINS,
    DEPICTS
};


struct Edge {
    uint32_t target_id;
    EdgeType type;
};

// ---------------- KNOWLEDGE GRAPH ----------------
class KnowledgeGraph {
public:
    KnowledgeGraph() = default;

    // Add directed edge: source -> target
    void add_edge(uint32_t source_id, uint32_t target_id, EdgeType type);

    // Get all neighbors (edges) of a node
    const std::vector<Edge>& get_neighbors(uint32_t node_id) const;

    // BFS 1-hop expansion (returns unique node IDs)
    std::vector<uint32_t> expand_1hop(const std::vector<uint32_t>& seed_ids) const;

private:
    // Adjacency list
    std::unordered_map<uint32_t, std::vector<Edge>> adjacency_list_;
};

#endif
#pragma once

#include <cstdint>
#include <iosfwd>
#include <queue>
#include <random>
#include <utility>
#include <vector>

#include "trifecta_core.hpp"

namespace trifecta {

class HNSWIndex {
public:
    HNSWIndex(size_t dim, size_t M = 16, size_t ef_construction = 200, size_t max_elements = 10000);

    void add_point(uint32_t global_id, const std::vector<float>& vec);
    std::vector<std::pair<uint32_t, float>> search(const std::vector<float>& query, size_t k, size_t ef) const;

    void save(std::ostream& os) const;
    void load(std::istream& is);

private:
    struct CompareByDistance {
        bool operator()(const std::pair<float, uint32_t>& a, const std::pair<float, uint32_t>& b) const {
            return a.first < b.first; // max priority queue (longest distance at top)
        }
    };
    
    struct CompareByDistanceMin {
        bool operator()(const std::pair<float, uint32_t>& a, const std::pair<float, uint32_t>& b) const {
            return a.first > b.first; // min priority queue (shortest distance at top)
        }
    };

    size_t dim_;
    size_t M_;
    size_t M_max_;
    size_t M_max0_;
    size_t ef_construction_;
    double mult_;
    
    std::vector<std::vector<float>> vectors_;
    std::vector<uint32_t> id_map_;
    
    int max_level_;
    uint32_t enter_point_;
    
    std::vector<std::vector<std::vector<uint32_t>>> links_; // level -> node -> links
    
    std::mt19937 generator_;
    
    int get_random_level();
    void check_dim(const std::vector<float>& vec) const;

    float get_distance(uint32_t a, uint32_t b) const;
    float get_distance(const std::vector<float>& vec, uint32_t a) const;

    std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, CompareByDistance> 
    search_layer(uint32_t ep, const std::vector<float>& query, size_t ef, int level) const;

    std::vector<uint32_t> select_neighbors(
        const std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, CompareByDistance>& candidates,
        size_t M) const;
};

} // namespace trifecta

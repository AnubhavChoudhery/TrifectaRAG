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
            return a.first < b.first;
        }
    };
    
    struct CompareByDistanceMin {
        bool operator()(const std::pair<float, uint32_t>& a, const std::pair<float, uint32_t>& b) const {
            return a.first > b.first;
        }
    };

    size_t dim_;
    size_t M_;
    size_t M_max_;
    size_t M_max0_;
    size_t ef_construction_;
    double mult_;
    
    // Flat contiguous storage: all vectors packed into one allocation.
    // vector i lives at offset i*dim_ .. (i+1)*dim_ - 1.
    std::vector<float> flat_vectors_;
    size_t             num_vectors_ = 0;

    std::vector<uint32_t> id_map_;
    
    int max_level_;
    uint32_t enter_point_;
    
    std::vector<std::vector<std::vector<uint32_t>>> links_;
    
    std::mt19937 generator_;
    
    int get_random_level();

    // Distance = 1.0 - dot(a, b).  Vectors are pre-normalized so dot == cosine.
    float get_distance(uint32_t a, uint32_t b) const;
    float get_distance(const float* query, uint32_t a) const;

    const float* vec_ptr(uint32_t internal_id) const {
        return flat_vectors_.data() + static_cast<size_t>(internal_id) * dim_;
    }

    std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, CompareByDistance> 
    search_layer(uint32_t ep, const float* query, size_t ef, int level) const;

    void prune_neighbors(uint32_t node, int level, size_t max_links);

    std::vector<uint32_t> select_neighbors(
        const std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, CompareByDistance>& candidates,
        size_t M) const;
};

} // namespace trifecta

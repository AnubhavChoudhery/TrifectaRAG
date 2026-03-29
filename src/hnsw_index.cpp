#include "hnsw_index.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <queue>
#include <unordered_set>

namespace trifecta {

HNSWIndex::HNSWIndex(size_t dim, size_t M, size_t ef_construction, size_t max_elements)
    : dim_(dim), M_(M), M_max_(M), M_max0_(M * 2),
      ef_construction_(ef_construction),
      mult_(1.0 / log(1.0 * M_)),
      max_level_(-1), enter_point_(0),
      generator_(100)
{
    vectors_.reserve(max_elements);
    id_map_.reserve(max_elements);
}

int HNSWIndex::get_random_level() {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -log(distribution(generator_)) * mult_;
    return (int)r;
}

float HNSWIndex::get_distance(uint32_t a, uint32_t b) const {
    const auto& va = vectors_[a];
    const auto& vb = vectors_[b];
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < dim_; i++) {
        dot += va[i] * vb[i];
        norm_a += va[i] * va[i];
        norm_b += vb[i] * vb[i];
    }
    float d = dot / (sqrt(norm_a) * std::sqrt(norm_b));
    return 1.0f - d; // cosine similarity to distance
}

float HNSWIndex::get_distance(const std::vector<float>& vec, uint32_t a) const {
    const auto& va = vectors_[a];
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < dim_; i++) {
        dot += vec[i] * va[i];
        norm_a += vec[i] * vec[i];
        norm_b += va[i] * va[i];
    }
    float d = dot / (sqrt(norm_a) * std::sqrt(norm_b));
    return 1.0f - d;
}

std::vector<uint32_t> HNSWIndex::select_neighbors(
    const std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, CompareByDistance>& candidates,
    size_t M) {
    auto cands = candidates;
    std::vector<std::pair<float, uint32_t>> cands_vec;
    while (!cands.empty()) {
        cands_vec.push_back(cands.top());
        cands.pop();
    }
    std::sort(cands_vec.begin(), cands_vec.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });
    std::vector<uint32_t> res;
    for (size_t i = 0; i < std::min(M, cands_vec.size()); i++) {
        res.push_back(cands_vec[i].second);
    }
    return res;
}

std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, HNSWIndex::CompareByDistance> 
HNSWIndex::search_layer(uint32_t ep, const std::vector<float>& query, size_t ef, int level) {
    std::unordered_set<uint32_t> visited;
    visited.insert(ep);
    
    std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, CompareByDistanceMin> C;
    std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, CompareByDistance> W;
    
    float d = get_distance(query, ep);
    C.push({d, ep});
    W.push({d, ep});
    
    while (!C.empty()) {
        auto c = C.top();
        C.pop();
        auto f = W.top();
        if (c.first > f.first) break;
        
        for (uint32_t n : links_[level][c.second]) {
            if (visited.find(n) == visited.end()) {
                visited.insert(n);
                f = W.top();
                float dist = get_distance(query, n);
                if (dist < f.first || W.size() < ef) {
                    C.push({dist, n});
                    W.push({dist, n});
                    if (W.size() > ef) {
                        W.pop();
                    }
                }
            }
        }
    }
    return W;
}

void HNSWIndex::add_point(uint32_t global_id, const std::vector<float>& vec) {
    uint32_t id = vectors_.size();
    vectors_.push_back(vec);
    id_map_.push_back(global_id);
    int level = get_random_level();
    
    if (vectors_.size() == 1) {
        max_level_ = level;
        enter_point_ = id;
        links_.resize(max_level_ + 1);
        for (int i = 0; i <= max_level_; i++) {
            links_[i].push_back(std::vector<uint32_t>());
        }
        return;
    }

    if (level > (int)links_.size() - 1) {
        int old_size = links_.size();
        links_.resize(level + 1);
        for (int i = old_size; i <= level; i++) {
            links_[i].resize(id + 1, std::vector<uint32_t>());
        }
    }
    
    for (int i = 0; i <= (int)links_.size() - 1; i++) {
        if (links_[i].size() <= id) {
             links_[i].resize(id + 1, std::vector<uint32_t>());
        }
    }
    
    uint32_t ep = enter_point_;
    for (int l = max_level_; l > level; l--) {
        auto W = search_layer(ep, vec, 1, l);
        ep = select_neighbors(W, 1)[0];
    }
    
    for (int l = std::min(level, max_level_); l >= 0; l--) {
        auto W = search_layer(ep, vec, ef_construction_, l);
        auto neighbors = select_neighbors(W, l == 0 ? M_max0_ : M_max_);
        links_[l][id] = neighbors;
        
        for (uint32_t n : neighbors) {
            links_[l][n].push_back(id);
            if (links_[l][n].size() > (l == 0 ? M_max0_ : M_max_)) {
                // simple truncate
                links_[l][n].pop_back();
            }
        }
        ep = W.top().second;
    }
    
    if (level > max_level_) {
        max_level_ = level;
        enter_point_ = id;
    }
}

std::vector<std::pair<uint32_t, float>> HNSWIndex::search(const std::vector<float>& query, size_t k, size_t ef) {
    if (vectors_.empty()) return {};
    uint32_t ep = enter_point_;
    for (int l = max_level_; l > 0; l--) {
        auto W = search_layer(ep, query, 1, l);
        ep = select_neighbors(W, 1)[0];
    }
    auto W = search_layer(ep, query, ef, 0);
    std::vector<std::pair<uint32_t, float>> res;
    while (!W.empty()) {
        res.push_back({id_map_[W.top().second], 1.0f - W.top().first});
        W.pop();
    }
    std::reverse(res.begin(), res.end());
    if (res.size() > k) res.resize(k);
    return res;
}

} // namespace trifecta
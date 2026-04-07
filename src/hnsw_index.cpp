#include "hnsw_index.hpp"
#include "serialize_utils.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <stdexcept>
#include <string>
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
    return static_cast<int>(r);
}

void HNSWIndex::check_dim(const std::vector<float>& vec) const {
    if (vec.size() != dim_) {
        throw std::invalid_argument(
            "HNSWIndex: expected vector dimension " + std::to_string(dim_) +
            ", got " + std::to_string(vec.size()));
    }
}

float HNSWIndex::get_distance(uint32_t a, uint32_t b) const {
    const float sim = math::cosine_similarity(vectors_[a], vectors_[b]);
    return 1.0f - sim;
}

float HNSWIndex::get_distance(const std::vector<float>& vec, uint32_t a) const {
    const float sim = math::cosine_similarity(vec, vectors_[a]);
    return 1.0f - sim;
}

std::vector<uint32_t> HNSWIndex::select_neighbors(
    const std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, CompareByDistance>& candidates,
    size_t M) const {
    auto cands = candidates;
    std::vector<std::pair<float, uint32_t>> cands_vec;
    while (!cands.empty()) {
        cands_vec.push_back(cands.top());
        cands.pop();
    }
    std::sort(cands_vec.begin(), cands_vec.end(), [](const auto& x, const auto& y) {
        return x.first < y.first;
    });
    std::vector<uint32_t> res;
    res.reserve(std::min(M, cands_vec.size()));
    for (size_t i = 0; i < std::min(M, cands_vec.size()); ++i) {
        res.push_back(cands_vec[i].second);
    }
    return res;
}

std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, HNSWIndex::CompareByDistance> 
HNSWIndex::search_layer(uint32_t ep, const std::vector<float>& query, size_t ef, int level) const {
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
    check_dim(vec);

    uint32_t id = static_cast<uint32_t>(vectors_.size());
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
        auto picked = select_neighbors(W, 1);
        if (picked.empty()) {
            break;
        }
        ep = picked[0];
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
        if (!W.empty()) {
            ep = W.top().second;
        }
    }
    
    if (level > max_level_) {
        max_level_ = level;
        enter_point_ = id;
    }
}

std::vector<std::pair<uint32_t, float>> HNSWIndex::search(const std::vector<float>& query, size_t k, size_t ef) const {
    check_dim(query);
    if (vectors_.empty()) {
        return {};
    }
    uint32_t ep = enter_point_;
    for (int l = max_level_; l > 0; l--) {
        auto W = search_layer(ep, query, 1, l);
        auto picked = select_neighbors(W, 1);
        if (picked.empty()) {
            break;
        }
        ep = picked[0];
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

// =============================================================================
// Binary persistence
// =============================================================================

void HNSWIndex::save(std::ostream& os) const {
    io::write_u64(os, dim_);
    io::write_u64(os, M_);
    io::write_u64(os, M_max_);
    io::write_u64(os, M_max0_);
    io::write_u64(os, ef_construction_);
    io::write_f64(os, mult_);
    io::write_i32(os, max_level_);
    io::write_u32(os, enter_point_);

    const uint64_t n_vecs = vectors_.size();
    io::write_u64(os, n_vecs);
    for (const auto& v : vectors_) {
        os.write(reinterpret_cast<const char*>(v.data()),
                 static_cast<std::streamsize>(dim_ * sizeof(float)));
    }

    io::write_u64(os, id_map_.size());
    if (!id_map_.empty()) {
        os.write(reinterpret_cast<const char*>(id_map_.data()),
                 static_cast<std::streamsize>(id_map_.size() * sizeof(uint32_t)));
    }

    io::write_u64(os, links_.size());
    for (const auto& level : links_) {
        io::write_u64(os, level.size());
        for (const auto& node_links : level) {
            io::write_u64(os, node_links.size());
            if (!node_links.empty()) {
                os.write(reinterpret_cast<const char*>(node_links.data()),
                         static_cast<std::streamsize>(node_links.size() * sizeof(uint32_t)));
            }
        }
    }
}

void HNSWIndex::load(std::istream& is) {
    dim_              = static_cast<size_t>(io::read_u64(is));
    M_                = static_cast<size_t>(io::read_u64(is));
    M_max_            = static_cast<size_t>(io::read_u64(is));
    M_max0_           = static_cast<size_t>(io::read_u64(is));
    ef_construction_  = static_cast<size_t>(io::read_u64(is));
    mult_             = io::read_f64(is);
    max_level_        = io::read_i32(is);
    enter_point_      = io::read_u32(is);

    const uint64_t n_vecs = io::read_u64(is);
    vectors_.clear();
    vectors_.resize(static_cast<size_t>(n_vecs));
    for (auto& v : vectors_) {
        v.resize(dim_);
        is.read(reinterpret_cast<char*>(v.data()),
                static_cast<std::streamsize>(dim_ * sizeof(float)));
    }

    const uint64_t n_ids = io::read_u64(is);
    id_map_.resize(static_cast<size_t>(n_ids));
    if (n_ids > 0) {
        is.read(reinterpret_cast<char*>(id_map_.data()),
                static_cast<std::streamsize>(n_ids * sizeof(uint32_t)));
    }

    const uint64_t n_levels = io::read_u64(is);
    links_.clear();
    links_.resize(static_cast<size_t>(n_levels));
    for (auto& level : links_) {
        const uint64_t n_nodes = io::read_u64(is);
        level.resize(static_cast<size_t>(n_nodes));
        for (auto& node_links : level) {
            const uint64_t n_nbrs = io::read_u64(is);
            node_links.resize(static_cast<size_t>(n_nbrs));
            if (n_nbrs > 0) {
                is.read(reinterpret_cast<char*>(node_links.data()),
                        static_cast<std::streamsize>(n_nbrs * sizeof(uint32_t)));
            }
        }
    }
}

} // namespace trifecta
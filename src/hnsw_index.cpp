#include "hnsw_index.hpp"
#include "serialize_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <queue>
#include <stdexcept>
#include <string>

namespace trifecta {

HNSWIndex::HNSWIndex(size_t dim, size_t M, size_t ef_construction, size_t max_elements)
    : dim_(dim), M_(M), M_max_(M), M_max0_(M * 2),
      ef_construction_(ef_construction),
      mult_(1.0 / log(1.0 * M_)),
      num_vectors_(0),
      max_level_(-1), enter_point_(0),
      generator_(100)
{
    flat_vectors_.reserve(max_elements * dim);
    id_map_.reserve(max_elements);
}

int HNSWIndex::get_random_level() {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -log(distribution(generator_)) * mult_;
    return static_cast<int>(r);
}

// Distance metric: 1 - dot(a, b).  All stored vectors are L2-normalized
// at insertion time, so dot product equals cosine similarity.  This avoids
// two sqrt() calls per distance computation vs. full cosine_similarity().
float HNSWIndex::get_distance(uint32_t a, uint32_t b) const {
    const float* pa = vec_ptr(a);
    const float* pb = vec_ptr(b);
    float dot = 0.0f;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#elif defined(_MSC_VER)
#pragma loop(ivdep)
#endif
    for (size_t i = 0; i < dim_; ++i) {
        dot += pa[i] * pb[i];
    }
    return 1.0f - dot;
}

float HNSWIndex::get_distance(const float* query, uint32_t a) const {
    const float* pa = vec_ptr(a);
    float dot = 0.0f;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#elif defined(_MSC_VER)
#pragma loop(ivdep)
#endif
    for (size_t i = 0; i < dim_; ++i) {
        dot += query[i] * pa[i];
    }
    return 1.0f - dot;
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

void HNSWIndex::prune_neighbors(uint32_t node, int level, size_t max_links) {
    auto& nbrs = links_[level][node];
    if (nbrs.size() <= max_links) return;

    // Sort by distance to node, keep the closest max_links.
    std::vector<std::pair<float, uint32_t>> scored;
    scored.reserve(nbrs.size());
    for (uint32_t n : nbrs) {
        scored.push_back({get_distance(node, n), n});
    }
    std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });
    nbrs.clear();
    nbrs.reserve(max_links);
    for (size_t i = 0; i < max_links; ++i) {
        nbrs.push_back(scored[i].second);
    }
}

std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, HNSWIndex::CompareByDistance> 
HNSWIndex::search_layer(uint32_t ep, const float* query, size_t ef, int level) const {
    // Visited set: use flat vector<bool> for O(1) lookup when node count is
    // moderate, falling back to unordered_set for very sparse graphs where
    // allocating num_vectors_ bools would waste memory.
    std::vector<bool> visited(num_vectors_, false);
    visited[ep] = true;
    
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
            if (!visited[n]) {
                visited[n] = true;
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
    if (vec.size() != dim_) {
        throw std::invalid_argument(
            "HNSWIndex: expected vector dimension " + std::to_string(dim_) +
            ", got " + std::to_string(vec.size()));
    }

    // Normalize and append to flat storage.
    size_t old_size = flat_vectors_.size();
    flat_vectors_.resize(old_size + dim_);
    float* dest = flat_vectors_.data() + old_size;
    std::memcpy(dest, vec.data(), dim_ * sizeof(float));
    math::normalize_inplace_raw(dest, dim_);

    uint32_t id = static_cast<uint32_t>(num_vectors_);
    ++num_vectors_;
    id_map_.push_back(global_id);
    int level = get_random_level();
    
    if (num_vectors_ == 1) {
        max_level_ = level;
        enter_point_ = id;
        links_.resize(max_level_ + 1);
        for (int i = 0; i <= max_level_; i++) {
            links_[i].push_back(std::vector<uint32_t>());
        }
        return;
    }

    if (level > (int)links_.size() - 1) {
        int old_lvl_size = links_.size();
        links_.resize(level + 1);
        for (int i = old_lvl_size; i <= level; i++) {
            links_[i].resize(id + 1, std::vector<uint32_t>());
        }
    }
    
    for (int i = 0; i <= (int)links_.size() - 1; i++) {
        if (links_[i].size() <= id) {
             links_[i].resize(id + 1, std::vector<uint32_t>());
        }
    }
    
    const float* query_ptr = vec_ptr(id);

    uint32_t ep = enter_point_;
    for (int l = max_level_; l > level; l--) {
        auto W = search_layer(ep, query_ptr, 1, l);
        auto picked = select_neighbors(W, 1);
        if (picked.empty()) {
            break;
        }
        ep = picked[0];
    }
    
    for (int l = std::min(level, max_level_); l >= 0; l--) {
        auto W = search_layer(ep, query_ptr, ef_construction_, l);
        size_t max_links = l == 0 ? M_max0_ : M_max_;
        auto neighbors = select_neighbors(W, max_links);
        links_[l][id] = neighbors;
        
        for (uint32_t n : neighbors) {
            links_[l][n].push_back(id);
            prune_neighbors(n, l, max_links);
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
    if (query.size() != dim_) {
        throw std::invalid_argument(
            "HNSWIndex: expected query dimension " + std::to_string(dim_) +
            ", got " + std::to_string(query.size()));
    }
    if (num_vectors_ == 0) {
        return {};
    }

    // Normalize query into a local buffer (caller's vector is const).
    std::vector<float> nq(query);
    math::normalize_inplace(nq);
    const float* query_ptr = nq.data();

    uint32_t ep = enter_point_;
    for (int l = max_level_; l > 0; l--) {
        auto W = search_layer(ep, query_ptr, 1, l);
        auto picked = select_neighbors(W, 1);
        if (picked.empty()) {
            break;
        }
        ep = picked[0];
    }
    auto W = search_layer(ep, query_ptr, ef, 0);
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

    io::write_u64(os, num_vectors_);
    if (num_vectors_ > 0) {
        os.write(reinterpret_cast<const char*>(flat_vectors_.data()),
                 static_cast<std::streamsize>(num_vectors_ * dim_ * sizeof(float)));
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

    num_vectors_ = static_cast<size_t>(io::read_u64(is));
    flat_vectors_.resize(num_vectors_ * dim_);
    if (num_vectors_ > 0) {
        is.read(reinterpret_cast<char*>(flat_vectors_.data()),
                static_cast<std::streamsize>(num_vectors_ * dim_ * sizeof(float)));
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

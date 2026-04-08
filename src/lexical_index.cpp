/**
 * =============================================================================
 * lexical_index.cpp — Phase 3: Lexical Engine (Inverted Index + BM25)
 * =============================================================================
 */

#include "lexical_index.hpp"
#include "serialize_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <unordered_map>

namespace trifecta {

namespace {

bool is_word_byte(unsigned char c) noexcept {
    return std::isalnum(c) != 0;
}

}  // namespace

std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    cur.reserve(32);

    for (unsigned char c : text) {
        if (is_word_byte(c)) {
            cur.push_back(static_cast<char>(std::tolower(c)));
        } else {
            if (!cur.empty()) {
                out.push_back(std::move(cur));
                cur.clear();
            }
        }
    }
    if (!cur.empty()) {
        out.push_back(std::move(cur));
    }
    return out;
}

void LexicalIndex::rebuild_inverted_for_term(const std::string& term) {
    auto it_tf = term_doc_tf_.find(term);
    if (it_tf == term_doc_tf_.end() || it_tf->second.empty()) {
        inverted_index_.erase(term);
        return;
    }
    std::vector<uint32_t> ids;
    ids.reserve(it_tf->second.size());
    for (const auto& p : it_tf->second) {
        ids.push_back(p.first);
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    inverted_index_[term] = std::move(ids);
}

void LexicalIndex::remove_document(uint32_t global_id) {
    auto doc_it = doc_term_tf_.find(global_id);
    if (doc_it == doc_term_tf_.end()) {
        return;
    }

    const std::uint32_t len = doc_lengths_.at(global_id);

    for (const auto& term_tf : doc_it->second) {
        const std::string& term = term_tf.first;
        auto t_it = term_doc_tf_.find(term);
        if (t_it == term_doc_tf_.end()) {
            continue;
        }
        t_it->second.erase(global_id);
        if (t_it->second.empty()) {
            term_doc_tf_.erase(t_it);
        }
    }
    inverted_dirty_ = true;

    doc_term_tf_.erase(doc_it);
    doc_lengths_.erase(global_id);
    if (document_count_ > 0) {
        --document_count_;
    }
    sum_doc_token_len_ -= len;
}

void LexicalIndex::add_document(uint32_t global_id, const std::string& text) {
    remove_document(global_id);

    const std::vector<std::string> tokens = tokenize(text);
    if (tokens.empty()) {
        return;
    }

    std::unordered_map<std::string, std::uint32_t> tf;
    tf.reserve(16);
    for (const std::string& t : tokens) {
        ++tf[t];
    }

    const std::uint32_t len = static_cast<std::uint32_t>(tokens.size());
    doc_lengths_[global_id] = len;
    doc_term_tf_[global_id] = tf;

    for (const auto& p : tf) {
        term_doc_tf_[p.first][global_id] = p.second;
    }

    inverted_dirty_ = true;
    ++document_count_;
    sum_doc_token_len_ += len;
}

void LexicalIndex::rebuild_inverted_index() {
    inverted_index_.clear();
    inverted_index_.reserve(term_doc_tf_.size());
    for (const auto& term_entry : term_doc_tf_) {
        std::vector<uint32_t> ids;
        ids.reserve(term_entry.second.size());
        for (const auto& p : term_entry.second) {
            ids.push_back(p.first);
        }
        std::sort(ids.begin(), ids.end());
        inverted_index_[term_entry.first] = std::move(ids);
    }
    inverted_dirty_ = false;
}

void LexicalIndex::ensure_index_built() {
    if (inverted_dirty_) {
        rebuild_inverted_index();
    }
}

float LexicalIndex::average_document_length() const noexcept {
    if (document_count_ == 0) {
        return 0.0f;
    }
    return static_cast<float>(sum_doc_token_len_) /
           static_cast<float>(document_count_);
}

std::uint32_t LexicalIndex::document_frequency(const std::string& term) const {
    auto it = term_doc_tf_.find(term);
    if (it == term_doc_tf_.end()) {
        return 0;
    }
    return static_cast<std::uint32_t>(it->second.size());
}

float LexicalIndex::idf(std::size_t num_docs, std::uint32_t df) noexcept {
    if (num_docs == 0 || df == 0) {
        return 0.0f;
    }
    const float N  = static_cast<float>(num_docs);
    const float nq = static_cast<float>(df);
    return std::log(1.0f + (N - nq + 0.5f) / (nq + 0.5f));
}

std::vector<std::pair<uint32_t, float>>
LexicalIndex::score_query(const std::string& query, std::size_t max_results) {
    ensure_index_built();
    const std::vector<std::string> q_tokens = tokenize(query);
    if (q_tokens.empty() || document_count_ == 0) {
        return {};
    }

    const float avgdl = average_document_length();
    if (avgdl <= 0.0f) {
        return {};
    }

    const std::size_t N = document_count_;

    std::unordered_map<uint32_t, float> acc;
    acc.reserve(32);

    for (const std::string& term : q_tokens) {
        auto t_it = term_doc_tf_.find(term);
        if (t_it == term_doc_tf_.end()) {
            continue;
        }
        const std::uint32_t df = static_cast<std::uint32_t>(t_it->second.size());
        const float idf_t      = idf(N, df);

        for (const auto& doc_tf : t_it->second) {
            const uint32_t gid = doc_tf.first;
            const std::uint32_t tf = doc_tf.second;
            const std::uint32_t dl = doc_lengths_.at(gid);
            const float len_ratio =
                static_cast<float>(dl) / avgdl;
            const float denom =
                static_cast<float>(tf) +
                k1_ * (1.0f - b_ + b_ * len_ratio);
            if (denom <= 0.0f) {
                continue;
            }
            const float score =
                idf_t * (static_cast<float>(tf) * (k1_ + 1.0f)) / denom;
            acc[gid] += score;
        }
    }

    std::vector<std::pair<uint32_t, float>> out;
    out.reserve(acc.size());
    for (auto& p : acc) {
        out.push_back(std::move(p));
    }

    auto cmp = [](const std::pair<uint32_t, float>& a,
                  const std::pair<uint32_t, float>& b) {
        if (a.second != b.second) return a.second > b.second;
        return a.first < b.first;
    };

    if (max_results > 0 && out.size() > max_results) {
        std::partial_sort(out.begin(), out.begin() + static_cast<long>(max_results),
                          out.end(), cmp);
        out.resize(max_results);
    } else {
        std::sort(out.begin(), out.end(), cmp);
    }
    return out;
}

// =============================================================================
// Binary persistence
// =============================================================================

void LexicalIndex::save(std::ostream& os) const {
    io::write_f32(os, k1_);
    io::write_f32(os, b_);
    io::write_u64(os, document_count_);
    io::write_u64(os, sum_doc_token_len_);

    io::write_u64(os, doc_lengths_.size());
    for (const auto& p : doc_lengths_) {
        io::write_u32(os, p.first);
        io::write_u32(os, p.second);
    }

    io::write_u64(os, term_doc_tf_.size());
    for (const auto& term_entry : term_doc_tf_) {
        io::write_str(os, term_entry.first);
        io::write_u64(os, term_entry.second.size());
        for (const auto& doc_tf : term_entry.second) {
            io::write_u32(os, doc_tf.first);
            io::write_u32(os, doc_tf.second);
        }
    }
}

void LexicalIndex::load(std::istream& is) {
    inverted_index_.clear();
    term_doc_tf_.clear();
    doc_term_tf_.clear();
    doc_lengths_.clear();

    k1_                = io::read_f32(is);
    b_                 = io::read_f32(is);
    document_count_    = static_cast<std::size_t>(io::read_u64(is));
    sum_doc_token_len_ = io::read_u64(is);

    const uint64_t n_docs = io::read_u64(is);
    for (uint64_t i = 0; i < n_docs; ++i) {
        uint32_t gid = io::read_u32(is);
        uint32_t len = io::read_u32(is);
        doc_lengths_[gid] = len;
    }

    const uint64_t n_terms = io::read_u64(is);
    for (uint64_t i = 0; i < n_terms; ++i) {
        std::string term = io::read_str(is);
        const uint64_t n_entries = io::read_u64(is);
        auto& doc_map = term_doc_tf_[term];
        for (uint64_t j = 0; j < n_entries; ++j) {
            uint32_t gid = io::read_u32(is);
            uint32_t tf  = io::read_u32(is);
            doc_map[gid] = tf;
            doc_term_tf_[gid][term] = tf;
        }
        rebuild_inverted_for_term(term);
    }
}

}  // namespace trifecta

#include "lexical_index.hpp"

#include <sstream>
#include <algorithm>
#include <cmath>
#include <unordered_set>

LexicalIndex::LexicalIndex()
    : total_docs_(0), avg_doc_length_(0.0f) {}


std::vector<std::string> LexicalIndex::tokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    std::string current;

    for (char c : text) {
        if (std::isalnum(c)) {
            current += std::tolower(c);
        } else {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
        }
    }

    if (!current.empty()) {
        tokens.push_back(current);
    }

    return tokens;
}

// ---------------- ADD DOCUMENT ----------------
void LexicalIndex::add_document(uint32_t global_id, const std::string& text) {
    auto tokens = tokenize(text);

    if (tokens.empty()) return;

    std::unordered_map<std::string, uint32_t> term_freq;
    std::unordered_set<std::string> seen_terms;

    for (const auto& token : tokens) {
        term_freq[token]++;
    }

    // Update inverted index + doc frequency
    for (const auto& [term, freq] : term_freq) {
        inverted_index_[term].push_back(global_id);

        if (seen_terms.insert(term).second) {
            doc_freqs_[term]++;
        }
    }

    doc_term_freqs_[global_id] = std::move(term_freq);
    doc_lengths_[global_id] = tokens.size();

    total_docs_++;
}

// ---------------- FINALIZE ----------------
void LexicalIndex::finalize() {
    if (total_docs_ == 0) return;

    uint64_t total_length = 0;
    for (const auto& [doc_id, length] : doc_lengths_) {
        total_length += length;
    }

    avg_doc_length_ = static_cast<float>(total_length) / total_docs_;
}

// ---------------- BM25 ----------------
float LexicalIndex::compute_bm25(
    uint32_t doc_id,
    const std::string& term,
    uint32_t term_freq
) const {
    auto df_it = doc_freqs_.find(term);
    if (df_it == doc_freqs_.end()) return 0.0f;

    float df = static_cast<float>(df_it->second);

    // IDF
    float idf = std::log(
        (total_docs_ - df + 0.5f) / (df + 0.5f) + 1.0f
    );

    float doc_len = static_cast<float>(doc_lengths_.at(doc_id));

    float numerator = term_freq * (k1_ + 1.0f);
    float denominator = term_freq +
        k1_ * (1.0f - b_ + b_ * (doc_len / avg_doc_length_));

    return idf * (numerator / denominator);
}

// ---------------- SEARCH ----------------
std::vector<std::pair<uint32_t, float>>
LexicalIndex::search(const std::string& query) const {
    std::vector<std::pair<uint32_t, float>> results;

    auto query_tokens = tokenize(query);
    if (query_tokens.empty()) return results;

    std::unordered_map<uint32_t, float> scores;

    for (const auto& term : query_tokens) {
        auto it = inverted_index_.find(term);
        if (it == inverted_index_.end()) continue;

        const auto& doc_list = it->second;

        for (uint32_t doc_id : doc_list) {
            auto tf_it = doc_term_freqs_.at(doc_id).find(term);
            if (tf_it == doc_term_freqs_.at(doc_id).end()) continue;

            uint32_t tf = tf_it->second;

            float score = compute_bm25(doc_id, term, tf);
            scores[doc_id] += score;
        }
    }

    // Convert to vector
    for (const auto& [doc_id, score] : scores) {
        results.emplace_back(doc_id, score);
    }

    // Sort descending
    std::sort(results.begin(), results.end(),
        [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

    return results;
}
/**
 * =============================================================================
 * lexical_index.hpp — Phase 3: Lexical Engine (Inverted Index + BM25)
 * =============================================================================
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trifecta {

/**
 * Tokenize UTF-8 text as ASCII words: lowercase; split on whitespace and
 * non-alphanumeric delimiters. Contiguous [0-9A-Za-z] runs become tokens.
 */
[[nodiscard]] std::vector<std::string> tokenize(const std::string& text);

/**
 * LexicalIndex — inverted index with BM25 scoring (Okapi-style IDF).
 */
class LexicalIndex {
public:
    static constexpr float kDefaultK1 = 1.5f;
    static constexpr float kDefaultB  = 0.75f;

    LexicalIndex() = default;

    /** Ingest one document keyed by registry global_id. Replaces prior text for the same id. */
    void add_document(uint32_t global_id, const std::string& text);

    /** Remove a document from the index if present. */
    void remove_document(uint32_t global_id);

    [[nodiscard]] std::size_t document_count() const noexcept { return document_count_; }
    [[nodiscard]] float       average_document_length() const noexcept;

    /** Document frequency: number of distinct documents containing `term`. */
    [[nodiscard]] std::uint32_t document_frequency(const std::string& term) const;

    /** Core inverted index: token -> sorted unique global_ids containing the term.
     *  Triggers a lazy rebuild if the index is dirty. */
    [[nodiscard]] const std::unordered_map<std::string, std::vector<uint32_t>>& inverted_index() {
        ensure_index_built();
        return inverted_index_;
    }

    /**
     * BM25 scores for documents matching at least one query token.
     * Results sorted descending by score, then ascending by global_id.
     * Lazily rebuilds the inverted index if dirty (deferred from add_document).
     * @param max_results  0 = return all, >0 = return at most this many.
     */
    [[nodiscard]] std::vector<std::pair<uint32_t, float>>
    score_query(const std::string& query, std::size_t max_results = 0);

    void set_bm25_params(float k1, float b) noexcept {
        k1_ = k1;
        b_  = b;
    }

    void save(std::ostream& os) const;
    void load(std::istream& is);

private:
    void rebuild_inverted_index();
    void rebuild_inverted_for_term(const std::string& term);
    void ensure_index_built();

    static float idf(std::size_t num_docs, std::uint32_t df) noexcept;

    bool inverted_dirty_ = false;
    std::unordered_map<std::string, std::vector<uint32_t>> inverted_index_;

    /** term -> (global_id -> term frequency in that document). */
    std::unordered_map<std::string, std::unordered_map<uint32_t, std::uint32_t>> term_doc_tf_;

    /** global_id -> per-term tf for efficient removal / replace. */
    std::unordered_map<uint32_t, std::unordered_map<std::string, std::uint32_t>> doc_term_tf_;

    std::unordered_map<uint32_t, std::uint32_t> doc_lengths_;

    std::size_t document_count_      = 0;
    std::uint64_t sum_doc_token_len_   = 0;

    float k1_ = kDefaultK1;
    float b_  = kDefaultB;
};

}  // namespace trifecta

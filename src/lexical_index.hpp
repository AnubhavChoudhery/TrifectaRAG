#ifndef LEXICAL_INDEX_HPP
#define LEXICAL_INDEX_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

class LexicalIndex {
public:
    LexicalIndex();

    // Add a document with a global ID
    void add_document(uint32_t global_id, const std::string& text);

    // Finalize stats after indexing all docs
    void finalize();

    // BM25 search
    std::vector<std::pair<uint32_t, float>> search(const std::string& query) const;

private:
    // Tokenizer
    std::vector<std::string> tokenize(const std::string& text) const;

    // BM25 scoring helper
    float compute_bm25(
        uint32_t doc_id,
        const std::string& term,
        uint32_t term_freq
    ) const;

private:
    // Inverted index: term -> list of doc IDs
    std::unordered_map<std::string, std::vector<uint32_t>> inverted_index_;

    // Term frequencies per document: doc_id -> (term -> freq)
    std::unordered_map<uint32_t, std::unordered_map<std::string, uint32_t>> doc_term_freqs_;

    // Document lengths: doc_id -> length
    std::unordered_map<uint32_t, uint32_t> doc_lengths_;

    // Document frequency: term -> number of docs containing term
    std::unordered_map<std::string, uint32_t> doc_freqs_;

    // Stats
    uint32_t total_docs_;
    float avg_doc_length_;

    // BM25 params
    const float k1_ = 1.5f;
    const float b_  = 0.75f;
};

#endif
/**
 * Phase 3 — Lexical Index (BM25) tests
 */

#include <lexical_index.hpp>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define REQUIRE(cond)                                                          \
    do {                                                                       \
        if (!(cond)) {                                                         \
            throw std::runtime_error("REQUIRE failed: " #cond);               \
        }                                                                      \
    } while (false)

static int g_tests_run    = 0;
static int g_tests_passed = 0;

#define TEST(name)                                                           \
    static void name();                                                      \
    struct _Reg_##name {                                                     \
        _Reg_##name() { run_test(#name, name); }                             \
    } _reg_##name;                                                           \
    static void name()

static void run_test(const char* name, void (*fn)()) {
    ++g_tests_run;
    try {
        fn();
        ++g_tests_passed;
        std::cout << "  [PASS] " << name << "\n";
    } catch (const std::exception& e) {
        std::cout << "  [FAIL] " << name << " — " << e.what() << "\n";
    } catch (...) {
        std::cout << "  [FAIL] " << name << " — unknown exception\n";
    }
}

// -----------------------------------------------------------------------------
TEST(lexical_basic_indexing_and_search) {
    LexicalIndex index;

    index.add_document(1, "the quick brown fox");
    index.add_document(2, "quick brown");
    index.add_document(3, "lazy dog");

    index.finalize();

    auto results = index.search("quick");

    REQUIRE(!results.empty());
    REQUIRE(results[0].first == 2u || results[0].first == 1u);
}
// -----------------------------------------------------------------------------
TEST(lexical_empty_index_returns_empty) {
    LexicalIndex index;
    index.finalize();

    auto results = index.search("anything");
    REQUIRE(results.empty());
}


TEST(lexical_single_document_matches_itself) {
    LexicalIndex index;

    index.add_document(42, "machine learning system");
    index.finalize();

    auto results = index.search("machine");

    REQUIRE(results.size() == 1u);
    REQUIRE(results[0].first == 42u);
}


TEST(lexical_no_matching_terms_returns_empty) {
    LexicalIndex index;

    index.add_document(1, "apple banana");
    index.add_document(2, "orange mango");

    index.finalize();

    auto results = index.search("car engine");

    REQUIRE(results.empty());
}


TEST(lexical_case_insensitivity) {
    LexicalIndex index;

    index.add_document(1, "Machine Learning");
    index.finalize();

    auto results = index.search("machine");

    REQUIRE(!results.empty());
    REQUIRE(results[0].first == 1u);
}


TEST(lexical_multiple_terms_accumulate_score) {
    LexicalIndex index;

    index.add_document(1, "deep learning neural network");
    index.add_document(2, "deep neural");
    index.add_document(3, "network only");

    index.finalize();

    auto results = index.search("deep neural network");

    REQUIRE(results.size() >= 2u);
    REQUIRE(results[0].second >= results[1].second);
}

// -----------------------------------------------------------------------------
TEST(lexical_results_sorted_by_score_nonincreasing) {
    LexicalIndex index;

    index.add_document(1, "quick fox");
    index.add_document(2, "quick");
    index.add_document(3, "fox");

    index.finalize();

    auto results = index.search("quick fox");

    REQUIRE(results.size() == 3u);

    for (std::size_t i = 1; i < results.size(); ++i) {
        REQUIRE(results[i - 1].second >= results[i].second);
    }
}


TEST(lexical_repeated_terms_increase_score) {
    LexicalIndex index;

    index.add_document(1, "word word word");
    index.add_document(2, "word");

    index.finalize();

    auto results = index.search("word");

    REQUIRE(results.size() == 2u);
    REQUIRE(results[0].first == 1u); // higher TF should rank higher
}


TEST(lexical_handles_punctuation) {
    LexicalIndex index;

    index.add_document(1, "hello, world!");
    index.finalize();

    auto results = index.search("hello");

    REQUIRE(!results.empty());
    REQUIRE(results[0].first == 1u);
}


int main() {
    std::cout << "\n=== Trifecta Phase 3 — Lexical BM25 tests ===\n\n";
    std::cout << "────────────────────────────────────────────\n";
    std::cout << "Results: " << g_tests_passed << " / " << g_tests_run << " passed.\n";

    if (g_tests_passed != g_tests_run) {
        std::cout << (g_tests_run - g_tests_passed) << " FAILURE(S).\n\n";
        return 1;
    }

    std::cout << "ALL TESTS PASSED.\n\n";
    return 0;
}
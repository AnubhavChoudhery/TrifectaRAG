/**
 * Phase 3 — LexicalIndex (tokenizer, inverted index, BM25)
 * Phase 4 — KnowledgeGraph (adjacency, BFS one-hop)
 */

#include <knowledge_graph.hpp>
#include <lexical_index.hpp>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#define REQUIRE(cond)                                                          \
    do {                                                                       \
        if (!(cond)) {                                                         \
            throw std::runtime_error("REQUIRE failed: " #cond);                \
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

static void assert_near(float a, float b, float eps, const char* msg) {
    if (std::fabs(a - b) > eps) {
        throw std::runtime_error(std::string(msg) + " got " + std::to_string(a) +
                                 " expected " + std::to_string(b));
    }
}

using trifecta::EdgeType;
using trifecta::KnowledgeGraph;
using trifecta::LexicalIndex;
using trifecta::tokenize;

// -----------------------------------------------------------------------------
TEST(tokenize_lowercase_splits_punctuation) {
    auto t = tokenize("Hello, world! 42x");
    REQUIRE(t.size() == 3u);
    REQUIRE(t[0] == "hello");
    REQUIRE(t[1] == "world");
    REQUIRE(t[2] == "42x");
}

TEST(tokenize_empty_and_whitespace) {
    REQUIRE(tokenize("").empty());
    REQUIRE(tokenize("   \t\n").empty());
}

// -----------------------------------------------------------------------------
TEST(lexical_inverted_index_and_df_avgdl) {
    LexicalIndex idx;
    idx.add_document(0u, "the quick brown");
    idx.add_document(1u, "the lazy dog");

    REQUIRE(idx.document_count() == 2u);
    assert_near(idx.average_document_length(), 3.0f, 1e-5f, "avgdl");

    REQUIRE(idx.document_frequency("the") == 2u);
    REQUIRE(idx.document_frequency("quick") == 1u);
    REQUIRE(idx.document_frequency("missing") == 0u);

    const auto& inv = idx.inverted_index();
    REQUIRE(inv.at("the").size() == 2u);
    std::unordered_set<uint32_t> the_docs(inv.at("the").begin(), inv.at("the").end());
    REQUIRE(the_docs.count(0u) == 1u);
    REQUIRE(the_docs.count(1u) == 1u);
}

TEST(lexical_replace_document_updates_stats) {
    LexicalIndex idx;
    idx.add_document(5u, "a b c");
    REQUIRE(idx.document_count() == 1u);
    REQUIRE(idx.document_frequency("a") == 1u);

    idx.add_document(5u, "a");
    REQUIRE(idx.document_count() == 1u);
    REQUIRE(idx.document_frequency("b") == 0u);
    REQUIRE(idx.document_frequency("a") == 1u);
    assert_near(idx.average_document_length(), 1.0f, 1e-5f, "avgdl after replace");

    idx.remove_document(5u);
    REQUIRE(idx.document_count() == 0u);
    REQUIRE(idx.inverted_index().empty());
}

TEST(bm25_empty_query_and_corpus) {
    LexicalIndex idx;
    REQUIRE(idx.score_query("").empty());
    REQUIRE(idx.score_query("hello").empty());

    idx.add_document(0u, "hello world");
    REQUIRE(idx.score_query("").empty());
}

TEST(bm25_scores_ordering_and_positive) {
    LexicalIndex idx;
    idx.add_document(0u, "alpha beta gamma");
    idx.add_document(1u, "alpha alpha beta");
    idx.add_document(2u, "delta");

    auto r = idx.score_query("alpha");
    REQUIRE(r.size() == 2u);
    REQUIRE(r[0].second >= r[1].second);
    REQUIRE(r[0].second > 0.0f);
    REQUIRE(r[1].second > 0.0f);
    std::unordered_set<uint32_t> ids;
    ids.insert(r[0].first);
    ids.insert(r[1].first);
    REQUIRE(ids.count(0u) == 1u);
    REQUIRE(ids.count(1u) == 1u);
}

TEST(bm25_golden_single_document) {
    LexicalIndex idx;
    idx.add_document(7u, "test test");
    REQUIRE(idx.document_count() == 1u);
    assert_near(idx.average_document_length(), 2.0f, 1e-5f, "avgdl");

    const float N  = 1.0f;
    const float df = 1.0f;
    const float idf =
        std::log(1.0f + (N - df + 0.5f) / (df + 0.5f));
    const float k1 = 1.5f;
    const float b  = 0.75f;
    const float tf = 2.0f;
    const float dl = 2.0f;
    const float avgdl = 2.0f;
    const float denom =
        tf + k1 * (1.0f - b + b * (dl / avgdl));
    const float expected = idf * (tf * (k1 + 1.0f)) / denom;

    auto r = idx.score_query("test");
    REQUIRE(r.size() == 1u);
    REQUIRE(r[0].first == 7u);
    assert_near(r[0].second, expected, 1e-4f, "bm25 golden");
}

TEST(bm25_default_k1_b_used) {
    LexicalIndex idx;
    REQUIRE(idx.document_count() == 0u);
    // Defaults exposed on class; scoring path uses k1_/b_ (verified indirectly via
    // stable ordering vs extreme params below).
    idx.add_document(0u, "term");
    idx.add_document(1u, "term term");
    auto base = idx.score_query("term");
    REQUIRE(base.size() == 2u);

    idx.set_bm25_params(3.0f, 0.0f);
    auto t = idx.score_query("term");
    REQUIRE(t.size() == 2u);
    REQUIRE(t[0].second != base[0].second || t[1].second != base[1].second);
}

// -----------------------------------------------------------------------------
TEST(graph_add_edge_and_adjacency) {
    KnowledgeGraph g;
    g.add_edge(0u, 10u, EdgeType::RELATES_TO);
    g.add_edge(0u, 11u, EdgeType::EXPLAINS);
    const auto& a = g.adjacency();
    REQUIRE(a.at(0u).size() == 2u);
    REQUIRE(a.at(0u)[0].target_id == 10u);
    REQUIRE(a.at(0u)[0].type == EdgeType::RELATES_TO);
    REQUIRE(a.at(0u)[1].target_id == 11u);
    REQUIRE(a.at(0u)[1].type == EdgeType::EXPLAINS);
}

TEST(graph_bfs_one_hop_union_and_dedupe) {
    KnowledgeGraph g;
    g.add_edge(0u, 1u, EdgeType::RELATES_TO);
    g.add_edge(0u, 2u, EdgeType::DEPICTS);
    g.add_edge(3u, 2u, EdgeType::EXPLAINS);
    g.add_edge(3u, 2u, EdgeType::RELATES_TO);

    auto n = g.bfs_one_hop_neighbors(std::vector<uint32_t>{0u, 3u});
    REQUIRE(n.size() == 2u);
    REQUIRE(n[0] == 1u);
    REQUIRE(n[1] == 2u);
}

TEST(graph_bfs_missing_seed_no_out_edges) {
    KnowledgeGraph g;
    g.add_edge(1u, 2u, EdgeType::RELATES_TO);
    auto n = g.bfs_one_hop_neighbors(std::vector<uint32_t>{99u});
    REQUIRE(n.empty());
}

TEST(graph_bfs_empty_seeds) {
    KnowledgeGraph g;
    g.add_edge(0u, 1u, EdgeType::RELATES_TO);
    auto n = g.bfs_one_hop_neighbors({});
    REQUIRE(n.empty());
}

// -----------------------------------------------------------------------------
int main() {
    std::cout << "Phase 3 & 4 tests\n";
    std::cout << "Summary: " << g_tests_passed << " / " << g_tests_run << " passed\n";
    return (g_tests_passed == g_tests_run) ? 0 : 1;
}

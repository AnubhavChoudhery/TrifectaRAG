/**
 * Phase 2 — HNSW index tests (approximate NN; smoke + API + small exact checks).
 */

#include <hnsw_index.hpp>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <trifecta_core.hpp>

/** Release-safe check (assert is a no-op when NDEBUG is set). */
#define REQUIRE(cond)                                                          \
    do {                                                                       \
        if (!(cond)) {                                                         \
            throw std::runtime_error("REQUIRE failed: " #cond);              \
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

static void assert_near(float actual, float expected, float tol, const char* msg) {
    if (std::fabs(actual - expected) > tol) {
        throw std::runtime_error(
            std::string(msg) + ": got " + std::to_string(actual) +
            " expected " + std::to_string(expected));
    }
}

using trifecta::HNSWIndex;
using trifecta::math::cosine_similarity;

// -----------------------------------------------------------------------------
TEST(hnsw_basic_three_points) {
    HNSWIndex index(4, 16, 200, 100);
    std::vector<float> v1 = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> v2 = {0.9f, 0.1f, 0.0f, 0.0f};
    std::vector<float> v3 = {0.0f, 0.0f, 1.0f, 0.0f};

    index.add_point(1, v1);
    index.add_point(2, v2);
    index.add_point(3, v3);

    auto results = index.search(v1, 2, 50);
    REQUIRE(results.size() >= 2);
    REQUIRE(results[0].first == 1u);
}

// -----------------------------------------------------------------------------
TEST(hnsw_empty_index_returns_empty) {
    HNSWIndex index(4, 16, 200, 100);
    std::vector<float> q = {1.0f, 0.0f, 0.0f, 0.0f};
    auto r = index.search(q, 5, 50);
    REQUIRE(r.empty());
}

// -----------------------------------------------------------------------------
TEST(hnsw_single_point_returns_self) {
    HNSWIndex index(3, 8, 100, 50);
    std::vector<float> v = {0.1f, 0.2f, 0.3f};
    index.add_point(42, v);
    auto r = index.search(v, 3, 20);
    REQUIRE(r.size() == 1u);
    REQUIRE(r[0].first == 42u);
    assert_near(r[0].second, 1.0f, 1e-5f, "self-similarity");
}

// -----------------------------------------------------------------------------
TEST(hnsw_k_larger_than_count_caps_results) {
    HNSWIndex index(4, 16, 200, 100);
    index.add_point(10, {1.0f, 0.0f, 0.0f, 0.0f});
    index.add_point(20, {0.0f, 1.0f, 0.0f, 0.0f});
    auto r = index.search({1.0f, 0.0f, 0.0f, 0.0f}, 100, 50);
    REQUIRE(r.size() == 2u);
}

// -----------------------------------------------------------------------------
TEST(hnsw_add_point_wrong_dimension_throws) {
    HNSWIndex index(4, 16, 200, 100);
    bool threw = false;
    try {
        index.add_point(1, {1.0f, 0.0f}); // wrong dim
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    REQUIRE(threw);
}

// -----------------------------------------------------------------------------
TEST(hnsw_search_wrong_dimension_throws) {
    HNSWIndex index(4, 16, 200, 100);
    index.add_point(1, {1.0f, 0.0f, 0.0f, 0.0f});
    bool threw = false;
    try {
        static_cast<void>(index.search({1.0f, 0.0f}, 1, 10));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    REQUIRE(threw);
}

// -----------------------------------------------------------------------------
/** Brute-force argmax cosine similarity; returns internal index 0..n-1. */
static std::size_t brute_force_nearest(
    const std::vector<float>& query,
    const std::vector<std::vector<float>>& points) {
    std::size_t best_i = 0;
    float best_sim     = -2.0f;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const float s = cosine_similarity(query, points[i]);
        if (s > best_sim) {
            best_sim = s;
            best_i   = i;
        }
    }
    return best_i;
}

// -----------------------------------------------------------------------------
TEST(hnsw_small_graph_matches_brute_force_top1) {
    constexpr int dim = 6;
    constexpr int n   = 12;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<std::vector<float>> points;
    points.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        std::vector<float> v(static_cast<std::size_t>(dim));
        for (int j = 0; j < dim; ++j) {
            v[static_cast<std::size_t>(j)] = dist(rng);
        }
        points.push_back(std::move(v));
    }

    HNSWIndex index(static_cast<std::size_t>(dim), 16, 200, 5000);
    for (int i = 0; i < n; ++i) {
        index.add_point(static_cast<std::uint32_t>(i), points[static_cast<std::size_t>(i)]);
    }

    for (int q = 0; q < n; q += 3) {
        const auto& query = points[static_cast<std::size_t>(q)];
        const std::size_t exact = brute_force_nearest(query, points);
        auto nn = index.search(query, 1, 200);
        REQUIRE(!nn.empty());
        REQUIRE(nn[0].first == static_cast<std::uint32_t>(exact));
    }
}

// -----------------------------------------------------------------------------
TEST(hnsw_results_sorted_by_similarity_nonincreasing) {
    HNSWIndex index(4, 16, 200, 200);
    index.add_point(1, {1.0f, 0.0f, 0.0f, 0.0f});
    index.add_point(2, {0.9f, 0.1f, 0.0f, 0.0f});
    index.add_point(3, {0.0f, 0.0f, 1.0f, 0.0f});
    auto r = index.search({1.0f, 0.0f, 0.0f, 0.0f}, 3, 80);
    REQUIRE(r.size() == 3u);
    for (std::size_t i = 1; i < r.size(); ++i) {
        REQUIRE(r[i - 1].second >= r[i].second);
    }
}

// -----------------------------------------------------------------------------
int main() {
    std::cout << "\n=== Trifecta Phase 2 — HNSW tests ===\n\n";
    std::cout << "─────────────────────────────────────\n";
    std::cout << "Results: " << g_tests_passed << " / " << g_tests_run << " passed.\n";
    if (g_tests_passed != g_tests_run) {
        std::cout << (g_tests_run - g_tests_passed) << " FAILURE(S).\n\n";
        return 1;
    }
    std::cout << "ALL TESTS PASSED.\n\n";
    return 0;
}

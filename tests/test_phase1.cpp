/**
 * =============================================================================
 * tests/test_phase1.cpp — Phase 1 Unit Tests
 * =============================================================================
 *
 * Validates:
 *   1. GlobalRegistry  — node ingestion, id assignment, bounds checking
 *   2. math::cosine_similarity — correctness + edge cases
 *   3. math::dot_product, l2_norm, normalize_inplace, l2_distance_sq
 *
 * Self-contained: no test framework dependency. Uses assert() + a tiny
 * helper macro. Compile and run:
 *
 *   g++ -std=c++17 -O2 -march=native -Isrc \
 *       tests/test_phase1.cpp src/trifecta_core.cpp -o test_phase1
 *   ./test_phase1
 *
 * =============================================================================
 */

#include <trifecta_core.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

// Suppress "set but not used" on variables whose sole purpose is assertion.
#define USE(x) static_cast<void>(x)

// ── Minimal test runner ──────────────────────────────────────────────────────

static int g_tests_run    = 0;
static int g_tests_passed = 0;

#define TEST(name)                                                           \
    static void name();                                                      \
    struct _Reg_##name {                                                     \
        _Reg_##name() { run_test(#name, name); }                             \
    } _reg_##name;                                                           \
    static void name()

static void run_test(const char* name, void(*fn)()) {
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

// Helper: check that |actual - expected| < tolerance
static void assert_near(float actual, float expected,
                        float tol, const char* msg) {
    if (std::fabs(actual - expected) > tol) {
        throw std::runtime_error(
            std::string(msg) + ": got " + std::to_string(actual) +
            " expected " + std::to_string(expected));
    }
}

using namespace trifecta;

// =============================================================================
// GlobalRegistry tests
// =============================================================================

TEST(registry_starts_empty) {
    GlobalRegistry reg;
    assert(reg.empty());
    assert(reg.size() == 0);
}

TEST(registry_add_node_assigns_sequential_ids) {
    GlobalRegistry reg;
    const uint32_t id0 = reg.add_node(Modality::TEXT,  "doc:0");
    const uint32_t id1 = reg.add_node(Modality::IMAGE, "img:1");
    const uint32_t id2 = reg.add_node(Modality::TEXT,  "doc:2");
    USE(id0); USE(id1); USE(id2);

    assert(id0 == 0);
    assert(id1 == 1);
    assert(id2 == 2);
    assert(reg.size() == 3);
}

TEST(registry_get_node_returns_correct_data) {
    GlobalRegistry reg;
    static_cast<void>(reg.add_node(Modality::TEXT,  "source: arxiv, chunk: 5"));
    static_cast<void>(reg.add_node(Modality::IMAGE, "file: fig1.png"));

    const NodeData& n0 = reg.get_node(0); USE(n0);
    assert(n0.global_id == 0);
    assert(n0.type == Modality::TEXT);
    assert(n0.metadata_payload == "source: arxiv, chunk: 5");
    assert(n0.is_text());
    assert(!n0.is_image());

    const NodeData& n1 = reg.get_node(1); USE(n1);
    assert(n1.global_id == 1);
    assert(n1.type == Modality::IMAGE);
    assert(n1.is_image());
}

TEST(registry_get_node_throws_on_invalid_id) {
    GlobalRegistry reg;
    static_cast<void>(reg.add_node(Modality::TEXT, "only node"));

    bool threw = false; USE(threw);
    try {
        static_cast<void>(reg.get_node(999));
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw && "get_node should throw std::out_of_range for invalid id");
}

TEST(registry_try_get_node_safe_access) {
    GlobalRegistry reg;
    static_cast<void>(reg.add_node(Modality::TEXT, "hello"));

    const auto found     = reg.try_get_node(0);  USE(found);
    const auto not_found = reg.try_get_node(42); USE(not_found);

    assert(found.has_value());
    assert(!not_found.has_value());
    assert(found->get().global_id == 0);
}

TEST(registry_reserve_does_not_change_size) {
    GlobalRegistry reg;
    reg.reserve(1'000'000);
    assert(reg.size() == 0);
    assert(reg.empty());
}

TEST(registry_raw_nodes_reflects_all_nodes) {
    GlobalRegistry reg;
    for (int i = 0; i < 10; ++i) {
        static_cast<void>(reg.add_node(Modality::TEXT, "chunk:" + std::to_string(i)));
    }
    const auto& nodes = reg.raw_nodes(); USE(nodes);
    assert(nodes.size() == 10);
    for (uint32_t i = 0; i < 10; ++i) {
        assert(nodes[i].global_id == i);
    }
}

TEST(modality_to_str_works) {
    assert(std::string(modality_to_str(Modality::TEXT))  == "TEXT");
    assert(std::string(modality_to_str(Modality::IMAGE)) == "IMAGE");
}

// =============================================================================
// math::l2_norm tests
// =============================================================================

TEST(l2_norm_unit_vector) {
    // ||[1,0,0,0]|| == 1.0
    std::vector<float> v = {1.0f, 0.0f, 0.0f, 0.0f};
    assert_near(math::l2_norm(v), 1.0f, 1e-6f, "l2_norm unit vec");
}

TEST(l2_norm_pythagorean_triple) {
    // ||[3,4]|| == 5
    std::vector<float> v = {3.0f, 4.0f};
    assert_near(math::l2_norm(v), 5.0f, 1e-5f, "l2_norm pythagorean");
}

TEST(l2_norm_zero_vector) {
    std::vector<float> z = {0.0f, 0.0f, 0.0f};
    assert_near(math::l2_norm(z), 0.0f, 1e-10f, "l2_norm zero");
}

TEST(l2_norm_large_dimension) {
    // 1536-dim all-ones vector: norm == sqrt(1536)
    const int DIM = 1536;
    std::vector<float> v(DIM, 1.0f);
    float expected = std::sqrt(static_cast<float>(DIM));
    assert_near(math::l2_norm(v), expected, 1e-3f, "l2_norm 1536-dim");
}

// =============================================================================
// math::dot_product tests
// =============================================================================

TEST(dot_product_orthogonal_vectors) {
    std::vector<float> a = {1.0f, 0.0f};
    std::vector<float> b = {0.0f, 1.0f};
    assert_near(math::dot_product(a, b), 0.0f, 1e-7f, "dot orthogonal");
}

TEST(dot_product_parallel_vectors) {
    std::vector<float> a = {2.0f, 3.0f};
    std::vector<float> b = {2.0f, 3.0f};
    // dot([2,3],[2,3]) = 4+9 = 13
    assert_near(math::dot_product(a, b), 13.0f, 1e-5f, "dot parallel");
}

TEST(dot_product_size_mismatch_throws) {
    std::vector<float> a = {1.0f, 2.0f};
    std::vector<float> b = {1.0f};
    bool threw = false; USE(threw);
    try { static_cast<void>(math::dot_product(a, b)); } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
}

// =============================================================================
// math::cosine_similarity tests
// =============================================================================

TEST(cosine_similarity_identical_vectors) {
    // cos(v, v) == 1.0 for any non-zero v
    std::vector<float> v = {0.1f, 0.5f, -0.3f, 0.9f, 0.2f};
    assert_near(math::cosine_similarity(v, v), 1.0f, 1e-6f,
                "cosine identical");
}

TEST(cosine_similarity_opposite_vectors) {
    // cos(v, -v) == -1.0
    std::vector<float> v  = {1.0f,  2.0f, -1.0f};
    std::vector<float> nv = {-1.0f, -2.0f,  1.0f};
    assert_near(math::cosine_similarity(v, nv), -1.0f, 1e-6f,
                "cosine opposite");
}

TEST(cosine_similarity_orthogonal_vectors) {
    // cos(e1, e2) == 0.0
    std::vector<float> e1 = {1.0f, 0.0f, 0.0f};
    std::vector<float> e2 = {0.0f, 1.0f, 0.0f};
    assert_near(math::cosine_similarity(e1, e2), 0.0f, 1e-7f,
                "cosine orthogonal");
}

TEST(cosine_similarity_45_degree_vectors) {
    // cos([1,0], [1,1]/sqrt(2)) == 1/sqrt(2) ≈ 0.7071
    std::vector<float> a = {1.0f, 0.0f};
    std::vector<float> b = {1.0f, 1.0f};   // not normalized — cosine handles it
    float expected = 1.0f / std::sqrt(2.0f);
    assert_near(math::cosine_similarity(a, b), expected, 1e-6f,
                "cosine 45deg");
}

TEST(cosine_similarity_zero_vector_returns_zero) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> z = {0.0f, 0.0f, 0.0f};
    // Degenerate case: should return 0, not NaN
    const float result = math::cosine_similarity(a, z); USE(result);
    assert(result == 0.0f && "zero-norm vector should return 0.0f");
}

TEST(cosine_similarity_high_dim) {
    // 768-dim: cos(v, v) == 1.0
    const int DIM = 768;
    std::vector<float> v(DIM);
    for (int i = 0; i < DIM; ++i) v[i] = static_cast<float>(i + 1) * 0.001f;
    assert_near(math::cosine_similarity(v, v), 1.0f, 1e-5f,
                "cosine 768-dim self");
}

TEST(cosine_similarity_result_clamped) {
    // Verify result never exceeds [-1, 1] even for numerically tricky inputs.
    std::vector<float> a(128, 1e-20f);
    std::vector<float> b(128, 1e-20f);
    const float result = math::cosine_similarity(a, b); USE(result);
    assert(result >= -1.0f && result <= 1.0f && "result must be clamped");
}

TEST(cosine_similarity_size_mismatch_throws) {
    std::vector<float> a(4, 1.0f);
    std::vector<float> b(5, 1.0f);
    bool threw = false; USE(threw);
    try { static_cast<void>(math::cosine_similarity(a, b)); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
}

// =============================================================================
// math::normalize_inplace tests
// =============================================================================

TEST(normalize_inplace_produces_unit_vector) {
    std::vector<float> v = {3.0f, 4.0f};
    math::normalize_inplace(v);
    assert_near(math::l2_norm(v), 1.0f, 1e-6f, "normalize unit norm");
    assert_near(v[0], 0.6f, 1e-6f, "normalize x component");
    assert_near(v[1], 0.8f, 1e-6f, "normalize y component");
}

TEST(normalize_inplace_zero_vector_no_crash) {
    // Should be a no-op — no NaN, no divide-by-zero.
    std::vector<float> z = {0.0f, 0.0f, 0.0f};
    math::normalize_inplace(z);
    assert(z[0] == 0.0f && z[1] == 0.0f && z[2] == 0.0f);
}

TEST(normalized_dot_equals_cosine) {
    // After normalization, dot_product should equal cosine_similarity
    std::vector<float> a = {1.0f, 2.0f, -1.0f, 0.5f};
    std::vector<float> b = {0.3f, -1.0f, 2.0f, 1.5f};

    float cos_before = math::cosine_similarity(a, b);

    math::normalize_inplace(a);
    math::normalize_inplace(b);

    float dot_after = math::dot_product(a, b);
    assert_near(dot_after, cos_before, 1e-5f,
                "dot(normalized) == cosine(original)");
}

// =============================================================================
// math::l2_distance_sq tests
// =============================================================================

TEST(l2_distance_sq_same_point_is_zero) {
    std::vector<float> v = {1.0f, 2.0f, 3.0f};
    assert_near(math::l2_distance_sq(v, v), 0.0f, 1e-7f, "l2_dist same");
}

TEST(l2_distance_sq_known_value) {
    // dist_sq([0,0], [3,4]) == 9 + 16 == 25
    std::vector<float> a = {0.0f, 0.0f};
    std::vector<float> b = {3.0f, 4.0f};
    assert_near(math::l2_distance_sq(a, b), 25.0f, 1e-5f,
                "l2_dist_sq 3-4-5");
}

TEST(l2_distance_sq_size_mismatch_throws) {
    std::vector<float> a(3, 1.0f);
    std::vector<float> b(4, 1.0f);
    bool threw = false; USE(threw);
    try { static_cast<void>(math::l2_distance_sq(a, b)); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
}

// =============================================================================
// main — prints summary
// =============================================================================

int main() {
    std::cout << "\n=== Trifecta Phase 1 — Unit Tests ===\n\n";
    // Tests are auto-registered via static constructors above.
    std::cout << "\n─────────────────────────────────────\n";
    std::cout << "Results: " << g_tests_passed << " / "
              << g_tests_run << " passed.\n";
    if (g_tests_passed == g_tests_run) {
        std::cout << "ALL TESTS PASSED.\n\n";
        return 0;
    } else {
        std::cout << (g_tests_run - g_tests_passed)
                  << " FAILURE(S). See above.\n\n";
        return 1;
    }
}

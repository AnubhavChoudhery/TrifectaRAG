/**
 * =============================================================================
 * trifecta_core.cpp — Phase 1: Math Implementation
 * =============================================================================
 *
 * PURPOSE:
 *   Implements the math utilities declared in trifecta_core.hpp. The
 *   GlobalRegistry class is entirely inline (header-only implementation via
 *   the class body), so this file focuses exclusively on the `trifecta::math`
 *   namespace functions that require careful numerical handling.
 *
 * COMPILATION:
 *   The SIMD-friendly loops here require optimization flags to vectorize:
 *
 *     GCC / Clang (Linux/macOS):
 *       g++ -std=c++17 -O2 -march=native -funroll-loops \
 *           -ftree-vectorize -ffast-math \
 *           src/trifecta_core.cpp -c -o trifecta_core.o
 *
 *     MSVC (Windows):
 *       cl /std:c++17 /O2 /arch:AVX2 /fp:fast src/trifecta_core.cpp /c
 *
 *   NOTE on -ffast-math / /fp:fast:
 *     These flags allow the compiler to reorder floating-point operations
 *     (e.g., FMA fusion, reassociation). This is safe here because we do NOT
 *     rely on strict IEEE 754 associativity — the tiny numerical differences
 *     are irrelevant for similarity ranking purposes.
 *
 * =============================================================================
 */

#include "trifecta_core.hpp"
#include "serialize_utils.hpp"

#include <cmath>       // std::sqrt, std::fabs
#include <stdexcept>   // std::invalid_argument
#include <string>      // std::to_string

namespace trifecta {
namespace math {

// =============================================================================
// Internal helper: validate that two vectors are non-empty and same-length.
// Centralising this check keeps the public functions lean.
// =============================================================================
static inline void validate_pair(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 const char* caller) {
    if (a.empty() || b.empty()) {
        throw std::invalid_argument(
            std::string(caller) + ": vectors must not be empty.");
    }
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            std::string(caller) + ": size mismatch — a.size()=" +
            std::to_string(a.size()) + " b.size()=" +
            std::to_string(b.size()) + ".");
    }
}

// =============================================================================
// l2_norm
// =============================================================================
/**
 * Computes the Euclidean norm (magnitude) of a vector:
 *   ||v|| = sqrt( sum_i( v[i]^2 ) )
 *
 * The loop is a single pass over the data. At -O2 with AVX2, this compiles
 * to a sequence of vfmadd231ps (fused multiply-add) instructions over 8-wide
 * float lanes, yielding ~4× throughput vs. scalar.
 *
 * noexcept: An empty vector returns 0.0f. The caller decides what to do.
 */
float l2_norm(const std::vector<float>& v) noexcept {
    float sq_sum = 0.0f;
    const std::size_t n = v.size();
    const float* __restrict data = v.data();   // __restrict: no aliasing hint

    // ── SIMD-friendly reduction ───────────────────────────────────────────────
    // The compiler sees a simple scalar accumulation with no loop-carried
    // dependency beyond sq_sum, which it can break into multiple accumulators
    // (accumulator variable renaming) and vectorize across SIMD lanes.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep       // GCC/Clang: assert no loop-carried vector dependencies
#elif defined(_MSC_VER)
#pragma loop(ivdep)     // MSVC equivalent
#endif
    for (std::size_t i = 0; i < n; ++i) {
        sq_sum += data[i] * data[i];
    }

    return std::sqrt(sq_sum);
}

// =============================================================================
// dot_product
// =============================================================================
/**
 * Computes the inner product of two same-length float vectors:
 *   dot(a, b) = sum_i( a[i] * b[i] )
 *
 * Equivalent to cosine_similarity when both vectors are pre-normalized.
 * The HNSW engine (Phase 2) will pre-normalize all stored embeddings onto
 * the unit sphere, making dot_product the fast inner-loop metric.
 */
float dot_product(const std::vector<float>& a, const std::vector<float>& b) {
    validate_pair(a, b, "dot_product");

    float dot = 0.0f;
    const std::size_t n = a.size();
    const float* __restrict pa = a.data();
    const float* __restrict pb = b.data();

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#elif defined(_MSC_VER)
#pragma loop(ivdep)
#endif
    for (std::size_t i = 0; i < n; ++i) {
        dot += pa[i] * pb[i];
    }

    return dot;
}

// =============================================================================
// cosine_similarity  ← THE HOT PATH
// =============================================================================
/**
 * Cosine Similarity — Single-Pass, Three-Accumulator Implementation
 * -----------------------------------------------------------------
 *
 *   cosine_sim(a, b) = dot(a,b) / (||a|| * ||b||)
 *
 * ALGORITHM DECISION — WHY A SINGLE PASS?
 * ----------------------------------------
 * A naïve implementation calls dot_product(a,b), l2_norm(a), l2_norm(b)
 * in sequence — THREE passes over the data. For a 1536-dimensional embedding
 * (OpenAI text-embedding-3-small), that is 3 × 1536 × 4 bytes = ~18 KB of
 * data per similarity call. With an L1 cache of ~32 KB, a two-vector
 * comparison barely fits; a three-pass approach risks evicting the second
 * vector before the third pass begins.
 *
 * The single-pass approach below accumulates dot, sq_a, sq_b in one loop.
 * All three variables are independent across iterations, so the CPU's out-
 * of-order execution unit pipelines them simultaneously. Modern Intel/AMD
 * cores have 2–3 FMA execution ports; this structure fully saturates them.
 *
 * NUMERICAL PRECISION:
 *   Computing sqrt at the end (rather than accumulating a running norm) is
 *   more numerically stable because floating-point sqrt has 0.5 ULP error,
 *   which is the best achievable.
 *
 * RETURN VALUE CLAMPING:
 *   Due to floating-point rounding, dot / (norm_a * norm_b) can drift
 *   slightly outside [-1, 1] for nearly-identical vectors. We clamp to
 *   prevent NaN propagation upstream (e.g., arccos in reranking).
 */
float cosine_similarity(const std::vector<float>& a,
                        const std::vector<float>& b) {
    validate_pair(a, b, "cosine_similarity");

    // Three independent accumulators — the compiler promotes each to its own
    // SIMD register lane, enabling instruction-level parallelism.
    float dot  = 0.0f;   // dot product:  sum( a[i] * b[i] )
    float sq_a = 0.0f;   // norm²(a):     sum( a[i] * a[i] )
    float sq_b = 0.0f;   // norm²(b):     sum( b[i] * b[i] )

    const std::size_t n  = a.size();
    const float* __restrict pa = a.data();
    const float* __restrict pb = b.data();

    // ── Core SIMD-friendly loop ───────────────────────────────────────────────
    //
    // At AVX2 with -ffast-math, this typically unrolls to blocks of 8 floats
    // per iteration using vmovups / vfmadd231ps instructions. Example output
    // from Godbolt (GCC 13, -O3 -march=haswell -ffast-math):
    //
    //   .L3:
    //     vmovups   ymm1, [rdi + rax*4]     ; load 8 floats from a
    //     vmovups   ymm2, [rsi + rax*4]     ; load 8 floats from b
    //     vfmadd231ps ymm0, ymm1, ymm2      ; dot += a[i..i+7] * b[i..i+7]
    //     vfmadd231ps ymm3, ymm1, ymm1      ; sq_a += a² (8 lanes)
    //     vfmadd231ps ymm4, ymm2, ymm2      ; sq_b += b² (8 lanes)
    //     add rax, 8
    //     cmp rax, rdx
    //     jl  .L3
    //
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#elif defined(_MSC_VER)
#pragma loop(ivdep)
#endif
    for (std::size_t i = 0; i < n; ++i) {
        const float ai = pa[i];
        const float bi = pb[i];
        dot  += ai * bi;   // fused multiply-add (FMA) candidate
        sq_a += ai * ai;   // fused multiply-add (FMA) candidate
        sq_b += bi * bi;   // fused multiply-add (FMA) candidate
    }

    // Guard against zero-norm vectors (all-zero embeddings are degenerate).
    // Returning 0.0f is the correct semantic: undefined direction → no match.
    constexpr float kEpsilon = 1e-10f;
    const float denom = std::sqrt(sq_a) * std::sqrt(sq_b);
    if (denom < kEpsilon) {
        return 0.0f;
    }

    // Clamp to [-1, 1] to absorb floating-point rounding drift.
    const float raw = dot / denom;
    return std::max(-1.0f, std::min(1.0f, raw));
}

// =============================================================================
// normalize_inplace
// =============================================================================
/**
 * Converts a vector to a unit vector (L2 norm = 1.0).
 *
 * PRE-NORMALIZATION STRATEGY:
 *   The HNSW engine (Phase 2) stores unit-normalized embeddings. This allows
 *   it to use the cheaper dot_product() instead of the full cosine_similarity()
 *   during graph traversal — saving two sqrt() calls per distance evaluation.
 *   On a graph with ef_search=200 and 100K nodes, that is ~40K saved sqrts
 *   per query, a measurable latency win.
 *
 * This function is a no-op on zero-norm vectors to avoid NaN propagation.
 */
void normalize_inplace(std::vector<float>& v) noexcept {
    const float norm = l2_norm(v);
    constexpr float kEpsilon = 1e-10f;
    if (norm < kEpsilon) return;   // degenerate zero vector — leave as-is

    const float inv_norm = 1.0f / norm;   // multiply is cheaper than divide
    const std::size_t n = v.size();
    float* __restrict data = v.data();

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#elif defined(_MSC_VER)
#pragma loop(ivdep)
#endif
    for (std::size_t i = 0; i < n; ++i) {
        data[i] *= inv_norm;   // multiply by reciprocal — one FP op vs. divide
    }
}

// =============================================================================
// l2_distance_sq
// =============================================================================
/**
 * Squared Euclidean distance:
 *   l2_dist_sq(a, b) = sum_i( (a[i] - b[i])^2 )
 *
 * Returns the SQUARED distance to avoid an expensive sqrt(). For ranking
 * purposes (is node X closer than node Y?), squaring preserves ordering since
 * sqrt is monotone. Only call sqrt when you need the actual distance value.
 */
float l2_distance_sq(const std::vector<float>& a,
                     const std::vector<float>& b) {
    validate_pair(a, b, "l2_distance_sq");

    float dist_sq = 0.0f;
    const std::size_t n = a.size();
    const float* __restrict pa = a.data();
    const float* __restrict pb = b.data();

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#elif defined(_MSC_VER)
#pragma loop(ivdep)
#endif
    for (std::size_t i = 0; i < n; ++i) {
        const float diff = pa[i] - pb[i];
        dist_sq += diff * diff;
    }

    return dist_sq;
}

} // namespace math

// =============================================================================
// GlobalRegistry — binary persistence
// =============================================================================

void GlobalRegistry::save(std::ostream& os) const {
    io::write_u64(os, nodes_.size());
    for (const auto& n : nodes_) {
        io::write_u32(os, n.global_id);
        io::write_u8(os, static_cast<uint8_t>(n.type));
        io::write_str(os, n.metadata_payload);
    }
}

void GlobalRegistry::load(std::istream& is) {
    nodes_.clear();
    const uint64_t count = io::read_u64(is);
    nodes_.reserve(static_cast<std::size_t>(count));
    for (uint64_t i = 0; i < count; ++i) {
        uint32_t gid = io::read_u32(is);
        auto mod = static_cast<Modality>(io::read_u8(is));
        std::string meta = io::read_str(is);
        nodes_.emplace_back(gid, mod, std::move(meta));
    }
}

} // namespace trifecta

/**
 * =============================================================================
 * trifecta_core.hpp — Phase 1: Foundation & Memory Registry
 * =============================================================================
 *
 * PURPOSE:
 *   This header defines the foundational data structures for the Trifecta
 *   Multi-Modal GraphRAG backend. It establishes the Global Memory Registry,
 *   which serves as the single source of truth for all ingested chunks (text
 *   or image). Every other engine (HNSW, BM25, Knowledge Graph) refers back
 *   to nodes stored here via a lightweight `uint32_t global_id`.
 *
 * DESIGN PHILOSOPHY:
 *   - Cache Locality:  NodeData is laid out to minimize padding. The hot
 *     fields (global_id, type) sit at the front so a cache line fetch during
 *     iteration gives the engine what it needs immediately.
 *   - Zero External Dependencies:  Pure C++17 Standard Library only.
 *   - SIMD-Friendly Math:  cosine_similarity() uses a tight loop that modern
 *     compilers (GCC -O2, Clang -O2, MSVC /O2) auto-vectorize to AVX2/SSE4.2.
 *     The pragma hints below make that contract explicit.
 *
 * THREAD SAFETY:
 *   GlobalRegistry is NOT thread-safe for concurrent writes. Wrap add_node()
 *   calls in an external mutex when ingesting from multiple producer threads.
 *   Read-only operations (get_node, size) are safe under a shared_lock model.
 *
 * =============================================================================
 */

#pragma once

// ─── Standard Library Includes ───────────────────────────────────────────────
#include <algorithm>      // std::max, std::min
#include <cassert>        // assert() for invariant checks
#include <cmath>          // std::sqrt, std::fabs
#include <cstdint>        // uint8_t, uint32_t — fixed-width types
#include <iosfwd>         // forward decl for save/load
#include <limits>         // std::numeric_limits
#include <optional>       // std::optional for safe node lookup
#include <stdexcept>      // std::out_of_range, std::runtime_error
#include <string>         // std::string for metadata payloads
#include <vector>         // std::vector — the only container used here

namespace trifecta {

// =============================================================================
// SECTION 1 — Modality Enum
// =============================================================================

/**
 * Modality
 * --------
 * Encodes the data type of an ingested chunk. Stored as uint8_t (1 byte)
 * rather than int (4 bytes) to keep NodeData compact and maximize the number
 * of nodes that fit in a single cache line.
 *
 * Future extensions (VIDEO, AUDIO, TABLE) can be added without changing the
 * underlying storage type because uint8_t supports 256 distinct values.
 */
enum class Modality : uint8_t {
    TEXT  = 0,   ///< Chunk originates from a text document (PDF, web page, etc.)
    IMAGE = 1,   ///< Chunk originates from an image (CLIP embedding, etc.)
};

// Convenience helper: convert Modality to a human-readable string.
inline const char* modality_to_str(Modality m) noexcept {
    switch (m) {
        case Modality::TEXT:  return "TEXT";
        case Modality::IMAGE: return "IMAGE";
        default:              return "UNKNOWN";
    }
}

// =============================================================================
// SECTION 2 — NodeData Struct
// =============================================================================

/**
 * NodeData
 * --------
 * The fundamental unit of the Global Memory Registry. Represents one ingested
 * chunk — a paragraph of text, an image caption, etc.
 *
 * Memory Layout (on a 64-bit system with standard ABI):
 *
 *   Offset │ Field            │ Size   │ Notes
 *   ───────┼──────────────────┼────────┼──────────────────────────────────────
 *     0    │ global_id        │ 4 B    │ Primary key, dense 0-based index
 *     4    │ type             │ 1 B    │ Modality enum (uint8_t)
 *     5    │ [padding]        │ 3 B    │ Compiler inserts to align std::string
 *     8    │ metadata_payload │ 24+ B  │ SSO string (pointer + size + capacity)
 *
 * Total hot-path fields (global_id + type) fit in the first 8 bytes — one
 * 64-bit register load. The metadata_payload string is accessed only when
 * returning results to Python, so it being off the hot path is acceptable.
 *
 * IMPORTANT: We deliberately do NOT store the float embedding vector inside
 * NodeData. Embeddings live in the HNSW engine's own contiguous flat array,
 * indexed by global_id. This separation avoids polluting the registry cache
 * lines with large float arrays during graph traversal.
 */
struct NodeData {
    uint32_t global_id;          ///< Unique, dense, 0-based identifier
    Modality type;               ///< TEXT or IMAGE

    /**
     * metadata_payload
     * ----------------
     * Arbitrary UTF-8 string. Examples:
     *   TEXT  → "source: arxiv:2401.12345, page: 7, chunk: 3"
     *   IMAGE → "alt_text: A diagram showing..., file: fig3.png"
     *
     * Kept as std::string for simplicity; switch to a string_view into an
     * external metadata store if memory pressure becomes a concern.
     */
    std::string metadata_payload;

    // Default constructor — used when resizing the registry vector.
    NodeData() noexcept
        : global_id(0), type(Modality::TEXT), metadata_payload() {}

    // Primary constructor used by GlobalRegistry::add_node().
    NodeData(uint32_t id, Modality modality, std::string payload) noexcept
        : global_id(id),
          type(modality),
          metadata_payload(std::move(payload))   // move-construct to avoid copy
    {}

    // Convenience: is this chunk textual?
    [[nodiscard]] bool is_text()  const noexcept { return type == Modality::TEXT;  }
    [[nodiscard]] bool is_image() const noexcept { return type == Modality::IMAGE; }
};

// =============================================================================
// SECTION 3 — GlobalRegistry Class
// =============================================================================

/**
 * GlobalRegistry
 * --------------
 * The single source of truth for all ingested chunks. Owns a contiguous
 * std::vector<NodeData> so that sequential scans enjoy hardware prefetching.
 *
 * Key invariant: registry_[i].global_id == i for all valid indices.
 * This identity mapping makes O(1) random access trivial — no hash lookup
 * needed; just nodes_[global_id].
 *
 * Usage (from Python via pybind11 in later phases):
 *
 *   GlobalRegistry registry;
 *   registry.reserve(1'000'000);
 *   uint32_t id = registry.add_node(Modality::TEXT, "source: doc1.pdf, chunk: 0");
 *   const NodeData& node = registry.get_node(id);
 */
class GlobalRegistry {
public:
    // ─── Construction ────────────────────────────────────────────────────────

    /**
     * Default constructor. The internal vector starts empty; call reserve()
     * upfront if you know the approximate number of chunks to avoid repeated
     * heap reallocations and their associated cache-thrashing copy cost.
     */
    GlobalRegistry() = default;

    /**
     * reserve(n)
     * ----------
     * Pre-allocates backing storage for n nodes. Call this before bulk
     * ingestion to eliminate mid-loop vector reallocations, which would
     * invalidate any raw pointers/references cached by other engines.
     *
     * IMPORTANT: Other engines (HNSW, BM25) that cache raw pointers into
     * this vector MUST NOT hold live pointers across calls to add_node()
     * unless reserve() was called beforehand with a sufficient capacity.
     */
    void reserve(std::size_t n) {
        nodes_.reserve(n);
    }

    // ─── Write Operations ────────────────────────────────────────────────────

    /**
     * add_node(modality, metadata) → global_id
     * -----------------------------------------
     * Ingests a new chunk into the registry. Assigns the next available
     * global_id (sequential, 0-based), constructs a NodeData in-place, and
     * returns the new id to the caller.
     *
     * Complexity: Amortized O(1) — std::vector push_back.
     *
     * @param modality       TEXT or IMAGE
     * @param metadata_payload  Arbitrary UTF-8 metadata string
     * @return               The newly assigned global_id
     * @throws std::overflow_error if the registry exceeds UINT32_MAX nodes
     */
    [[nodiscard]] uint32_t add_node(Modality modality,
                                    std::string metadata_payload) {
        // Guard against exceeding the id space.
        if (nodes_.size() >= static_cast<std::size_t>(
                std::numeric_limits<uint32_t>::max())) {
            throw std::overflow_error(
                "GlobalRegistry: exceeded maximum node capacity (UINT32_MAX).");
        }

        const uint32_t new_id = static_cast<uint32_t>(nodes_.size());
        nodes_.emplace_back(new_id, modality, std::move(metadata_payload));
        return new_id;
    }

    // ─── Read Operations ─────────────────────────────────────────────────────

    /**
     * get_node(id) → const NodeData&
     * --------------------------------
     * O(1) direct-index access. Throws std::out_of_range if id is invalid.
     * Prefer this in hot paths where you've already validated the id.
     */
    [[nodiscard]] const NodeData& get_node(uint32_t id) const {
        if (id >= nodes_.size()) {
            throw std::out_of_range(
                "GlobalRegistry::get_node — id " + std::to_string(id) +
                " out of range (size=" + std::to_string(nodes_.size()) + ").");
        }
        return nodes_[id];
    }

    /**
     * try_get_node(id) → std::optional<std::reference_wrapper<const NodeData>>
     * --------------------------------------------------------------------------
     * Safe lookup that returns std::nullopt instead of throwing.
     * Use in validation paths where an invalid id is a recoverable condition.
     */
    [[nodiscard]] std::optional<std::reference_wrapper<const NodeData>>
    try_get_node(uint32_t id) const noexcept {
        if (id >= nodes_.size()) return std::nullopt;
        return std::cref(nodes_[id]);
    }

    /**
     * size() → number of registered nodes.
     */
    [[nodiscard]] std::size_t size() const noexcept {
        return nodes_.size();
    }

    /**
     * empty() → true if no nodes have been ingested yet.
     */
    [[nodiscard]] bool empty() const noexcept {
        return nodes_.empty();
    }

    /**
     * raw_nodes() → const reference to the underlying vector.
     * --------------------------------------------------------
     * Grants read-only access to the full contiguous node array. Used by the
     * HNSW and BM25 engines to iterate the registry during index rebuilds
     * without copying data.
     */
    [[nodiscard]] const std::vector<NodeData>& raw_nodes() const noexcept {
        return nodes_;
    }

    // ─── Binary Persistence ─────────────────────────────────────────────────

    void save(std::ostream& os) const;
    void load(std::istream& is);

private:
    std::vector<NodeData> nodes_;
};

// =============================================================================
// SECTION 4 — SIMD-Friendly Cosine Similarity
// =============================================================================

namespace math {

/**
 * cosine_similarity(a, b) → float in [-1.0, 1.0]
 * -------------------------------------------------
 * Computes the cosine similarity between two dense float vectors:
 *
 *   cosine_sim(a, b) = dot(a, b) / (||a|| * ||b||)
 *
 * WHY THIS LOOP IS FAST
 * ---------------------
 * 1. Single-pass: dot product, norm_a², and norm_b² are accumulated in the
 *    same loop iteration, maximising instruction-level parallelism (ILP) and
 *    avoiding a second memory pass over the (potentially large) arrays.
 *
 * 2. Auto-vectorization: The three accumulator variables (dot, sq_a, sq_b)
 *    are independent across iterations. Combined with the `restrict`-style
 *    semantics of passing by const-ref (no aliasing), compilers reliably emit
 *    packed SIMD instructions (AVX2 vfmadd, SSE4 dpps) at -O2/-O3.
 *
 * 3. No heap allocation: operates directly on the caller's vector data.
 *
 * COMPILER HINTS:
 *   On GCC/Clang: compile with -O2 -march=native (enables AVX2 if supported).
 *   On MSVC:      compile with /O2 /arch:AVX2.
 *   The #pragma ivdep below tells ICC/MSVC the loop has no loop-carried
 *   dependencies, unlocking vectorization in stricter compilation modes.
 *
 * @param a  First vector (e.g., document embedding)
 * @param b  Second vector (e.g., query embedding)
 * @return   Cosine similarity, or 0.0f if either vector is zero-norm.
 * @throws   std::invalid_argument if a.size() != b.size() or either is empty.
 */
[[nodiscard]] float cosine_similarity(const std::vector<float>& a,
                                      const std::vector<float>& b);

/**
 * dot_product(a, b) → float
 * --------------------------
 * Raw inner product with no normalization. Useful when vectors are
 * pre-normalized (unit-sphere HNSW optimization) because dot == cosine then.
 *
 * @throws std::invalid_argument if sizes differ.
 */
[[nodiscard]] float dot_product(const std::vector<float>& a,
                                const std::vector<float>& b);

/**
 * l2_norm(v) → float
 * -------------------
 * Euclidean (L2) norm of a vector. Used internally by cosine_similarity and
 * exposed for pre-normalization workflows.
 */
[[nodiscard]] float l2_norm(const std::vector<float>& v) noexcept;

/**
 * normalize_inplace(v)
 * ---------------------
 * Divides every element by the L2 norm, converting v to a unit vector.
 * No-op if the norm is effectively zero (avoids NaN on zero vectors).
 */
void normalize_inplace(std::vector<float>& v) noexcept;

/**
 * normalize_inplace_raw(data, n)
 * -------------------------------
 * Same as normalize_inplace but operates on a raw float pointer + length.
 * Used by HNSWIndex to normalize vectors directly in flat storage.
 */
void normalize_inplace_raw(float* data, std::size_t n) noexcept;

/**
 * l2_distance_sq(a, b) → float
 * ------------------------------
 * Squared Euclidean distance — cheaper than L2 (avoids sqrt) and sufficient
 * for distance comparisons. Used by the HNSW engine (Phase 2) as a secondary
 * metric.
 *
 * @throws std::invalid_argument if sizes differ.
 */
[[nodiscard]] float l2_distance_sq(const std::vector<float>& a,
                                   const std::vector<float>& b);

} // namespace math

} // namespace trifecta

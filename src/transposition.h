#pragma once
// transposition.h — Custom open-addressing transposition table (TT)
//
// Design choices:
//   • Power-of-2 size so index = hash & mask (no modulo).
//   • Each bucket is exactly 16 bytes — fits in a cache line pair.
//   • Replacement strategy: "always replace" for same-depth, "depth-preferred"
//     for different depths.  This keeps the most tactically relevant entries.
//   • Thread safety: intentionally lock-free with relaxed atomics.  Occasional
//     hash collisions are tolerable — they just cause a re-search.
//
// TODO: Experiment with a two-tier TT (depth-preferred + always-replace halves)
// TODO: Prefetch TT entry during move loop to hide cache latency

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>

// ─── Entry flags ──────────────────────────────────────────────────────────────
enum TTFlag : uint8_t {
    TT_NONE  = 0,
    TT_EXACT = 1,   ///< score is exact (PV node)
    TT_LOWER = 2,   ///< score >= beta  (cut node / lower bound)
    TT_UPPER = 3,   ///< score <= alpha (all node / upper bound)
};

// ─── TT entry (16 bytes) ──────────────────────────────────────────────────────
#pragma pack(push, 1)
struct TTEntry {
    uint32_t key32;      ///< Upper 32 bits of Zobrist hash (collision check)
    int32_t  score;      ///< Centipawn score from the perspective of side-to-move
    uint32_t best_move;  ///< Packed Move::data of the best move (0 = none)
    uint8_t  depth;      ///< Search depth at which this entry was stored
    uint8_t  flag;       ///< TTFlag
    uint16_t age;        ///< Engine half-move clock (for aging out stale entries)
};
#pragma pack(pop)
static_assert(sizeof(TTEntry) == 16, "TTEntry must be 16 bytes");

// ─── Transposition Table ──────────────────────────────────────────────────────

class TranspositionTable {
public:
    /// Allocate a TT of exactly `mb` megabytes (rounded down to power-of-2 entries).
    explicit TranspositionTable(size_t mb = 128) { resize(mb); }

    ~TranspositionTable() {
#if defined(_WIN32) || defined(__MINGW32__)
        _aligned_free(table_);
#else
        std::free(table_);
#endif
    }

    // Non-copyable, movable
    TranspositionTable(const TranspositionTable&)            = delete;
    TranspositionTable& operator=(const TranspositionTable&) = delete;

    /// Resize the table (clears all entries).
    void resize(size_t mb) {
        size_t bytes    = mb * 1024ULL * 1024ULL;
        size_t capacity = bytes / sizeof(TTEntry);
        // Round down to power of two
        capacity = 1ULL << (63 - __builtin_clzll(capacity | 1));
        mask_    = capacity - 1;

        std::free(table_);
        // Aligned allocation for cache-line friendliness
        // Use _aligned_malloc on Windows/MinGW, aligned_alloc elsewhere
#if defined(_WIN32) || defined(__MINGW32__)
        table_ = static_cast<TTEntry*>(
            _aligned_malloc(capacity * sizeof(TTEntry), 64));
#else
        table_ = static_cast<TTEntry*>(
            aligned_alloc(64, capacity * sizeof(TTEntry)));
#endif
        if (!table_) throw std::bad_alloc{};
        clear();
    }

    /// Wipe all entries (called at start of new game).
    void clear() {
        std::memset(table_, 0, (mask_ + 1) * sizeof(TTEntry));
        generation_ = 0;
    }

    /// Advance the generation counter (call at each new root search).
    void new_search() { ++generation_; }

    // ── Probe ──────────────────────────────────────────────────────────────

    /// Returns true if a usable entry was found.
    /// `score` and `best_move` are filled in on hit.
    /// `flag` tells you whether the score is exact, a lower, or upper bound.
    bool probe(uint64_t hash, int depth, int alpha, int beta,
               int& score, uint32_t& best_move, TTFlag& flag) const {
        TTEntry& e = table_[hash & mask_];
        if (e.key32 != static_cast<uint32_t>(hash >> 32)) return false;

        best_move = e.best_move;  // Always return the best move hint

        if (e.depth < depth) return false;  // Entry too shallow

        flag = static_cast<TTFlag>(e.flag);
        score = e.score;

        if (flag == TT_EXACT)                        return true;
        if (flag == TT_LOWER && score >= beta)        return true;
        if (flag == TT_UPPER && score <= alpha)       return true;
        return false;
    }

    // ── Store ──────────────────────────────────────────────────────────────

    void store(uint64_t hash, int depth, int score,
               uint32_t best_move, TTFlag flag) {
        TTEntry& e = table_[hash & mask_];

        // Replacement policy: prefer deeper entries; always replace stale ones.
        bool is_stale = (e.age != generation_);
        bool deeper   = (depth >= static_cast<int>(e.depth));
        if (!is_stale && !deeper) return;

        e.key32     = static_cast<uint32_t>(hash >> 32);
        e.score     = static_cast<int32_t>(score);
        e.best_move = best_move;
        e.depth     = static_cast<uint8_t>(depth);
        e.flag      = static_cast<uint8_t>(flag);
        e.age       = generation_;
    }

    // ── Diagnostics ────────────────────────────────────────────────────────

    /// Approximate fill rate in per-mille (0–1000).
    int hashfull() const {
        size_t sample = std::min(static_cast<size_t>(1000), mask_ + 1);
        int used = 0;
        for (size_t i = 0; i < sample; ++i)
            if (table_[i].flag != TT_NONE) ++used;
        return static_cast<int>(used * 1000 / sample);
    }

    size_t capacity() const { return mask_ + 1; }

private:
    TTEntry* table_{nullptr};
    size_t   mask_{0};
    uint16_t generation_{0};
};
#pragma once
// chess_board.h — Bitboard-based chess board representation
// Uses magic bitboards for O(1) sliding-piece move generation.
//
// Board layout (little-endian rank-file mapping):
//   bit 0 = a1, bit 1 = b1, ..., bit 63 = h8
//
// TODO: Add 960-chess support (castling rights generalisation)

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// ─── Basic types ──────────────────────────────────────────────────────────────

using Bitboard = uint64_t;
using Square   = int;   // 0-63

constexpr Square NO_SQ = -1;

// clang-format off
enum Piece  { PAWN=0, KNIGHT, BISHOP, ROOK, QUEEN, KING, PIECE_NONE=6 };
enum Color  { WHITE=0, BLACK=1 };
enum File   { FA=0,FB,FC,FD,FE,FF,FG,FH };
enum Rank   { R1=0,R2,R3,R4,R5,R6,R7,R8 };
// clang-format on

inline constexpr Square make_square(File f, Rank r) { return r * 8 + f; }
inline constexpr File   sq_file(Square s)           { return static_cast<File>(s & 7); }
inline constexpr Rank   sq_rank(Square s)           { return static_cast<Rank>(s >> 3); }
inline constexpr Bitboard bb(Square s)              { return 1ULL << s; }

// ─── Move encoding (packed into 32 bits) ─────────────────────────────────────
// Bits  0- 5: from square
// Bits  6-11: to square
// Bits 12-14: promotion piece (0=none, 1=N, 2=B, 3=R, 4=Q)
// Bits 15-17: move flags  (0=quiet, 1=capture, 2=ep, 3=castle-K, 4=castle-Q)
struct Move {
    uint32_t data{0};

    Move() = default;
    explicit Move(Square from, Square to,
                  int promo = 0, int flags = 0)
        : data(static_cast<uint32_t>(from)
             | (static_cast<uint32_t>(to)    << 6)
             | (static_cast<uint32_t>(promo) << 12)
             | (static_cast<uint32_t>(flags) << 15)) {}

    Square from()  const { return static_cast<Square>(data & 0x3F); }
    Square to()    const { return static_cast<Square>((data >> 6) & 0x3F); }
    int    promo() const { return (data >> 12) & 0x7; }
    int    flags() const { return (data >> 15) & 0x7; }
    bool   is_null()    const { return data == 0; }
    bool   is_capture() const { return (flags() & 1) != 0; }

    bool operator==(Move o) const { return data == o.data; }
    bool operator!=(Move o) const { return data != o.data; }

    // Returns algebraic notation, e.g. "e2e4", "e7e8q"
    std::string uci() const;
};

// Null move sentinel
inline const Move NULL_MOVE{};

// ─── Magic bitboard tables ────────────────────────────────────────────────────
// Each square has a magic number and a shift; attacks are looked up in a
// pre-computed table indexed by (occ * magic) >> shift.

struct MagicEntry {
    Bitboard mask;       ///< Relevant occupancy mask for this square
    Bitboard magic;      ///< Magic multiplier
    int      shift;      ///< Right-shift amount (64 - popcount(mask))
    Bitboard* attacks;   ///< Pointer into the shared flat attack table
};

// ─── Board state ──────────────────────────────────────────────────────────────

struct BoardState {
    // Occupancy by piece & color: pieces[color][piece]
    Bitboard pieces[2][6]{};
    Bitboard occ[2]{};      ///< Combined occupancy per color
    Bitboard all_occ{};     ///< Union of both

    Color    side_to_move{WHITE};
    uint8_t  castling{0b1111}; ///< bits: WK WQ BK BQ
    Square   ep_sq{NO_SQ};     ///< En-passant target square
    int      fifty{0};         ///< Half-move clock
    int      full_move{1};

    uint64_t hash{0};          ///< Zobrist hash (updated incrementally)
};

// ─── Main board class ─────────────────────────────────────────────────────────

class ChessBoard {
public:
    ChessBoard();

    // ── Setup ──────────────────────────────────────────────────────────────
    void set_startpos();
    bool set_fen(std::string_view fen);
    std::string get_fen() const;

    // ── Move generation ────────────────────────────────────────────────────
    /// Generates all pseudo-legal moves; legality is checked on make().
    std::vector<Move> gen_moves() const;
    /// Generates only captures + promotions (for quiescence).
    std::vector<Move> gen_captures() const;

    // ── Make / unmake ──────────────────────────────────────────────────────
    /// Returns false if the move leaves own king in check (illegal).
    bool make_move(Move m);
    void unmake_move();

    // ── Queries ────────────────────────────────────────────────────────────
    bool in_check() const;
    bool is_attacked(Square sq, Color by) const;
    bool is_draw() const;   ///< 50-move rule OR insufficient material

    Color  side() const { return state_.side_to_move; }
    Square ep_sq() const { return state_.ep_sq; }
    Bitboard occ(Color c) const { return state_.occ[c]; }
    Bitboard all_occ()    const { return state_.all_occ; }
    Bitboard pieces(Color c, Piece p) const { return state_.pieces[c][p]; }
    uint64_t hash()       const { return state_.hash; }

    // ── Neural network input ───────────────────────────────────────────────
    // Fills a float[12][8][8] tensor (12 planes: 6 pieces × 2 colors).
    // Used by the CNN evaluator.  Planes are from White's perspective.
    void fill_nn_input(float* out) const;

    // ── Debug ──────────────────────────────────────────────────────────────
    void print() const;

private:
    // ── Magic bitboard helpers ─────────────────────────────────────────────
    static void  init_magics();
    static bool  magics_ready_;
    static MagicEntry rook_magic_[64];
    static MagicEntry bishop_magic_[64];
    static Bitboard   rook_attacks_[102400];
    static Bitboard   bishop_attacks_[5248];

    Bitboard rook_attacks(Square sq, Bitboard occ) const;
    Bitboard bishop_attacks(Square sq, Bitboard occ) const;
    Bitboard queen_attacks(Square sq, Bitboard occ) const;

    // ── Static attack tables ───────────────────────────────────────────────
    static Bitboard knight_attacks_[64];
    static Bitboard king_attacks_[64];
    static Bitboard pawn_attacks_[2][64];  ///< [color][square]
    static void init_static_attacks();

    // ── Move generation helpers ────────────────────────────────────────────
    void gen_pawn_moves(std::vector<Move>& ml, bool captures_only) const;
    void gen_knight_moves(std::vector<Move>& ml, bool captures_only) const;
    void gen_bishop_moves(std::vector<Move>& ml, bool captures_only) const;
    void gen_rook_moves(std::vector<Move>& ml, bool captures_only) const;
    void gen_queen_moves(std::vector<Move>& ml, bool captures_only) const;
    void gen_king_moves(std::vector<Move>& ml, bool captures_only) const;
    void gen_castling(std::vector<Move>& ml) const;

    // ── Zobrist ────────────────────────────────────────────────────────────
    static uint64_t zobrist_piece_[2][6][64];
    static uint64_t zobrist_ep_[64];
    static uint64_t zobrist_castle_[16];
    static uint64_t zobrist_stm_;
    static void init_zobrist();
    uint64_t compute_hash() const;

    // ── State stack ────────────────────────────────────────────────────────
    BoardState state_;
    std::vector<BoardState> history_;  ///< For unmake; also used for repetition
};

// ─── Inline utilities ─────────────────────────────────────────────────────────

/// Count set bits (population count)
inline int popcount(Bitboard b) {
#ifdef __GNUC__
    return __builtin_popcountll(b);
#else
    // MSVC / fallback
    int n = 0;
    while (b) { ++n; b &= b - 1; }
    return n;
#endif
}

/// Index of least-significant bit; undefined if b==0
inline Square lsb(Bitboard b) {
#ifdef __GNUC__
    return __builtin_ctzll(b);
#else
    Square sq = 0;
    while (!(b & 1)) { b >>= 1; ++sq; }
    return sq;
#endif
}

/// Pop and return LSB
inline Square pop_lsb(Bitboard& b) {
    Square s = lsb(b);
    b &= b - 1;
    return s;
}

// ─── FEN constants ────────────────────────────────────────────────────────────
constexpr std::string_view START_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

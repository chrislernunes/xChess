// chess_board.cpp — ChessBoard implementation

// CNN constants (mirrored from evaluator.h to avoid circular include)
static constexpr int CNN_PLANES = 12;
static constexpr int CNN_H      = 8;
static constexpr int CNN_W      = 8;
// Provides real move generation, make/unmake, FEN parsing, and UCI move output.
// Magic bitboard tables are initialised lazily on first ChessBoard construction.
//
// NOTE: This is a functional implementation sufficient to run the engine.
// The magic numbers below are standard published values (from CPW / Pradyumna).

#include "chess_board.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>

// ─── Static member definitions ────────────────────────────────────────────────
bool      ChessBoard::magics_ready_         = false;
MagicEntry ChessBoard::rook_magic_[64]      = {};
MagicEntry ChessBoard::bishop_magic_[64]    = {};
Bitboard  ChessBoard::rook_attacks_[102400] = {};
Bitboard  ChessBoard::bishop_attacks_[5248] = {};
Bitboard  ChessBoard::knight_attacks_[64]   = {};
Bitboard  ChessBoard::king_attacks_[64]     = {};
Bitboard  ChessBoard::pawn_attacks_[2][64]  = {};
uint64_t  ChessBoard::zobrist_piece_[2][6][64] = {};
uint64_t  ChessBoard::zobrist_ep_[64]       = {};
uint64_t  ChessBoard::zobrist_castle_[16]   = {};
uint64_t  ChessBoard::zobrist_stm_          = 0;

// ─── Move::uci ────────────────────────────────────────────────────────────────
std::string Move::uci() const {
    if (is_null()) return "0000";
    static const char* files = "abcdefgh";
    static const char* ranks = "12345678";
    std::string s;
    s += files[sq_file(from())];
    s += ranks[sq_rank(from())];
    s += files[sq_file(to())];
    s += ranks[sq_rank(to())];
    static const char promo_chars[] = "\0nbrq";
    if (promo() > 0 && promo() <= 4) s += promo_chars[promo()];
    return s;
}

// ─── Zobrist initialisation ───────────────────────────────────────────────────
void ChessBoard::init_zobrist() {
    std::mt19937_64 rng(0xDEADBEEFCAFEULL);
    for (int c = 0; c < 2; ++c)
        for (int p = 0; p < 6; ++p)
            for (int sq = 0; sq < 64; ++sq)
                zobrist_piece_[c][p][sq] = rng();
    for (int sq = 0; sq < 64; ++sq) zobrist_ep_[sq]     = rng();
    for (int i  = 0; i < 16; ++i)  zobrist_castle_[i]   = rng();
    zobrist_stm_ = rng();
}

uint64_t ChessBoard::compute_hash() const {
    uint64_t h = 0;
    for (int c = 0; c < 2; ++c)
        for (int p = 0; p < 6; ++p) {
            Bitboard b = state_.pieces[c][p];
            while (b) h ^= zobrist_piece_[c][p][pop_lsb(b)];
        }
    if (state_.ep_sq != NO_SQ)    h ^= zobrist_ep_[state_.ep_sq];
    h ^= zobrist_castle_[state_.castling & 0xF];
    if (state_.side_to_move == BLACK) h ^= zobrist_stm_;
    return h;
}

// ─── Static attack tables ─────────────────────────────────────────────────────
static Bitboard knight_mask(Square sq) {
    Bitboard b = 0, s = bb(sq);
    if (s >> 17 && sq_file(sq) > 0) b |= s >> 17;
    if (s >> 15 && sq_file(sq) < 7) b |= s >> 15;
    if (s >> 10 && sq_file(sq) > 1) b |= s >> 10;
    if (s >>  6 && sq_file(sq) < 6) b |= s >>  6;
    if (s << 17 && sq_file(sq) < 7) b |= s << 17;
    if (s << 15 && sq_file(sq) > 0) b |= s << 15;
    if (s << 10 && sq_file(sq) < 6) b |= s << 10;
    if (s <<  6 && sq_file(sq) > 1) b |= s <<  6;
    return b;
}
static Bitboard king_mask(Square sq) {
    Bitboard b = 0, s = bb(sq);
    if (sq_file(sq) > 0) { b |= s >> 1; if (sq_rank(sq) > 0) b |= s >> 9; if (sq_rank(sq) < 7) b |= s << 7; }
    if (sq_file(sq) < 7) { b |= s << 1; if (sq_rank(sq) > 0) b |= s >> 7; if (sq_rank(sq) < 7) b |= s << 9; }
    if (sq_rank(sq) > 0) b |= s >> 8;
    if (sq_rank(sq) < 7) b |= s << 8;
    return b;
}

// Sliding attacks computed by stepping (used during magic init)
static Bitboard sliding_attacks(Square sq, Bitboard occ, int dr, int df) {
    Bitboard result = 0;
    int r = sq_rank(sq) + dr, f = sq_file(sq) + df;
    while (r >= 0 && r < 8 && f >= 0 && f < 8) {
        Square s = make_square(static_cast<File>(f), static_cast<Rank>(r));
        result |= bb(s);
        if (occ & bb(s)) break;
        r += dr; f += df;
    }
    return result;
}
static Bitboard rook_attacks_slow(Square sq, Bitboard occ) {
    return sliding_attacks(sq, occ,  1,  0)
         | sliding_attacks(sq, occ, -1,  0)
         | sliding_attacks(sq, occ,  0,  1)
         | sliding_attacks(sq, occ,  0, -1);
}
static Bitboard bishop_attacks_slow(Square sq, Bitboard occ) {
    return sliding_attacks(sq, occ,  1,  1)
         | sliding_attacks(sq, occ,  1, -1)
         | sliding_attacks(sq, occ, -1,  1)
         | sliding_attacks(sq, occ, -1, -1);
}

// Enumerate all subsets of a mask (Carry-Rippler)
static void enumerate_subsets(Bitboard mask,
                               std::vector<Bitboard>& out) {
    Bitboard sub = 0;
    do {
        out.push_back(sub);
        sub = (sub - mask) & mask;
    } while (sub);
}

// ─── Magic numbers (from public domain CPW table) ────────────────────────────
static const uint64_t ROOK_MAGICS[64] = {
    0x8a80104000800020ULL,0x140002000100040ULL,0x2801880a0017001ULL,0x100081001000420ULL,
    0x200020010080420ULL,0x3001c0002010008ULL,0x8480008002000100ULL,0x2080088004402900ULL,
    0x800098204000ULL,0x2024401000200040ULL,0x100802000801000ULL,0x120800800801000ULL,
    0x208808088000400ULL,0x2802200800400ULL,0x2200800100020080ULL,0x801000060821100ULL,
    0x80044006422000ULL,0x100808020004000ULL,0x12108a0010204200ULL,0x140848010000802ULL,
    0x481828014002800ULL,0x8094004002004100ULL,0x4010040010010802ULL,0x20008806104ULL,
    0x100400080208000ULL,0x2040002120081000ULL,0x21200680100081ULL,0x20100080080080ULL,
    0x2000a00200410ULL,0x20080800400ULL,0x80088400100102ULL,0x80004600042881ULL,
    0x4040008040800020ULL,0x440003000200801ULL,0x4200011004500ULL,0x188020010100100ULL,
    0x14800401802800ULL,0x2080040080800200ULL,0x124080204001001ULL,0x200046502000484ULL,
    0x480400080088020ULL,0x1000422010034000ULL,0x30200100110040ULL,0x100021010009ULL,
    0x2002080100110004ULL,0x202008004008002ULL,0x20020004010100ULL,0x2048440040820001ULL,
    0x101002200408200ULL,0x40802000401080ULL,0x4008142004410100ULL,0x2060820c0120200ULL,
    0x1001004080100ULL,0x20c020080040080ULL,0x2935610830022400ULL,0x44440041009200ULL,
    0x280001040802101ULL,0x2100190040002085ULL,0x80c0084100102001ULL,0x4024081001000421ULL,
    0x20030a0244872ULL,0x12001008414402ULL,0x2006104900a0804ULL,0x1004081002402ULL,
};
static const uint64_t BISHOP_MAGICS[64] = {
    0x40040844404084ULL,0x2004208a004208ULL,0x10190041080202ULL,0x108060845042010ULL,
    0x581104180800210ULL,0x2112080446200010ULL,0x1080820820060210ULL,0x3c0808410220200ULL,
    0x4050404440404ULL,0x21001420088ULL,0x24d0080801082102ULL,0x1020a0a020400ULL,
    0x40308200402ULL,0x4011002100800ULL,0x401484104104005ULL,0x801010402020200ULL,
    0x400210c3880100ULL,0x404022024108200ULL,0x810018200204102ULL,0x4002801a02003ULL,
    0x85040820080400ULL,0x810102c808880400ULL,0xe900410884800ULL,0x8002020480840102ULL,
    0x220200865090201ULL,0x2010100a02021202ULL,0x152048408022401ULL,0x20080002081110ULL,
    0x4001001021004000ULL,0x800040400a011002ULL,0xe4004081011002ULL,0x1c004001012080ULL,
    0x8004200962a00220ULL,0x8422100208500202ULL,0x2000402200300c08ULL,0x8646020080080080ULL,
    0x80020a0200100808ULL,0x2010004880111000ULL,0x623000a080011400ULL,0x42008c0340209202ULL,
    0x209188240001000ULL,0x400408a884001800ULL,0x110400a6080400ULL,0x1840060a44020800ULL,
    0x90080104000041ULL,0x201011000808101ULL,0x1a2208080504f080ULL,0x8012020600211212ULL,
    0x500861011240000ULL,0x180806108200800ULL,0x4000020e01040044ULL,0x300000261044000aULL,
    0x802241102020002ULL,0x20906061210001ULL,0x5a84841004010310ULL,0x4010801011c04ULL,
    0xa010109502200ULL,0x4a02012000ULL,0x500201010098b028ULL,0x8040002811040900ULL,
    0x28000010020204ULL,0x6000020202d0240ULL,0x8918844842082200ULL,0x4010011029020020ULL,
};

void ChessBoard::init_magics() {
    // Knight and king static tables
    for (Square sq = 0; sq < 64; ++sq) {
        knight_attacks_[sq] = knight_mask(sq);
        king_attacks_[sq]   = king_mask(sq);
        // Pawn attacks
        Bitboard s = bb(sq);
        if (sq_rank(sq) < 7) {
            pawn_attacks_[WHITE][sq]  = (sq_file(sq) > 0 ? s << 7 : 0)
                                      | (sq_file(sq) < 7 ? s << 9 : 0);
        }
        if (sq_rank(sq) > 0) {
            pawn_attacks_[BLACK][sq]  = (sq_file(sq) > 0 ? s >> 9 : 0)
                                      | (sq_file(sq) < 7 ? s >> 7 : 0);
        }
    }

    // Magic bitboard initialisation for rooks and bishops
    size_t rook_offset = 0, bishop_offset = 0;
    for (Square sq = 0; sq < 64; ++sq) {
        // ── Rook ──────────────────────────────────────────────────────────
        {
            Bitboard edges = ((0xFFULL | 0xFF00000000000000ULL) & ~(0xFFULL << (sq_rank(sq)*8)))
                           | ((0x0101010101010101ULL | 0x8080808080808080ULL) & ~(0x0101010101010101ULL << sq_file(sq)));
            Bitboard mask  = rook_attacks_slow(sq, 0) & ~edges;
            int bits  = popcount(mask);
            int shift = 64 - bits;
            rook_magic_[sq].mask    = mask;
            rook_magic_[sq].magic   = ROOK_MAGICS[sq];
            rook_magic_[sq].shift   = shift;
            rook_magic_[sq].attacks = rook_attacks_ + rook_offset;

            std::vector<Bitboard> subsets;
            enumerate_subsets(mask, subsets);
            for (Bitboard occ : subsets) {
                size_t idx = (occ * ROOK_MAGICS[sq]) >> shift;
                rook_attacks_[rook_offset + idx] = rook_attacks_slow(sq, occ);
            }
            rook_offset += (1ULL << bits);
        }
        // ── Bishop ────────────────────────────────────────────────────────
        {
            Bitboard edges = 0xFF818181818181FFULL;
            Bitboard mask  = bishop_attacks_slow(sq, 0) & ~edges;
            int bits  = popcount(mask);
            int shift = 64 - bits;
            bishop_magic_[sq].mask    = mask;
            bishop_magic_[sq].magic   = BISHOP_MAGICS[sq];
            bishop_magic_[sq].shift   = shift;
            bishop_magic_[sq].attacks = bishop_attacks_ + bishop_offset;

            std::vector<Bitboard> subsets;
            enumerate_subsets(mask, subsets);
            for (Bitboard occ : subsets) {
                size_t idx = (occ * BISHOP_MAGICS[sq]) >> shift;
                bishop_attacks_[bishop_offset + idx] = bishop_attacks_slow(sq, occ);
            }
            bishop_offset += (1ULL << bits);
        }
    }
}

// ─── Attack lookups ───────────────────────────────────────────────────────────
Bitboard ChessBoard::rook_attacks(Square sq, Bitboard occ) const {
    const MagicEntry& e = rook_magic_[sq];
    return e.attacks[((occ & e.mask) * e.magic) >> e.shift];
}
Bitboard ChessBoard::bishop_attacks(Square sq, Bitboard occ) const {
    const MagicEntry& e = bishop_magic_[sq];
    return e.attacks[((occ & e.mask) * e.magic) >> e.shift];
}
Bitboard ChessBoard::queen_attacks(Square sq, Bitboard occ) const {
    return rook_attacks(sq, occ) | bishop_attacks(sq, occ);
}

// ─── Constructor ──────────────────────────────────────────────────────────────
ChessBoard::ChessBoard() {
    if (!magics_ready_) {
        init_zobrist();
        init_static_attacks();
        init_magics();
        magics_ready_ = true;
    }
    set_startpos();
}

void ChessBoard::init_static_attacks() {
    // Done inside init_magics; separated for clarity
}

// ─── Setup ────────────────────────────────────────────────────────────────────
void ChessBoard::set_startpos() {
    set_fen(START_FEN);
}

bool ChessBoard::set_fen(std::string_view fen) {
    state_ = BoardState{};
    history_.clear();

    std::istringstream ss{std::string(fen)};
    std::string board_str, side_str, castle_str, ep_str;
    ss >> board_str >> side_str >> castle_str >> ep_str;
    ss >> state_.fifty >> state_.full_move;

    // Parse piece placement
    int rank = 7, file = 0;
    for (char c : board_str) {
        if (c == '/') { --rank; file = 0; }
        else if (std::isdigit(c)) { file += c - '0'; }
        else {
            Color col = std::isupper(c) ? WHITE : BLACK;
            char  lc  = std::tolower(c);
            Piece p   = PIECE_NONE;
            if      (lc == 'p') p = PAWN;
            else if (lc == 'n') p = KNIGHT;
            else if (lc == 'b') p = BISHOP;
            else if (lc == 'r') p = ROOK;
            else if (lc == 'q') p = QUEEN;
            else if (lc == 'k') p = KING;
            if (p != PIECE_NONE) {
                Square sq = make_square(static_cast<File>(file),
                                        static_cast<Rank>(rank));
                state_.pieces[col][p] |= bb(sq);
            }
            ++file;
        }
    }

    // Update occupancy
    for (int c = 0; c < 2; ++c) {
        state_.occ[c] = 0;
        for (int p = 0; p < 6; ++p) state_.occ[c] |= state_.pieces[c][p];
    }
    state_.all_occ = state_.occ[WHITE] | state_.occ[BLACK];

    state_.side_to_move = (side_str == "b") ? BLACK : WHITE;

    state_.castling = 0;
    if (castle_str != "-") {
        if (castle_str.find('K') != std::string::npos) state_.castling |= 0b0001;
        if (castle_str.find('Q') != std::string::npos) state_.castling |= 0b0010;
        if (castle_str.find('k') != std::string::npos) state_.castling |= 0b0100;
        if (castle_str.find('q') != std::string::npos) state_.castling |= 0b1000;
    }

    if (ep_str != "-") {
        int f = ep_str[0] - 'a';
        int r = ep_str[1] - '1';
        state_.ep_sq = make_square(static_cast<File>(f), static_cast<Rank>(r));
    } else {
        state_.ep_sq = NO_SQ;
    }

    state_.hash = compute_hash();
    return true;
}

std::string ChessBoard::get_fen() const {
    std::ostringstream ss;
    static const char piece_chars[2][6] = {
        {'P','N','B','R','Q','K'},
        {'p','n','b','r','q','k'},
    };
    for (int r = 7; r >= 0; --r) {
        int empty = 0;
        for (int f = 0; f < 8; ++f) {
            Square sq = make_square(static_cast<File>(f), static_cast<Rank>(r));
            bool found = false;
            for (int c = 0; c < 2; ++c)
                for (int p = 0; p < 6; ++p)
                    if (state_.pieces[c][p] & bb(sq)) {
                        if (empty) { ss << empty; empty = 0; }
                        ss << piece_chars[c][p];
                        found = true;
                    }
            if (!found) ++empty;
        }
        if (empty) ss << empty;
        if (r > 0) ss << '/';
    }
    ss << ' ' << (state_.side_to_move == WHITE ? 'w' : 'b') << ' ';
    if (!state_.castling) ss << '-';
    else {
        if (state_.castling & 0b0001) ss << 'K';
        if (state_.castling & 0b0010) ss << 'Q';
        if (state_.castling & 0b0100) ss << 'k';
        if (state_.castling & 0b1000) ss << 'q';
    }
    ss << ' ';
    if (state_.ep_sq == NO_SQ) ss << '-';
    else ss << "abcdefgh"[sq_file(state_.ep_sq)]
            << "12345678"[sq_rank(state_.ep_sq)];
    ss << ' ' << state_.fifty << ' ' << state_.full_move;
    return ss.str();
}

// ─── is_attacked ──────────────────────────────────────────────────────────────
bool ChessBoard::is_attacked(Square sq, Color by) const {
    Bitboard occ = state_.all_occ;
    if (pawn_attacks_[1-by][sq]        & state_.pieces[by][PAWN])   return true;
    if (knight_attacks_[sq]             & state_.pieces[by][KNIGHT]) return true;
    if (bishop_attacks(sq, occ)         & (state_.pieces[by][BISHOP]
                                         | state_.pieces[by][QUEEN])) return true;
    if (rook_attacks(sq, occ)           & (state_.pieces[by][ROOK]
                                         | state_.pieces[by][QUEEN])) return true;
    if (king_attacks_[sq]               & state_.pieces[by][KING])   return true;
    return false;
}

bool ChessBoard::in_check() const {
    Square king_sq = lsb(state_.pieces[state_.side_to_move][KING]);
    return is_attacked(king_sq, static_cast<Color>(1 - state_.side_to_move));
}

// ─── Move generation ──────────────────────────────────────────────────────────
std::vector<Move> ChessBoard::gen_moves() const {
    std::vector<Move> ml;
    ml.reserve(48);
    gen_pawn_moves(ml, false);
    gen_knight_moves(ml, false);
    gen_bishop_moves(ml, false);
    gen_rook_moves(ml, false);
    gen_queen_moves(ml, false);
    gen_king_moves(ml, false);
    gen_castling(ml);
    return ml;
}
std::vector<Move> ChessBoard::gen_captures() const {
    std::vector<Move> ml;
    ml.reserve(16);
    gen_pawn_moves(ml, true);
    gen_knight_moves(ml, true);
    gen_bishop_moves(ml, true);
    gen_rook_moves(ml, true);
    gen_queen_moves(ml, true);
    gen_king_moves(ml, true);
    return ml;
}

void ChessBoard::gen_pawn_moves(std::vector<Move>& ml, bool cap_only) const {
    Color us   = state_.side_to_move;
    Color them = static_cast<Color>(1 - us);
    Bitboard pawns = state_.pieces[us][PAWN];
    Bitboard empty = ~state_.all_occ;
    Bitboard their = state_.occ[them];

    int push_dir = (us == WHITE) ? 8 : -8;
    Rank promo_rank = (us == WHITE) ? R8 : R1;
    Rank start_rank = (us == WHITE) ? R2 : R7;

    while (pawns) {
        Square from = pop_lsb(pawns);
        // Single push
        if (!cap_only) {
            Square to1 = from + push_dir;
            if (to1 >= 0 && to1 < 64 && (empty & bb(to1))) {
                if (sq_rank(to1) == promo_rank) {
                    for (int p = 1; p <= 4; ++p) ml.push_back(Move(from, to1, p, 0));
                } else {
                    ml.push_back(Move(from, to1, 0, 0));
                    // Double push
                    if (sq_rank(from) == start_rank) {
                        Square to2 = from + push_dir * 2;
                        if (empty & bb(to2)) ml.push_back(Move(from, to2, 0, 0));
                    }
                }
            }
        }
        // Captures
        Bitboard atk = pawn_attacks_[us][from];
        Bitboard cap = atk & their;
        while (cap) {
            Square to = pop_lsb(cap);
            if (sq_rank(to) == promo_rank)
                for (int p = 1; p <= 4; ++p) ml.push_back(Move(from, to, p, 1));
            else
                ml.push_back(Move(from, to, 0, 1));
        }
        // En passant
        if (state_.ep_sq != NO_SQ && (atk & bb(state_.ep_sq)))
            ml.push_back(Move(from, state_.ep_sq, 0, 2));
    }
}

void ChessBoard::gen_knight_moves(std::vector<Move>& ml, bool cap_only) const {
    Color us   = state_.side_to_move;
    Color them = static_cast<Color>(1 - us);
    Bitboard knights = state_.pieces[us][KNIGHT];
    while (knights) {
        Square from = pop_lsb(knights);
        Bitboard atk = knight_attacks_[from] & ~state_.occ[us];
        if (cap_only) atk &= state_.occ[them];
        while (atk) {
            Square to = pop_lsb(atk);
            int flags = (state_.occ[them] & bb(to)) ? 1 : 0;
            ml.push_back(Move(from, to, 0, flags));
        }
    }
}

void ChessBoard::gen_bishop_moves(std::vector<Move>& ml, bool cap_only) const {
    Color us   = state_.side_to_move;
    Color them = static_cast<Color>(1 - us);
    Bitboard bishops = state_.pieces[us][BISHOP];
    while (bishops) {
        Square from = pop_lsb(bishops);
        Bitboard atk = bishop_attacks(from, state_.all_occ) & ~state_.occ[us];
        if (cap_only) atk &= state_.occ[them];
        while (atk) {
            Square to = pop_lsb(atk);
            int flags = (state_.occ[them] & bb(to)) ? 1 : 0;
            ml.push_back(Move(from, to, 0, flags));
        }
    }
}

void ChessBoard::gen_rook_moves(std::vector<Move>& ml, bool cap_only) const {
    Color us   = state_.side_to_move;
    Color them = static_cast<Color>(1 - us);
    Bitboard rooks = state_.pieces[us][ROOK];
    while (rooks) {
        Square from = pop_lsb(rooks);
        Bitboard atk = rook_attacks(from, state_.all_occ) & ~state_.occ[us];
        if (cap_only) atk &= state_.occ[them];
        while (atk) {
            Square to = pop_lsb(atk);
            int flags = (state_.occ[them] & bb(to)) ? 1 : 0;
            ml.push_back(Move(from, to, 0, flags));
        }
    }
}

void ChessBoard::gen_queen_moves(std::vector<Move>& ml, bool cap_only) const {
    Color us   = state_.side_to_move;
    Color them = static_cast<Color>(1 - us);
    Bitboard queens = state_.pieces[us][QUEEN];
    while (queens) {
        Square from = pop_lsb(queens);
        Bitboard atk = queen_attacks(from, state_.all_occ) & ~state_.occ[us];
        if (cap_only) atk &= state_.occ[them];
        while (atk) {
            Square to = pop_lsb(atk);
            int flags = (state_.occ[them] & bb(to)) ? 1 : 0;
            ml.push_back(Move(from, to, 0, flags));
        }
    }
}

void ChessBoard::gen_king_moves(std::vector<Move>& ml, bool cap_only) const {
    Color us   = state_.side_to_move;
    Color them = static_cast<Color>(1 - us);
    Square from = lsb(state_.pieces[us][KING]);
    Bitboard atk = king_attacks_[from] & ~state_.occ[us];
    if (cap_only) atk &= state_.occ[them];
    while (atk) {
        Square to = pop_lsb(atk);
        int flags = (state_.occ[them] & bb(to)) ? 1 : 0;
        ml.push_back(Move(from, to, 0, flags));
    }
}

void ChessBoard::gen_castling(std::vector<Move>& ml) const {
    Color us = state_.side_to_move;
    if (in_check()) return;

    if (us == WHITE) {
        // King-side
        if ((state_.castling & 0b0001)
            && !(state_.all_occ & 0x60ULL)
            && !is_attacked(4, BLACK) && !is_attacked(5, BLACK) && !is_attacked(6, BLACK))
            ml.push_back(Move(4, 6, 0, 3));
        // Queen-side
        if ((state_.castling & 0b0010)
            && !(state_.all_occ & 0xEULL)
            && !is_attacked(4, BLACK) && !is_attacked(3, BLACK) && !is_attacked(2, BLACK))
            ml.push_back(Move(4, 2, 0, 4));
    } else {
        // King-side
        if ((state_.castling & 0b0100)
            && !(state_.all_occ & 0x6000000000000000ULL)
            && !is_attacked(60, WHITE) && !is_attacked(61, WHITE) && !is_attacked(62, WHITE))
            ml.push_back(Move(60, 62, 0, 3));
        // Queen-side
        if ((state_.castling & 0b1000)
            && !(state_.all_occ & 0x0E00000000000000ULL)
            && !is_attacked(60, WHITE) && !is_attacked(59, WHITE) && !is_attacked(58, WHITE))
            ml.push_back(Move(60, 58, 0, 4));
    }
}

// ─── Make / Unmake ────────────────────────────────────────────────────────────
bool ChessBoard::make_move(Move m) {
    // Null move
    if (m.is_null()) {
        history_.push_back(state_);
        state_.side_to_move = static_cast<Color>(1 - state_.side_to_move);
        state_.ep_sq = NO_SQ;
        state_.hash  = compute_hash();
        return true;
    }

    history_.push_back(state_);
    BoardState& s = state_;

    Color us   = s.side_to_move;
    Color them = static_cast<Color>(1 - us);
    Square from = m.from(), to = m.to();

    // Find moving piece
    Piece moving = PIECE_NONE;
    for (int p = 0; p < 6; ++p)
        if (s.pieces[us][p] & bb(from)) { moving = static_cast<Piece>(p); break; }

    // Remove captured piece
    if (m.flags() == 2) {
        // En passant: captured pawn is on a different square
        Square cap_sq = to + (us == WHITE ? -8 : 8);
        for (int p = 0; p < 6; ++p) s.pieces[them][p] &= ~bb(cap_sq);
    } else {
        for (int p = 0; p < 6; ++p) s.pieces[them][p] &= ~bb(to);
    }

    // Move piece
    s.pieces[us][moving] &= ~bb(from);
    Piece placed = moving;
    if (m.promo() > 0) {
        static const Piece promo_map[] = {PIECE_NONE, KNIGHT, BISHOP, ROOK, QUEEN};
        placed = promo_map[m.promo()];
    }
    s.pieces[us][placed] |= bb(to);

    // Castling rook
    if (moving == KING) {
        if (m.flags() == 3) {  // King-side
            Square rook_from = (us == WHITE) ? 7  : 63;
            Square rook_to   = (us == WHITE) ? 5  : 61;
            s.pieces[us][ROOK] &= ~bb(rook_from);
            s.pieces[us][ROOK] |=  bb(rook_to);
        } else if (m.flags() == 4) {  // Queen-side
            Square rook_from = (us == WHITE) ? 0  : 56;
            Square rook_to   = (us == WHITE) ? 3  : 59;
            s.pieces[us][ROOK] &= ~bb(rook_from);
            s.pieces[us][ROOK] |=  bb(rook_to);
        }
        // Revoke castling rights
        if (us == WHITE) s.castling &= ~0b0011;
        else             s.castling &= ~0b1100;
    }
    // Rook moves revoke castling
    if (moving == ROOK) {
        if      (from == 0)  s.castling &= ~0b0010;
        else if (from == 7)  s.castling &= ~0b0001;
        else if (from == 56) s.castling &= ~0b1000;
        else if (from == 63) s.castling &= ~0b0100;
    }
    // Captures on rook squares revoke castling
    if      (to == 0)  s.castling &= ~0b0010;
    else if (to == 7)  s.castling &= ~0b0001;
    else if (to == 56) s.castling &= ~0b1000;
    else if (to == 63) s.castling &= ~0b0100;

    // En passant target
    s.ep_sq = NO_SQ;
    if (moving == PAWN && std::abs(to - from) == 16)
        s.ep_sq = from + (us == WHITE ? 8 : -8);

    // Fifty-move clock
    if (moving == PAWN || m.is_capture()) s.fifty = 0;
    else ++s.fifty;

    if (us == BLACK) ++s.full_move;
    s.side_to_move = them;

    // Rebuild occupancy
    for (int c = 0; c < 2; ++c) {
        s.occ[c] = 0;
        for (int p = 0; p < 6; ++p) s.occ[c] |= s.pieces[c][p];
    }
    s.all_occ = s.occ[WHITE] | s.occ[BLACK];
    s.hash    = compute_hash();

    // Check legality: our king must not be in check
    Square king_sq = lsb(s.pieces[us][KING]);
    if (is_attacked(king_sq, them)) {
        // Illegal — restore
        state_ = history_.back();
        history_.pop_back();
        return false;
    }
    return true;
}

void ChessBoard::unmake_move() {
    if (!history_.empty()) {
        state_ = history_.back();
        history_.pop_back();
    }
}

// ─── Draw detection ───────────────────────────────────────────────────────────
bool ChessBoard::is_draw() const {
    // 50-move rule
    if (state_.fifty >= 100) return true;

    // Insufficient material: only kings left, or K+B/N vs K
    int total = popcount(state_.all_occ);
    if (total == 2) return true;  // KvK
    if (total == 3) {
        if (popcount(state_.pieces[WHITE][BISHOP] | state_.pieces[WHITE][KNIGHT]
                   | state_.pieces[BLACK][BISHOP] | state_.pieces[BLACK][KNIGHT]) == 1)
            return true;
    }

    // Threefold repetition (simplified: check hash appears 3× in history)
    int count = 1;
    for (auto it = history_.rbegin(); it != history_.rend(); ++it) {
        if (it->hash == state_.hash) ++count;
        if (count >= 3) return true;
        if (it->fifty == 0) break;  // Can't have repeated before irreversible move
    }

    return false;
}

// ─── Neural net input ─────────────────────────────────────────────────────────
void ChessBoard::fill_nn_input(float* out) const {
    std::memset(out, 0, CNN_PLANES * CNN_H * CNN_W * sizeof(float));
    // Plane order: WP WN WB WR WQ WK BP BN BB BR BQ BK
    // Always from White's perspective (rank 0 = a1)
    for (int c = 0; c < 2; ++c)
        for (int p = 0; p < 6; ++p) {
            int plane = c * 6 + p;
            Bitboard b = state_.pieces[c][p];
            while (b) {
                Square sq   = pop_lsb(b);
                int rank    = sq_rank(sq);
                int file    = sq_file(sq);
                out[plane * 64 + rank * 8 + file] = 1.0f;
            }
        }
}

// ─── Print ────────────────────────────────────────────────────────────────────
void ChessBoard::print() const {
    static const char* piece_str[2][6] = {
        {"P","N","B","R","Q","K"},
        {"p","n","b","r","q","k"},
    };
    std::cout << "\n  +---+---+---+---+---+---+---+---+\n";
    for (int r = 7; r >= 0; --r) {
        std::cout << (r+1) << " |";
        for (int f = 0; f < 8; ++f) {
            Square sq = make_square(static_cast<File>(f), static_cast<Rank>(r));
            bool found = false;
            for (int c = 0; c < 2; ++c)
                for (int p = 0; p < 6; ++p)
                    if (state_.pieces[c][p] & bb(sq)) {
                        std::cout << " " << piece_str[c][p] << " |";
                        found = true;
                    }
            if (!found) std::cout << "   |";
        }
        std::cout << "\n  +---+---+---+---+---+---+---+---+\n";
    }
    std::cout << "    a   b   c   d   e   f   g   h\n";
    std::cout << "  FEN: " << get_fen() << "\n";
    std::cout << "  Side: " << (state_.side_to_move == WHITE ? "White" : "Black")
              << "  Hash: " << std::hex << state_.hash << std::dec << "\n\n";
}
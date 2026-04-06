// engine.cpp — Hybrid alpha-beta + MCTS search implementation
#include "engine.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

using Clock = std::chrono::steady_clock;
using ms    = std::chrono::milliseconds;

// ─── Constants ────────────────────────────────────────────────────────────────
static constexpr int INF          = 1'000'000;
static constexpr int MATE_SCORE   = 900'000;
static constexpr int DRAW_SCORE   = 0;
static constexpr int NMP_DEPTH    = 3;   // Null-move pruning minimum depth
static constexpr int NMP_REDUCE   = 3;   // Null-move reduction
static constexpr int LMR_MIN_MOVE = 3;   // Start LMR after this many moves
static constexpr int LMR_DEPTH    = 3;   // Minimum depth for LMR

// ─── Engine constructor ───────────────────────────────────────────────────────
Engine::Engine(SearchMode mode, EvalMode eval, size_t tt_mb)
    : evaluator_(eval), tt_(tt_mb), search_mode_(mode) {
    board_.set_startpos();
}

void Engine::new_game() {
    board_.set_startpos();
    tt_.clear();
    clear_killers();
    clear_history();
    nodes_ = 0;
}

// ─── Top-level search dispatcher ──────────────────────────────────────────────
SearchResult Engine::search(const SearchLimits& limits) {
    stop_flag_.store(false, std::memory_order_relaxed);
    nodes_       = 0;
    seldepth_    = 0;
    search_start_ = Clock::now();
    allotted_ms_ = calc_allotted_time(limits);
    tt_.new_search();

    SearchMode effective = search_mode_;
    if (effective == SearchMode::AUTO) {
        // Simple heuristic: use MCTS if in complex middlegame
        auto moves = board_.gen_moves();
        effective = (moves.size() > 35) ? SearchMode::MCTS : SearchMode::ALPHABETA;
    }

    if (effective == SearchMode::MCTS)
        return search_mcts(limits);
    return search_ab(limits);
}

// ─────────────────────────────────────────────────────────────────────────────
// ALPHA-BETA SEARCH
// ─────────────────────────────────────────────────────────────────────────────

SearchResult Engine::search_ab(const SearchLimits& limits) {
    clear_killers();
    clear_history();

    int max_depth = (limits.depth > 0 && limits.depth < MAX_DEPTH)
                    ? limits.depth : MAX_DEPTH - 1;

    SearchResult best{};
    best.score = -INF;
    std::vector<Move> best_pv;

    // Iterative deepening
    for (int depth = 1; depth <= max_depth; ++depth) {
        if (time_up()) break;
        if (stop_flag_.load(std::memory_order_relaxed)) break;

        int score = negamax(depth, -INF, INF, 0, /*null_ok=*/true);

        if (stop_flag_.load(std::memory_order_relaxed) && depth > 1) break;

        best.score = score;
        best.depth = depth;

        // Retrieve PV from TT
        {
            std::vector<Move> pv;
            ChessBoard tmp = board_;
            for (int i = 0; i < depth; ++i) {
                uint32_t tt_move = 0;
                int dummy_score;
                TTFlag dummy_flag;
                tt_.probe(tmp.hash(), 0, -INF, INF, dummy_score, tt_move, dummy_flag);
                if (!tt_move) break;
                Move m; m.data = tt_move;
                pv.push_back(m);
                if (!tmp.make_move(m)) break;
            }
            best_pv = pv;
            if (!pv.empty()) best.best_move = pv[0];
            if (pv.size() > 1) best.ponder_move = pv[1];
        }

        int elapsed = static_cast<int>(
            std::chrono::duration_cast<ms>(Clock::now() - search_start_).count());
        emit_info(depth, score, nodes_, elapsed, best_pv);
    }

    best.nodes    = nodes_;
    best.seldepth = seldepth_;
    best.hashfull = tt_.hashfull();
    best.time_ms  = static_cast<int>(
        std::chrono::duration_cast<ms>(Clock::now() - search_start_).count());

    return best;
}

int Engine::negamax(int depth, int alpha, int beta, int ply, bool null_ok) {
    if (stop_flag_.load(std::memory_order_relaxed) || time_up()) return 0;

    seldepth_ = std::max(seldepth_, ply);

    // ── Draws ──────────────────────────────────────────────────────────────
    if (ply > 0 && (board_.is_draw() || is_draw_by_repetition()))
        return DRAW_SCORE;

    // ── TT probe ───────────────────────────────────────────────────────────
    uint32_t tt_move = 0;
    int tt_score;
    TTFlag tt_flag;
    if (tt_.probe(board_.hash(), depth, alpha, beta, tt_score, tt_move, tt_flag)) {
        if (ply > 0) return tt_score;
    }

    // ── Quiescence at depth 0 ──────────────────────────────────────────────
    if (depth <= 0) return quiesce(alpha, beta, ply);

    bool in_check = board_.in_check();

    // ── Null-move pruning ──────────────────────────────────────────────────
    // Skip if: in check, at low depth, or position is a zugzwang candidate
    if (null_ok && !in_check && depth >= NMP_DEPTH) {
        // TODO: better zugzwang detection (e.g. only pawns left)
        board_.make_move(NULL_MOVE);
        int null_score = -negamax(depth - 1 - NMP_REDUCE, -beta, -beta + 1,
                                   ply + 1, /*null_ok=*/false);
        board_.unmake_move();
        if (null_score >= beta) return beta;  // Null-move cutoff
    }

    // ── Move generation and ordering ───────────────────────────────────────
    auto moves = board_.gen_moves();
    if (moves.empty()) {
        // Checkmate or stalemate
        return in_check ? -(MATE_SCORE - ply) : DRAW_SCORE;
    }
    order_moves(moves, ply, tt_move);

    Move  best_move;
    int   best_score = -INF;
    bool  raised_alpha = false;
    int   move_idx = 0;

    for (Move m : moves) {
        if (!board_.make_move(m)) { ++move_idx; continue; }
        ++nodes_;

        int score;

        // ── Late move reduction (LMR) ──────────────────────────────────
        if (move_idx >= LMR_MIN_MOVE && depth >= LMR_DEPTH
            && !in_check && !m.is_capture() && m.promo() == 0) {
            // Reduced-depth search
            int reduce = 1 + static_cast<int>(std::log(depth) * std::log(move_idx) / 2.0);
            score = -negamax(depth - 1 - reduce, -alpha - 1, -alpha, ply + 1, true);
            // Re-search at full depth if it unexpectedly raises alpha
            if (score > alpha && score < beta)
                score = -negamax(depth - 1, -beta, -alpha, ply + 1, true);
        } else if (move_idx > 0) {
            // Principal variation search — zero-window first
            score = -negamax(depth - 1, -alpha - 1, -alpha, ply + 1, true);
            if (score > alpha && score < beta)
                score = -negamax(depth - 1, -beta, -alpha, ply + 1, true);
        } else {
            // First move: full window
            score = -negamax(depth - 1, -beta, -alpha, ply + 1, true);
        }

        board_.unmake_move();

        if (score > best_score) {
            best_score = score;
            best_move  = m;
        }
        if (score > alpha) {
            alpha       = score;
            raised_alpha = true;
            // ── Killer + history update ────────────────────────────────
            if (!m.is_capture()) {
                if (killers_[ply][0] != m) {
                    killers_[ply][1] = killers_[ply][0];
                    killers_[ply][0] = m;
                }
                history_[board_.side()][m.from()][m.to()] +=
                    depth * depth;  // Depth-squared bonus
            }
        }
        if (alpha >= beta) {
            // Beta cutoff
            tt_.store(board_.hash(), depth, best_score,
                      best_move.data, TT_LOWER);
            return beta;
        }
        ++move_idx;
    }

    // ── Store TT ───────────────────────────────────────────────────────────
    TTFlag flag = raised_alpha ? TT_EXACT : TT_UPPER;
    tt_.store(board_.hash(), depth, best_score, best_move.data, flag);

    return best_score;
}

int Engine::quiesce(int alpha, int beta, int ply) {
    ++nodes_;
    seldepth_ = std::max(seldepth_, ply);

    int stand_pat = evaluator_.eval(board_);

    if (stand_pat >= beta) return beta;
    if (stand_pat > alpha) alpha = stand_pat;

    auto captures = board_.gen_captures();
    order_moves(captures, ply, 0);

    for (Move m : captures) {
        if (!board_.make_move(m)) continue;
        int score = -quiesce(-beta, -alpha, ply + 1);
        board_.unmake_move();
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

// ─────────────────────────────────────────────────────────────────────────────
// MCTS SEARCH
// ─────────────────────────────────────────────────────────────────────────────

MCTSNode::MCTSNode(const ChessBoard& b, Move m, MCTSNode* par)
    : board(b), move(m), parent(par) {
    untried_moves = board.gen_moves();
    is_terminal   = untried_moves.empty();
}

double MCTSNode::ucb1(double exploration) const {
    if (visits == 0) return 1e9;
    double q = total_value / visits;
    double u = exploration * std::sqrt(std::log(parent->visits) / visits);
    return q + u;
}

MCTSNode* MCTSNode::best_child() const {
    MCTSNode* best = nullptr;
    double    best_val = -1e18;
    for (auto& child : children) {
        double v = child->ucb1();
        if (v > best_val) { best_val = v; best = child.get(); }
    }
    return best;
}

MCTSNode* MCTSNode::select() {
    MCTSNode* node = this;
    while (!node->is_terminal && node->untried_moves.empty())
        node = node->best_child();
    return node;
}

MCTSNode* MCTSNode::expand() {
    if (untried_moves.empty()) return this;
    // Pick an untried move at random
    int idx = rand() % static_cast<int>(untried_moves.size());
    Move m  = untried_moves[idx];
    untried_moves.erase(untried_moves.begin() + idx);

    ChessBoard child_board = board;
    if (!child_board.make_move(m)) return this;

    children.push_back(std::make_unique<MCTSNode>(child_board, m, this));
    return children.back().get();
}

double Engine::rollout(MCTSNode* node) {
    // Use the neural net / HCE evaluator for a static rollout
    // (no random playout — this is more like AlphaZero than UCT)
    float val = static_cast<float>(evaluator_.eval(node->board)) / 600.0f;
    // Clamp to [-1, 1]
    val = std::max(-1.0f, std::min(1.0f, val));
    // Negate if black to move (we always want White's perspective)
    if (node->board.side() == BLACK) val = -val;
    return static_cast<double>(val);
}

SearchResult Engine::search_mcts(const SearchLimits& limits) {
    (void)limits;
    auto root = std::make_unique<MCTSNode>(board_);

    int iterations = 0;
    while (!time_up() && !stop_flag_.load(std::memory_order_relaxed)) {
        // 1. Select
        MCTSNode* node = root->select();
        // 2. Expand
        if (!node->is_terminal) node = node->expand();
        // 3. Simulate (static eval as rollout)
        double val = rollout(node);
        // 4. Backpropagate
        MCTSNode* cur = node;
        while (cur) {
            cur->visits++;
            // Negate at each ply (alternating sides)
            cur->total_value += (cur->board.side() == WHITE) ? val : -val;
            cur = cur->parent;
        }
        ++nodes_;
        ++iterations;
    }

    // Best move = child with most visits
    SearchResult res{};
    int  best_visits = -1;
    for (auto& child : root->children) {
        if (child->visits > best_visits) {
            best_visits   = child->visits;
            res.best_move = child->move;
        }
    }

    res.nodes    = nodes_;
    res.depth    = static_cast<int>(std::log2(iterations + 1));
    res.hashfull = 0;
    res.time_ms  = static_cast<int>(
        std::chrono::duration_cast<ms>(Clock::now() - search_start_).count());
    return res;
}

// ─── Move ordering ────────────────────────────────────────────────────────────
// Priority (highest first):
//   1. TT best move
//   2. Winning captures (MVV-LVA)
//   3. Killer moves
//   4. History heuristic
//   5. Quiet moves (FIFO)

void Engine::order_moves(std::vector<Move>& moves, int ply,
                          uint32_t tt_move) const {
    std::stable_sort(moves.begin(), moves.end(),
        [&](Move a, Move b) {
            return move_score(a, ply, tt_move) > move_score(b, ply, tt_move);
        });
}

int Engine::move_score(Move m, int ply, uint32_t tt_move) const {
    if (m.data == tt_move) return 1'000'000;

    if (m.is_capture()) {
        // MVV-LVA: Most Valuable Victim / Least Valuable Aggressor
        // (We don't track piece types in Move, so just give a flat bonus)
        // TODO: track captured piece type in Move flags for proper MVV-LVA
        return 800'000;
    }

    if (m.promo() != 0) return 900'000;

    // Killer moves
    if (ply < MAX_DEPTH) {
        if (killers_[ply][0] == m) return 700'000;
        if (killers_[ply][1] == m) return 699'000;
    }

    // History
    Color c = board_.side();
    return history_[c][m.from()][m.to()];
}

// ─── Time management ──────────────────────────────────────────────────────────
int Engine::calc_allotted_time(const SearchLimits& lim) const {
    if (lim.infinite) return INT32_MAX;
    if (lim.movetime_ms > 0) return lim.movetime_ms;

    int time_left = (board_.side() == WHITE) ? lim.wtime_ms : lim.btime_ms;
    int inc       = (board_.side() == WHITE) ? lim.winc_ms  : lim.binc_ms;

    if (time_left <= 0) return 1000;  // Fallback

    int moves_left = (lim.movestogo > 0) ? lim.movestogo : 30;
    int allotted   = time_left / moves_left + inc / 2;
    // Never use more than 20% of remaining time on a single move
    allotted = std::min(allotted, time_left / 5);
    return std::max(allotted, 50);
}

bool Engine::time_up() const {
    if (allotted_ms_ == INT32_MAX) return false;
    auto elapsed = std::chrono::duration_cast<ms>(
        Clock::now() - search_start_).count();
    return elapsed >= allotted_ms_;
}

// ─── UCI info output ──────────────────────────────────────────────────────────
void Engine::emit_info(int depth, int score, uint64_t nodes, int time_ms,
                        const std::vector<Move>& pv) {
    if (!info_cb_) return;
    std::ostringstream oss;
    oss << "info depth " << depth
        << " score cp " << score
        << " nodes " << nodes
        << " time " << time_ms
        << " nps " << (time_ms > 0 ? nodes * 1000 / time_ms : 0)
        << " hashfull " << tt_.hashfull()
        << " pv";
    for (Move m : pv) oss << " " << m.uci();
    info_cb_(oss.str());
}

// ─── Repetition detection ──────────────────────────────────────────────────────
bool Engine::is_draw_by_repetition() const {
    // Handled inside ChessBoard::is_draw() via history_ comparison
    return false;  // Placeholder
}

// ─── Housekeeping ─────────────────────────────────────────────────────────────
void Engine::clear_killers() {
    for (int d = 0; d < MAX_DEPTH; ++d)
        killers_[d][0] = killers_[d][1] = NULL_MOVE;
}

void Engine::clear_history() {
    std::memset(history_, 0, sizeof(history_));
}

// ─── Perft ────────────────────────────────────────────────────────────────────
uint64_t Engine::perft(int depth) {
    return perft_internal(board_, depth);
}

uint64_t Engine::perft_internal(ChessBoard& b, int depth) {
    if (depth == 0) return 1;
    auto moves = b.gen_moves();
    uint64_t total = 0;
    for (Move m : moves) {
        if (!b.make_move(m)) continue;
        total += perft_internal(b, depth - 1);
        b.unmake_move();
    }
    return total;
}
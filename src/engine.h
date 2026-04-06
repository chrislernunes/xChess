#pragma once
// engine.h — Hybrid alpha-beta + MCTS search controller
//
// Search strategy:
//   • Alpha-beta with iterative deepening (ID) handles tactical depth.
//     Standard enhancements: null-move pruning, LMR, futility pruning,
//     killer heuristic, history heuristic, TT move ordering.
//   • MCTS handles strategic breadth at the root.  UCB1 selection, neural
//     net value + policy from the evaluator, virtual loss for (future) parallel.
//   • HybridMode::AUTO switches to MCTS when the AB tree is too wide
//     (branching factor > threshold at root) or time is tight.
//
// Scores are always in centipawns from White's perspective internally;
// the UCI interface converts to side-to-move perspective before output.
//
// TODO: Add aspiration windows around iterative deepening scores
// TODO: Add late-endgame tablebase probe (Syzygy)
// TODO: Implement parallel MCTS with virtual loss

#include "chess_board.h"
#include "evaluator.h"
#include "transposition.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <vector>

// ─── Search mode ──────────────────────────────────────────────────────────────
enum class SearchMode {
    ALPHABETA,  ///< Pure minimax + alpha-beta
    MCTS,       ///< Pure MCTS
    AUTO,       ///< Engine picks based on position complexity
};

// ─── Search limits ────────────────────────────────────────────────────────────
struct SearchLimits {
    int  depth{64};          ///< Max depth (0 = unlimited)
    int  movetime_ms{0};     ///< Exact thinking time in ms (0 = use wtime/btime)
    int  wtime_ms{0};
    int  btime_ms{0};
    int  winc_ms{0};
    int  binc_ms{0};
    int  movestogo{0};
    bool infinite{false};
    bool ponder{false};
};

// ─── Search result ────────────────────────────────────────────────────────────
struct SearchResult {
    Move    best_move;
    Move    ponder_move;
    int     score;           ///< Centipawns, side-to-move perspective
    int     depth;
    int     seldepth;
    uint64_t nodes;
    int     time_ms;
    int     hashfull;        ///< TT fill in per-mille
};

// ─── MCTS node ────────────────────────────────────────────────────────────────
struct MCTSNode {
    ChessBoard  board;          ///< Board state at this node
    Move        move;           ///< Move that led here (null for root)
    MCTSNode*   parent{nullptr};

    std::vector<std::unique_ptr<MCTSNode>> children;
    std::vector<Move>                      untried_moves;

    double  total_value{0.0};
    int     visits{0};
    bool    is_terminal{false};

    explicit MCTSNode(const ChessBoard& b, Move m = NULL_MOVE,
                      MCTSNode* par = nullptr);

    double ucb1(double exploration = 1.414) const;
    MCTSNode* best_child() const;
    MCTSNode* select();
    MCTSNode* expand();
};

// ─── Engine ───────────────────────────────────────────────────────────────────
class Engine {
public:
    explicit Engine(SearchMode mode  = SearchMode::AUTO,
                    EvalMode   eval  = EvalMode::HCE,
                    size_t     tt_mb = 128);

    // ── Configuration ──────────────────────────────────────────────────────
    void set_search_mode(SearchMode m) { search_mode_ = m; }
    void set_eval_mode(EvalMode m)     { evaluator_.set_mode(m); }
    void set_tt_size(size_t mb)        { tt_.resize(mb); }
    bool load_models(const std::string& mlp, const std::string& cnn) {
        return evaluator_.load_models(mlp, cnn);
    }

    // ── Board management ───────────────────────────────────────────────────
    void new_game();
    ChessBoard& board() { return board_; }

    // ── Search ─────────────────────────────────────────────────────────────
    SearchResult search(const SearchLimits& limits);

    /// Stop an ongoing search (called from UCI "stop" or signal handler).
    void stop() { stop_flag_.store(true, std::memory_order_relaxed); }

    /// Output UCI info lines during search (set to nullptr to disable).
    using InfoCallback = std::function<void(const std::string&)>;
    void set_info_callback(InfoCallback cb) { info_cb_ = std::move(cb); }

    // ── Perft (move generation correctness test) ───────────────────────────
    uint64_t perft(int depth);

    // ── Diagnostics ────────────────────────────────────────────────────────
    uint64_t nodes_searched() const { return nodes_; }

private:
    ChessBoard      board_;
    Evaluator       evaluator_;
    TranspositionTable tt_;
    SearchMode      search_mode_;
    std::atomic<bool> stop_flag_{false};
    InfoCallback    info_cb_;

    // Stats
    uint64_t nodes_{0};
    int      seldepth_{0};
    std::chrono::steady_clock::time_point search_start_;
    int      allotted_ms_{0};

    // Killer moves: [depth][slot0, slot1]
    static constexpr int MAX_DEPTH = 128;
    Move killers_[MAX_DEPTH][2]{};

    // History heuristic [color][from][to]
    int history_[2][64][64]{};

    // ── Alpha-beta search ──────────────────────────────────────────────────
    SearchResult search_ab(const SearchLimits& limits);
    int          negamax(int depth, int alpha, int beta, int ply,
                         bool null_ok);
    int          quiesce(int alpha, int beta, int ply);

    // ── MCTS search ───────────────────────────────────────────────────────
    SearchResult search_mcts(const SearchLimits& limits);
    double       rollout(MCTSNode* node);

    // ── Move ordering ──────────────────────────────────────────────────────
    void order_moves(std::vector<Move>& moves, int ply, uint32_t tt_move) const;
    int  move_score(Move m, int ply, uint32_t tt_move) const;

    // ── Time management ────────────────────────────────────────────────────
    int  calc_allotted_time(const SearchLimits& lim) const;
    bool time_up() const;

    // ── UCI info printing ──────────────────────────────────────────────────
    void emit_info(int depth, int score, uint64_t nodes, int time_ms,
                   const std::vector<Move>& pv);

    // ── Utility ────────────────────────────────────────────────────────────
    bool is_draw_by_repetition() const;
    void clear_killers();
    void clear_history();
    uint64_t perft_internal(ChessBoard& b, int depth);
};

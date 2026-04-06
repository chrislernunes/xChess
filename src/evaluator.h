#pragma once
// evaluator.h — Evaluation interface with three levels:
//   1. HCE  — hand-crafted evaluation (material + piece-square tables)
//   2. MLP  — small multi-layer perceptron via LibTorch
//   3. CNN  — convolutional net treating the board as a 12×8×8 image
//
// The evaluator is selected at runtime via EvalMode.
// All scores are in centipawns from White's perspective.
//
// TODO: Implement NNUE-style incremental update for MLP eval

#include "chess_board.h"
#include <array>
#include <memory>
#include <string>
#include <vector>

#ifdef USE_TORCH
#include <torch/script.h>
#endif

// ─── Evaluation mode ──────────────────────────────────────────────────────────
enum class EvalMode {
    HCE,  ///< Hand-crafted evaluation (always available)
    MLP,  ///< Small MLP neural net
    CNN,  ///< CNN neural net (best quality, slowest single-pos)
};

// ─── Hand-crafted evaluation constants ───────────────────────────────────────
// Material values in centipawns
constexpr int PIECE_VALUE[6] = {
    100,   // Pawn
    320,   // Knight
    330,   // Bishop
    500,   // Rook
    900,   // Queen
    20000  // King (used only for check detection, never traded)
};

// Piece-square tables (middlegame) — White's perspective, a1=0, h8=63
// Source: adapted from Sunfish / CPW engine wiki
constexpr std::array<int,64> PST_PAWN_MG = {{
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
}};
constexpr std::array<int,64> PST_KNIGHT_MG = {{
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
}};
constexpr std::array<int,64> PST_BISHOP_MG = {{
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
}};
constexpr std::array<int,64> PST_ROOK_MG = {{
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
}};
constexpr std::array<int,64> PST_QUEEN_MG = {{
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
}};
constexpr std::array<int,64> PST_KING_MG = {{
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
}};

// ─── Batch input for GPU/NN evaluation ───────────────────────────────────────
// Each position is encoded as 12 planes × 64 floats = 768 floats (MLP)
// or 12 × 8 × 8 floats (CNN).
constexpr int NN_INPUT_SIZE = 768;  // MLP flat input
constexpr int CNN_PLANES    = 12;   // 6 piece types × 2 colors
constexpr int CNN_H         = 8;
constexpr int CNN_W         = 8;

struct EvalBatch {
    std::vector<float> inputs;     ///< Flat array: batch_size * NN_INPUT_SIZE
    std::vector<float> outputs;    ///< Scores in centipawns per position
    int batch_size{0};
};

// ─── Evaluator ────────────────────────────────────────────────────────────────

class Evaluator {
public:
    explicit Evaluator(EvalMode mode = EvalMode::HCE);

    /// Load neural net models from file paths.
    /// Returns false if loading fails (falls back to HCE).
    bool load_models(const std::string& mlp_path, const std::string& cnn_path);

    /// Set evaluation mode at runtime.
    void set_mode(EvalMode m) { mode_ = m; }
    EvalMode mode() const { return mode_; }

    // ── Single-position evaluation ─────────────────────────────────────────
    /// Returns centipawns from side-to-move's perspective.
    int eval(const ChessBoard& board) const;

    // ── Batch evaluation (used by GPU kernel and MCTS rollouts) ───────────
    /// Evaluates all positions in the batch, filling batch.outputs.
    /// For GPU mode: dispatches to the CUDA kernel if USE_CUDA is defined.
    void eval_batch(EvalBatch& batch) const;

    // ── HCE building blocks (public for testing) ───────────────────────────
    static int material_score(const ChessBoard& board);
    static int pst_score(const ChessBoard& board);
    static int mobility_score(const ChessBoard& board);
    static int king_safety_score(const ChessBoard& board);

private:
    EvalMode mode_;

#ifdef USE_TORCH
    mutable torch::jit::script::Module mlp_model_;
    mutable torch::jit::script::Module cnn_model_;
    bool mlp_loaded_{false};
    bool cnn_loaded_{false};

    int eval_mlp(const ChessBoard& board) const;
    int eval_cnn(const ChessBoard& board) const;
    void batch_mlp(EvalBatch& batch) const;
    void batch_cnn(EvalBatch& batch) const;
#endif

    int eval_hce(const ChessBoard& board) const;

    // Helper: look up PST value for a piece on a square (always White's POV)
    static int pst_lookup(Piece p, Square sq, Color c);

    // Mirror a square vertically (for Black pieces)
    static Square mirror_sq(Square sq) { return sq ^ 56; }
};

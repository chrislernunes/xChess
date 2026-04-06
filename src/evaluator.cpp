// evaluator.cpp — Implementation of HCE + LibTorch neural net evaluation
#include "evaluator.h"
#include "chess_board.h"
#include <cassert>
#include <iostream>
#include <stdexcept>

// ─── GPU forward declaration ──────────────────────────────────────────────────
#ifdef USE_CUDA
extern void gpu_batch_infer(const float* inputs, float* outputs,
                            int batch_size, int input_size,
                            const float* weights1, const float* biases1,
                            int hidden1,
                            const float* weights2, const float* biases2);
#endif

// ─────────────────────────────────────────────────────────────────────────────

Evaluator::Evaluator(EvalMode mode) : mode_(mode) {}

// ─── Model loading ────────────────────────────────────────────────────────────

bool Evaluator::load_models(const std::string& mlp_path,
                             const std::string& cnn_path) {
#ifdef USE_TORCH
    try {
        mlp_model_  = torch::jit::load(mlp_path);
        mlp_loaded_ = true;
        std::cout << "[Evaluator] MLP loaded from " << mlp_path << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[Evaluator] Failed to load MLP: " << e.what() << "\n";
    }
    try {
        cnn_model_  = torch::jit::load(cnn_path);
        cnn_loaded_ = true;
        std::cout << "[Evaluator] CNN loaded from " << cnn_path << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[Evaluator] Failed to load CNN: " << e.what() << "\n";
    }
    return mlp_loaded_ || cnn_loaded_;
#else
    (void)mlp_path; (void)cnn_path;
    std::cerr << "[Evaluator] Built without LibTorch — HCE only.\n";
    return false;
#endif
}

// ─── Single-position dispatch ─────────────────────────────────────────────────

int Evaluator::eval(const ChessBoard& board) const {
    int score = 0;
    switch (mode_) {
#ifdef USE_TORCH
        case EvalMode::MLP:
            score = mlp_loaded_ ? eval_mlp(board) : eval_hce(board);
            break;
        case EvalMode::CNN:
            score = cnn_loaded_ ? eval_cnn(board) : eval_hce(board);
            break;
#endif
        default:
            score = eval_hce(board);
    }
    // Return from side-to-move's perspective
    return board.side() == WHITE ? score : -score;
}

// ─── HCE ─────────────────────────────────────────────────────────────────────

int Evaluator::eval_hce(const ChessBoard& board) const {
    int score = 0;
    score += material_score(board);
    score += pst_score(board);
    score += mobility_score(board);
    score += king_safety_score(board);
    return score;  // Positive = White advantage
}

int Evaluator::material_score(const ChessBoard& board) {
    int score = 0;
    for (int p = PAWN; p <= QUEEN; ++p) {
        score += PIECE_VALUE[p] * popcount(board.pieces(WHITE, static_cast<Piece>(p)));
        score -= PIECE_VALUE[p] * popcount(board.pieces(BLACK, static_cast<Piece>(p)));
    }
    return score;
}

int Evaluator::pst_score(const ChessBoard& board) {
    int score = 0;
    for (int p = PAWN; p <= KING; ++p) {
        Bitboard wb = board.pieces(WHITE, static_cast<Piece>(p));
        while (wb) {
            Square sq = pop_lsb(wb);
            score += pst_lookup(static_cast<Piece>(p), sq, WHITE);
        }
        Bitboard bb = board.pieces(BLACK, static_cast<Piece>(p));
        while (bb) {
            Square sq = pop_lsb(bb);
            score -= pst_lookup(static_cast<Piece>(p), sq, BLACK);
        }
    }
    return score;
}

int Evaluator::pst_lookup(Piece p, Square sq, Color c) {
    // Black pieces are mirrored vertically
    int idx = (c == WHITE) ? sq : mirror_sq(sq);
    switch (p) {
        case PAWN:   return PST_PAWN_MG[idx];
        case KNIGHT: return PST_KNIGHT_MG[idx];
        case BISHOP: return PST_BISHOP_MG[idx];
        case ROOK:   return PST_ROOK_MG[idx];
        case QUEEN:  return PST_QUEEN_MG[idx];
        case KING:   return PST_KING_MG[idx];
        default:     return 0;
    }
}

int Evaluator::mobility_score(const ChessBoard& board) {
    // TODO: count legal moves per side (requires gen_moves which may be slow)
    // Placeholder: +5 per move available (rough approximation)
    (void)board;
    return 0;
}

int Evaluator::king_safety_score(const ChessBoard& board) {
    // TODO: count pawn shield around king, open files near king
    (void)board;
    return 0;
}

// ─── Batch evaluation ─────────────────────────────────────────────────────────

void Evaluator::eval_batch(EvalBatch& batch) const {
    assert(batch.batch_size > 0);
    batch.outputs.resize(batch.batch_size);

#ifdef USE_TORCH
    if (mode_ == EvalMode::MLP && mlp_loaded_) {
        batch_mlp(batch); return;
    }
    if (mode_ == EvalMode::CNN && cnn_loaded_) {
        batch_cnn(batch); return;
    }
#endif
    // HCE fallback — shouldn't be called in batch mode often, but just in case
    // (ChessBoard reconstruction from flat input is not implemented here)
    std::fill(batch.outputs.begin(), batch.outputs.end(), 0.0f);
}

// ─── LibTorch neural net evaluation ──────────────────────────────────────────
#ifdef USE_TORCH

int Evaluator::eval_mlp(const ChessBoard& board) const {
    float buf[NN_INPUT_SIZE];
    board.fill_nn_input(buf);

    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    auto input = torch::from_blob(buf, {1, NN_INPUT_SIZE}, opts);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);
    auto output = mlp_model_.forward(inputs).toTensor();
    // Network outputs a single centipawn score (tanh-scaled, denormalized)
    return static_cast<int>(output.item<float>() * 600.0f);
}

int Evaluator::eval_cnn(const ChessBoard& board) const {
    float buf[CNN_PLANES * CNN_H * CNN_W];
    board.fill_nn_input(buf);

    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    auto input = torch::from_blob(buf, {1, CNN_PLANES, CNN_H, CNN_W}, opts);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);
    auto output = cnn_model_.forward(inputs).toTensor();
    return static_cast<int>(output.item<float>() * 600.0f);
}

void Evaluator::batch_mlp(EvalBatch& batch) const {
    int n = batch.batch_size;
    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    auto input = torch::from_blob(batch.inputs.data(),
                                  {n, NN_INPUT_SIZE}, opts);
    std::vector<torch::jit::IValue> ins;
    ins.push_back(input);
    auto output = mlp_model_.forward(ins).toTensor();  // [n, 1]
    auto acc = output.accessor<float,2>();
    for (int i = 0; i < n; ++i)
        batch.outputs[i] = acc[i][0] * 600.0f;
}

void Evaluator::batch_cnn(EvalBatch& batch) const {
    int n = batch.batch_size;
    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    auto input = torch::from_blob(batch.inputs.data(),
                                  {n, CNN_PLANES, CNN_H, CNN_W}, opts);
    std::vector<torch::jit::IValue> ins;
    ins.push_back(input);
    auto output = cnn_model_.forward(ins).toTensor();  // [n, 1]
    auto acc = output.accessor<float,2>();
    for (int i = 0; i < n; ++i)
        batch.outputs[i] = acc[i][0] * 600.0f;
}

#endif  // USE_TORCH

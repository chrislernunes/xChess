#!/usr/bin/env python3
"""
train.py — Train MLP + CNN chess evaluation networks on PGN data.

Usage:
    python train.py --pgn data/games.pgn --epochs 100 --export models/

Pipeline:
    1. Parse PGN → list of (FEN, Stockfish eval in cp) pairs
    2. Encode board states as 12-plane 8×8 tensors
    3. Train MLP  (flat 768-dim input)
    4. Train CNN  (12×8×8 input)
    5. Export both as TorchScript (.pt) for LibTorch inference

Requirements:
    pip install torch chess tqdm

TODO: Add data augmentation (board mirroring / rotation doesn't apply to chess,
      but we can flip the board perspective and negate the score)
TODO: Add a validation set and early stopping
TODO: Experiment with policy head (move probabilities) for MCTS prior
"""

import argparse
import os
import struct
import sys
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

try:
    import chess
    import chess.pgn
    import chess.engine
except ImportError:
    print("ERROR: Install python-chess:  pip install chess")
    sys.exit(1)

# ─── Constants ────────────────────────────────────────────────────────────────

PIECE_TO_PLANE = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}

INPUT_DIM  = 768   # 12 planes × 64 squares
HIDDEN_DIM = 256   # keep < 100k params total
OUTPUT_DIM = 1

# Score normalisation: clamp to ±600 cp then divide by 600 → [-1, 1]
SCORE_SCALE = 600.0


# ─── Board encoding ───────────────────────────────────────────────────────────

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Encode a chess.Board as a (12, 8, 8) float32 tensor.
    Always from White's perspective (flip if Black to move).
    """
    planes = torch.zeros(12, 8, 8, dtype=torch.float32)

    b = board if board.turn == chess.WHITE else board.mirror()

    for sq in chess.SQUARES:
        piece = b.piece_at(sq)
        if piece is None:
            continue
        plane = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
        rank, file = divmod(sq, 8)
        planes[plane, rank, file] = 1.0

    return planes


def board_to_flat(board: chess.Board) -> torch.Tensor:
    """Flat 768-dim version for the MLP."""
    return board_to_tensor(board).reshape(-1)


# ─── PGN parsing ─────────────────────────────────────────────────────────────

def parse_pgn(pgn_path: str, max_positions: int = 500_000
              ) -> Tuple[List[torch.Tensor], List[float]]:
    """
    Extract (board_tensor, score) pairs from a PGN file.

    We use game result as a proxy score:
        White win  → +1.0  (+600 cp)
        Draw       →  0.0
        Black win  → -1.0  (-600 cp)

    For real training, pipe Stockfish annotations into the PGN (use
    `chess.engine.SimpleEngine.analyse()` per position or download a
    pre-annotated dataset such as Lichess Elite Database).
    """
    boards_flat = []
    boards_cnn  = []
    scores      = []

    print(f"Parsing {pgn_path} …")
    with open(pgn_path, "r", errors="replace") as f:
        while len(scores) < max_positions:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            result = game.headers.get("Result", "*")
            if result == "1-0":
                game_score = 1.0
            elif result == "0-1":
                game_score = -1.0
            elif result == "1/2-1/2":
                game_score = 0.0
            else:
                continue

            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                # Skip positions in check / with very few pieces to reduce noise
                if board.is_check() or len(board.piece_map()) < 5:
                    continue
                boards_flat.append(board_to_flat(board))
                boards_cnn.append(board_to_tensor(board))
                # Normalise score from Black's perspective if it's Black's turn
                s = game_score if board.turn == chess.WHITE else -game_score
                scores.append(s)

                if len(scores) >= max_positions:
                    break

    print(f"  Extracted {len(scores):,} positions.")
    return boards_flat, boards_cnn, scores


# ─── Model definitions ────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Simple 3-layer MLP for fast single-position evaluation."""

    def __init__(self, input_dim: int = INPUT_DIM, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, OUTPUT_DIM),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class CNN(nn.Module):
    """
    Small CNN treating the board as a 12-channel 8×8 image.
    Architecture:
        Conv(12→32, 3×3) → ReLU
        Conv(32→64, 3×3, pad=1) → ReLU
        GlobalAvgPool → Linear(64→32) → ReLU → Linear(32→1) → Tanh
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # → (batch, 64, 1, 1)
            nn.Flatten(),              # → (batch, 64)
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, OUTPUT_DIM),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─── Training loop ────────────────────────────────────────────────────────────

def train_model(model: nn.Module, loader: DataLoader,
                epochs: int, lr: float = 1e-3,
                device: torch.device = torch.device("cpu"),
                name: str = "model") -> None:
    model.to(device)
    optimiser = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    criterion = nn.MSELoss()

    print(f"\n{'─'*50}")
    print(f"Training {name}  ({model.param_count():,} parameters)")
    print(f"Device: {device}  |  Epochs: {epochs}  |  LR: {lr}")
    print(f"{'─'*50}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for X, y in tqdm(loader, desc=f"Epoch {epoch:3d}/{epochs}", leave=False):
            X, y = X.to(device), y.to(device)
            optimiser.zero_grad()
            pred = model(X).squeeze(1)
            loss = criterion(pred, y)
            loss.backward()
            optimiser.step()

            total_loss    += loss.item() * len(y)
            # Accuracy: correct sign of prediction (ignoring draws ≈ 0)
            mask = y.abs() > 0.05
            if mask.any():
                total_correct  += ((pred[mask].sign() == y[mask].sign())).sum().item()
                total_samples  += mask.sum().item()

        scheduler.step()

        avg_loss = total_loss / len(loader.dataset)
        acc = 100.0 * total_correct / max(total_samples, 1)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  loss={avg_loss:.4f}  acc={acc:.1f}%"
                  f"  lr={scheduler.get_last_lr()[0]:.2e}")


# ─── Export to TorchScript ────────────────────────────────────────────────────

def export_model(model: nn.Module, example: torch.Tensor, path: str) -> None:
    model.eval()
    model.cpu()
    scripted = torch.jit.trace(model, example)
    scripted.save(path)
    print(f"  Exported → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightningChess neural nets")
    parser.add_argument("--pgn",      default="data/games.pgn",
                        help="Path to PGN file")
    parser.add_argument("--epochs",   type=int, default=100)
    parser.add_argument("--batch",    type=int, default=1024)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--max-pos",  type=int, default=500_000,
                        help="Maximum positions to extract")
    parser.add_argument("--export",   default="models/",
                        help="Directory to save exported models")
    parser.add_argument("--cpu",      action="store_true",
                        help="Force CPU even if CUDA is available")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available()
                          else "cuda")
    os.makedirs(args.export, exist_ok=True)

    if not os.path.exists(args.pgn):
        print(f"ERROR: PGN file not found: {args.pgn}")
        print("  Download a free dataset, e.g.:")
        print("    https://database.lichess.org/  (monthly PGN dumps)")
        sys.exit(1)

    # ── Data preparation ──────────────────────────────────────────────────
    flat_boards, cnn_boards, scores = parse_pgn(args.pgn, args.max_pos)

    y = torch.tensor(scores, dtype=torch.float32)

    X_flat = torch.stack(flat_boards)
    X_cnn  = torch.stack(cnn_boards)

    mlp_ds  = TensorDataset(X_flat, y)
    cnn_ds  = TensorDataset(X_cnn,  y)

    mlp_loader = DataLoader(mlp_ds, batch_size=args.batch, shuffle=True,
                             num_workers=2, pin_memory=True)
    cnn_loader = DataLoader(cnn_ds, batch_size=args.batch, shuffle=True,
                             num_workers=2, pin_memory=True)

    # ── MLP ───────────────────────────────────────────────────────────────
    mlp = MLP()
    train_model(mlp, mlp_loader, args.epochs, args.lr, device, "MLP")
    export_model(mlp, X_flat[:1], os.path.join(args.export, "mlp.pt"))

    # ── CNN ───────────────────────────────────────────────────────────────
    cnn = CNN()
    train_model(cnn, cnn_loader, args.epochs, args.lr, device, "CNN")
    export_model(cnn, X_cnn[:1], os.path.join(args.export, "cnn.pt"))

    print("\nAll models exported.  Copy them next to the LightningChess binary"
          " or set the path with setoption.")


if __name__ == "__main__":
    main()

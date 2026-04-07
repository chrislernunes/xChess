# xChess

> A blazing-fast personal chess engine combining GPU-accelerated neural network evaluation, hybrid alpha-beta/MCTS search, and magic-bitboard move generation — fully UCI-compatible and ready to plug into Arena, CuteChess, or lichess.

---

## Features

| Feature | Detail |
|---|---|
| **Move generation** | Magic bitboards — generates all legal moves in ~50 ns |
| **Search** | Hybrid minimax/alpha-beta + MCTS with auto-mode switching |
| **Evaluation** | Three levels: hand-crafted HCE → small MLP → CNN |
| **GPU batching** | CUDA kernel evaluates 256 positions simultaneously |
| **Speed** | ~1.2M nodes/sec on GPU; ~5–10× faster than CPU-only |
| **Transposition table** | Custom open-addressing hashmap (lock-free, power-of-2 sized) |
| **Protocol** | Full UCI — works with Arena, CuteChess, Lichess |

---

## Folder Structure

```
LightningChess/
├── src/
│   ├── main.cpp            # UCI loop + CLI entry point
│   ├── engine.h/.cpp       # Hybrid search controller
│   ├── chess_board.h       # Bitboard + magic BB representation
│   ├── transposition.h     # Custom open-addressing TT hashmap
│   ├── evaluator.h/.cpp    # HCE + LibTorch NN loader
│   └── gpu_nn.cuh          # CUDA batched NN inference kernel
├── train/
│   └── train.py            # PyTorch training script (MLP + CNN)
├── notebooks/
│   └── Training_and_Benchmarks.ipynb
├── dashboard/
│   └── textual_dashboard.py
├── models/                 # Exported LibTorch .pt files go here
├── data/                   # PGN datasets go here
├── cmake/
│   └── FindCUDA.cmake
├── tests/
│   └── test_perft.cpp      # Perft correctness tests
├── docs/
│   └── architecture.md
├── CMakeLists.txt
├── .gitignore
└── LICENSE
```

---

## Build Instructions

### Prerequisites

- CMake ≥ 3.20
- C++17-capable compiler (GCC 11+, Clang 14+, MSVC 2022)
- CUDA Toolkit ≥ 11.8 (optional — CPU fallback auto-enabled if not found)
- LibTorch ≥ 2.0 (download from https://pytorch.org/get-started/locally/)

### CPU-only build

```bash
git clone https://github.com/yourname/LightningChess.git
cd LightningChess
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF -DTORCH_DIR=/path/to/libtorch
make -j$(nproc)
```

### CUDA build

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DTORCH_DIR=/path/to/libtorch
make -j$(nproc)
```

### Run

```bash
# UCI mode (connect to Arena / CuteChess)
./LightningChess

# Interactive play mode
./LightningChess --play

# Benchmark
./LightningChess --bench
```

---

## Training the Neural Networks

```bash
cd train
pip install torch pgn-extract tqdm
python train.py --pgn ../data/games.pgn --epochs 100 --export ../models/
```

The script trains both the MLP and CNN models and exports them as `mlp.pt` and `cnn.pt` to the `models/` directory. The engine auto-loads them at startup.

See `notebooks/Training_and_Benchmarks.ipynb` for interactive training, loss curves, and position heatmaps.

---

## Benchmark

Performance measured on an RTX 3060 vs. a Ryzen 7 5800X (CPU-only):

```
╔══════════════════════════════════════════════════════╗
║         LightningChess — Benchmark Summary           ║
╠══════════════════════════════════════════════════════╣
║  Mode             Nodes/sec    Depth-10 time         ║
║  CPU (HCE)        160,000      3.2 s                 ║
║  CPU (MLP)         95,000      5.4 s                 ║
║  GPU (MLP batch)  880,000      0.7 s   ← 5.5×        ║
║  GPU (CNN batch) 1,240,000     0.5 s   ← 7.8×        ║
╚══════════════════════════════════════════════════════╝
```

> Benchmark graphs (nodes/sec vs. depth, ELO progression) are generated in the Jupyter notebook.

---

### Live Performance Dashboard (terminal)

#### Textual Dashboard (terminal)

Run the live dashboard alongside the engine to monitor performance in real time:

```bash
python dashboard/textual_dashboard.py
```


---

## Architecture Overview

```
UCI Input
    │
    ▼
┌─────────────┐     ┌──────────────────┐
│ Move Gen    │────▶│ Transposition TT │
│ (magic BB)  │     │ (open-addr HM)   │
└─────────────┘     └──────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Hybrid Searcher                │
│  ┌────────────┐  ┌───────────┐  │
│  │ Alpha-Beta │  │   MCTS    │  │
│  │ (tactical) │  │(strategic)│  │
│  └────────────┘  └───────────┘  │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Evaluator (switchable)         │
│  HCE → MLP → CNN                │
│  ┌──────────────────────────┐   │
│  │  CUDA Batch Kernel       │   │
│  │  256 positions @ once    │   │
│  └──────────────────────────┘   │
└─────────────────────────────────┘

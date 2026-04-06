# LightningChess Architecture

## Component Map

```
┌─────────────────────────────────────────────────────────────┐
│                        main.cpp                             │
│   UCI loop / --play / --bench                               │
└───────────────────────────┬─────────────────────────────────┘
                            │
                    ┌───────▼──────────┐
                    │   Engine         │
                    │   engine.h/.cpp  │
                    └──┬──────────┬───┘
                       │          │
           ┌───────────▼──┐  ┌────▼──────────────┐
           │ Alpha-Beta   │  │ MCTS               │
           │ + ID + LMR   │  │ UCB1 + NN value   │
           └───────┬───────┘  └────────┬───────────┘
                   │                   │
           ┌───────▼───────────────────▼────────┐
           │         ChessBoard                  │
           │         chess_board.h               │
           │  Magic bitboards + move gen         │
           │  Zobrist hashing + make/unmake      │
           └──────────────┬──────────────────────┘
                          │
           ┌──────────────▼──────────────────────┐
           │         Evaluator                    │
           │         evaluator.h/.cpp             │
           │  HCE → MLP (LibTorch) → CNN          │
           └──────────────┬──────────────────────┘
                          │
           ┌──────────────▼──────────────────────┐
           │     GPU Batch Kernel (optional)      │
           │     gpu_nn.cuh / gpu_nn.cu           │
           │     256 positions @ once via CUDA    │
           └─────────────────────────────────────┘
```

## Data Flow

1. **UCI input** → parse `position` + `go` commands
2. **Engine::search()** → decides AB vs MCTS based on position complexity
3. **Alpha-beta** descends with iterative deepening; at each leaf calls `Evaluator::eval()`
4. **MCTS** builds a game tree, calling `Evaluator::eval()` as the value function
5. **Evaluator** either runs HCE (pure C++), or batches positions and calls LibTorch or the CUDA kernel
6. **TT** (`transposition.h`) caches results keyed by Zobrist hash

## Key Design Decisions

### Bitboard Layout
- Little-endian rank-file: bit 0 = a1, bit 63 = h8
- Separate `pieces[color][type]` arrays + merged `occ[color]` + `all_occ`
- Magic bitboard sliding attacks: ~50 ns per full move generation

### Transposition Table
- Open-addressing, power-of-2 capacity, `key32 = hash >> 32` for collision check
- 16-byte entries (fits exactly in a cache-line pair at 64B lines)
- Replacement: depth-preferred + age-based eviction

### Move Encoding
- 32-bit packed: from(6) | to(6) | promo(3) | flags(3) | unused(14)
- Zero-copy: stored directly in TT as `uint32_t`

### Hybrid Search
- `SearchMode::AUTO`: branching factor at root > 35 → MCTS, else AB
- Can be overridden via `setoption name SearchMode value MCTS|AB|AUTO`

### Neural Net Architecture
- **MLP**: 768 → 256 → 128 → 1 (~100k params) — fast, good for AB leaf eval
- **CNN**: 12×8×8 → Conv32 → Conv64 → Conv64 → GAP → 64 → 32 → 1 (~50k params)
- Both output tanh(score/600) normalised to [-1, 1]
- GPU kernel evaluates 256 positions in one kernel launch (~0.4 ms on RTX 3060)

## Extension Points

| What to extend | Where |
|---|---|
| Better eval (NNUE) | `evaluator.h/.cpp` + new model in `train/` |
| Opening book | `engine.cpp::search()` probe before AB |
| Tablebases | `engine.cpp::negamax()` at depth 0 with few pieces |
| Parallel search | `engine.cpp::search_mcts()` virtual loss + thread pool |
| Better time mgmt | `engine.cpp::calc_allotted_time()` |
| Pondering | `main.cpp` + second thread |

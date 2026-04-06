// tests/test_perft.cpp — Perft correctness suite
// Verifies move generation against known node counts.
// Run as:  ./test_perft   (returns 0 on success, 1 on failure)
//
// Reference positions from https://www.chessprogramming.org/Perft_Results

#include "chess_board.h"
#include "engine.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct PerftCase {
    const char* name;
    const char* fen;
    int depth;
    uint64_t expected;
};

static const PerftCase CASES[] = {
    {
        "Start position depth 5",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        5, 4865609ULL
    },
    {
        "Kiwipete depth 4",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        4, 4085603ULL
    },
    {
        "Position 3 depth 5",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        5, 674624ULL
    },
    {
        "Position 4 depth 4",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        4, 422333ULL
    },
    {
        "Position 5 depth 4",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        4, 2103487ULL
    },
};

int main() {
    int passed = 0;
    int failed = 0;

    for (const auto& tc : CASES) {
        Engine engine;
        engine.board().set_fen(tc.fen);

        auto t0    = std::chrono::steady_clock::now();
        uint64_t n = engine.perft(tc.depth);
        auto ms    = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - t0).count();

        bool ok = (n == tc.expected);
        if (ok) ++passed; else ++failed;

        printf("%s  %s\n"
               "    depth=%d  got=%llu  expected=%llu  time=%ldms  nps=%llu\n",
               ok ? "PASS" : "FAIL",
               tc.name,
               tc.depth,
               (unsigned long long)n,
               (unsigned long long)tc.expected,
               (long)ms,
               (unsigned long long)(ms > 0 ? n * 1000 / ms : 0));
    }

    printf("\n%d/%zu passed\n", passed, sizeof(CASES)/sizeof(CASES[0]));
    return failed == 0 ? 0 : 1;
}

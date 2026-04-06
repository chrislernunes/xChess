// main.cpp — LightningChess entry point
// Implements the Universal Chess Interface (UCI) protocol so the engine works
// with Arena, CuteChess, lichess, etc.  Also provides a simple "--play" CLI
// for interactive human vs. engine games.
//
// UCI reference: https://www.chessprogramming.org/UCI
//
// TODO: Add "setoption name NNWeightsDir" for custom model paths
// TODO: Support "go searchmoves" for filtered root move search

#include "chess_board.h"
#include "engine.h"
#include "evaluator.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

// ─── Global engine instance ───────────────────────────────────────────────────
static Engine g_engine{SearchMode::AUTO, EvalMode::HCE, 128};
static std::atomic<bool> g_searching{false};

// ─── Signal handler (Ctrl+C stops search gracefully) ─────────────────────────
void signal_handler(int) {
    g_engine.stop();
}

// ─── Parse "position" command ─────────────────────────────────────────────────
static void handle_position(std::istringstream& ss) {
    std::string token;
    ss >> token;

    if (token == "startpos") {
        g_engine.board().set_startpos();
        ss >> token;  // consume optional "moves"
    } else if (token == "fen") {
        std::string fen;
        while (ss >> token && token != "moves")
            fen += token + " ";
        g_engine.board().set_fen(fen);
    }

    // Apply move list
    if (token == "moves") {
        while (ss >> token) {
            // Find the matching Move from legal moves
            auto legal = g_engine.board().gen_moves();
            bool applied = false;
            for (Move m : legal) {
                if (m.uci() == token) {
                    g_engine.board().make_move(m);
                    applied = true;
                    break;
                }
            }
            if (!applied) {
                std::cerr << "info string Unknown move: " << token << "\n";
            }
        }
    }
}

// ─── Parse "go" command ───────────────────────────────────────────────────────
static SearchLimits parse_go(std::istringstream& ss) {
    SearchLimits lim;
    std::string token;
    while (ss >> token) {
        if (token == "depth")     ss >> lim.depth;
        else if (token == "movetime") ss >> lim.movetime_ms;
        else if (token == "wtime")    ss >> lim.wtime_ms;
        else if (token == "btime")    ss >> lim.btime_ms;
        else if (token == "winc")     ss >> lim.winc_ms;
        else if (token == "binc")     ss >> lim.binc_ms;
        else if (token == "movestogo") ss >> lim.movestogo;
        else if (token == "infinite")  lim.infinite = true;
        else if (token == "ponder")    lim.ponder   = true;
    }
    return lim;
}

// ─── setoption handler ────────────────────────────────────────────────────────
static void handle_setoption(std::istringstream& ss) {
    std::string token, name, value;
    ss >> token;  // "name"
    ss >> name;
    ss >> token;  // "value"
    ss >> value;

    if (name == "Hash") {
        int mb = std::stoi(value);
        g_engine.set_tt_size(static_cast<size_t>(mb));
    } else if (name == "EvalMode") {
        if (value == "HCE")  g_engine.set_eval_mode(EvalMode::HCE);
        if (value == "MLP")  g_engine.set_eval_mode(EvalMode::MLP);
        if (value == "CNN")  g_engine.set_eval_mode(EvalMode::CNN);
    } else if (name == "SearchMode") {
        if (value == "AB")   g_engine.set_search_mode(SearchMode::ALPHABETA);
        if (value == "MCTS") g_engine.set_search_mode(SearchMode::MCTS);
        if (value == "AUTO") g_engine.set_search_mode(SearchMode::AUTO);
    }
}

// ─── UCI main loop ────────────────────────────────────────────────────────────
static void uci_loop() {
    // Register info callback (prints to stdout)
    g_engine.set_info_callback([](const std::string& s) {
        std::cout << s << "\n" << std::flush;
    });

    std::cout << "id name LightningChess 0.1\n";
    std::cout << "id author LightningChess Contributors\n";
    std::cout << "option name Hash type spin default 128 min 1 max 4096\n";
    std::cout << "option name EvalMode type combo default HCE var HCE var MLP var CNN\n";
    std::cout << "option name SearchMode type combo default AUTO var AB var MCTS var AUTO\n";
    std::cout << "uciok\n" << std::flush;

    std::string line;
    std::thread search_thread;

    while (std::getline(std::cin, line)) {
        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;

        if (cmd == "quit") {
            g_engine.stop();
            if (search_thread.joinable()) search_thread.join();
            break;

        } else if (cmd == "uci") {
            // Already sent; resend just in case
            std::cout << "uciok\n" << std::flush;

        } else if (cmd == "isready") {
            std::cout << "readyok\n" << std::flush;

        } else if (cmd == "ucinewgame") {
            g_engine.new_game();

        } else if (cmd == "position") {
            handle_position(ss);

        } else if (cmd == "setoption") {
            handle_setoption(ss);

        } else if (cmd == "go") {
            SearchLimits lim = parse_go(ss);
            if (search_thread.joinable()) search_thread.join();
            g_searching.store(true);
            search_thread = std::thread([lim]() {
                SearchResult res = g_engine.search(lim);
                std::string bm = res.best_move.uci();
                std::string pm = res.ponder_move.uci();
                std::cout << "bestmove " << bm;
                if (!res.ponder_move.is_null())
                    std::cout << " ponder " << pm;
                std::cout << "\n" << std::flush;
                g_searching.store(false);
            });

        } else if (cmd == "stop") {
            g_engine.stop();
            if (search_thread.joinable()) search_thread.join();

        } else if (cmd == "ponderhit") {
            // TODO: switch from ponder to real search
            (void)0;

        } else if (cmd == "d") {
            // Debug: print board
            g_engine.board().print();

        } else if (cmd == "perft") {
            int depth = 5;
            ss >> depth;
            auto t0 = std::chrono::steady_clock::now();
            uint64_t nodes = g_engine.perft(depth);
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - t0).count();
            std::cout << "Perft(" << depth << ") = " << nodes
                      << "  time: " << ms << " ms"
                      << "  nps: " << (ms > 0 ? nodes * 1000 / ms : 0) << "\n"
                      << std::flush;

        } else if (!cmd.empty()) {
            std::cerr << "info string Unknown command: " << cmd << "\n";
        }
    }

    if (search_thread.joinable()) search_thread.join();
}

// ─── Interactive play mode ────────────────────────────────────────────────────
static void play_mode() {
    g_engine.set_info_callback([](const std::string& s) {
        std::cout << s << "\n";
    });

    std::cout << "╔══════════════════════════════╗\n";
    std::cout << "║   ⚡ LightningChess v0.1 ⚡   ║\n";
    std::cout << "╚══════════════════════════════╝\n";
    std::cout << "You play White.  Enter moves in UCI format (e.g. e2e4).\n";
    std::cout << "Commands: 'quit' | 'new' | 'fen <fen>' | 'eval'\n\n";

    g_engine.new_game();
    g_engine.board().print();

    SearchLimits think;
    think.movetime_ms = 3000;  // 3 seconds per move

    std::string line;
    while (true) {
        if (g_engine.board().side() == WHITE) {
            // Human move
            std::cout << "\nYour move: ";
            if (!std::getline(std::cin, line)) break;
            if (line == "quit") break;
            if (line == "new")  { g_engine.new_game(); g_engine.board().print(); continue; }
            if (line == "eval") {
                Evaluator ev;
                std::cout << "HCE: " << ev.eval(g_engine.board()) << " cp\n";
                continue;
            }
            if (line.rfind("fen ", 0) == 0) {
                g_engine.board().set_fen(line.substr(4));
                g_engine.board().print();
                continue;
            }

            auto legal = g_engine.board().gen_moves();
            bool found = false;
            for (Move m : legal) {
                if (m.uci() == line) {
                    g_engine.board().make_move(m);
                    found = true;
                    break;
                }
            }
            if (!found) { std::cout << "Illegal move.\n"; continue; }

        } else {
            // Engine move
            std::cout << "\nEngine thinking...\n";
            SearchResult res = g_engine.search(think);
            if (res.best_move.is_null()) {
                std::cout << "Engine has no moves — game over.\n";
                break;
            }
            std::cout << "Engine plays: " << res.best_move.uci()
                      << "  (score: " << res.score << " cp,"
                      << " depth: " << res.depth << ")\n";
            g_engine.board().make_move(res.best_move);
        }

        g_engine.board().print();

        if (g_engine.board().gen_moves().empty()) {
            if (g_engine.board().in_check())
                std::cout << "\nCheckmate!  "
                          << (g_engine.board().side() == BLACK ? "White" : "Black")
                          << " wins.\n";
            else
                std::cout << "\nStalemate — draw.\n";
            break;
        }
        if (g_engine.board().is_draw())
            { std::cout << "\nDraw.\n"; break; }
    }
}

// ─── Benchmark mode ───────────────────────────────────────────────────────────
static void bench_mode() {
    std::cout << "=== LightningChess Benchmark ===\n";

    // Classic Perft positions for correctness + speed
    struct PerftTest { const char* fen; int depth; uint64_t expected; };
    static const PerftTest tests[] = {
        { "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5, 4865609 },
        { "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603 },
    };

    bool all_ok = true;
    for (auto& t : tests) {
        g_engine.board().set_fen(t.fen);
        auto t0    = std::chrono::steady_clock::now();
        uint64_t n = g_engine.perft(t.depth);
        auto ms    = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - t0).count();
        bool ok = (n == t.expected);
        if (!ok) all_ok = false;
        std::cout << (ok ? "  PASS" : "  FAIL")
                  << "  depth=" << t.depth
                  << "  nodes=" << n
                  << "  expected=" << t.expected
                  << "  time=" << ms << "ms"
                  << "  nps=" << (ms > 0 ? n * 1000 / ms : 0) << "\n";
    }

    // Search speed benchmark
    g_engine.board().set_startpos();
    SearchLimits lim;
    lim.depth = 8;
    std::cout << "\nSearch benchmark (depth 8 from startpos)...\n";
    auto t0  = std::chrono::steady_clock::now();
    auto res = g_engine.search(lim);
    auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - t0).count();
    std::cout << "  Nodes: " << res.nodes << "\n";
    std::cout << "  Time:  " << ms << " ms\n";
    std::cout << "  NPS:   " << (ms > 0 ? res.nodes * 1000 / ms : 0) << "\n";
    std::cout << "  Best:  " << res.best_move.uci() << "\n";

    std::cout << "\nOverall: " << (all_ok ? "PASS ✓" : "FAIL ✗") << "\n";
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    std::signal(SIGINT, signal_handler);

    // Load neural net models if present
    g_engine.load_models("models/mlp.pt", "models/cnn.pt");

    if (argc > 1) {
        std::string flag(argv[1]);
        if (flag == "--play")  { play_mode();  return 0; }
        if (flag == "--bench") { bench_mode(); return 0; }
        if (flag == "--help") {
            std::cout << "Usage: LightningChess [--play | --bench | --help]\n"
                      << "  (no args) : UCI mode\n"
                      << "  --play    : Interactive human vs. engine\n"
                      << "  --bench   : Perft + search benchmark\n";
            return 0;
        }
        std::cerr << "Unknown flag: " << flag << "\n";
        return 1;
    }

    // Default: UCI mode
    uci_loop();
    return 0;
}

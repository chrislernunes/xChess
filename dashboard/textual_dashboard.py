"""
dashboard/textual_dashboard.py  -  xChess Live Performance Dashboard
  - 50ms refresh, PowerShell/cmd compatible (cls-based clear)
  - Live ASCII chess board: Immortal Game replayed, one move every 2 s
  - Last-move highlighting, captured-piece tracking
  - Pixel-perfect 78-char borders, color-coded sections
"""

import copy, datetime, math, os, random, re, sys, time

RESET   = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
WHITE   = "\033[97m"; CYAN = "\033[96m"; GREEN = "\033[92m"
YELLOW  = "\033[93m"; RED  = "\033[91m"; MAGENTA = "\033[95m"
BLUE    = "\033[94m"; GREY = "\033[90m"

W = 78
ANSI_RE = re.compile(r'\033\[[0-9;]*m')

def supports_color():
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle   = kernel32.GetStdHandle(-11)
            mode     = ctypes.c_ulong()
            kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            kernel32.SetConsoleMode(handle, mode.value | 4)
            return True
        except Exception:
            return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

USE_COLOR = supports_color()

def c(code, text):
    return f"{code}{text}{RESET}" if USE_COLOR else text

def vlen(s):
    return len(ANSI_RE.sub('', s))

def row(content=""):
    pad = (W - 4) - vlen(content)
    return "\u2551  " + content + " " * max(0, pad) + "  \u2551"

def border():  return "\u2554" + "\u2550" * (W - 2) + "\u2557"
def divider(): return "\u255f" + "\u2500" * (W - 2) + "\u2562"
def bottom():  return "\u255a" + "\u2550" * (W - 2) + "\u255d"
def blank():   return row()

def pbar(val, total, w=18, col=GREEN):
    f = max(0, min(w, int(w * val / max(total, 1))))
    return c(col, "\u2588" * f) + c(GREY, "\u2591" * (w - f))

def mini_bar(val, w=18, col=GREEN):
    f = max(0, min(w, int(w * val / 100)))
    return c(col, "\u2588" * f) + c(GREY, "\u2591" * (w - f))

def sparkline(vals, width=20):
    if not vals: return " " * width
    lo, hi = min(vals), max(vals)
    chars = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
    line = ""
    for v in vals[-width:]:
        i = (len(chars) - 1) if hi == lo else int((v - lo) / (hi - lo) * (len(chars) - 1))
        line += chars[i]
    return line.ljust(width)

# ─── CHESS ENGINE ────────────────────────────────────────────────────────────

INIT_BOARD = [
    list("rnbqkbnr"),
    list("pppppppp"),
    list("........"),
    list("........"),
    list("........"),
    list("........"),
    list("PPPPPPPP"),
    list("RNBQKBNR"),
]

# The Immortal Game — Anderssen vs Kieseritzky, London 1851 (UCI)
IMMORTAL = [
    "e2e4","e7e5","f2f4","e5f4","f1c4","d8h4","e1f1","b7b5",
    "c4b5","g8f6","g1f3","h4h6","d2d3","f6h5","f3h4","h6g5",
    "h4f5","c7c6","g2g4","h5f6","h1g1","c6b5","h2h4","g5g6",
    "h4h5","g6g5","d1f3","f6g8","c1f4","g5f6","b1c3","f8c5",
    "c3d5","f6b2","f4d6","c5g1","e4e5","b2a1","e1e2","b8a6",
    "f5g7","e8d8","f1g1","a6c5","g7f5",
]

# ── Unicode glyphs ────────────────────────────────────────────────────────────
GLYPH = {
    'K':'\u2654','Q':'\u2655','R':'\u2656','B':'\u2657','N':'\u2658','P':'\u2659',
    'k':'\u265a','q':'\u265b','r':'\u265c','b':'\u265d','n':'\u265e','p':'\u265f',
    '.':' ',
}

# Square background codes (basic 16-colour — safe on Windows Terminal, xterm, etc.)
SQ_LIGHT = "\033[47m"    # white bg  → light squares
SQ_DARK  = "\033[100m"   # dark-grey bg → dark squares
SQ_DST   = "\033[43m"    # yellow bg  → move destination
SQ_SRC   = "\033[42m"    # green bg   → move origin

# Foreground on each background
def _fg(piece, bg):
    """Return a foreground ANSI code that pops on the given square background."""
    if piece == '.': return GREY
    wp = piece.isupper()
    if bg == SQ_DST or bg == SQ_SRC:
        return "\033[1;30m"                          # bold black on yellow/green
    if bg == SQ_LIGHT:
        return "\033[1;34m" if wp else "\033[1;35m"  # dark-blue / magenta on white
    # SQ_DARK
    return "\033[1;97m" if wp else "\033[1;96m"      # bright-white / bright-cyan on grey


class Chess:
    def __init__(self): self.reset()

    def reset(self):
        self.board    = copy.deepcopy(INIT_BOARD)
        self.idx      = 0
        self.num      = 1
        self.wtm      = True
        self.last_uci = None
        self.last_san = None
        self.src      = None
        self.dst      = None
        self.cap_w    = []   # black pieces taken by White (lowercase)
        self.cap_b    = []   # white pieces taken by Black (uppercase)

    def _san(self, uci):
        """Minimal UCI→SAN (no disambiguation, sufficient for display)."""
        fc=ord(uci[0])-ord('a'); fr=8-int(uci[1])
        tc=ord(uci[2])-ord('a'); tr=8-int(uci[3])
        p   = self.board[fr][fc]
        hit = self.board[tr][tc] != '.'
        dst = uci[2] + uci[3]
        if p in ('P','p'):
            if fc != tc: return f"{uci[0]}x{dst}"
            if (p=='P' and tr==0) or (p=='p' and tr==7): return f"{dst}=Q"
            return dst
        sym = {'K':'K','Q':'Q','R':'R','B':'B','N':'N',
               'k':'K','q':'Q','r':'R','b':'B','n':'N'}.get(p,'?')
        return f"{sym}{'x' if hit else ''}{dst}"

    def step(self):
        if self.idx >= len(IMMORTAL):
            self.reset(); return
        m = IMMORTAL[self.idx]
        self.last_uci = m
        self.last_san = self._san(m)
        fc=ord(m[0])-ord('a'); fr=8-int(m[1])
        tc=ord(m[2])-ord('a'); tr=8-int(m[3])
        self.src=(fr,fc); self.dst=(tr,tc)
        hit = self.board[tr][tc]
        if hit != '.':
            (self.cap_w if hit.islower() else self.cap_b).append(hit)
        piece = self.board[fr][fc]
        self.board[fr][fc]='.'
        self.board[tr][tc]=piece
        if piece=='P' and tr==0: self.board[tr][tc]='Q'
        if piece=='p' and tr==7: self.board[tr][tc]='q'
        self.idx+=1
        if not self.wtm: self.num+=1
        self.wtm=not self.wtm


_chess      = Chess()
_MOVE_TICKS = 6     # one move every 300 ms


def board_section(tick):
    """10 row()-strings: coloured chess board left, info panel right."""
    if tick > 0 and tick % _MOVE_TICKS == 0:
        _chess.step()

    b   = _chess.board
    src = _chess.src
    dst = _chess.dst

    # ── cell renderer: bg-coloured square + unicode glyph ─────────────────
    def cell(ri, ci):
        p      = b[ri][ci]
        light  = (ri + ci) % 2 == 0
        at_src = (ri, ci) == src
        at_dst = (ri, ci) == dst
        bg = (SQ_DST  if at_dst else
              SQ_SRC  if at_src else
              SQ_LIGHT if light  else SQ_DARK)
        fg  = _fg(p, bg)
        gly = GLYPH.get(p, p)
        # two terminal columns: glyph + space, both on the coloured bg
        return f"{bg}{fg}{gly} \033[0m"

    # ── board lines  (visual width = 4 + 16 + 1 = 21, pad to BW=23) ──────
    BW     = 23
    blines = []
    blines.append(c(GREY, "    a b c d e f g h "))          # header
    for ri in range(8):
        rank = 8 - ri
        ln   = c(GREY, f"  {rank} ")
        for ci in range(8): ln += cell(ri, ci)
        ln  += c(GREY, str(rank))
        blines.append(ln)
    blines.append(c(GREY, "    a b c d e f g h "))          # footer
    # 10 lines total

    # ── info panel (right column) ─────────────────────────────────────────
    san   = _chess.last_san or "---"
    uci   = _chess.last_uci or "------"
    num   = _chess.num
    turn  = c(WHITE+BOLD, "\u25cf White") if _chess.wtm else c(CYAN, "\u25cf Black")
    prog  = _chess.idx; total = len(IMMORTAL)
    pw    = 12
    pf    = max(0, min(pw, int(pw * prog / total)))
    pbar_ = c(CYAN,"\u2588"*pf) + c(GREY,"\u2591"*(pw-pf))

    # glyphs for captured pieces
    def glyphs(pieces):
        return "".join(GLYPH.get(p, p) for p in pieces) or "\u2014"

    cw_g = glyphs(_chess.cap_w)          # black pieces white captured
    cb_g = glyphs(_chess.cap_b)          # white pieces black captured
    cw_v = len(_chess.cap_w); cb_v = len(_chess.cap_b)

    ilines = [
        c(BOLD+BLUE,  ""),                                        # [0]
        f"  Move {num:>3d}  {turn}",                                          # [1]
        f"  {c(YELLOW+BOLD, san):<}  {c(GREY,'('+uci+')')}",                 # [2]
        f"  {pbar_}  {c(GREY, str(prog)+'/'+str(total))}",                   # [3]
        "",                                                                    # [4]
        f"  {c(GREEN+BOLD,'W+')} {c(CYAN,  cw_g[:16])}  {c(GREY,'x'+str(cw_v))}",  # [5]
        f"  {c(RED+BOLD,  'B+')} {c(WHITE, cb_g[:16])}  {c(GREY,'x'+str(cb_v))}",  # [6]
        "",                                                                    # [7]
        c(GREY, "  \u25a8 light  \u25a9 dark  \u2593 src  \u2588 dst"),  # [8] legend # [8] legend
        f"  {c(GREY,'')}",                       # [9]
    ]

    out = []
    for bl, il in zip(blines, ilines):
        pad = max(0, BW - vlen(bl))
        out.append(row(bl + " " * pad + il))
    return out


# ─── SIMULATED ENGINE METRICS ────────────────────────────────────────────────

_t0 = time.time()

def sim_metrics(tick):
    t          = time.time() - _t0
    nps        = int(1_240_000 * (1 + 0.12 * math.sin(t * 1.7) + random.uniform(-0.02, 0.02)))
    depth      = 18 + int(3 * abs(math.sin(t * 0.3)))
    seldepth   = depth + random.randint(4, 9)
    score      = int(35 * math.sin(t * 0.5) + random.uniform(-5, 5))
    alpha      = score - random.randint(20, 60)
    beta       = score + random.randint(20, 60)
    nodes      = int(nps * max(t, 0.001))
    tt_fill    = min(999, int(120 + 60 * math.sin(t * 0.2)))
    gpu_util   = min(99, int(88 + 6 * math.sin(t * 2.1) + random.uniform(-2, 2)))
    gpu_x      = round(7.8 + 0.3 * math.sin(t * 0.8), 1)
    mcts_pct   = int(72 + 8 * math.sin(t * 0.4))
    win_L8     = int(68 + 4  * math.sin(t * 0.15))
    win_L9     = int(50 + 10 * math.sin(t * 0.22))
    loss       = round(max(0.001, 0.023 + 0.002 * math.sin(t * 0.7)), 4)
    epoch      = min(100, int(42 + t * 0.05))
    acc        = round(94.0 + 1.5 * math.sin(t * 0.3), 1)
    elo        = 2870 + int(5 * math.sin(t * 0.05))
    if not hasattr(sim_metrics, "elo_hist"): sim_metrics.elo_hist = []
    sim_metrics.elo_hist.append(elo)
    if len(sim_metrics.elo_hist) > 20: sim_metrics.elo_hist.pop(0)
    moves = ["e2e4","d2d4","g1f3","c2c4","e4e5","d4d5"]
    pvs   = ["e2e4 e7e5 g1f3 b8c6","d2d4 d7d5 c2c4 e7e6","g1f3 d7d5 d2d4 g8f6"]
    return dict(nps=nps, depth=depth, seldepth=seldepth, score=score,
                alpha=alpha, beta=beta, nodes=nodes, tt_fill=tt_fill,
                gpu_util=gpu_util, gpu_x=gpu_x, mcts_pct=mcts_pct,
                win_L8=win_L8, win_L9=win_L9, loss=loss, epoch=epoch,
                acc=acc, elo=elo, elo_hist=list(sim_metrics.elo_hist),
                best=moves[tick % len(moves)],
                pv=pvs[tick % len(pvs)])

def score_col(s):
    if s > 50:  return GREEN
    if s > 0:   return CYAN
    if s < -50: return RED
    if s < 0:   return YELLOW
    return WHITE

# ─── FRAME RENDERER ──────────────────────────────────────────────────────────

def render(tick, m):
    now = (datetime.datetime.now()
           .strftime("%A, %B %d, %Y  %I:%M:%S.%f")[:-3] + " IST")
    L   = []

    # ── title ────────────────────────────────────────────────────────────────
    L.append(c(CYAN + BOLD, border()))
    title = "xChess"
    pl = (W - 2 - len(title)) // 2
    pr = (W - 2 - len(title)) - pl
    L.append(c(CYAN, "\u2551") +
              c(BOLD + YELLOW, " " * pl + title + " " * pr) +
              c(CYAN, "\u2551"))
    L.append(c(CYAN, divider()))

    # ── SEARCH ───────────────────────────────────────────────────────────────
    L.append(row(c(BOLD + BLUE, "  SEARCH")))
    L.append(blank())
    sc = f"{m['score']:+d} cp"
    L.append(row(f"  {'Best Move':<24}  {c(BOLD+WHITE, m['best'])}"
                 f"    Score: {c(score_col(m['score']), sc)}"))
    L.append(row(f"  {'Depth / Seldepth':<24}  "
                 f"{c(CYAN, str(m['depth']))} / {c(CYAN, str(m['seldepth']))}"))
    L.append(row(f"  {'Alpha / Beta window':<24}  "
                 f"{c(YELLOW, str(m['alpha']))} / {c(YELLOW, str(m['beta']))}"))
    L.append(row(f"  {'Principal Variation':<24}  {c(DIM+WHITE, m['pv'])}"))
    L.append(blank())

    # ── LIVE BOARD ───────────────────────────────────────────────────────────
    L.append(c(CYAN, divider()))
    for br in board_section(tick):
        L.append(br)
    L.append(blank())

    # ── PERFORMANCE ──────────────────────────────────────────────────────────
    L.append(c(CYAN, divider()))
    L.append(row(c(BOLD + BLUE, "  PERFORMANCE")))
    L.append(blank())
    nps_s   = f"{m['nps']:>12,}"
    nodes_s = f"{m['nodes']:>14,}"
    L.append(row(f"  {'Nodes / second':<24}  {c(GREEN+BOLD, nps_s)}"))
    L.append(row(f"  {'Total nodes searched':<24}  {c(GREEN, nodes_s)}"))
    L.append(row(f"  {'TT hashfull':<24}  "
                 f"{c(MAGENTA, str(m['tt_fill']))}/1000  ({m['tt_fill']/10:.1f}%)"))
    L.append(row(f"  {'Engine ELO (est.)':<24}  "
                 f"{c(BOLD+YELLOW, str(m['elo']))}  Trend: {c(YELLOW, sparkline(m['elo_hist']))}"))
    L.append(row(f"  {'Win rate vs SF L8':<24}  "
                 f"{mini_bar(m['win_L8'])}  {c(GREEN, str(m['win_L8'])+'%')}"))
    L.append(row(f"  {'Win rate vs SF L9':<24}  "
                 f"{mini_bar(m['win_L9'], col=CYAN)}  {c(CYAN, str(m['win_L9'])+'%')}"))
    L.append(blank())

    # ── GPU & HYBRID SEARCH ──────────────────────────────────────────────────
    L.append(c(CYAN, divider()))
    L.append(row(c(BOLD + BLUE, "  GPU  &  HYBRID SEARCH")))
    L.append(blank())
    L.append(row(f"  {'GPU Utilization':<24}  "
                 f"{pbar(m['gpu_util'], 100)}  {c(BOLD+GREEN, str(m['gpu_util'])+'%')}"))
    L.append(row(f"  {'GPU Speedup':<24}  "
                 f"{c(CYAN+BOLD, str(m['gpu_x'])+'x')}  vs CPU    Batch: {c(WHITE,'256')}"))
    L.append(row(f"  {'Active model':<24}  "
                 f"{c(YELLOW,'CNN')}    Mode: {c(CYAN,'AUTO')}    MCTS: {c(MAGENTA,str(m['mcts_pct'])+'%')}"))
    L.append(blank())

    # ── TRAINING ─────────────────────────────────────────────────────────────
    L.append(c(CYAN, divider()))
    L.append(row(c(BOLD + BLUE, "  TRAINING")))
    L.append(blank())
    L.append(row(f"  {'Epoch':<24}  "
                 f"{pbar(m['epoch'], 100, col=CYAN)}  {c(CYAN+BOLD, str(m['epoch']))}/100"))
    lc = GREEN if m['loss'] < 0.05 else YELLOW
    L.append(row(f"  {'Loss / Accuracy':<24}  "
                 f"{c(lc, str(m['loss']))}    Acc: {c(GREEN, str(m['acc'])+'%')}"))
    L.append(blank())

    # ── FOOTER ───────────────────────────────────────────────────────────────
    L.append(c(CYAN, divider()))
    L.append(row(c(DIM + WHITE, f"  Updated : {now}")))
    L.append(row(c(GREY,        "  Ctrl+C to exit")))
    L.append(c(CYAN + BOLD, bottom()))

    return "\n".join(L)


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

def cls():
    if sys.platform == "win32":
        os.system("cls")
    else:
        sys.stdout.write("\033[2J\033[H"); sys.stdout.flush()

def main():
    cls()
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleCursorInfo(
                ctypes.windll.kernel32.GetStdHandle(-11),
                ctypes.byref(type("CONSOLE_CURSOR_INFO", (ctypes.Structure,),
                    {"_fields_": [("dwSize", ctypes.c_ulong),
                                  ("bVisible", ctypes.c_bool)]})(1, False)))
        except Exception:
            pass
    else:
        sys.stdout.write("\033[?25l"); sys.stdout.flush()

    tick = 0
    try:
        while True:
            frame = render(tick, sim_metrics(tick))
            sys.stdout.write("\033[H" + frame + "\n")
            sys.stdout.flush()
            tick += 1
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        if sys.platform != "win32":
            sys.stdout.write("\033[?25h"); sys.stdout.flush()
        print("\n\nDashboard stopped.")

if __name__ == "__main__":
    main()
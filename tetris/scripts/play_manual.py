"""
Joue à Tetris au clavier dans le terminal (mode curses, temps réel).

Touches :
    Flèche gauche / droite : déplacer
    Flèche bas             : soft drop
    Flèche haut / X        : rotate CW
    Z                      : rotate CCW
    Espace                 : hard drop
    C                      : hold
    R                      : restart
    Q                      : quitter

Lancement :
    python scripts/play_manual.py
"""

import curses
import sys
import time
from pathlib import Path

# Ajoute la racine du projet au PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tetris.env import Tetris
from tetris.env.constants import PIECE_IDS, BOARD_BUFFER, BOARD_WIDTH

# Cible 60 FPS
FRAME_TIME = 1.0 / 60.0


# Couleurs curses par type de pièce (initialisées dans main)
COLOR_PAIRS = {}


def init_colors():
    """Init des paires de couleurs curses pour chaque pièce."""
    curses.start_color()
    curses.use_default_colors()
    # Couleurs proches du Guideline standard
    color_map = {
        "I": curses.COLOR_CYAN,
        "O": curses.COLOR_YELLOW,
        "T": curses.COLOR_MAGENTA,
        "S": curses.COLOR_GREEN,
        "Z": curses.COLOR_RED,
        "L": curses.COLOR_YELLOW,   # orange n'existe pas, on prend yellow
        "J": curses.COLOR_BLUE,
    }
    for i, (name, color) in enumerate(color_map.items(), start=1):
        curses.init_pair(i, color, -1)
        COLOR_PAIRS[PIECE_IDS[name] + 1] = i


def draw(stdscr, game):
    """Affiche le board + UI."""
    stdscr.erase()
    h, w = stdscr.getmaxyx()

    # Construit la grille à afficher (board + pièce courante superposée)
    display = game.grid.copy()
    if not game.game_over:
        piece_id = PIECE_IDS[game.current.type] + 1
        for col, row in game.current.blocks():
            if 0 <= row < display.shape[0] and 0 <= col < BOARD_WIDTH:
                display[row, col] = piece_id

    # Ghost piece (où la pièce va tomber si hard drop)
    ghost_blocks = []
    if not game.game_over:
        ghost_y_offset = 0
        while True:
            test_blocks = [(c, r + ghost_y_offset + 1) for c, r in game.current.blocks()]
            valid = True
            for c, r in test_blocks:
                if c < 0 or c >= BOARD_WIDTH or r >= display.shape[0]:
                    valid = False; break
                if r >= 0 and game.grid[r, c] != 0:
                    valid = False; break
            if not valid:
                break
            ghost_y_offset += 1
        if ghost_y_offset > 0:
            ghost_blocks = [(c, r + ghost_y_offset) for c, r in game.current.blocks()]

    # Affichage
    y_start = 1
    x_start = 2

    # Buffer (2 lignes au-dessus)
    for r_offset, r in enumerate(range(BOARD_BUFFER - 2, BOARD_BUFFER)):
        y = y_start + r_offset
        x = x_start
        stdscr.addstr(y, x, " ")
        for c in range(BOARD_WIDTH):
            v = display[r, c]
            if v != 0:
                pair = COLOR_PAIRS.get(v, 0)
                stdscr.addstr(y, x + 1 + c * 2, "▓▓", curses.color_pair(pair))
            else:
                stdscr.addstr(y, x + 1 + c * 2, "··", curses.A_DIM)

    # Top border
    stdscr.addstr(y_start + 2, x_start, "┌" + "──" * BOARD_WIDTH + "┐")

    # Board visible
    for r in range(BOARD_BUFFER, display.shape[0]):
        y = y_start + 3 + (r - BOARD_BUFFER)
        stdscr.addstr(y, x_start, "│")
        for c in range(BOARD_WIDTH):
            v = display[r, c]
            cell_x = x_start + 1 + c * 2
            if v != 0:
                pair = COLOR_PAIRS.get(v, 0)
                stdscr.addstr(y, cell_x, "██", curses.color_pair(pair))
            elif (c, r) in ghost_blocks:
                stdscr.addstr(y, cell_x, "░░", curses.A_DIM)
            else:
                stdscr.addstr(y, cell_x, "  ")
        stdscr.addstr(y, x_start + 1 + BOARD_WIDTH * 2, "│")

    # Bottom border
    stdscr.addstr(y_start + 3 + 20, x_start, "└" + "──" * BOARD_WIDTH + "┘")

    # Side panel (next, hold, score)
    panel_x = x_start + BOARD_WIDTH * 2 + 5
    stdscr.addstr(y_start + 2, panel_x, f"Score : {game.score}")
    stdscr.addstr(y_start + 3, panel_x, f"Lines : {game.lines_total}")
    stdscr.addstr(y_start + 4, panel_x, f"Level : {game.level}")
    stdscr.addstr(y_start + 6, panel_x, f"Next  : {game.next_piece}")
    stdscr.addstr(y_start + 7, panel_x, f"Hold  : {game.hold or '-'}")

    # Help
    stdscr.addstr(y_start + 10, panel_x, "← →   move")
    stdscr.addstr(y_start + 11, panel_x, "↓     soft drop")
    stdscr.addstr(y_start + 12, panel_x, "↑/X   rotate CW")
    stdscr.addstr(y_start + 13, panel_x, "Z     rotate CCW")
    stdscr.addstr(y_start + 14, panel_x, "Space hard drop")
    stdscr.addstr(y_start + 15, panel_x, "C     hold")
    stdscr.addstr(y_start + 16, panel_x, "R     restart")
    stdscr.addstr(y_start + 17, panel_x, "Q     quit")

    if game.game_over:
        msg = "*** GAME OVER ***"
        y_msg = y_start + 12
        x_msg = x_start + (BOARD_WIDTH * 2 - len(msg)) // 2
        stdscr.addstr(y_msg, x_msg, msg, curses.A_BOLD | curses.A_BLINK)

    stdscr.refresh()


def handle_key(game, key):
    """Mappe une touche curses vers une action du jeu."""
    if key == curses.KEY_LEFT:
        game.move(-1, 0)
    elif key == curses.KEY_RIGHT:
        game.move(1, 0)
    elif key == curses.KEY_DOWN:
        game.soft_drop()
    elif key == curses.KEY_UP or key in (ord("x"), ord("X")):
        game.rotate(1)
    elif key in (ord("z"), ord("Z")):
        game.rotate(-1)
    elif key == ord(" "):
        game.hard_drop()
    elif key in (ord("c"), ord("C")):
        game.hold_piece()


def main(stdscr):
    curses.curs_set(0)        # cache le curseur
    stdscr.nodelay(True)      # getch non-bloquant
    stdscr.timeout(int(FRAME_TIME * 1000))
    init_colors()

    game = Tetris()
    last_frame = time.time()

    while True:
        # Limite à 60 FPS
        now = time.time()
        elapsed = now - last_frame
        if elapsed < FRAME_TIME:
            time.sleep(FRAME_TIME - elapsed)
        last_frame = time.time()

        # Lecture clavier (non-bloquant)
        key = stdscr.getch()

        if key == ord("q") or key == ord("Q"):
            break
        elif key in (ord("r"), ord("R")):
            game = Tetris()
        elif key != -1:
            if not game.game_over:
                handle_key(game, key)

        # Tick du jeu (gravité + lock delay)
        if not game.game_over:
            game.tick()

        # Affichage
        draw(stdscr, game)


if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass
    print("Merci d'avoir joué !")

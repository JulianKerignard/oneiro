"""
Moteur de jeu Tetris Guideline (compatible tetr.io).

Convention de coordonnées :
- (col, row) avec row=0 en HAUT de la grille
- grid[row][col] : 0 si vide, id de pièce + 1 si occupé
- spawn dans le buffer du haut (rows 0..BOARD_BUFFER-1, invisibles)
- board visible : rows BOARD_BUFFER..BOARD_BUFFER+BOARD_HEIGHT-1
"""

import numpy as np
import random
from .constants import (
    BOARD_WIDTH, BOARD_HEIGHT, BOARD_BUFFER,
    PIECES, PIECE_IDS, SHAPES, PIECE_BOX_SIZE,
    KICKS_JLSTZ, KICKS_I, SPAWN_COL,
    SCORE_TABLE, LOCK_DELAY, GRAVITY_INTERVAL,
)


class Piece:
    """Une pièce active sur le board."""
    __slots__ = ("type", "rotation", "x", "y")

    def __init__(self, piece_type, x, y, rotation=0):
        self.type = piece_type
        self.rotation = rotation
        self.x = x
        self.y = y

    def blocks(self):
        """Retourne les positions absolues (col, row) des 4 blocs."""
        return [(self.x + dx, self.y + dy) for dx, dy in SHAPES[self.type][self.rotation]]


class Tetris:
    """Moteur de jeu Tetris Guideline."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.reset()

    # ------------------------------------------------------------ setup

    def reset(self):
        total_h = BOARD_HEIGHT + BOARD_BUFFER
        self.grid = np.zeros((total_h, BOARD_WIDTH), dtype=np.int8)
        self.bag = []
        self.hold = None
        self.hold_used = False
        self.score = 0
        self.lines_total = 0
        self.level = 1
        self.game_over = False
        self.lock_timer = 0
        self.gravity_timer = 0
        self.current = self._spawn_piece()
        self.next_piece = self._next_from_bag()

    # ------------------------------------------------------------ bag

    def _refill_bag(self):
        """7-bag : tirage random des 7 pièces sans répétition."""
        new_bag = list(PIECES)
        self.rng.shuffle(new_bag)
        self.bag.extend(new_bag)

    def _next_from_bag(self):
        if not self.bag:
            self._refill_bag()
        return self.bag.pop(0)

    # ------------------------------------------------------------ spawn

    def _spawn_piece(self, forced_type=None):
        """Fait apparaître une nouvelle pièce. Détecte game over si collision."""
        ptype = forced_type if forced_type else self._next_from_bag()
        # spawn position : centré horizontalement, dans le buffer du haut
        spawn_x = SPAWN_COL
        spawn_y = BOARD_BUFFER - PIECE_BOX_SIZE[ptype]  # juste au-dessus du board visible
        piece = Piece(ptype, spawn_x, spawn_y, rotation=0)
        if not self._is_valid(piece):
            self.game_over = True
        self.hold_used = False
        self.lock_timer = 0
        self.gravity_timer = 0
        return piece

    # ------------------------------------------------------------ validation

    def _is_valid(self, piece):
        """Vérifie qu'une pièce est dans des positions valides (pas hors board, pas en collision)."""
        for col, row in piece.blocks():
            if col < 0 or col >= BOARD_WIDTH:
                return False
            if row >= BOARD_HEIGHT + BOARD_BUFFER:
                return False
            if row >= 0 and self.grid[row, col] != 0:
                return False
        return True

    # ------------------------------------------------------------ actions

    def move(self, dx, dy):
        """Tente de déplacer la pièce de (dx, dy). Retourne True si succès."""
        if self.game_over:
            return False
        candidate = Piece(self.current.type, self.current.x + dx, self.current.y + dy,
                          self.current.rotation)
        if self._is_valid(candidate):
            self.current = candidate
            return True
        return False

    def rotate(self, direction):
        """Rotation avec SRS wall kicks. direction = +1 (CW) ou -1 (CCW)."""
        if self.game_over:
            return False
        if self.current.type == "O":
            return True  # O ne tourne pas

        from_rot = self.current.rotation
        to_rot = (from_rot + direction) % 4

        kicks = KICKS_I if self.current.type == "I" else KICKS_JLSTZ
        offsets = kicks.get((from_rot, to_rot), [(0, 0)])

        for dx, dy in offsets:
            # Convention SRS : dy positif = vers le haut sur l'écran
            # Notre convention : row positif = vers le bas. On inverse dy.
            candidate = Piece(
                self.current.type,
                self.current.x + dx,
                self.current.y - dy,
                to_rot,
            )
            if self._is_valid(candidate):
                self.current = candidate
                return True
        return False

    def hard_drop(self):
        """Fait tomber la pièce jusqu'au sol et lock immédiat."""
        if self.game_over:
            return 0
        dropped = 0
        while self.move(0, 1):
            dropped += 1
        self._lock_piece()
        return dropped

    def soft_drop(self):
        """Descend d'1 case."""
        return self.move(0, 1)

    def hold_piece(self):
        """Met la pièce en hold (ou échange avec celle déjà en hold)."""
        if self.game_over or self.hold_used:
            return False
        if self.hold is None:
            self.hold = self.current.type
            self.current = self._spawn_piece()
        else:
            new_type = self.hold
            self.hold = self.current.type
            self.current = self._spawn_piece(forced_type=new_type)
        self.hold_used = True
        return True

    # ------------------------------------------------------------ lock + clear

    def _lock_piece(self):
        """Pose la pièce dans la grille, clear les lignes pleines, spawn next."""
        piece_id = PIECE_IDS[self.current.type] + 1  # +1 pour distinguer du vide (0)
        for col, row in self.current.blocks():
            if 0 <= row < self.grid.shape[0] and 0 <= col < BOARD_WIDTH:
                self.grid[row, col] = piece_id

        lines = self._clear_lines()
        if lines > 0:
            self.score += SCORE_TABLE.get(lines, 0) * self.level
            self.lines_total += lines

        self.current = self._spawn_piece(forced_type=self.next_piece)
        self.next_piece = self._next_from_bag()

    def _clear_lines(self):
        """Détecte et supprime les lignes pleines. Retourne le nombre clearé."""
        full_rows = []
        for r in range(self.grid.shape[0]):
            if np.all(self.grid[r] != 0):
                full_rows.append(r)
        if not full_rows:
            return 0
        # Supprimer les lignes pleines et faire descendre le reste
        keep_mask = np.ones(self.grid.shape[0], dtype=bool)
        keep_mask[full_rows] = False
        kept = self.grid[keep_mask]
        new_grid = np.zeros_like(self.grid)
        new_grid[-len(kept):] = kept  # les lignes restantes vont en bas
        self.grid = new_grid
        return len(full_rows)

    # ------------------------------------------------------------ tick (gravité + lock delay)

    def tick(self):
        """Avance d'1 frame : gravité + gestion du lock delay."""
        if self.game_over:
            return

        # Test si la pièce touche le sol (peut pas descendre)
        touching = not self._can_move(0, 1)

        if touching:
            self.lock_timer += 1
            if self.lock_timer >= LOCK_DELAY:
                self._lock_piece()
        else:
            self.lock_timer = 0
            self.gravity_timer += 1
            if self.gravity_timer >= GRAVITY_INTERVAL:
                self.move(0, 1)
                self.gravity_timer = 0

    def _can_move(self, dx, dy):
        candidate = Piece(self.current.type, self.current.x + dx, self.current.y + dy,
                          self.current.rotation)
        return self._is_valid(candidate)

    # ------------------------------------------------------------ rendering console

    def render(self):
        """Affichage console (pour debug + jeu manuel)."""
        # Grille avec pièce courante superposée
        display = self.grid.copy()
        if not self.game_over:
            piece_id = PIECE_IDS[self.current.type] + 1
            for col, row in self.current.blocks():
                if 0 <= row < display.shape[0] and 0 <= col < BOARD_WIDTH:
                    display[row, col] = piece_id

        lines = []
        # Buffer (2 lignes au-dessus du board pour voir la pièce qui spawn)
        for r in range(BOARD_BUFFER - 2, BOARD_BUFFER):
            row_str = " "
            for c in range(BOARD_WIDTH):
                v = display[r, c]
                row_str += "▓▓" if v != 0 else "··"
            row_str += " "
            lines.append(row_str)
        lines.append("┌" + "──" * BOARD_WIDTH + "┐")
        # Board visible
        for r in range(BOARD_BUFFER, display.shape[0]):
            row_str = "│"
            for c in range(BOARD_WIDTH):
                v = display[r, c]
                row_str += "██" if v != 0 else "  "
            row_str += "│"
            lines.append(row_str)
        lines.append("└" + "──" * BOARD_WIDTH + "┘")
        lines.append(f"Score: {self.score}  Lines: {self.lines_total}  Level: {self.level}")
        lines.append(f"Next: {self.next_piece}  Hold: {self.hold or '-'}")
        if self.game_over:
            lines.append("  *** GAME OVER ***")
        return "\n".join(lines)

    def __str__(self):
        return self.render()

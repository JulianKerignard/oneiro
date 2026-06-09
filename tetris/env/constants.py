"""
Tetris Guideline constants (SRS - Super Rotation System).
Compatible avec tetr.io, Tetris Effect, Puyo Puyo Tetris, etc.
"""

import numpy as np

# Dimensions standard Tetris Guideline
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
BOARD_BUFFER = 4  # rangées invisibles au-dessus pour le spawn

# Identifiants des pièces
PIECES = ["I", "O", "T", "S", "Z", "L", "J"]
PIECE_IDS = {name: i for i, name in enumerate(PIECES)}

# Couleurs Guideline standard (pour rendering)
COLORS = {
    "I": (0, 240, 240),
    "O": (240, 240, 0),
    "T": (160, 0, 240),
    "S": (0, 240, 0),
    "Z": (240, 0, 0),
    "L": (240, 160, 0),
    "J": (0, 0, 240),
}

# Formes des pièces (rotation 0 = spawn orientation)
# Chaque pièce = liste des 4 positions (col, row) occupées
# Référentiel : (0,0) = coin haut-gauche de la matrice englobante
SHAPES_R0 = {
    "I": [(0, 1), (1, 1), (2, 1), (3, 1)],  # ligne horizontale au milieu d'une 4x4
    "O": [(1, 0), (2, 0), (1, 1), (2, 1)],  # carré 2x2 centré dans une 4x4
    "T": [(1, 0), (0, 1), (1, 1), (2, 1)],  # T pointant vers le haut
    "S": [(1, 0), (2, 0), (0, 1), (1, 1)],  # S
    "Z": [(0, 0), (1, 0), (1, 1), (2, 1)],  # Z
    "L": [(2, 0), (0, 1), (1, 1), (2, 1)],  # L
    "J": [(0, 0), (0, 1), (1, 1), (2, 1)],  # J
}

# Tailles de matrice englobante (pour rotation correcte)
# I tourne dans une 4x4, les autres dans une 3x3 (sauf O qui ne tourne pas)
PIECE_BOX_SIZE = {
    "I": 4,
    "O": 2,
    "T": 3, "S": 3, "Z": 3, "L": 3, "J": 3,
}


def rotate_cw(positions, box_size):
    """Rotation horaire 90° dans une matrice carrée box_size×box_size."""
    return [(box_size - 1 - row, col) for (col, row) in positions]


def rotate_ccw(positions, box_size):
    """Rotation antihoraire 90°."""
    return [(row, box_size - 1 - col) for (col, row) in positions]


def compute_all_rotations(piece_name):
    """Génère les 4 rotations d'une pièce depuis sa forme spawn (R0)."""
    box = PIECE_BOX_SIZE[piece_name]
    r0 = SHAPES_R0[piece_name]
    r1 = rotate_cw(r0, box)
    r2 = rotate_cw(r1, box)
    r3 = rotate_cw(r2, box)
    return [r0, r1, r2, r3]


# Pré-calcul des 4 rotations pour chaque pièce
# SHAPES[piece_name][rotation_index] = liste de (col, row)
SHAPES = {name: compute_all_rotations(name) for name in PIECES}


# SRS Wall Kick tables
# Quand une rotation classique est bloquée par un mur ou bloc,
# on essaie plusieurs offsets dans l'ordre. Le premier qui marche est utilisé.
#
# Format : KICKS_JLSTZ[(from_rotation, to_rotation)] = liste de (dx, dy) à essayer
# Source : Tetris Wiki, SRS standard

KICKS_JLSTZ = {
    (0, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (1, 0): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    (1, 2): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    (2, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (2, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
    (3, 2): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (3, 0): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (0, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
}

KICKS_I = {
    (0, 1): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
    (1, 0): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
    (1, 2): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
    (2, 1): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
    (2, 3): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
    (3, 2): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
    (3, 0): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
    (0, 3): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
}


# Position de spawn (colonne du coin haut-gauche de la matrice englobante)
# Guideline : centré horizontalement, sur les rangées 19-20 (juste au-dessus du board visible)
SPAWN_COL = 3   # colonne 3 pour pièces 3x3, ajusté à 3 pour I (4-wide en col 3)


# Actions disponibles (interface RL)
ACTIONS = {
    0: "noop",
    1: "left",
    2: "right",
    3: "rotate_cw",
    4: "rotate_ccw",
    5: "soft_drop",
    6: "hard_drop",
    7: "hold",
}
NUM_ACTIONS = len(ACTIONS)


# Scoring Guideline standard
SCORE_TABLE = {
    1: 100,   # Single
    2: 300,   # Double
    3: 500,   # Triple
    4: 800,   # Tetris (clear 4 lignes d'un coup)
}
# Bonus T-spin / B2B viendront plus tard (V2)


# Constantes de timing (en steps)
LOCK_DELAY = 30        # 30 frames pour bouger après touch ground (Guideline = 0.5s à 60fps)
GRAVITY_INTERVAL = 60  # 1 case par seconde au niveau 1 (60 frames)

"""
Wrapper RL pour Tetris.

Action space (high-level, à la Cold Clear) :
    0-39  : placements         (action // 10 = rotation 0-3, action % 10 = colonne 0-9)
    40    : hold               (swap avec la pièce en réserve)
    Total : 41 actions discrètes

Observation space :
    grid (24×10 binaire)        : 240
    one_hot(current piece, 7)   : 7
    one_hot(next[0], 7)         : 7
    one_hot(next[1], 7)         : 7
    one_hot(next[2], 7)         : 7
    one_hot(hold, 8)            : 8   (7 pièces + 1 "vide")
    Total                        : 276 (float32)

Reward shaping (volontairement minimal, pénalité pour actions invalides) :
    + delta_score / 100         : récompense de score (≈ +1 single, +8 tetris)
    - 0.1                       : si action invalide (place hors-board ou hold déjà utilisé)
    - 1.0                       : à la mort (game over)
"""

import numpy as np
from .game import Tetris, Piece
from .constants import (
    BOARD_WIDTH, BOARD_HEIGHT, BOARD_BUFFER,
    PIECES, PIECE_IDS, SHAPES, PIECE_BOX_SIZE,
    NUM_ACTIONS,
)

# Espace d'observation
GRID_FLAT_SIZE = (BOARD_HEIGHT + BOARD_BUFFER) * BOARD_WIDTH   # 24 * 10 = 240
NUM_PIECES = 7                                                  # I, O, T, S, Z, L, J
HOLD_VOCAB = 8                                                  # 7 pièces + "vide"
N_NEXT_LOOKAHEAD = 3                                            # 3 prochaines visibles

OBS_DIM = (
    GRID_FLAT_SIZE                       # grille
    + NUM_PIECES                          # current
    + NUM_PIECES * N_NEXT_LOOKAHEAD       # 3 next
    + HOLD_VOCAB                          # hold
)
# 240 + 7 + 21 + 8 = 276

# Action space
ACTION_DIM = 41   # 40 placements + 1 hold
HOLD_ACTION = 40

# Rewards
INVALID_ACTION_PENALTY = -0.1
GAME_OVER_PENALTY = -1.0
SCORE_NORMALIZER = 100.0   # divise delta_score, donc +1 par single (Tetris = +8)

# Reward shaping coefficients (rééquilibrés : le shaping ne doit pas dominer
# la récompense de clear de lignes, sinon l'agent évite tout placement à risque
# et ne complète jamais de ligne).
SHAPING_MAX_HEIGHT = 0.005     # pénalité par unité de hauteur max ajoutée
SHAPING_HOLES = 0.05            # pénalité par trou nouveau créé
SHAPING_BUMPINESS = 0.01        # pénalité par unité de bumpiness ajoutée

# Bonus de line clear décroissant (curriculum-like) :
# Au début : gros bonus pour aider l'agent à découvrir l'intérêt des lignes.
# Quand il a clearé beaucoup → bonus diminue vers 0, le score natif (+1 single) suffit.
# Évite que le critic soit déstabilisé par des rewards trop forts en permanence.
SHAPING_LINE_CLEAR_BASE = 5.0     # bonus initial par ligne
SHAPING_LINE_CLEAR_DECAY = 0.1    # vitesse de décroissance (50% après 10 lignes lifetime)

# Potential-based reward shaping (Ng & Russell 1999) sur le remplissage des lignes.
#   φ(grid) = ∑ (fill_ratio_par_ligne)²
#   shaped_reward += SHAPING_LINE_FILL × (φ(s_new) - φ(s_old))
# Le carré = signal plus fort sur les lignes presque pleines (gradient vers le clear).
# Le DELTA = pas de stagnation possible, l'agent doit toujours progresser pour gagner.
# Théorème Ng & Russell : la policy optimale est PRÉSERVÉE (on guide l'exploration).
SHAPING_LINE_FILL = 0.5


def compute_grid_features(grid):
    """
    Calcule les features de la grille pour le reward shaping.

    Args:
        grid : np.array (H, W) avec 0=vide, >0=occupé

    Returns:
        dict avec :
            max_height       : hauteur de la colonne la plus haute
            aggregate_height : somme des hauteurs de toutes les colonnes
            holes            : nb de cases vides sous des blocs
            bumpiness        : somme des |delta hauteur| entre colonnes voisines
    """
    H, W = grid.shape
    occupied = (grid != 0)

    # Hauteur de chaque colonne (= H - première ligne occupée, ou 0 si colonne vide)
    heights = np.zeros(W, dtype=np.int32)
    for c in range(W):
        col = occupied[:, c]
        first = np.argmax(col) if col.any() else H
        heights[c] = H - first

    # Trous : cases vides sous le premier bloc de chaque colonne
    holes = 0
    for c in range(W):
        if heights[c] > 0:
            top_row = H - heights[c]
            holes += int((~occupied[top_row:, c]).sum())

    bumpiness = int(np.abs(np.diff(heights)).sum())

    # Potential φ pour le potential-based shaping :
    # ∑ (fill_per_line)² → signal quadratique vers les lignes presque pleines.
    fill_per_line = occupied.sum(axis=1) / W   # (H,) ratio dans [0, 1]
    potential = float(((fill_per_line) ** 2).sum())

    return {
        "max_height":       int(heights.max()),
        "aggregate_height": int(heights.sum()),
        "holes":            holes,
        "bumpiness":        bumpiness,
        "potential":        potential,
    }


class TetrisEnv:
    """Interface RL pour Tetris (gym-style)."""

    def __init__(self, seed=None, invalid_penalty=None, max_episode_steps=None,
                 reward_shaping=False):
        """
        Args:
            seed                : seed pour le RNG du jeu
            invalid_penalty      : reward négatif pour action invalide (None = défaut -0.1)
            max_episode_steps    : si fourni, truncate l'épisode après N steps
            reward_shaping        : si True, ajoute du dense reward basé sur hauteur/trous/bumpiness
        """
        self.game = Tetris(seed=seed)
        self.action_dim = ACTION_DIM
        self.obs_dim = OBS_DIM
        self.invalid_penalty = invalid_penalty if invalid_penalty is not None else INVALID_ACTION_PENALTY
        self.max_episode_steps = max_episode_steps
        self.reward_shaping = reward_shaping
        self.episode_step = 0
        self._prev_features = None
        # Compteur lifetime de lignes clearées (pour le bonus décroissant).
        # Persiste à travers les épisodes pour cet env (pas reset par reset()).
        self.lifetime_lines_cleared = 0
        # PERF #5 : cache du action mask. Le mask ne change qu'après step() qui modifie
        # la pièce courante. Évite de recalculer 41 placements à chaque appel.
        self._cached_mask = None

    # ============================================================== gym API

    def reset(self, seed=None):
        if seed is not None:
            self.game = Tetris(seed=seed)
        else:
            self.game.reset()
        self.episode_step = 0
        if self.reward_shaping:
            self._prev_features = compute_grid_features(self.game.grid)
        self._cached_mask = None   # invalide le cache après reset
        return self._get_obs()

    def step(self, action):
        """Applique une action high-level. Retourne (obs, reward, done, info)."""
        if self.game.game_over:
            return self._get_obs(), 0.0, True, {"invalid": False}

        prev_score = self.game.score
        invalid = False

        if action == HOLD_ACTION:
            ok = self.game.hold_piece()
            if not ok:
                invalid = True
        else:
            rotation = action // 10
            column = action % 10
            ok = self._apply_placement(rotation, column)
            if not ok:
                invalid = True

        prev_lines = self.game.lines_total

        # Calcul du reward
        if invalid:
            reward = self.invalid_penalty
        else:
            reward = (self.game.score - prev_score) / SCORE_NORMALIZER

            # Reward shaping (dense signal basé sur la grille)
            if self.reward_shaping:
                new_features = compute_grid_features(self.game.grid)
                if self._prev_features is not None:
                    delta_h = new_features["max_height"] - self._prev_features["max_height"]
                    delta_holes = new_features["holes"] - self._prev_features["holes"]
                    delta_bump = new_features["bumpiness"] - self._prev_features["bumpiness"]
                    delta_pot = new_features["potential"] - self._prev_features["potential"]
                    reward -= SHAPING_MAX_HEIGHT * max(0, delta_h)
                    reward -= SHAPING_HOLES * delta_holes
                    reward -= SHAPING_BUMPINESS * delta_bump
                    # Potential-based : signe libre (pas de max(0,...))
                    #   - placement vers ligne pleine → delta > 0 → bonus
                    #   - clear → delta < 0 LOCALEMENT, compensé par le clear bonus
                    reward += SHAPING_LINE_FILL * delta_pot
                self._prev_features = new_features

                # Bonus de line clear DÉCROISSANT (curriculum-like)
                new_lines = self.game.lines_total - prev_lines
                if new_lines > 0:
                    bonus_per_line = SHAPING_LINE_CLEAR_BASE / (
                        1.0 + self.lifetime_lines_cleared * SHAPING_LINE_CLEAR_DECAY
                    )
                    reward += bonus_per_line * new_lines
                    self.lifetime_lines_cleared += new_lines

        if self.game.game_over:
            reward += GAME_OVER_PENALTY

        self.episode_step += 1
        # Truncate si on dépasse max_episode_steps
        if self.max_episode_steps is not None and self.episode_step >= self.max_episode_steps:
            if not self.game.game_over:
                self.game.game_over = True

        self._cached_mask = None   # PERF #5 : invalide le cache après step (current piece a changé)
        return self._get_obs(), reward, self.game.game_over, {"invalid": invalid}

    # ============================================================== action mask

    def get_action_mask(self) -> np.ndarray:
        """
        Retourne un mask boolean de taille action_dim.
        True = action valide depuis l'état courant.

        Utilisé pour empêcher l'actor de prendre des actions invalides
        (rotation/colonne impossible, ou hold déjà utilisé).

        PERF #5 : cache invalidé seulement après step() ou reset().
        """
        if self._cached_mask is not None:
            return self._cached_mask

        mask = np.zeros(self.action_dim, dtype=bool)

        if self.game.game_over:
            self._cached_mask = mask
            return mask  # tout invalide après game over

        # Hold : valide si pas déjà utilisé ce drop
        if not self.game.hold_used:
            mask[HOLD_ACTION] = True

        piece_type = self.game.current.type

        # O ne tourne pas : seules les actions avec rotation=0 sont possibles
        rotations = [0] if piece_type == "O" else range(4)

        for rotation in rotations:
            for column in range(BOARD_WIDTH):
                action = rotation * 10 + column
                if action >= 40:
                    continue
                # Test horizontal : la pièce tient-elle dans le board ?
                test_piece = Piece(piece_type, column, 0, rotation)
                if not all(0 <= c < BOARD_WIDTH for c, _ in test_piece.blocks()):
                    continue
                # Test au spawn : peut-on poser la pièce sans collision immédiate ?
                spawn_y = BOARD_BUFFER - PIECE_BOX_SIZE[piece_type]
                candidate = Piece(piece_type, column, spawn_y, rotation)
                if self.game._is_valid(candidate):
                    mask[action] = True

        self._cached_mask = mask
        return mask

    # ============================================================== placement

    def _apply_placement(self, rotation, target_col):
        """
        Place la pièce courante à la rotation et colonne demandées.
        Retourne True si succès, False si action invalide.
        """
        piece_type = self.game.current.type

        # Construit une pièce hypothétique : rotation appliquée, à la colonne cible
        # On positionne la pièce tout en haut, puis on simule la chute via hard_drop
        if piece_type == "O" and rotation != 0:
            # O ne tourne pas — toute action avec rotation > 0 est invalide
            return False

        # Vérifie d'abord que la pièce tient horizontalement à cette colonne et rotation
        test_piece = Piece(piece_type, target_col, 0, rotation)
        for col, _ in test_piece.blocks():
            if col < 0 or col >= BOARD_WIDTH:
                return False

        # On configure la pièce courante avec la rotation cible et la colonne cible
        # à la position de spawn (tout en haut)
        spawn_y = BOARD_BUFFER - PIECE_BOX_SIZE[piece_type]
        candidate = Piece(piece_type, target_col, spawn_y, rotation)
        if not self.game._is_valid(candidate):
            return False

        self.game.current = candidate
        # Hard drop pour lock immédiatement
        self.game.hard_drop()
        return True

    # ============================================================== observation

    def _get_obs(self):
        """Construit le vecteur d'observation (276 dim)."""
        # 1. Grille binaire (board + buffer)
        grid_bin = (self.game.grid != 0).astype(np.float32).flatten()

        # 2. Current piece
        current_oh = self._piece_one_hot(self.game.current.type)

        # 3. Next pieces lookahead (3)
        next_pieces = self._peek_next(N_NEXT_LOOKAHEAD)
        next_oh = np.concatenate([self._piece_one_hot(p) for p in next_pieces])

        # 4. Hold piece (8 dim : 7 pièces + 1 vide)
        hold_oh = np.zeros(HOLD_VOCAB, dtype=np.float32)
        if self.game.hold is None:
            hold_oh[7] = 1.0   # "vide"
        else:
            hold_oh[PIECE_IDS[self.game.hold]] = 1.0

        return np.concatenate([grid_bin, current_oh, next_oh, hold_oh]).astype(np.float32)

    def _piece_one_hot(self, piece_type):
        oh = np.zeros(NUM_PIECES, dtype=np.float32)
        if piece_type is not None and piece_type in PIECE_IDS:
            oh[PIECE_IDS[piece_type]] = 1.0
        return oh

    def _peek_next(self, n):
        """Retourne les n prochaines pièces sans modifier l'état du bag."""
        # game.next_piece = la 1ère prochaine (déjà tirée)
        # game.bag = pile des suivantes
        upcoming = [self.game.next_piece]
        # On copie le bag pour ne pas le modifier
        bag_copy = list(self.game.bag)
        rng_copy = self.game.rng.getstate()

        while len(upcoming) < n:
            if not bag_copy:
                # Re-shuffle simulé (sans toucher au vrai rng)
                temp_bag = list(PIECES)
                # Génère un shuffle déterministe basé sur l'état rng courant
                import random as _r
                tmp_rng = _r.Random()
                tmp_rng.setstate(rng_copy)
                tmp_rng.shuffle(temp_bag)
                bag_copy.extend(temp_bag)
                # Note : ce n'est pas parfaitement aligné avec ce qui sortira réellement
                # mais c'est suffisant pour donner un signal à l'agent
            upcoming.append(bag_copy.pop(0))

        return upcoming[:n]

    # ============================================================== rendering passthrough

    def render(self):
        return self.game.render()

    def __str__(self):
        return self.render()

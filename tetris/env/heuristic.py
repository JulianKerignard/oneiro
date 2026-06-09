"""
Heuristique Dellacherie (1996) pour Tetris.

Référence : Pierre Dellacherie, algorithme du joueur "Pierre Dellacherie"
qui a tenu le record du monde Tetris des années 90.

Pour chaque action valide :
  1. Simule le placement via deepcopy de l'env
  2. Calcule les features (aggregate_height, holes, bumpiness, lines_cleared)
  3. Score = somme pondérée des features (coefs Dellacherie)
  4. Retourne l'action avec le meilleur score

Utilisé pour le pretrain du replay buffer : injecte ~500-1500 line clears
dans les premières 10k transitions, ce que l'agent random ne ferait jamais.
"""

import copy
import numpy as np

from .env import compute_grid_features, HOLD_ACTION


# ============================== Coefficients Dellacherie (1996)
# Validés empiriquement sur des millions de games.
# Le seul qui est POSITIF : lines_cleared (récompense le clear).
# Les autres sont des PÉNALITÉS (hauteur, trous, surface chaotique).
COEF_AGGREGATE_HEIGHT = -0.510066
COEF_LINES_CLEARED   = +0.760666
COEF_HOLES           = -0.356629
COEF_BUMPINESS       = -0.184483


def dellacherie_score(features, lines_cleared):
    """Score d'un état post-placement selon Dellacherie."""
    return (
        COEF_AGGREGATE_HEIGHT * features["aggregate_height"]
        + COEF_LINES_CLEARED * lines_cleared
        + COEF_HOLES * features["holes"]
        + COEF_BUMPINESS * features["bumpiness"]
    )


def select_heuristic_action(env):
    """
    Choisit la meilleure action selon Dellacherie.

    Args:
        env : TetrisEnv (état non modifié)

    Returns:
        action : int dans [0, 40]
    """
    mask = env.get_action_mask()
    valid_placements = [a for a in np.where(mask)[0] if a != HOLD_ACTION]

    # Edge case : aucune action de placement valide (game over imminent)
    if not valid_placements:
        if mask[HOLD_ACTION]:
            return HOLD_ACTION
        return 0  # safety, sera invalid mais l'env gère

    best_action = valid_placements[0]
    best_score = -float("inf")
    prev_lines = env.game.lines_total

    for action in valid_placements:
        # Simulation : deepcopy de l'env, on step dessus, on lit le résultat
        env_sim = copy.deepcopy(env)
        env_sim.step(action)
        # Si l'action a causé game over, score très négatif (à éviter)
        if env_sim.game.game_over:
            score = -1e6
        else:
            features = compute_grid_features(env_sim.game.grid)
            lines_cleared = env_sim.game.lines_total - prev_lines
            score = dellacherie_score(features, lines_cleared)

        if score > best_score:
            best_score = score
            best_action = action

    return int(best_action)

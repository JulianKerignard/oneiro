"""
Wrapper Crafter compatible avec notre pipeline Dreamer.

Crafter (Hafner 2021) : 2D survival game, 22 achievements, pixels 64×64×3.

API compatible TetrisEnv :
    reset(seed=None) → obs (3, 64, 64) float32 [0, 1]
    step(action)     → obs, reward, done, info
    get_action_mask()→ np.array(17, dtype=bool) — toutes valides
    obs_shape        : (3, 64, 64)
    action_dim       : 17

Différences avec TetrisEnv :
    - obs est une IMAGE (3 channels) au lieu d'un vecteur (276 dim)
    - pas d'action invalide possible (Crafter ne mask rien)
    - reward sparse (achievements rares)
    - info contient 'achievements' (dict de 22 flags)
"""

import numpy as np
import crafter   # package pip


# ============================== Constants
OBS_SHAPE = (3, 64, 64)        # (C, H, W) après normalisation
OBS_DIM = 3 * 64 * 64          # 12288 (flat) — pour cohérence si besoin
ACTION_DIM = 17

ACHIEVEMENTS = (
    "collect_coal", "collect_diamond", "collect_drink", "collect_iron",
    "collect_sapling", "collect_stone", "collect_wood",
    "defeat_skeleton", "defeat_zombie",
    "eat_cow", "eat_plant",
    "make_iron_pickaxe", "make_iron_sword",
    "make_stone_pickaxe", "make_stone_sword",
    "make_wood_pickaxe", "make_wood_sword",
    "place_furnace", "place_plant", "place_stone", "place_table",
    "wake_up",
)
N_ACHIEVEMENTS = len(ACHIEVEMENTS)   # 22

ACTION_NAMES = (
    "noop", "move_left", "move_right", "move_up", "move_down",
    "do", "sleep",
    "place_stone", "place_table", "place_furnace", "place_plant",
    "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
    "make_wood_sword", "make_stone_sword", "make_iron_sword",
)

# Indices d'actions utiles au diagnostic (l'anti-spam sleep/noop a été retiré :
# désactivé depuis v14 car il interférait avec les hyperparams du paper, et la
# machinerie tournait à chaque step pour une pénalité toujours nulle).
NOOP_ACTION_IDX = 0
SLEEP_ACTION_IDX = 6


class CrafterEnv:
    """Wrapper Crafter compatible TetrisEnv-style."""

    obs_shape = OBS_SHAPE
    obs_dim = OBS_DIM
    action_dim = ACTION_DIM

    def __init__(self, seed=None, length=10000):
        """
        Args:
            seed   : seed du RNG (optionnel)
            length : max episode steps (défaut 10000, comme Crafter standard)
        """
        self._length = length          # conservé : reset(seed=) recrée l'env et doit le repasser
        self._env = crafter.Env(seed=seed, length=length)
        self._episode_step = 0
        self._unlocked_this_episode = set()
        self._cached_mask = np.ones(ACTION_DIM, dtype=bool)   # toutes valides toujours
        # DIAGNOSTIC : max d'inventaire atteint dans l'épisode + tentatives par action.
        # Le gate du craft est ARITHMÉTIQUE (data.yaml) : place_table coûte wood=2,
        # make_wood_pickaxe wood=1+table, collect_stone exige la pioche → ≥3 bois.
        # Or collect_wood ne donne +1 qu'UNE fois : les bois 2 et 3 ne rapportent RIEN.
        # Sans ces compteurs, impossible de distinguer "n'essaie pas" / "essaie mais
        # n'a pas les ressources" / "a les ressources mais l'action échoue".
        self._inv_max = {}
        self._action_counts = np.zeros(ACTION_DIM, dtype=np.int64)

    # ============================================================== gym API

    def reset(self, seed=None):
        """Reset l'env, retourne obs (3, 64, 64) float32."""
        if seed is not None:
            # Crafter ne supporte pas seed dynamique : recréer l'env.
            # /!\ repasser `length` — sans lui l'env repartait sur le défaut 10000
            # et un env construit avec une autre durée la perdait silencieusement.
            self._env = crafter.Env(seed=seed, length=self._length)
        obs = self._env.reset()
        self._episode_step = 0
        self._unlocked_this_episode = set()
        self._inv_max = {}
        self._action_counts[:] = 0
        return self._normalize_obs(obs)

    def step(self, action):
        """Applique une action, retourne (obs, reward, done, info)."""
        action_int = int(action)
        obs, reward, done, info = self._env.step(action_int)
        self._episode_step += 1

        # DIAGNOSTIC : max d'inventaire vu + histogramme des actions tentées
        if 0 <= action_int < ACTION_DIM:
            self._action_counts[action_int] += 1
        for item, qty in (info.get("inventory") or {}).items():
            if qty > self._inv_max.get(item, 0):
                self._inv_max[item] = int(qty)

        # Tracker les achievements DÉCROCHÉS pendant cet episode
        achievements = info.get("achievements", {})
        for name, val in achievements.items():
            if val > 0 and name not in self._unlocked_this_episode:
                self._unlocked_this_episode.add(name)

        # info enrichi pour les stats agent
        info["n_achievements_episode"] = len(self._unlocked_this_episode)
        info["invalid"] = False    # Crafter n'a pas d'action invalide

        return self._normalize_obs(obs), float(reward), bool(done), info

    # ============================================================== action mask

    def get_action_mask(self) -> np.ndarray:
        """
        Mask des actions valides. Pour Crafter, toutes sont toujours valides
        (l'agent doit apprendre laquelle est utile selon le contexte).
        """
        return self._cached_mask

    # ============================================================== observation

    @staticmethod
    def _normalize_obs(obs):
        """
        Convertit obs Crafter (H, W, C) uint8 [0, 255] en (C, H, W) float32 [0, 1].
        Format adapté pour CNN PyTorch (channel-first).
        """
        return obs.transpose(2, 0, 1).astype(np.float32) / 255.0

    # ============================================================== rendering / debug

    def render(self):
        """Retourne l'obs courante en uint8 (H, W, C) pour affichage."""
        return self._env.render()

    @property
    def n_unlocked_episode(self):
        """Nombre d'achievements débloqués sur l'episode courant."""
        return len(self._unlocked_this_episode)

    @property
    def unlocked_names(self):
        """Noms des achievements débloqués sur l'episode courant (copie)."""
        return set(self._unlocked_this_episode)

    @property
    def inv_max(self):
        """Max d'inventaire atteint dans l'épisode courant (dict item → qty)."""
        return dict(self._inv_max)

    @property
    def action_counts(self):
        """Histogramme des actions tentées dans l'épisode courant (copie)."""
        return self._action_counts.copy()


__all__ = [
    "CrafterEnv",
    "OBS_SHAPE", "OBS_DIM", "ACTION_DIM",
    "ACHIEVEMENTS", "ACTION_NAMES", "N_ACHIEVEMENTS",
]

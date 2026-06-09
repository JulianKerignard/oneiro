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

# Anti sleep/noop spam (corrige le mode collapse "agent dort en boucle")
NOOP_ACTION_IDX = 0
SLEEP_ACTION_IDX = 6
SPAM_THRESHOLD = 3          # gardé pour compat mais ineffectif si SPAM_PENALTY = 0
SPAM_PENALTY = 0.0          # DÉSACTIVÉ pour v14 : interférait avec hyperparams paper
                             # Le sleep spam sera géré par adaptive_alpha + auto_explore à la place


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
        self._env = crafter.Env(seed=seed, length=length)
        self._episode_step = 0
        self._unlocked_this_episode = set()
        self._cached_mask = np.ones(ACTION_DIM, dtype=bool)   # toutes valides toujours
        # Tracking anti-spam (sleep/noop)
        self._consecutive_sleep = 0
        self._consecutive_noop = 0

    # ============================================================== gym API

    def reset(self, seed=None):
        """Reset l'env, retourne obs (3, 64, 64) float32."""
        if seed is not None:
            # Crafter ne supporte pas seed dynamique : recréer l'env
            self._env = crafter.Env(seed=seed)
        obs = self._env.reset()
        self._episode_step = 0
        self._unlocked_this_episode = set()
        self._consecutive_sleep = 0
        self._consecutive_noop = 0
        return self._normalize_obs(obs)

    def step(self, action):
        """Applique une action, retourne (obs, reward, done, info)."""
        action_int = int(action)
        obs, reward, done, info = self._env.step(action_int)
        self._episode_step += 1

        # Tracker les achievements DÉCROCHÉS pendant cet episode
        achievements = info.get("achievements", {})
        for name, val in achievements.items():
            if val > 0 and name not in self._unlocked_this_episode:
                self._unlocked_this_episode.add(name)

        # === Pénalité anti-spam (sleep / noop)
        # L'agent peut spam SLEEP pour rester en vie + récolter wake_up.
        # On pénalise les actions identiques consécutives au-delà de SPAM_THRESHOLD.
        # Le wake_up achievement reste intact (donné par Crafter).
        spam_penalty = 0.0
        if action_int == SLEEP_ACTION_IDX:
            self._consecutive_sleep += 1
            self._consecutive_noop = 0
            if self._consecutive_sleep > SPAM_THRESHOLD:
                spam_penalty = SPAM_PENALTY * (self._consecutive_sleep - SPAM_THRESHOLD)
        elif action_int == NOOP_ACTION_IDX:
            self._consecutive_noop += 1
            self._consecutive_sleep = 0
            if self._consecutive_noop > SPAM_THRESHOLD:
                spam_penalty = SPAM_PENALTY * (self._consecutive_noop - SPAM_THRESHOLD)
        else:
            self._consecutive_sleep = 0
            self._consecutive_noop = 0

        # info enrichi pour les stats agent
        info["n_achievements_episode"] = len(self._unlocked_this_episode)
        info["invalid"] = False    # Crafter n'a pas d'action invalide
        info["spam_penalty"] = spam_penalty

        return self._normalize_obs(obs), float(reward - spam_penalty), bool(done), info

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


__all__ = [
    "CrafterEnv",
    "OBS_SHAPE", "OBS_DIM", "ACTION_DIM",
    "ACHIEVEMENTS", "ACTION_NAMES", "N_ACHIEVEMENTS",
]

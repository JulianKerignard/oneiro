"""
Replay Buffer pour le World Model.

Stocke des transitions (obs, action, reward, next_obs, done) en arrays numpy
parallèles avec FIFO circulaire. Supporte deux modes de sampling :

  - sample()           : transitions individuelles (utile pour debug, eval)
  - sample_sequences() : séquences contiguës de N steps (pour entraîner le RSSM)

Le flag `dones` marque les frontières d'épisode. Une séquence sampled peut
contenir un game over au milieu : c'est le flag qui permet aux composants
downstream (RSSM, critic, imagination) de les gérer correctement.

Format mémoire :
    obs       : float32  (capacity, obs_dim)
    actions   : int32    (capacity,)
    rewards   : float32  (capacity,)
    next_obs  : float32  (capacity, obs_dim)
    dones     : bool     (capacity,)

Pour Tetris (obs_dim=276, capacity=100k), le buffer pèse ~225 MB en RAM.
"""

from pathlib import Path
import numpy as np


class ReplayBuffer:
    """Replay buffer FIFO circulaire avec sampling de séquences."""

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs_dim = obs_dim

        # 5 arrays parallèles
        self.obs       = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions   = np.zeros(capacity, dtype=np.int32)
        self.rewards   = np.zeros(capacity, dtype=np.float32)
        self.next_obs  = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones     = np.zeros(capacity, dtype=bool)

        self.idx = 0     # prochain index d'écriture
        self.size = 0    # nombre d'éléments actuellement stockés

    # ----------------------------------------------------------- write

    def add(self, obs, action, reward, next_obs, done):
        """Ajoute une transition au buffer (FIFO circulaire)."""
        self.obs[self.idx]      = obs
        self.actions[self.idx]  = action
        self.rewards[self.idx]  = reward
        self.next_obs[self.idx] = next_obs
        self.dones[self.idx]    = done

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # ----------------------------------------------------------- sampling

    def sample(self, batch_size: int) -> dict:
        """
        Sample batch_size transitions individuelles aléatoires.

        Utile pour : debug, eval, training d'un agent model-free (DQN, SAC).
        Pour entraîner un WM, utiliser sample_sequences() à la place.

        Returns:
            dict de 5 arrays de shape (batch_size, ...).
        """
        if self.size < batch_size:
            raise ValueError(
                f"Buffer trop petit pour sample : {self.size} < {batch_size}"
            )
        indices = np.random.randint(0, self.size, batch_size)
        return {
            "obs":      self.obs[indices],
            "actions":  self.actions[indices],
            "rewards":  self.rewards[indices],
            "next_obs": self.next_obs[indices],
            "dones":    self.dones[indices],
        }

    def sample_sequences(self, batch_size: int, seq_len: int) -> dict:
        """
        Sample batch_size séquences contiguës de seq_len steps consécutifs.

        Le sampling tolère les boundaries d'épisode au milieu d'une séquence.
        Le flag `dones` permet aux composants downstream de les gérer
        (reset RSSM, cut return, stop imagination).

        Returns:
            dict avec shapes :
                obs       : (B, T, obs_dim)
                actions   : (B, T)
                rewards   : (B, T)
                next_obs  : (B, T, obs_dim)
                dones     : (B, T)
        """
        if self.size < seq_len:
            raise ValueError(
                f"Buffer trop petit pour sample_sequences : {self.size} < {seq_len}"
            )

        # Indices de départ valides : [0, size - seq_len]
        max_start = self.size - seq_len
        starts = np.random.randint(0, max_start + 1, batch_size)

        # Construit la matrice d'indices (B, T) via broadcasting
        indices = starts[:, None] + np.arange(seq_len)[None, :]

        return {
            "obs":      self.obs[indices],
            "actions":  self.actions[indices],
            "rewards":  self.rewards[indices],
            "next_obs": self.next_obs[indices],
            "dones":    self.dones[indices],
        }

    # ----------------------------------------------------------- info

    def __len__(self):
        return self.size

    def is_full(self):
        return self.size == self.capacity

    def memory_usage_mb(self):
        """Mémoire utilisée par le buffer en MB."""
        bytes_total = (
            self.obs.nbytes
            + self.actions.nbytes
            + self.rewards.nbytes
            + self.next_obs.nbytes
            + self.dones.nbytes
        )
        return bytes_total / 1e6

    # ----------------------------------------------------------- save / load

    def save(self, path):
        """Sauvegarde le buffer compressé dans un fichier .npz."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            obs      = self.obs[:self.size],
            actions  = self.actions[:self.size],
            rewards  = self.rewards[:self.size],
            next_obs = self.next_obs[:self.size],
            dones    = self.dones[:self.size],
            idx      = self.idx,
            size     = self.size,
            capacity = self.capacity,
            obs_dim  = self.obs_dim,
        )

    def load(self, path):
        """Charge le buffer depuis un fichier .npz."""
        data = np.load(path)

        if int(data["obs_dim"]) != self.obs_dim:
            raise ValueError(
                f"obs_dim mismatch : file={int(data['obs_dim'])}, buffer={self.obs_dim}"
            )
        if int(data["capacity"]) > self.capacity:
            raise ValueError(
                f"capacity du fichier ({int(data['capacity'])}) "
                f"> capacity du buffer ({self.capacity})"
            )

        loaded_size = int(data["size"])
        self.obs[:loaded_size]      = data["obs"]
        self.actions[:loaded_size]  = data["actions"]
        self.rewards[:loaded_size]  = data["rewards"]
        self.next_obs[:loaded_size] = data["next_obs"]
        self.dones[:loaded_size]    = data["dones"]

        self.idx = int(data["idx"])
        self.size = loaded_size


class ImageReplayBuffer:
    """
    Replay buffer pour observations IMAGE (Crafter, Minecraft).

    Stocke obs en uint8 (×4 moins de mémoire que float32) et convertit
    en float32 [0, 1] au moment du sample.

    Format mémoire :
        obs      : uint8   (capacity, C, H, W)
        actions  : int32   (capacity,)
        rewards  : float32 (capacity,)
        next_obs : uint8   (capacity, C, H, W)
        dones    : bool    (capacity,)

    Pour Crafter (3×64×64, capacity=50k) : ~1.2 GB (vs 4.8 GB en float32).
    """

    def __init__(self, capacity: int, obs_shape: tuple):
        self.capacity = capacity
        self.obs_shape = tuple(obs_shape)   # ex: (3, 64, 64)

        self.obs       = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions   = np.zeros(capacity, dtype=np.int32)
        self.rewards   = np.zeros(capacity, dtype=np.float32)
        self.next_obs  = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.dones     = np.zeros(capacity, dtype=bool)

        self.idx = 0
        self.size = 0

    def _to_uint8(self, obs):
        """Convertit float32 [0,1] → uint8 [0,255]."""
        if obs.dtype == np.uint8:
            return obs
        return (np.clip(obs, 0.0, 1.0) * 255.0).astype(np.uint8)

    def _to_float32(self, obs_uint8):
        """Convertit uint8 [0,255] → float32 [0,1]."""
        return obs_uint8.astype(np.float32) / 255.0

    def add(self, obs, action, reward, next_obs, done):
        """Ajoute une transition (FIFO circulaire). obs/next_obs en float32 [0,1] ou uint8."""
        self.obs[self.idx]      = self._to_uint8(obs)
        self.actions[self.idx]  = action
        self.rewards[self.idx]  = reward
        self.next_obs[self.idx] = self._to_uint8(next_obs)
        self.dones[self.idx]    = done

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        """Sample batch_size transitions individuelles. obs converti en float32."""
        if self.size < batch_size:
            raise ValueError(f"Buffer trop petit : {self.size} < {batch_size}")
        indices = np.random.randint(0, self.size, batch_size)
        return {
            "obs":      self._to_float32(self.obs[indices]),
            "actions":  self.actions[indices],
            "rewards":  self.rewards[indices],
            "next_obs": self._to_float32(self.next_obs[indices]),
            "dones":    self.dones[indices],
        }

    def sample_sequences(self, batch_size: int, seq_len: int) -> dict:
        """Sample batch_size séquences contiguës. obs converti en float32."""
        if self.size < seq_len:
            raise ValueError(f"Buffer trop petit pour seq : {self.size} < {seq_len}")
        max_start = self.size - seq_len
        starts = np.random.randint(0, max_start + 1, batch_size)
        indices = starts[:, None] + np.arange(seq_len)[None, :]
        return {
            "obs":      self._to_float32(self.obs[indices]),
            "actions":  self.actions[indices],
            "rewards":  self.rewards[indices],
            "next_obs": self._to_float32(self.next_obs[indices]),
            "dones":    self.dones[indices],
        }

    def __len__(self):
        return self.size

    def is_full(self):
        return self.size == self.capacity

    def memory_usage_mb(self):
        total = (
            self.obs.nbytes + self.actions.nbytes + self.rewards.nbytes
            + self.next_obs.nbytes + self.dones.nbytes
        )
        return total / 1e6

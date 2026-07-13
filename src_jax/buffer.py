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
import jax
import jax.numpy as jnp


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


# ============================================================================
# ImageReplayBufferJAX : buffer GPU-resident pour fix le bottleneck sample_batch
# ============================================================================
#
# Profiling Modal L4 a montré que sample_batch = 36% du temps total à cause de :
#   1. Buffer numpy CPU → cast float32 + normalize côté CPU (cher)
#   2. device_put de (B, T, 3, 64, 64) float32 à chaque sample (~1.5 MB transfer)
#
# Cette version garde le buffer en uint8 sur GPU. Le sample fait gather +
# cast/normalize en un seul jit kernel. Un staging numpy permet de batcher les
# add() durant la collecte (sinon 1 device_put + at[].set() par step = sync GPU).


from functools import partial


@partial(jax.jit, static_argnames=("seq_len",))
def _gather_sequences_impl(obs_buf, actions_buf, rewards_buf, dones_buf,
                           env_idx, starts, seq_len):
    """
    Pure function jit-compilée : gather + cast/normalize uint8 → float32.
    Tout reste sur GPU (pas de host roundtrip).

    Buffers en layout per-env (E, P, ...) : une séquence = un seul env
    (FIX interleaving : avant, le stockage plat entrelaçait les n_envs et
    chaque "séquence contiguë" changeait d'env à chaque step — le RSSM
    apprenait une dynamique inter-envs fictive).

    `seq_len` est static (re-trace si change) car utilisé par jnp.arange.
    En pratique SEQ_LEN est fixé pour tout le training donc une seule compile.
    """
    # (B, T) positions temporelles dans l'env choisi
    t_indices = starts[:, None] + jnp.arange(seq_len)[None, :]
    e_indices = env_idx[:, None]  # (B, 1) broadcast sur T

    obs_u8  = obs_buf[e_indices, t_indices]      # (B, T, C, H, W) uint8
    actions = actions_buf[e_indices, t_indices]  # (B, T) int32
    rewards = rewards_buf[e_indices, t_indices]  # (B, T) float32
    dones   = dones_buf[e_indices, t_indices]    # (B, T) float32

    # Cast + normalize côté GPU (cheap, fused dans le kernel XLA)
    obs = obs_u8.astype(jnp.float32) / 255.0

    return {
        "obs":     obs,
        "actions": actions,
        "rewards": rewards,
        "dones":   dones,
    }


class ImageReplayBufferJAX:
    """
    Replay buffer JAX-resident sur GPU pour observations IMAGE, per-env.

    FIX INTERLEAVING (audit v16) : l'ancienne version stockait les n_envs
    dans un tableau plat dans l'ordre de collecte (env0_t, env1_t, ...,
    env15_t, env0_t+1, ...). Les "séquences contiguës" données au RSSM
    changeaient donc d'environnement À CHAQUE STEP — une dynamique fictive.
    Cette version stocke par env : (n_envs, per_env_cap, ...) ; une séquence
    samplée vit entièrement dans UN env = vraie trajectoire temporelle
    (les frontières d'épisode restent marquées par `dones`).

    Le buffer vit sur le device (uint8). Le sample fait un seul gather
    jit-compilé qui retourne des jax.Array float32 [0,1] sur GPU — aucun
    transfer host↔device dans la hot loop.

    L'add() utilise un staging numpy pour batcher les push (sans staging,
    chaque add ferait 1 device_put + at[].set() = sync GPU).

    /!\\ flush() est appelé automatiquement avant chaque sample_sequences().

    API :
      - add(obs, action, reward, next_obs, done, env_id) : env_id REQUIS
      - sample_sequences(key, batch_size, seq_len) : prend une PRNGKey,
        retourne des jax.Array sur GPU (PAS de device_put côté caller)

    Caveat wrap-around (hérité de l'ancienne version, toléré) : quand un
    buffer per-env est plein, une séquence peut chevaucher le point
    d'écriture (discontinuité ancienne→récente sans done). Avec 500k/16 =
    31 250 steps/env, ça n'arrive qu'après ~15k iter.
    """

    def __init__(self, capacity: int, obs_shape: tuple, n_envs: int = 1,
                 staging_capacity: int = 256, data_sharding=None):
        self.n_envs = int(n_envs)
        # capacity = budget TOTAL, réparti également entre les envs
        self.per_env_cap = capacity // self.n_envs
        self.capacity = self.per_env_cap * self.n_envs
        self.obs_shape = tuple(obs_shape)  # ex: (3, 64, 64)

        # Data-parallel : sharding du batch samplé sur l'axe `data` (dim 0 = B).
        # None → comportement identique à avant (placement par défaut). Réglé
        # depuis main() une fois le mesh créé (1 device → sharding trivial).
        self.data_sharding = data_sharding

        E, P = self.n_envs, self.per_env_cap
        # Buffers GPU layout per-env (uint8 pour obs, float32 pour le reste)
        self.obs = jax.device_put(
            np.zeros((E, P, *self.obs_shape), dtype=np.uint8))
        self.actions = jax.device_put(np.zeros((E, P), dtype=np.int32))
        self.rewards = jax.device_put(np.zeros((E, P), dtype=np.float32))
        self.dones = jax.device_put(np.zeros((E, P), dtype=np.float32))

        # Tracking côté Python, par env
        self.size_env = np.zeros(E, dtype=np.int64)
        self.ptr_env = np.zeros(E, dtype=np.int64)

        # Staging numpy pour batcher les add() (avec env_id par transition)
        self._staging_capacity = int(staging_capacity)
        self._staging_size = 0
        self._staging_obs = np.zeros(
            (self._staging_capacity, *self.obs_shape), dtype=np.uint8)
        self._staging_actions = np.zeros(
            (self._staging_capacity,), dtype=np.int32)
        self._staging_rewards = np.zeros(
            (self._staging_capacity,), dtype=np.float32)
        self._staging_dones = np.zeros(
            (self._staging_capacity,), dtype=np.float32)
        self._staging_env_ids = np.zeros(
            (self._staging_capacity,), dtype=np.int64)

    # --------------------------------------------------------------- write

    def _to_uint8(self, obs):
        """Convertit float32 [0,1] → uint8 [0,255] si nécessaire."""
        if obs.dtype == np.uint8:
            return obs
        return (np.clip(obs, 0.0, 1.0) * 255.0).astype(np.uint8)

    def add(self, obs, action, reward, next_obs, done, env_id: int = 0):
        """
        Add une transition de l'env `env_id`. Staging numpy, flush auto.

        Note : on ne stocke PAS next_obs séparément — au moment du sample,
        next_obs[t] = obs[t+1] dans la séquence (le RSSM s'en sert via le
        slicing T-1). Cf. train_dreamer_jax.py qui n'utilise jamais
        batch["next_obs"] directement.
        """
        i = self._staging_size
        self._staging_obs[i] = self._to_uint8(obs)
        self._staging_actions[i] = action
        self._staging_rewards[i] = reward
        self._staging_dones[i] = float(bool(done))
        self._staging_env_ids[i] = int(env_id)
        self._staging_size += 1

        if self._staging_size >= self._staging_capacity:
            self._flush_staging()

    def _flush_staging(self):
        """Scatter le staging vers les buffers GPU per-env via at[].set()."""
        if self._staging_size == 0:
            return

        n = self._staging_size
        env_ids = self._staging_env_ids[:n]
        # Position destination par env : chaque env avance son propre ptr.
        # L'ordre chronologique par env est préservé (la collecte est
        # séquentielle, le staging conserve l'ordre d'arrivée).
        positions = np.empty(n, dtype=np.int64)
        for e in np.unique(env_ids):
            mask = env_ids == e
            k = int(mask.sum())
            positions[mask] = (self.ptr_env[e] + np.arange(k)) % self.per_env_cap
            self.ptr_env[e] = (self.ptr_env[e] + k) % self.per_env_cap
            self.size_env[e] = min(self.size_env[e] + k, self.per_env_cap)

        e_idx = jnp.asarray(env_ids)
        p_idx = jnp.asarray(positions)

        # at[(e, p)].set() : scatter 2D. XLA gère in-place quand il peut.
        self.obs = self.obs.at[e_idx, p_idx].set(
            jnp.asarray(self._staging_obs[:n]))
        self.actions = self.actions.at[e_idx, p_idx].set(
            jnp.asarray(self._staging_actions[:n]))
        self.rewards = self.rewards.at[e_idx, p_idx].set(
            jnp.asarray(self._staging_rewards[:n]))
        self.dones = self.dones.at[e_idx, p_idx].set(
            jnp.asarray(self._staging_dones[:n]))

        self._staging_size = 0

    def flush(self):
        """API publique pour forcer un flush (appelé auto avant sample)."""
        self._flush_staging()

    # --------------------------------------------------------------- sample

    def sample_sequences(self, key, batch_size: int, seq_len: int,
                         priority_frac: float = 0.0) -> dict:
        """
        Sample batch_size séquences de seq_len steps consécutifs, chacune
        entièrement dans UN env (vraie trajectoire temporelle).

        Args:
            key : jax.random.PRNGKey pour le sampling (env, start).
            batch_size : nombre de séquences.
            seq_len : longueur de chaque séquence.
            priority_frac : accepté pour compat d'API avec le buffer CPU ;
                le prioritized replay n'est PAS implémenté côté GPU (le chemin
                TPU/production utilise ImageReplayBufferCPU). Ignoré ici.

        Returns:
            dict avec jax.Array sur GPU :
              obs     : (B, T, C, H, W) float32 [0,1]
              actions : (B, T) int32
              rewards : (B, T) float32
              dones   : (B, T) float32

        /!\\ Le caller ne doit PAS faire jax.device_put() : la batch est déjà
            sur GPU. Et ne doit pas re-cast les dtypes.
        """
        # Flush staging avant sample (sinon transitions récentes manquent)
        self.flush()

        # La collecte est symétrique entre envs → on sample uniformément
        # un env puis un start dans [0, min_size - seq_len].
        min_size = int(self.size_env.min())
        if min_size < seq_len:
            raise ValueError(
                f"Buffer trop petit pour seq : min(size_env)={min_size} < {seq_len}")

        max_start = min_size - seq_len
        k_env, k_start = jax.random.split(key)
        env_idx = jax.random.randint(k_env, (batch_size,), 0, self.n_envs)
        starts = jax.random.randint(k_start, (batch_size,), 0, max_start + 1)

        # Gather + cast/normalize en un seul kernel jit
        batch = _gather_sequences_impl(
            self.obs, self.actions, self.rewards, self.dones,
            env_idx, starts, seq_len,
        )
        # Data-parallel : shard la dim 0 (B) sur l'axe `data`. Sur 1 device le
        # sharding est trivial (no-op fonctionnel). Sur N devices, XLA propage
        # le split dans les train steps jit (GSPMD).
        if self.data_sharding is not None:
            batch = jax.device_put(batch, self.data_sharding)
        return batch

    # --------------------------------------------------------------- info

    def __len__(self):
        return int(self.size_env.sum()) + self._staging_size

    def is_full(self):
        return bool((self.size_env == self.per_env_cap).all())

    def memory_usage_mb(self):
        """VRAM utilisée par les buffers GPU (uint8 + float32)."""
        # Tailles théoriques (ne compte pas le staging numpy CPU).
        obs_bytes = self.capacity * int(np.prod(self.obs_shape))  # uint8
        actions_bytes = self.capacity * 4   # int32
        rewards_bytes = self.capacity * 4   # float32
        dones_bytes = self.capacity * 4     # float32
        return (obs_bytes + actions_bytes + rewards_bytes + dones_bytes) / 1e6


# ============================================================================
# ImageReplayBufferCPU : buffer RAM-resident (host) — pour GPU à faible VRAM
# ============================================================================
#
# Variante de ImageReplayBufferJAX pour les GPU dont la VRAM ne tient pas le
# buffer 1M (~12.3GB), typiquement RTX 3080 (10GB). Le buffer vit en RAM
# système (numpy) ; seul le BATCH samplé (~12.6MB) est transféré sur GPU.
#
# Trade-off vs la version GPU-resident :
#   - libère ~12.3GB de VRAM (le buffer 1M tient alors dans 32GB de RAM)
#   - re-paie un transfert host->device par sample (la latence qu'on avait
#     éliminée). Impact mesuré attendu : -20 à -40% d'ips.
#
# Même API et même layout per-env (E, P, ...) que la version JAX (fix H_308 :
# une séquence samplée vit dans UN seul env). Pas de staging : en RAM on écrit
# directement (le staging GPU servait à batcher les transferts device).


class ImageReplayBufferCPU:
    """Replay buffer per-env stocké en RAM (host). API miroir de ImageReplayBufferJAX."""

    def __init__(self, capacity: int, obs_shape: tuple, n_envs: int = 1,
                 data_sharding=None, **_):
        self.n_envs = int(n_envs)
        self.per_env_cap = capacity // self.n_envs
        self.capacity = self.per_env_cap * self.n_envs
        self.obs_shape = tuple(obs_shape)

        # Data-parallel : sharding du batch samplé sur l'axe `data` (dim 0 = B).
        # C'est le chemin réel sur TPU (--buffer_device cpu). None → placement
        # par défaut (identique à avant le data-parallel).
        self.data_sharding = data_sharding

        E, P = self.n_envs, self.per_env_cap
        # Buffers RAM (numpy). obs en uint8 (×4 moins que float32).
        self.obs = np.zeros((E, P, *self.obs_shape), dtype=np.uint8)
        self.actions = np.zeros((E, P), dtype=np.int32)
        self.rewards = np.zeros((E, P), dtype=np.float32)
        self.dones = np.zeros((E, P), dtype=np.float32)

        self.size_env = np.zeros(E, dtype=np.int64)
        self.ptr_env = np.zeros(E, dtype=np.int64)

    def _to_uint8(self, obs):
        if obs.dtype == np.uint8:
            return obs
        return (np.clip(obs, 0.0, 1.0) * 255.0).astype(np.uint8)

    def add(self, obs, action, reward, next_obs, done, env_id: int = 0):
        """Écrit une transition de l'env `env_id` directement en RAM (FIFO per-env)."""
        e = int(env_id)
        p = int(self.ptr_env[e])
        self.obs[e, p] = self._to_uint8(obs)
        self.actions[e, p] = action
        self.rewards[e, p] = reward
        self.dones[e, p] = float(bool(done))
        self.ptr_env[e] = (p + 1) % self.per_env_cap
        self.size_env[e] = min(self.size_env[e] + 1, self.per_env_cap)

    def flush(self):
        """No-op (pas de staging GPU en mode CPU) — présent pour la compat API."""
        pass

    def sample_sequences(self, key, batch_size: int, seq_len: int,
                         priority_frac: float = 0.0) -> dict:
        """
        Sample batch_size séquences mono-env, gather en RAM puis transfert GPU.

        Le sampling des indices utilise la PRNGKey jax (reproductible, aligné
        sur la version GPU) ; le gather est numpy, et seul le batch est
        device_put (uint8 → cast float32 sur GPU).

        priority_frac : fraction du batch (0..1) forcée à contenir un ACHIEVEMENT
            (reward > 0.5) dans sa fenêtre. Prioritized replay par récompense —
            destiné au batch AC pour ré-ancrer l'imagination sur des états actifs
            (près d'un +1) au lieu de longues séquences plates. NB : sur-échantillonne
            TOUS les achievements confondus (le buffer ne stocke que reward=+1, pas
            QUEL achievement) → les fréquents (wake_up ~95%) dominent, les rares
            (place_table ~0.7%) sont boostés mais restent minoritaires.
        """
        min_size = int(self.size_env.min())
        if min_size < seq_len:
            raise ValueError(
                f"Buffer trop petit pour seq : min(size_env)={min_size} < {seq_len}")

        max_start = min_size - seq_len
        k_env, k_start, k_prio = jax.random.split(key, 3)
        env_idx = np.asarray(jax.random.randint(k_env, (batch_size,), 0, self.n_envs))
        starts = np.asarray(jax.random.randint(k_start, (batch_size,), 0, max_start + 1))

        # Prioritized replay (reward > 0.5) : remplace les n_prio premières
        # séquences par des fenêtres GARANTIES contenir un achievement.
        n_prio = int(round(batch_size * float(priority_frac)))
        if n_prio > 0:
            seed = int(jax.random.randint(k_prio, (), 0, 2_000_000_000))
            rng = np.random.default_rng(seed)
            # Positions (env, t) à reward > 0.5 dans la zone valide [0, min_size).
            nz_e, nz_t = np.nonzero(self.rewards[:, :min_size] > 0.5)
            if nz_e.size > 0:
                env_idx = np.array(env_idx)  # copie writable (np.asarray(jax) est read-only)
                starts = np.array(starts)
                pick = rng.integers(0, nz_e.size, size=n_prio)
                pe, pt = nz_e[pick], nz_t[pick]
                # start tel que la fenêtre [start, start+seq_len) couvre pt
                lo = np.maximum(0, pt - seq_len + 1)
                hi = np.minimum(pt, max_start)
                span = (hi - lo + 1).astype(np.float64)
                ps = (lo + rng.random(n_prio) * span).astype(np.int64)
                ps = np.clip(ps, 0, max_start)
                env_idx[:n_prio] = pe
                starts[:n_prio] = ps

        # Gather numpy (B, T) dans l'env choisi
        t_idx = starts[:, None] + np.arange(seq_len)[None, :]   # (B, T)
        e_idx = env_idx[:, None]                                # (B, 1)
        obs_u8 = self.obs[e_idx, t_idx]                         # (B, T, C, H, W) uint8
        actions = self.actions[e_idx, t_idx]
        rewards = self.rewards[e_idx, t_idx]
        dones = self.dones[e_idx, t_idx]

        # Transfert host->device : uint8 (4× plus léger que float32), cast sur GPU
        batch = {
            "obs": jnp.asarray(obs_u8, dtype=jnp.float32) / 255.0,
            "actions": jnp.asarray(actions),
            "rewards": jnp.asarray(rewards),
            "dones": jnp.asarray(dones),
        }
        # Data-parallel : shard la dim 0 (B) sur l'axe `data`. Sur 1 device le
        # sharding est trivial (no-op fonctionnel). Sur N devices, le batch est
        # splitté en N morceaux et XLA propage le DP dans les train steps jit.
        if self.data_sharding is not None:
            batch = jax.device_put(batch, self.data_sharding)
        return batch

    def __len__(self):
        return int(self.size_env.sum())

    def is_full(self):
        return bool((self.size_env == self.per_env_cap).all())

    def memory_usage_mb(self):
        """RAM (host) utilisée par les buffers."""
        obs_bytes = self.capacity * int(np.prod(self.obs_shape))  # uint8
        return (obs_bytes + self.capacity * 12) / 1e6  # +int32+2×float32

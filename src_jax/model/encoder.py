"""
Encoder pour le World Model — version JAX/Flax NNX.

Deux variantes :
    - Encoder    : MLP, pour observations vectorielles (Tetris, ~276 dim)
    - CNNEncoder : Conv2d, pour images (Crafter 64×64×3, Minecraft à venir)

Style DreamerV3, encoder DÉTERMINISTE. La stochasticité (sampling z) est gérée
par le RSSM en aval, pas ici.

Convention JAX : NHWC (vs NCHW PyTorch).
CNNEncoder accepte (B, C, H, W) ou (B, T, C, H, W) en entrée et transpose en interne.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class Encoder(nnx.Module):
    """MLP encoder : obs vectorielle → embedding (Tetris)."""

    def __init__(
        self,
        obs_dim: int = 276,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        *,
        rngs: nnx.Rngs,
    ):
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim

        self.linear1 = nnx.Linear(obs_dim, hidden_dim, rngs=rngs)
        self.norm1 = nnx.LayerNorm(hidden_dim, epsilon=1e-5, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(hidden_dim, epsilon=1e-5, rngs=rngs)
        self.out = nnx.Linear(hidden_dim, embed_dim, rngs=rngs)

    def __call__(self, obs: jax.Array) -> jax.Array:
        """
        Args:
            obs : (..., obs_dim)

        Returns:
            embedding : (..., embed_dim)
        """
        x = jax.nn.silu(self.norm1(self.linear1(obs)))
        x = jax.nn.silu(self.norm2(self.linear2(x)))
        return self.out(x)


class CNNEncoder(nnx.Module):
    """
    CNN encoder pour images RGB style DreamerV3 (Crafter 64×64×3).

    Architecture : 4 conv layers (downsample ×16) + linear vers embed_dim.
    Input  : (B, C, H, W) ou (B, T, C, H, W) — NCHW, float32 in [0, 1]
    Output : (..., embed_dim)

    Note JAX : les convolutions opèrent en NHWC en interne.
    L'input NCHW est transposé en entrée, jamais exposé à l'appelant.

    Params : ~1.2M (avec base_channels=32, embed_dim=128).
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 128,
        base_channels: int = 32,
        input_resolution: int = 64,
        *,
        rngs: nnx.Rngs,
    ):
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.input_resolution = input_resolution

        c = base_channels
        # Flax NNX Conv : (in_features, out_features, kernel_size, ...)
        # Opère sur NHWC en interne.
        # Architecture downsampling : 64 → 32 → 16 → 8 → 4
        self.conv1 = nnx.Conv(
            in_features=in_channels, out_features=c,
            kernel_size=(4, 4), strides=(2, 2), padding="SAME", rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            in_features=c, out_features=c * 2,
            kernel_size=(4, 4), strides=(2, 2), padding="SAME", rngs=rngs,
        )
        self.conv3 = nnx.Conv(
            in_features=c * 2, out_features=c * 4,
            kernel_size=(4, 4), strides=(2, 2), padding="SAME", rngs=rngs,
        )
        self.conv4 = nnx.Conv(
            in_features=c * 4, out_features=c * 8,
            kernel_size=(4, 4), strides=(2, 2), padding="SAME", rngs=rngs,
        )

        # Avec 4 strides /2 et SAME padding :
        # 64 → 32 → 16 → 8 → 4
        final_res = input_resolution // 16   # 4
        self.final_channels = c * 8          # 256
        flat_dim = self.final_channels * final_res * final_res  # 4096

        self.proj_linear = nnx.Linear(flat_dim, embed_dim, rngs=rngs)
        self.proj_norm = nnx.LayerNorm(embed_dim, epsilon=1e-5, rngs=rngs)

    def __call__(self, obs: jax.Array) -> jax.Array:
        """
        Args:
            obs : (B, C, H, W) ou (B, T, C, H, W) — NCHW float32 in [0, 1]

        Returns:
            embedding : (B, embed_dim) ou (B, T, embed_dim)
        """
        orig_shape = obs.shape
        if obs.ndim == 5:
            B, T = orig_shape[0], orig_shape[1]
            obs = obs.reshape(B * T, *orig_shape[2:])
            has_time = True
        else:
            has_time = False

        # NCHW → NHWC pour les convolutions JAX
        x = jnp.transpose(obs, (0, 2, 3, 1))   # (BT, H, W, C)

        x = jax.nn.silu(self.conv1(x))
        x = jax.nn.silu(self.conv2(x))
        x = jax.nn.silu(self.conv3(x))
        x = jax.nn.silu(self.conv4(x))

        # FIX parité PyTorch : transpose NHWC → NCHW avant flatten pour matcher
        # l'ordre des features que verra le proj_linear (PyTorch flatten est C-H-W,
        # JAX flatten brut serait H-W-C → ordre différent).
        x = jnp.transpose(x, (0, 3, 1, 2))      # NHWC → NCHW
        x = x.reshape(x.shape[0], -1)           # flatten (BT, flat_dim) en ordre C-H-W
        emb = self.proj_norm(self.proj_linear(x))  # (BT, embed_dim)

        if has_time:
            emb = emb.reshape(B, T, -1)
        return emb

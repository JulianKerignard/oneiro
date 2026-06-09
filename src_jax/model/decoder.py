"""
Decoder pour le World Model — version JAX/Flax NNX.

Deux variantes :
    - Decoder    : MLP, retourne LOGITS pour obs binaire (Tetris, BCEWithLogitsLoss)
    - CNNDecoder : Transposed conv, retourne RECONSTRUCTION pour images RGB
                    (Crafter, MSE loss style DreamerV3)

Convention JAX : NHWC en interne.
CNNDecoder retourne (B, C, H, W) NCHW en sortie pour compatibilité avec le reste
du pipeline (PyTorch convention côté données).
"""

import jax
import jax.numpy as jnp
from flax import nnx


class Decoder(nnx.Module):
    """MLP decoder : embedding → obs (logits binaires)."""

    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        obs_dim: int = 276,
        *,
        rngs: nnx.Rngs,
    ):
        self.embed_dim = embed_dim
        self.obs_dim = obs_dim

        self.linear1 = nnx.Linear(embed_dim, hidden_dim, rngs=rngs)
        self.norm1 = nnx.LayerNorm(hidden_dim, epsilon=1e-5, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(hidden_dim, epsilon=1e-5, rngs=rngs)
        self.out = nnx.Linear(hidden_dim, obs_dim, rngs=rngs)

    def __call__(self, embedding: jax.Array) -> jax.Array:
        """
        Args:
            embedding : (..., embed_dim)

        Returns:
            logits : (..., obs_dim) — LOGITS (pas de sigmoid appliqué).
        """
        x = jax.nn.silu(self.norm1(self.linear1(embedding)))
        x = jax.nn.silu(self.norm2(self.linear2(x)))
        return self.out(x)


class CNNDecoder(nnx.Module):
    """
    CNN decoder pour images RGB, symétrique à CNNEncoder.

    Architecture : linear → reshape → 4 transposed conv (upsample ×16).
    Input  : (B, state_dim) ou (B, T, state_dim)
    Output : (B, C, H, W) ou (B, T, C, H, W) — pixels en [0, 1] (sigmoid à la fin)
             Format NCHW pour compatibilité avec le pipeline de données.

    Note JAX : ConvTranspose opère en NHWC. On transpose NHWC → NCHW à la sortie.

    Params : ~1.2M avec base_channels=32.
    """

    def __init__(
        self,
        state_dim: int,
        out_channels: int = 3,
        base_channels: int = 32,
        output_resolution: int = 64,
        *,
        rngs: nnx.Rngs,
    ):
        self.state_dim = state_dim
        self.out_channels = out_channels
        self.output_resolution = output_resolution

        # On part d'une feature map 4×4 et on upscale ×16 → 64
        initial_res = output_resolution // 16   # 4
        self.initial_res = initial_res
        c = base_channels
        self.initial_channels = c * 8           # 256
        flat_dim = self.initial_channels * initial_res * initial_res  # 4096

        # Linear + LayerNorm pour passer de state à feature map
        self.proj_linear = nnx.Linear(state_dim, flat_dim, rngs=rngs)
        self.proj_norm = nnx.LayerNorm(flat_dim, epsilon=1e-5, rngs=rngs)

        # Flax NNX ConvTranspose : (in_features, out_features, kernel_size, ...)
        # Opère en NHWC. Transposed conv : 4 → 8 → 16 → 32 → 64
        self.deconv1 = nnx.ConvTranspose(
            in_features=c * 8, out_features=c * 4,
            kernel_size=(4, 4), strides=(2, 2), padding="SAME", rngs=rngs,
        )
        self.deconv2 = nnx.ConvTranspose(
            in_features=c * 4, out_features=c * 2,
            kernel_size=(4, 4), strides=(2, 2), padding="SAME", rngs=rngs,
        )
        self.deconv3 = nnx.ConvTranspose(
            in_features=c * 2, out_features=c,
            kernel_size=(4, 4), strides=(2, 2), padding="SAME", rngs=rngs,
        )
        self.deconv4 = nnx.ConvTranspose(
            in_features=c, out_features=out_channels,
            kernel_size=(4, 4), strides=(2, 2), padding="SAME", rngs=rngs,
        )

    def __call__(self, state: jax.Array) -> jax.Array:
        """
        Args:
            state : (B, state_dim) ou (B, T, state_dim)

        Returns:
            recon : (B, C, H, W) ou (B, T, C, H, W) — pixels en [0, 1], NCHW
        """
        orig_shape = state.shape
        if state.ndim == 3:
            B, T = orig_shape[0], orig_shape[1]
            state = state.reshape(B * T, orig_shape[2])
            has_time = True
        else:
            has_time = False

        x = self.proj_norm(self.proj_linear(state))              # (BT, flat_dim)
        # Reshape en feature map NHWC : (BT, H, W, C)
        x = x.reshape(-1, self.initial_res, self.initial_res, self.initial_channels)

        x = jax.nn.silu(self.deconv1(x))                          # (BT, 8, 8, c*4)
        x = jax.nn.silu(self.deconv2(x))                          # (BT, 16, 16, c*2)
        x = jax.nn.silu(self.deconv3(x))                          # (BT, 32, 32, c)
        x = self.deconv4(x)                                       # (BT, 64, 64, out_ch)
        x = jax.nn.sigmoid(x)

        # NHWC → NCHW pour compatibilité pipeline
        x = jnp.transpose(x, (0, 3, 1, 2))                       # (BT, C, H, W)

        if has_time:
            x = x.reshape(B, T, self.out_channels, self.output_resolution, self.output_resolution)
        return x

"""
Encoder pour le World Model.

Deux variantes :
    - Encoder    : MLP, pour observations vectorielles (Tetris, ~276 dim)
    - CNNEncoder : Conv2d, pour images (Crafter 64×64×3, Minecraft à venir)

Style DreamerV3, encoder DÉTERMINISTE. La stochasticité (sampling z) est gérée
par le RSSM en aval, pas ici.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """MLP encoder : obs vectorielle → embedding (Tetris)."""

    def __init__(self, obs_dim: int = 276, hidden_dim: int = 256, embed_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs : (..., obs_dim)

        Returns:
            embedding : (..., embed_dim)
        """
        return self.net(obs)


class CNNEncoder(nn.Module):
    """
    CNN encoder pour images RGB style DreamerV3 (Crafter 64×64×3).

    Architecture : 4 conv layers (downsample ×16) + linear vers embed_dim.
    Input  : (B, 3, 64, 64) ou (B, T, 3, 64, 64) — supports time dim.
    Output : (..., embed_dim)

    Params : ~1.2M (avec base_channels=32, embed_dim=128).
    Scalable : augmenter base_channels pour plus de capacité.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 128,
        base_channels: int = 32,
        input_resolution: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.input_resolution = input_resolution

        # Architecture downsampling : 64 → 32 → 16 → 8 → 4
        c = base_channels
        self.conv = nn.Sequential(
            # 64 → 32
            nn.Conv2d(in_channels, c, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            # 32 → 16
            nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            # 16 → 8
            nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            # 8 → 4
            nn.Conv2d(c * 4, c * 8, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
        )

        # Taille de la feature map finale après les 4 strides /2
        final_res = input_resolution // 16   # 64 / 16 = 4
        self.final_res = final_res
        self.final_channels = c * 8           # 32 * 8 = 256
        flat_dim = self.final_channels * final_res * final_res  # 256 * 4 * 4 = 4096

        self.proj = nn.Sequential(
            nn.Linear(flat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs : (B, C, H, W) ou (B, T, C, H, W) — float32 in [0, 1]

        Returns:
            embedding : (B, embed_dim) ou (B, T, embed_dim)
        """
        # Gestion de la dim temporelle : flatten B et T pour la conv 2D
        orig_shape = obs.shape
        if obs.dim() == 5:
            B, T = orig_shape[0], orig_shape[1]
            obs = obs.reshape(B * T, *orig_shape[2:])
            has_time = True
        else:
            has_time = False

        # Forward CNN
        x = self.conv(obs)                            # (BT, C_final, H', W')
        x = x.reshape(x.shape[0], -1)                 # flatten (BT, C_final*H'*W')
        emb = self.proj(x)                            # (BT, embed_dim)

        # Restore time dim si présente
        if has_time:
            emb = emb.reshape(B, T, -1)
        return emb

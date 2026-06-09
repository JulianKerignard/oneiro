"""
Decoder pour le World Model.

Deux variantes :
    - Decoder    : MLP, retourne LOGITS pour obs binaire (Tetris, BCEWithLogitsLoss)
    - CNNDecoder : Transposed conv, retourne RECONSTRUCTION pour images RGB
                    (Crafter, MSE loss style DreamerV3)
"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """MLP decoder : embedding → obs (logits binaires)."""

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256, obs_dim: int = 276):
        super().__init__()
        self.embed_dim = embed_dim
        self.obs_dim = obs_dim

        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding : (..., embed_dim)

        Returns:
            logits : (..., obs_dim) — LOGITS (pas de sigmoid appliqué).
        """
        return self.net(embedding)


class CNNDecoder(nn.Module):
    """
    CNN decoder pour images RGB, symétrique à CNNEncoder.

    Architecture : linear → reshape → 4 transposed conv (upsample ×16).
    Input  : (B, state_dim) ou (B, T, state_dim)
    Output : (..., 3, 64, 64) — reconstruction RGB en [0, 1] (sigmoid à la fin)

    Loss recommandée : MSE entre output et obs originale (style DreamerV3).

    Params : ~1.2M avec base_channels=32.
    """

    def __init__(
        self,
        state_dim: int,
        out_channels: int = 3,
        base_channels: int = 32,
        output_resolution: int = 64,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.out_channels = out_channels
        self.output_resolution = output_resolution

        # On part d'une feature map 4×4 et on upscale ×16 → 64
        initial_res = output_resolution // 16   # 4
        self.initial_res = initial_res
        c = base_channels
        self.initial_channels = c * 8           # 256
        self.flat_dim = self.initial_channels * initial_res * initial_res  # 4096

        # Linear pour passer de state à feature map
        self.proj = nn.Sequential(
            nn.Linear(state_dim, self.flat_dim),
            nn.LayerNorm(self.flat_dim),
        )

        # Transposed conv : 4 → 8 → 16 → 32 → 64
        self.deconv = nn.Sequential(
            # 4 → 8
            nn.ConvTranspose2d(c * 8, c * 4, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            # 8 → 16
            nn.ConvTranspose2d(c * 4, c * 2, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            # 16 → 32
            nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            # 32 → 64 (sortie)
            nn.ConvTranspose2d(c, out_channels, kernel_size=4, stride=2, padding=1),
            # sigmoid pour rester dans [0, 1] (matche notre wrapper CrafterEnv)
            nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state : (B, state_dim) ou (B, T, state_dim)

        Returns:
            recon : (B, C, H, W) ou (B, T, C, H, W) — pixels en [0, 1]
        """
        orig_shape = state.shape
        if state.dim() == 3:
            B, T = orig_shape[0], orig_shape[1]
            state = state.reshape(B * T, orig_shape[2])
            has_time = True
        else:
            has_time = False

        x = self.proj(state)                                    # (BT, flat_dim)
        x = x.reshape(-1, self.initial_channels, self.initial_res, self.initial_res)
        recon = self.deconv(x)                                  # (BT, C, H, W)

        if has_time:
            recon = recon.reshape(B, T, self.out_channels, self.output_resolution, self.output_resolution)
        return recon

"""
Random Network Distillation (RND) — exploration dirigée par curiosité.

Référence : Burda et al. 2019 "Exploration by Random Network Distillation".

Principe :
    - target_net : poids RANDOM fixés (gelés à l'init)
    - predictor_net : entraîné à imiter target_net sur les obs vues

    bonus(obs) = || predictor(obs) - target(obs) ||²

    - Obs vue souvent → predictor bon → erreur basse → bonus faible
    - Obs nouvelle    → predictor mauvais → erreur haute → bonus

Intuition : "count-based exploration" implicite. Le predictor "compte" via
sa capacité à prédire. Pas besoin de stocker les visites.

Usage :
    rnd = RNDModule(in_channels=3, embed_dim=128, base_channels=32)

    # Pendant collecte : bonus intrinsèque ajouté au reward
    bonus = rnd.compute_bonus(obs)       # (B,) float
    reward_total = env_reward + alpha_rnd × bonus_normalized

    # Pendant WM training : entraîner le predictor
    loss_rnd = rnd.train_loss(obs_batch)
    loss_total = loss_wm + loss_rnd
"""

import torch
import torch.nn as nn
from .encoder import CNNEncoder


class RNDModule(nn.Module):
    """
    Module RND : target gelé + predictor entraînable + running stats.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 128,
        base_channels: int = 32,
        input_resolution: int = 64,
        normalization_momentum: float = 0.99,
    ):
        super().__init__()
        # Target : random fixé pour toujours
        self.target = CNNEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            base_channels=base_channels,
            input_resolution=input_resolution,
        )
        for p in self.target.parameters():
            p.requires_grad = False
        self.target.eval()

        # Predictor : entraînable, init différent de target
        self.predictor = CNNEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            base_channels=base_channels,
            input_resolution=input_resolution,
        )
        # Re-init pour s'assurer qu'il est différent du target
        for m in self.predictor.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Running stats pour normaliser le bonus intrinsèque (stabilité critique pour PG)
        self.register_buffer("running_mean", torch.zeros(1))
        self.register_buffer("running_var", torch.ones(1))
        self.normalization_momentum = normalization_momentum

    def compute_bonus(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Calcule le bonus intrinsèque BRUT (avant normalisation).

        Args:
            obs : (B, C, H, W) ou (B, T, C, H, W) — float32 [0, 1]
        Returns:
            bonus : (B,) ou (B, T) — MSE error par sample
        """
        with torch.no_grad():
            target_emb = self.target(obs)
        pred_emb = self.predictor(obs)
        # MSE par sample (moyenne sur embed_dim)
        bonus = ((pred_emb - target_emb) ** 2).mean(dim=-1)
        return bonus

    def normalize_bonus(self, bonus: torch.Tensor) -> torch.Tensor:
        """
        Normalise le bonus par running std (DreamerV3 style stability).
        Met à jour les running stats avec EMA.
        """
        with torch.no_grad():
            cur_mean = bonus.mean().detach()
            cur_var = bonus.var().detach()
            # EMA update
            self.running_mean.copy_(
                self.normalization_momentum * self.running_mean
                + (1.0 - self.normalization_momentum) * cur_mean
            )
            self.running_var.copy_(
                self.normalization_momentum * self.running_var
                + (1.0 - self.normalization_momentum) * cur_var
            )
        std = torch.sqrt(self.running_var + 1e-8)
        # On normalise par std seulement (garder le signe positif du bonus)
        return bonus / std

    def train_loss(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Loss pour entraîner le predictor.
        = moyenne du bonus brut = MSE(predictor, target).

        Note : le gradient ne passe que dans predictor (target est frozen).
        """
        return self.compute_bonus(obs).mean()

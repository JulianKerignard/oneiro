"""
Critic — value function du World Model (DreamerV3 twohot symlog).

Prend en input l'état du RSSM (h+z, state_dim=384 par défaut) et retourne
des LOGITS sur 255 bins en espace symlog.

Pour avoir la value scalaire : critic.predict(state) (espace original).
Pour la loss au training      : critic.loss(state, returns_target).

Avantages vs MSE simple :
  - Gradient stable même quand les returns ont une grande dynamique
  - Pas d'explosion sur les rewards rares (line clears Tetris)
  - Standard DreamerV3, prouvé robuste sur 150+ envs sans tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import (
    symlog, symexp,
    twohot_encode, twohot_decode,
    N_BINS, BIN_MIN_SYMLOG, BIN_MAX_SYMLOG,
)


class Critic(nn.Module):
    """MLP : state (h+z) → distribution twohot sur 255 bins (espace symlog)."""

    def __init__(self, state_dim: int = 384, hidden_dim: int = 256, n_bins: int = N_BINS):
        super().__init__()
        self.n_bins = n_bins
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, n_bins),
        )
        bins = torch.linspace(BIN_MIN_SYMLOG, BIN_MAX_SYMLOG, n_bins)
        self.register_buffer("bins", bins)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns raw logits over n_bins."""
        return self.net(state)

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """Decode en value scalaire (espace original)."""
        logits = self.forward(state)
        return twohot_decode(logits, self.bins)

    def loss(self, state: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Cross-entropy entre logits prédits et twohot(target)."""
        logits = self.forward(state)
        with torch.no_grad():
            target_twohot = twohot_encode(target, self.bins)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(target_twohot * log_probs).sum(-1).mean()

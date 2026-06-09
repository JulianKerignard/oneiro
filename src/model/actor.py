"""
Actor — la policy du World Model.

Prend en input l'état du RSSM (h+z, state_dim=384 par défaut) et retourne
une distribution Categorical sur les 41 actions Tetris.

Training : policy gradient dans l'imagination du WM.
    loss = -(advantage × log π(a | s)) - entropy_coef × H(π)

L'actor est petit (~165k params) — le gros du compute est dans le WM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """MLP policy : state → Categorical distribution over actions."""

    def __init__(self, state_dim: int = 384, hidden_dim: int = 256, action_dim: int = 41):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state : (..., state_dim)

        Returns:
            logits : (..., action_dim) — raw logits (pas de softmax)
        """
        return self.net(state)

    def get_dist(self, state: torch.Tensor, mask: torch.Tensor = None) -> torch.distributions.Categorical:
        """
        Retourne la distribution Categorical sur les actions.

        Args:
            state : (..., state_dim)
            mask  : (..., action_dim) bool optionnel.
                    Si fourni, les logits des actions invalides sont mis à -inf,
                    leur probabilité devient 0 après softmax.
        """
        logits = self.forward(state)
        if mask is not None:
            # Met les logits invalides à une très grande négatif (softmax → ~0)
            very_neg = torch.tensor(-1e9, device=logits.device, dtype=logits.dtype)
            logits = torch.where(mask, logits, very_neg)
        return torch.distributions.Categorical(logits=logits)

    def act(self, state: torch.Tensor, deterministic: bool = False,
            mask: torch.Tensor = None) -> torch.Tensor:
        """
        Sample une action.

        Args:
            state : (..., state_dim)
            deterministic : si True, retourne argmax (eval). Sinon sample.
            mask  : (..., action_dim) bool optionnel pour masquer les actions invalides.

        Returns:
            action : (...) tensor d'int (action 0-40)
        """
        dist = self.get_dist(state, mask=mask)
        if deterministic:
            return dist.probs.argmax(dim=-1)
        return dist.sample()

    def log_prob_and_entropy(self, state: torch.Tensor, action: torch.Tensor,
                              mask: torch.Tensor = None):
        """
        Utile pour le training :
            log_prob de l'action prise
            entropie de la policy en ce state

        Args:
            state  : (..., state_dim)
            action : (...) int tensor
            mask   : (..., action_dim) bool optionnel
        """
        dist = self.get_dist(state, mask=mask)
        return dist.log_prob(action), dist.entropy()

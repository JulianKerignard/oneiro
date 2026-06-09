"""
Têtes (heads) du World Model.

  - RewardHead   : prédit le reward depuis l'état (h, z) du RSSM
                   Utilise SYMLOG + TWOHOT (DreamerV3 canonique) :
                   output = distribution catégorique sur 255 bins en espace symlog
                   loss = cross-entropy avec twohot target
                   ► robuste aux rewards rares de grande magnitude (line clears)
  - ContinueHead : prédit P(continue) — l'inverse de done. BCE classique.

Toutes deux consomment le state complet [h ; z] du RSSM.

Fonctions helpers :
  - symlog(x), symexp(x)        : compression log symétrique
  - twohot_encode(x, bins)       : encode scalaire → distribution twohot
  - twohot_decode(logits, bins)  : decode logits → scalaire (espérance + symexp)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================== Twohot config

N_BINS = 255                  # nombre de bins (DreamerV3 utilise 255)
BIN_MIN_SYMLOG = -20.0        # min en espace symlog (couvre returns énormes)
BIN_MAX_SYMLOG = 20.0


# ============================== symlog / symexp

def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symétric log : sign(x) * log(1 + |x|). Réduit la dynamique des outliers."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse de symlog : sign(x) * (exp(|x|) - 1). Décompresse."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


# ============================== twohot encoding / decoding

def twohot_encode(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """
    Encode scalaires x en distribution twohot sur les bins (espace symlog).

    Args:
        x    : (...,) scalaires en ESPACE ORIGINAL (pas symlog)
        bins : (n_bins,) en espace symlog (linspace régulier)

    Returns:
        (..., n_bins) two-hot : masse répartie sur les 2 bins voisins de symlog(x).
    """
    n_bins = bins.shape[0]
    x_sym = symlog(x)
    # PERF #1 : utiliser les constantes Python BIN_MIN/MAX_SYMLOG au lieu de bins[0].item()
    # (évite 2 syncs MPS→CPU par appel)
    x_clamped = torch.clamp(x_sym, BIN_MIN_SYMLOG, BIN_MAX_SYMLOG)

    # PERF #2 : calcul direct de l'index (bins est linspace régulier, pas besoin de searchsorted)
    # idx = (x - min) / step  où step = (max - min) / (n_bins - 1)
    bin_step = (BIN_MAX_SYMLOG - BIN_MIN_SYMLOG) / (n_bins - 1)
    idx_below = ((x_clamped - BIN_MIN_SYMLOG) / bin_step).long().clamp(0, n_bins - 2)
    idx_above = idx_below + 1

    bin_low = bins[idx_below]
    bin_high = bins[idx_above]

    # Pondération linéaire entre les 2 bins
    w_high = (x_clamped - bin_low) / (bin_high - bin_low + 1e-8)
    w_high = torch.clamp(w_high, 0.0, 1.0)
    w_low = 1.0 - w_high

    # Two-hot tensor
    twohot = torch.zeros(*x.shape, n_bins, device=x.device, dtype=bins.dtype)
    twohot.scatter_(-1, idx_below.unsqueeze(-1), w_low.unsqueeze(-1))
    twohot.scatter_(-1, idx_above.unsqueeze(-1), w_high.unsqueeze(-1))
    return twohot


def twohot_decode(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """
    Décode logits twohot en scalaire (espace ORIGINAL).

    Args:
        logits : (..., n_bins) raw logits
        bins   : (n_bins,) en espace symlog

    Returns:
        (...,) scalaires (espace original via symexp)
    """
    probs = F.softmax(logits, dim=-1)
    expected_symlog = (probs * bins).sum(-1)
    return symexp(expected_symlog)


# ============================== RewardHead (twohot)

class RewardHead(nn.Module):
    """
    MLP : state (h+z) → distribution twohot sur 255 bins en espace symlog.

    Output ATTENDU : logits bruts (loss = cross-entropy avec twohot target).
    Pour avoir un scalaire (e.g. dans imagination), utiliser .predict(state).
    """

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
        """Decode en scalaire (espace original)."""
        logits = self.forward(state)
        return twohot_decode(logits, self.bins)

    def loss(self, state: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Cross-entropy entre logits prédits et twohot(target)."""
        logits = self.forward(state)
        with torch.no_grad():
            target_twohot = twohot_encode(target, self.bins)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(target_twohot * log_probs).sum(-1).mean()


# ============================== ContinueHead (inchangée, BCE)

class ContinueHead(nn.Module):
    """MLP : state (h+z) → logit pour P(continue)."""

    def __init__(self, state_dim: int = 384, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns logit (sigmoid pas appliqué)."""
        return self.net(state).squeeze(-1)

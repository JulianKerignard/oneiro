"""
Têtes (heads) du World Model — version JAX/Flax NNX.

  - RewardHead   : prédit le reward depuis l'état (h, z) du RSSM
                   Utilise SYMLOG + TWOHOT (DreamerV3 canonique)
  - ContinueHead : prédit P(continue) — l'inverse de done. BCE classique.

Fonctions helpers :
  - symlog(x), symexp(x)        : compression log symétrique
  - twohot_encode(x, bins)       : encode scalaire → distribution twohot
  - twohot_decode(logits, bins)  : decode logits → scalaire (espérance + symexp)
"""

import jax
import jax.numpy as jnp
from flax import nnx


# ============================== Twohot config

N_BINS = 255                  # nombre de bins (DreamerV3 utilise 255)
BIN_MIN_SYMLOG = -20.0        # min en espace symlog (couvre returns énormes)
BIN_MAX_SYMLOG = 20.0


# ============================== symlog / symexp

def symlog(x: jax.Array) -> jax.Array:
    """Symétric log : sign(x) * log(1 + |x|). Réduit la dynamique des outliers."""
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x: jax.Array) -> jax.Array:
    """Inverse de symlog : sign(x) * (exp(|x|) - 1). Décompresse."""
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1.0)


# ============================== twohot encoding / decoding

def twohot_encode(x: jax.Array, bins: jax.Array) -> jax.Array:
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

    # Calcul direct de l'index (bins est linspace régulier, pas besoin de searchsorted)
    x_clamped = jnp.clip(x_sym, BIN_MIN_SYMLOG, BIN_MAX_SYMLOG)
    bin_step = (BIN_MAX_SYMLOG - BIN_MIN_SYMLOG) / (n_bins - 1)
    idx_below = jnp.clip(
        ((x_clamped - BIN_MIN_SYMLOG) / bin_step).astype(jnp.int32),
        0, n_bins - 2,
    )
    idx_above = idx_below + 1

    bin_low = bins[idx_below]
    bin_high = bins[idx_above]

    # Pondération linéaire entre les 2 bins
    w_high = jnp.clip((x_clamped - bin_low) / (bin_high - bin_low + 1e-8), 0.0, 1.0)
    w_low = 1.0 - w_high

    # Two-hot tensor via one_hot (JAX-friendly, pas de scatter_ mutable)
    one_hot_below = jax.nn.one_hot(idx_below, n_bins)   # (..., n_bins)
    one_hot_above = jax.nn.one_hot(idx_above, n_bins)   # (..., n_bins)

    twohot = (
        one_hot_below * w_low[..., None]
        + one_hot_above * w_high[..., None]
    )
    return twohot


def twohot_decode(logits: jax.Array, bins: jax.Array) -> jax.Array:
    """
    Décode logits twohot en scalaire (espace ORIGINAL).

    Args:
        logits : (..., n_bins) raw logits
        bins   : (n_bins,) en espace symlog

    Returns:
        (...,) scalaires (espace original via symexp)
    """
    probs = jax.nn.softmax(logits, axis=-1)
    expected_symlog = (probs * bins).sum(-1)
    return symexp(expected_symlog)


# ============================== RewardHead (twohot)

class RewardHead(nnx.Module):
    """
    MLP : state (h+z) → distribution twohot sur 255 bins en espace symlog.

    Output ATTENDU : logits bruts (loss = cross-entropy avec twohot target).
    Pour avoir un scalaire (e.g. dans imagination), utiliser .predict(state).
    """

    def __init__(
        self,
        state_dim: int = 384,
        hidden_dim: int = 256,
        n_bins: int = N_BINS,
        *,
        rngs: nnx.Rngs,
    ):
        self.n_bins = n_bins

        self.linear1 = nnx.Linear(state_dim, hidden_dim, rngs=rngs)
        self.norm1 = nnx.LayerNorm(hidden_dim, epsilon=1e-5, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(hidden_dim, epsilon=1e-5, rngs=rngs)
        self.out = nnx.Linear(hidden_dim, n_bins, rngs=rngs)

        # Zero init paper DreamerV3 (outscale=0.0) : reward predit ~0 au début,
        # évite bootstrap aléatoire des rewards imaginés.
        self.out.kernel.value = jnp.zeros_like(self.out.kernel.value)

        # Bins en espace symlog — constante non-paramètre
        self.bins = jnp.linspace(BIN_MIN_SYMLOG, BIN_MAX_SYMLOG, n_bins)

    def __call__(self, state: jax.Array) -> jax.Array:
        """Returns raw logits over n_bins."""
        x = jax.nn.silu(self.norm1(self.linear1(state)))
        x = jax.nn.silu(self.norm2(self.linear2(x)))
        return self.out(x)

    def predict(self, state: jax.Array) -> jax.Array:
        """Decode en scalaire (espace original)."""
        logits = self(state)
        return twohot_decode(logits, self.bins)

    def loss(self, state: jax.Array, target: jax.Array) -> jax.Array:
        """
        Cross-entropy entre logits prédits et twohot(target).

        Args:
            state  : (..., state_dim)
            target : (...,) scalaires en espace original

        Returns:
            loss scalaire (moyenne sur le batch)
        """
        logits = self(state)
        target_twohot = jax.lax.stop_gradient(twohot_encode(target, self.bins))
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return -(target_twohot * log_probs).sum(-1).mean()


# ============================== ContinueHead (BCE)

class ContinueHead(nnx.Module):
    """MLP : state (h+z) → logit pour P(continue)."""

    def __init__(
        self,
        state_dim: int = 384,
        hidden_dim: int = 256,
        *,
        rngs: nnx.Rngs,
    ):
        self.linear1 = nnx.Linear(state_dim, hidden_dim, rngs=rngs)
        self.norm1 = nnx.LayerNorm(hidden_dim, epsilon=1e-5, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(hidden_dim, epsilon=1e-5, rngs=rngs)
        self.out = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, state: jax.Array) -> jax.Array:
        """Returns logit (sigmoid pas appliqué). Shape : (...,)"""
        x = jax.nn.silu(self.norm1(self.linear1(state)))
        x = jax.nn.silu(self.norm2(self.linear2(x)))
        return self.out(x).squeeze(-1)

    def loss(self, state: jax.Array, target_continue: jax.Array) -> jax.Array:
        """
        Binary cross-entropy avec logits.

        Args:
            state           : (..., state_dim)
            target_continue : (...,) float32 in {0.0, 1.0} — 1.0 si l'épisode continue

        Returns:
            loss scalaire (moyenne sur le batch)
        """
        logits = self(state)
        # BCE avec logits : -[y * log σ(x) + (1-y) * log(1 - σ(x))]
        # = max(x, 0) - x*y + log(1 + exp(-|x|))
        return jnp.mean(
            jnp.maximum(logits, 0) - logits * target_continue + jnp.log1p(jnp.exp(-jnp.abs(logits)))
        )

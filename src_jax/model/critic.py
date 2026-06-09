"""
Critic — value function du World Model (DreamerV3 twohot symlog) — JAX/Flax NNX.

Prend en input l'état du RSSM (h+z, state_dim=384 par défaut) et retourne
des LOGITS sur 255 bins en espace symlog.

Pour avoir la value scalaire : critic.predict(state) (espace original).
Pour la loss au training      : critic.loss(state, returns_target).

Avantages vs MSE simple :
  - Gradient stable même quand les returns ont une grande dynamique
  - Pas d'explosion sur les rewards rares (line clears Tetris)
  - Standard DreamerV3, prouvé robuste sur 150+ envs sans tuning

Note Flax NNX : self.bins est un tableau JAX simple (non-paramètre, non-tracé
par l'optimiseur). Il sera transporté avec le module mais exclu des gradients.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from .heads import (
    symlog, symexp,
    twohot_encode, twohot_decode,
    N_BINS, BIN_MIN_SYMLOG, BIN_MAX_SYMLOG,
)


class Critic(nnx.Module):
    """MLP : state (h+z) → distribution twohot sur 255 bins (espace symlog)."""

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

        # Zero init paper DreamerV3 (outscale=0.0) : évite bootstrap aléatoire
        # de la value function au début du training.
        self.out.kernel.value = jnp.zeros_like(self.out.kernel.value)

        # Bins en espace symlog — constante non-paramètre.
        # Stocké comme attribut simple (jnp.ndarray), pas via nnx.Param.
        # L'optimiseur NNX filtre sur nnx.Param, donc bins ne sera JAMAIS mis à jour.
        self.bins = jnp.linspace(BIN_MIN_SYMLOG, BIN_MAX_SYMLOG, n_bins)

    def __call__(self, state: jax.Array) -> jax.Array:
        """Returns raw logits over n_bins."""
        x = jax.nn.silu(self.norm1(self.linear1(state)))
        x = jax.nn.silu(self.norm2(self.linear2(x)))
        return self.out(x)

    def predict(self, state: jax.Array) -> jax.Array:
        """Decode en value scalaire (espace original)."""
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

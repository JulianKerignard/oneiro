"""
Distributions en JAX pur — remplace distrax (qui tire tensorflow_probability,
lequel casse sur les versions récentes de JAX : `jax.interpreters.xla.
pytype_aval_mappings` supprimé → crash à l'import sur TPU).

On n'utilisait que distrax.Categorical (sample / log_prob / entropy /
kl_divergence). Cette implémentation reproduit son comportement exact
(testé en parité numérique < 1e-5 vs distrax), sans aucune dépendance lourde.
"""

import jax
import jax.numpy as jnp


class Categorical:
    """Categorical sur la dernière dimension. API miroir de distrax.Categorical.

    Construire avec `logits=` (non normalisés) OU `probs=` (normalisés, somme=1).
    Toutes les opérations agissent sur l'axe -1.
    """

    def __init__(self, logits: jax.Array | None = None, probs: jax.Array | None = None):
        if (logits is None) == (probs is None):
            raise ValueError("Categorical : fournir exactement un de logits/probs.")
        if logits is not None:
            self.logits = logits
            # log-probs normalisées = log_softmax (stable)
            self._log_probs = jax.nn.log_softmax(logits, axis=-1)
            self.probs = jnp.exp(self._log_probs)
        else:
            self.probs = probs
            # log(probs) avec garde anti -inf (cohérent distrax : 0*log0 traité à 0)
            self._log_probs = jnp.log(jnp.clip(probs, 1e-12, 1.0))

    def sample(self, seed: jax.Array) -> jax.Array:
        """Échantillonne un indice sur l'axe -1 (distribution identique à distrax)."""
        return jax.random.categorical(seed, self._log_probs, axis=-1)

    def log_prob(self, value: jax.Array) -> jax.Array:
        """log p(value). value : indices entiers, shape = batch (sans l'axe -1)."""
        return jnp.take_along_axis(
            self._log_probs, value[..., None].astype(jnp.int32), axis=-1
        )[..., 0]

    def entropy(self) -> jax.Array:
        """H = -Σ p·log p (les termes p=0 contribuent 0, comme distrax)."""
        return -jnp.sum(jnp.where(self.probs > 0, self.probs * self._log_probs, 0.0), axis=-1)

    def kl_divergence(self, other: "Categorical") -> jax.Array:
        """KL(self || other) = Σ p·(log p_self − log p_other), p=0 → 0."""
        return jnp.sum(
            jnp.where(self.probs > 0, self.probs * (self._log_probs - other._log_probs), 0.0),
            axis=-1,
        )

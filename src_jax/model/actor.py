"""
Actor — la policy du World Model — version JAX/Flax NNX.

Prend en input l'état du RSSM (h+z, state_dim=960 pour Crafter Palier 2)
et retourne une distribution Categorical sur les actions.

Notes JAX :
    - Pas de PRNG key stockée dans le module (stateless).
    - La key est passée explicitement à sample().
    - Categorical gère le sampling, log_prob et l'entropie.
    - mask (action invalide) : jnp.where(mask, logits, -1e9)

Training : policy gradient dans l'imagination du WM.
    loss = -(advantage × log π(a | s)) - entropy_coef × H(π)
"""

import jax
import jax.numpy as jnp
from flax import nnx
from .distributions import Categorical


class Actor(nnx.Module):
    """MLP policy : state → logits Categorical over actions."""

    def __init__(
        self,
        state_dim: int = 384,
        hidden_dim: int = 256,
        action_dim: int = 41,
        num_layers: int = 2,
        *,
        rngs: nnx.Rngs,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_layers = num_layers

        # Pile de num_layers blocs (Linear → LayerNorm → SiLU), réf dreamerv3-flax
        # (policy.py/mlp.py : MLP profond avant la tête). SiLU = activation projet.
        # nnx.data() : NNX >=0.12 traite un list brut comme statique → on l'enveloppe
        # pour que les sous-modules soient bien tracés/comptés par l'optimiseur.
        linears, norms = [], []
        in_dim = state_dim
        for _ in range(num_layers):
            linears.append(nnx.Linear(in_dim, hidden_dim, rngs=rngs))
            norms.append(nnx.LayerNorm(hidden_dim, epsilon=1e-5, rngs=rngs))
            in_dim = hidden_dim
        self.linears = nnx.data(linears)
        self.norms = nnx.data(norms)
        self.out = nnx.Linear(hidden_dim, action_dim, rngs=rngs)

    def __call__(self, state: jax.Array) -> jax.Array:
        """
        Args:
            state : (..., state_dim)

        Returns:
            logits : (..., action_dim) — raw logits (pas de softmax)
        """
        x = state
        for linear, norm in zip(self.linears, self.norms):
            x = jax.nn.silu(norm(linear(x)))
        return self.out(x)

    def get_dist(
        self,
        state: jax.Array,
        mask: jax.Array | None = None,
        unimix: float = 0.01,
    ) -> Categorical:
        """
        Retourne la distribution Categorical sur les actions.

        Paper DreamerV3 : 1% uniforme (unimix) prévient les KL spikes et
        les distributions déterministes.

        Args:
            state  : (..., state_dim)
            mask   : (..., action_dim) bool optionnel.
                     Si fourni, les logits des actions invalides sont mis à -1e9.
            unimix : proportion d'uniforme à mélanger (0.01 par défaut, paper DreamerV3).

        Returns:
            Categorical distribution
        """
        logits = self(state)
        if mask is not None:
            logits = jnp.where(mask, logits, jnp.full_like(logits, -1e9))

        # Unimix : (1 - unimix) * policy + unimix * uniform
        probs = jax.nn.softmax(logits, axis=-1)
        uniform = jnp.ones_like(probs) / probs.shape[-1]
        probs = (1.0 - unimix) * probs + unimix * uniform

        return Categorical(probs=probs)

    def sample(
        self,
        state: jax.Array,
        rng: jax.Array,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        """
        Sample une action.

        Args:
            state : (..., state_dim)
            rng   : jax.random.PRNGKey
            mask  : (..., action_dim) bool optionnel

        Returns:
            action : (...) int32
        """
        dist = self.get_dist(state, mask=mask)
        return dist.sample(seed=rng)

    def act_deterministic(
        self,
        state: jax.Array,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        """
        Retourne l'action déterministe (argmax — pour eval).

        Args:
            state : (..., state_dim)
            mask  : (..., action_dim) bool optionnel

        Returns:
            action : (...) int32
        """
        logits = self(state)
        if mask is not None:
            logits = jnp.where(mask, logits, jnp.full_like(logits, -1e9))
        return jnp.argmax(logits, axis=-1)

    def log_prob_and_entropy(
        self,
        state: jax.Array,
        action: jax.Array,
        mask: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Utile pour le training :
            log_prob de l'action prise
            entropie de la policy en ce state

        Args:
            state  : (..., state_dim)
            action : (...) int32
            mask   : (..., action_dim) bool optionnel

        Returns:
            (log_prob, entropy) — shapes (...,)
        """
        dist = self.get_dist(state, mask=mask)
        return dist.log_prob(action), dist.entropy()

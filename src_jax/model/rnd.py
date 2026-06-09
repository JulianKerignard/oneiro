"""
Random Network Distillation (RND) — exploration dirigée par curiosité.
Version JAX/Flax NNX.

Référence : Burda et al. 2019 "Exploration by Random Network Distillation".

Principe :
    - target_net : poids RANDOM fixés (gelés à l'init)
    - predictor_net : entraîné à imiter target_net sur les obs vues

    bonus(obs) = || predictor(obs) - target(obs) ||²

    - Obs vue souvent → predictor bon → erreur basse → bonus faible
    - Obs nouvelle    → predictor mauvais → erreur haute → bonus élevé

Particularités JAX vs PyTorch :
    1. Gel du target : pas de requires_grad=False en JAX. On utilise
       jax.lax.stop_gradient() sur la SORTIE du target dans compute_bonus().
       Les params du target restent dans le graphe mais le gradient ne
       remonte pas. L'optimizer sera filtré en Phase 4 pour ne mettre à
       jour que predictor (via nnx.state(rnd.predictor, nnx.Param)).

    2. Running stats mutables : on sous-classe nnx.Variable (RNDStats),
       équivalent de register_buffer PyTorch. Ces variables ne sont pas
       des nnx.Param, donc jamais incluses dans les updates d'optimizer.
       La mutation se fait via .value = ... dans normalize_bonus().

    3. Init predictor : on split la clé de base pour garantir que target
       et predictor ont des poids différents dès le départ.

Usage :
    rnd = RNDModule(in_channels=3, embed_dim=128, base_channels=32,
                    rngs=nnx.Rngs(0))

    # Pendant collecte
    bonus_raw = rnd.compute_bonus(obs)          # (B,)
    bonus_norm = rnd.normalize_bonus(bonus_raw) # (B,) + update EMA

    # Pendant WM training (predictor seulement)
    loss = rnd.train_loss(obs)   # scalaire
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx

from src_jax.model.encoder import CNNEncoder


class RNDStats(nnx.Variable):
    """
    Marker type pour les running statistics mutables (non-trainables).

    En Flax NNX, sous-classer nnx.Variable permet de créer un type
    de variable personnalisé. Les fonctions de filtrage (nnx.state, etc.)
    peuvent alors discriminer RNDStats de nnx.Param, ce qui garantit
    que l'optimizer ne touche jamais ces buffers.

    Équivalent de register_buffer() en PyTorch.
    """
    pass


class RNDModule(nnx.Module):
    """
    Module RND : target gelé + predictor entraînable + running stats.

    Args:
        in_channels           : canaux d'entrée (3 pour RGB)
        embed_dim             : dimension de l'embedding
        base_channels         : nombre de filtres de base du CNN
        input_resolution      : résolution spatiale (H = W = 64)
        normalization_momentum: momentum EMA des running stats (0.99)
        rngs                  : clés JAX/Flax NNX
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 128,
        base_channels: int = 32,
        input_resolution: int = 64,
        normalization_momentum: float = 0.99,
        *,
        rngs: nnx.Rngs,
    ):
        # Split la clé de base en deux pour garantir target ≠ predictor
        # rngs.default() retourne la prochaine clé du stream "default"
        base_key = rngs.default()
        target_key, predictor_key = jr.split(base_key, 2)

        target_rngs = nnx.Rngs(target_key)
        predictor_rngs = nnx.Rngs(predictor_key)

        # Target : poids random fixés — jamais mis à jour par l'optimizer
        # Le gel se fait via stop_gradient() dans compute_bonus(),
        # et par filtrage de l'optimizer en Phase 4.
        self.target = CNNEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            base_channels=base_channels,
            input_resolution=input_resolution,
            rngs=target_rngs,
        )

        # Predictor : poids différents du target (clé splittée différente)
        # C'est lui qui est optimisé pour imiter target.
        self.predictor = CNNEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            base_channels=base_channels,
            input_resolution=input_resolution,
            rngs=predictor_rngs,
        )

        # Running stats : nnx.Variable non-trainable (équivalent register_buffer)
        # Initialisées comme en PyTorch : mean=0, var=1
        # API Flax NNX moderne : get_value() / set_value() (ou [...]  / [...] = )
        self.running_mean = RNDStats(jnp.zeros(()))
        self.running_var = RNDStats(jnp.ones(()))
        self.normalization_momentum = normalization_momentum

    def compute_bonus(self, obs: jax.Array) -> jax.Array:
        """
        Calcule le bonus intrinsèque BRUT (avant normalisation).

        Le target est "gelé" via stop_gradient sur sa sortie : le gradient
        ne remonte pas à travers lui, même si ses params font partie du
        graphe. Seul le predictor reçoit des gradients.

        Args:
            obs   : (B, C, H, W) ou (B, T, C, H, W) — float32 [0, 1]
        Returns:
            bonus : (B,) ou (B, T) — MSE error par sample (mean sur embed_dim)
        """
        # stop_gradient sur target_emb = équivalent de torch.no_grad() pour target
        target_emb = jax.lax.stop_gradient(self.target(obs))
        pred_emb = self.predictor(obs)
        # MSE par sample, moyenné sur la dim embedding
        bonus = ((pred_emb - target_emb) ** 2).mean(axis=-1)
        return bonus

    def normalize_bonus(self, bonus: jax.Array) -> jax.Array:
        """
        Normalise le bonus par running std (style DreamerV3).
        Met à jour les running stats via EMA en place (mutation nnx.Variable).

        Note : la mutation de self.running_mean.value et self.running_var.value
        est permise par le design de nnx.Variable — c'est son rôle. En revanche,
        cette mutation ne sera pas tracée par jit si le module est utilisé tel
        quel dans une fonction jittée. Pour Phase 4, la gestion du state
        mutable devra utiliser nnx.split/merge explicitement.

        Args:
            bonus    : (B,) ou (B, T) — float32
        Returns:
            normalized: même shape que bonus
        """
        # Calcul EMA sans gradient (c'est une mise à jour de stat, pas un param)
        cur_mean = jax.lax.stop_gradient(bonus.mean())
        cur_var = jax.lax.stop_gradient(bonus.var())

        m = self.normalization_momentum
        new_mean = m * self.running_mean.get_value() + (1.0 - m) * cur_mean
        new_var = m * self.running_var.get_value() + (1.0 - m) * cur_var

        # Mutation en place via l'API Flax NNX moderne (set_value)
        self.running_mean.set_value(new_mean)
        self.running_var.set_value(new_var)

        std = jnp.sqrt(new_var + 1e-8)
        # Normalise par std uniquement (garde le signe positif du bonus)
        return bonus / std

    def train_loss(self, obs: jax.Array) -> jax.Array:
        """
        Loss pour entraîner le predictor.
        = moyenne du bonus brut = MSE(predictor(obs), target(obs)).

        Le gradient ne passe que dans predictor (stop_gradient dans compute_bonus).

        Args:
            obs  : (B, C, H, W) ou (B, T, C, H, W) — float32 [0, 1]
        Returns:
            loss : scalaire
        """
        return self.compute_bonus(obs).mean()

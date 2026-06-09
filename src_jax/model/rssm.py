"""
RSSM — Recurrent State Space Model — version JAX/Flax NNX.

Cœur du World Model Dreamer-style. Apprend la dynamique latente du jeu via :

  1. GRU      : h_t = f(h_{t-1}, z_{t-1}, a_{t-1})           — déterministe
  2. Prior    : z_t ~ p(z_t | h_t)                            — sans obs
  3. Posterior: z_t ~ q(z_t | h_t, embedding_t)               — avec obs

L'état complet du WM = (h, z) concaténés. C'est ce que les autres composants
(decoder, reward head, actor, critic) consomment.

Z est CATÉGORIQUE (DreamerV2/V3 style) :
    z = z_categories × z_classes (typiquement 24×24 = 576 valeurs one-hot pour Crafter P2)

Sampling via straight-through estimator :
    forward  : sample one-hot discret (no gradient)
    backward : passe par les probas (différentiable)

KL loss avec free bits GLOBAL et balancing canonique DreamerV3.

Notes JAX :
    - Pas de state mutable interne. Les états (h, z) sont passés en arguments.
    - Sampling : la PRNG key est passée explicitement (pas stockée).
    - observe_sequence / imagine_sequence utilisent jax.lax.scan pour
      compiler une seule boucle (vs T appels Python → recompilation O(T)).
    - Convention scan : on transpose (B, T, ...) → (T, B, ...) avant scan,
      puis on retranspose à la sortie.
"""

import jax
import jax.numpy as jnp
from flax import nnx
import distrax


# ============================================================== Custom GRU Cell

class CustomGRUCell(nnx.Module):
    """
    GRU cell qui matche EXACTEMENT PyTorch nn.GRUCell.

    Différence vs nnx.GRUCell standard :
      - dense_h utilise use_bias=True (les biais b_hr, b_hz, b_hn sont présents)
      - Pour le gate n : tanh(W_in·x + b_in + r * (W_hn·h + b_hn))
        (r multiplie aussi b_hn, conforme PyTorch).

    PyTorch nn.GRUCell formule officielle :
        r = sigmoid(W_ir·x + b_ir + W_hr·h + b_hr)
        z = sigmoid(W_iz·x + b_iz + W_hz·h + b_hz)
        n = tanh(W_in·x + b_in + r * (W_hn·h + b_hn))
        h' = (1 - z) * n + z * h

    Ordre des gates concaténés : (r, z, n) — identique à PyTorch (weight_ih ordonné rzn).

    API compatible avec le test de parité numérique :
      self.dense_i.kernel : (in, 3*hidden)
      self.dense_i.bias   : (3*hidden,)
      self.dense_h.kernel : (hidden, 3*hidden)
      self.dense_h.bias   : (3*hidden,)
    """

    def __init__(self, input_size: int, hidden_size: int, *, rngs: nnx.Rngs):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dense_i = nnx.Linear(input_size, 3 * hidden_size, use_bias=True, rngs=rngs)
        self.dense_h = nnx.Linear(hidden_size, 3 * hidden_size, use_bias=True, rngs=rngs)

    def __call__(self, h_prev: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """
        Args:
            h_prev : (B, hidden_size) état caché précédent
            x      : (B, input_size)  entrée

        Returns:
            (new_h, new_h) tuple — convention nnx.GRUCell : (carry, output) identiques.
        """
        gates_i = self.dense_i(x)        # (B, 3*hidden)
        gates_h = self.dense_h(h_prev)   # (B, 3*hidden)

        # Split en 3 dans l'ordre PyTorch : (r, z, n)
        r_i, z_i, n_i = jnp.split(gates_i, 3, axis=-1)
        r_h, z_h, n_h = jnp.split(gates_h, 3, axis=-1)

        r = jax.nn.sigmoid(r_i + r_h)
        z = jax.nn.sigmoid(z_i + z_h)
        # ATTENTION : pour n, r multiplie n_h qui inclut le bias b_hn (PyTorch-compat)
        n = jnp.tanh(n_i + r * n_h)

        h_new = (1.0 - z) * n + z * h_prev
        return h_new, h_new


# ============================================================== helpers

def symlog(x: jax.Array) -> jax.Array:
    """Symetric log : sign(x) * log(1 + |x|). Réduit la dynamique des outliers."""
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def sample_categorical_straight_through(
    logits: jax.Array,
    key: jax.Array,
    uniform_mix: float = 0.01,
) -> jax.Array:
    """
    Sample one-hot depuis une distribution catégorique avec
    straight-through estimator pour le gradient.

    Args:
        logits      : (..., num_classes) raw logits
        key         : jax.random.PRNGKey
        uniform_mix : proportion d'uniforme mélangée (anti-collapse, DreamerV3)

    Returns:
        sample : (..., num_classes) one-hot dans la dim -1.
                 Forward = one-hot, Backward = gradient via probs (ST estimator).
    """
    probs = jax.nn.softmax(logits, axis=-1)
    num_classes = probs.shape[-1]
    # Mix avec uniforme pour éviter les distributions dégénérées
    probs = (1.0 - uniform_mix) * probs + uniform_mix / num_classes

    # Sample catégorique via distrax (opère sur la dernière dim)
    sample_idx = distrax.Categorical(probs=probs).sample(seed=key)
    sample_onehot = jax.nn.one_hot(sample_idx, num_classes, dtype=probs.dtype)

    # Straight-through : forward = sample_onehot, backward = probs (différentiable)
    # Idiome JAX : x + stop_gradient(y - x) ⇒ valeur=y, gradient=x.
    return probs + jax.lax.stop_gradient(sample_onehot - probs)


# ============================================================== RSSM

class RSSM(nnx.Module):
    """Recurrent State Space Model avec z catégorique (Flax NNX)."""

    def __init__(
        self,
        embed_dim: int = 128,
        action_dim: int = 41,
        h_dim: int = 128,
        z_categories: int = 16,
        z_classes: int = 16,
        hidden_dim: int = 256,
        *,
        rngs: nnx.Rngs,
    ):
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.h_dim = h_dim
        self.z_categories = z_categories
        self.z_classes = z_classes
        self.z_dim = z_categories * z_classes
        self.hidden_dim = hidden_dim

        # ----- pre_gru : (z + action) → hidden_dim  (Linear + LayerNorm + ELU)
        self.pre_gru_linear = nnx.Linear(self.z_dim + action_dim, hidden_dim, rngs=rngs)
        self.pre_gru_norm = nnx.LayerNorm(hidden_dim, epsilon=1e-5, rngs=rngs)

        # ----- GRU cell : hidden_dim input, h_dim hidden
        # CustomGRUCell : matche PyTorch nn.GRUCell (bias_ih + bias_hh tous deux présents)
        # Signature : (h_prev, x) → (new_h, new_h)  (tuple comme nnx.GRUCell)
        self.gru = CustomGRUCell(input_size=hidden_dim, hidden_size=h_dim, rngs=rngs)

        # ----- Prior network : h → logits z   (sert pour l'imagination + KL)
        self.prior_linear1 = nnx.Linear(h_dim, hidden_dim, rngs=rngs)
        self.prior_norm = nnx.LayerNorm(hidden_dim, epsilon=1e-5, rngs=rngs)
        self.prior_linear2 = nnx.Linear(hidden_dim, self.z_dim, rngs=rngs)

        # ----- Posterior network : (h, embedding) → logits z   (sert au training)
        self.post_linear1 = nnx.Linear(h_dim + embed_dim, hidden_dim, rngs=rngs)
        self.post_norm = nnx.LayerNorm(hidden_dim, epsilon=1e-5, rngs=rngs)
        self.post_linear2 = nnx.Linear(hidden_dim, self.z_dim, rngs=rngs)

    # --------------------------------------------------------- info

    @property
    def state_dim(self) -> int:
        """Taille totale du state complet [h ; z]."""
        return self.h_dim + self.z_dim

    def init_state(self, batch_size: int) -> dict:
        """Initialise un état vide (h=0, z=0). Pas besoin de device en JAX."""
        return {
            "h": jnp.zeros((batch_size, self.h_dim)),
            "z": jnp.zeros((batch_size, self.z_dim)),
        }

    @staticmethod
    def get_state_vec(state: dict) -> jax.Array:
        """Concatène h et z en un seul vecteur (utile pour les heads)."""
        return jnp.concatenate([state["h"], state["z"]], axis=-1)

    # --------------------------------------------------------- internal nets

    def _pre_gru(self, x: jax.Array) -> jax.Array:
        """(z + action) → hidden_dim. Linear → LayerNorm → ELU."""
        return jax.nn.silu(self.pre_gru_norm(self.pre_gru_linear(x)))

    def _prior_net(self, h: jax.Array) -> jax.Array:
        """h → z_logits flat (z_dim,). Linear → LayerNorm → ELU → Linear."""
        x = jax.nn.silu(self.prior_norm(self.prior_linear1(h)))
        return self.prior_linear2(x)

    def _posterior_net(self, h_emb: jax.Array) -> jax.Array:
        """(h, embedding) concat → z_logits flat (z_dim,). Linear → LN → ELU → Linear."""
        x = jax.nn.silu(self.post_norm(self.post_linear1(h_emb)))
        return self.post_linear2(x)

    # --------------------------------------------------------- step (observe)

    def observe_step(
        self,
        prev_state: dict,
        prev_action: jax.Array,
        embedding: jax.Array,
        key: jax.Array,
    ) -> tuple[dict, jax.Array, jax.Array]:
        """
        Un step en mode OBSERVATION (training).
        Utilise le posterior (a accès à l'obs via l'embedding).

        Args:
            prev_state  : dict {h, z} de shapes (B, h_dim), (B, z_dim)
            prev_action : (B, action_dim) one-hot
            embedding   : (B, embed_dim) sortie de l'encoder pour obs_t
            key         : PRNGKey pour le sampling stochastique

        Returns:
            new_state    : dict {h, z}
            post_logits  : (B, z_cat, z_classes)
            prior_logits : (B, z_cat, z_classes)
        """
        # 1. Update h via GRU
        gru_input_raw = jnp.concatenate([prev_state["z"], prev_action], axis=-1)
        gru_input = self._pre_gru(gru_input_raw)
        h, _ = self.gru(prev_state["h"], gru_input)  # (new_h, output) — équivalents

        # 2. Posterior : z sampled depuis (h, embedding)
        post_input = jnp.concatenate([h, embedding], axis=-1)
        post_logits = self._posterior_net(post_input).reshape(
            -1, self.z_categories, self.z_classes
        )
        z_sample = sample_categorical_straight_through(post_logits, key)
        z = z_sample.reshape(-1, self.z_dim)

        # 3. Prior : logits depuis h seul (utilisé pour la KL loss)
        prior_logits = self._prior_net(h).reshape(
            -1, self.z_categories, self.z_classes
        )

        new_state = {"h": h, "z": z}
        return new_state, post_logits, prior_logits

    # --------------------------------------------------------- step (imagine)

    def imagine_step(
        self,
        prev_state: dict,
        prev_action: jax.Array,
        key: jax.Array,
    ) -> tuple[dict, jax.Array]:
        """
        Un step en mode IMAGINATION (rêve).
        Utilise le prior (pas d'obs disponible).

        Args:
            prev_state  : dict {h, z}
            prev_action : (B, action_dim) one-hot
            key         : PRNGKey pour le sampling z

        Returns:
            new_state    : dict {h, z}
            prior_logits : (B, z_cat, z_classes)
        """
        gru_input_raw = jnp.concatenate([prev_state["z"], prev_action], axis=-1)
        gru_input = self._pre_gru(gru_input_raw)
        h, _ = self.gru(prev_state["h"], gru_input)

        prior_logits = self._prior_net(h).reshape(
            -1, self.z_categories, self.z_classes
        )
        z_sample = sample_categorical_straight_through(prior_logits, key)
        z = z_sample.reshape(-1, self.z_dim)

        new_state = {"h": h, "z": z}
        return new_state, prior_logits

    # --------------------------------------------------------- sequence (observe)

    def observe_sequence(
        self,
        embeddings: jax.Array,
        actions_onehot: jax.Array,
        dones: jax.Array | None = None,
        initial_state: dict | None = None,
        *,
        key: jax.Array,
    ) -> dict:
        """
        Process une séquence complète en mode observation via jax.lax.scan.

        Args:
            embeddings     : (B, T, embed_dim)
            actions_onehot : (B, T, action_dim)
            dones          : (B, T) optionnel. Reset h/z après done.
            initial_state  : dict {h, z} optionnel, sinon zeros.
            key            : PRNGKey

        Returns:
            dict avec :
                h            : (B, T, h_dim)
                z            : (B, T, z_dim)
                post_logits  : (B, T, z_cat, z_classes)
                prior_logits : (B, T, z_cat, z_classes)
        """
        B, T, _ = embeddings.shape

        if initial_state is None:
            initial_state = self.init_state(B)

        # Prev_action initial = zéros
        prev_action_init = jnp.zeros((B, self.action_dim))

        # Transposer (B, T, ...) → (T, B, ...) car jax.lax.scan itère sur la dim 0
        emb_T = jnp.transpose(embeddings, (1, 0, 2))      # (T, B, embed_dim)
        act_T = jnp.transpose(actions_onehot, (1, 0, 2))  # (T, B, action_dim)
        if dones is not None:
            done_T = jnp.transpose(dones, (1, 0))         # (T, B)
            # Construire dones_prev : décalé d'un step
            # PyTorch : si t==0, pas de reset. Sinon, on regarde dones[:, t-1].
            # On crée donc done_prev[0] = 0 (jamais reset au step 0)
            #              done_prev[t] = done[t-1] pour t >= 1
            done_prev_T = jnp.concatenate(
                [jnp.zeros((1, B), dtype=done_T.dtype), done_T[:-1]],
                axis=0,
            )  # (T, B)
        else:
            done_prev_T = jnp.zeros((T, B), dtype=jnp.float32)

        # Construire la séquence des "previous actions" :
        # PyTorch : prev_action[t=0] = 0, prev_action[t>=1] = actions[t-1]
        prev_act_T = jnp.concatenate(
            [prev_action_init[None, ...], act_T[:-1]],
            axis=0,
        )  # (T, B, action_dim)

        # ---- step_fn pour jax.lax.scan ----
        # Carry : dict avec h, z, key (les actions sont dans xs car déjà décalées)
        # xs    : (emb_t, prev_action_t, done_prev_t) au timestep t
        def step_fn(carry, xs):
            h_prev, z_prev, key = carry["h"], carry["z"], carry["key"]
            emb_t, prev_action_t, done_prev_t = xs

            # Reset si done au step précédent : mask (B, 1) à appliquer
            # done_prev_t shape (B,) → on broadcast sur h/z
            done_mask = (1.0 - done_prev_t.astype(jnp.float32))[:, None]  # (B, 1)
            h_prev = h_prev * done_mask
            z_prev = z_prev * done_mask
            prev_action_t = prev_action_t * done_mask

            # Split key : subkey pour ce step, key continue dans le carry
            key, subkey = jax.random.split(key)

            new_state, post_logits, prior_logits = self.observe_step(
                {"h": h_prev, "z": z_prev}, prev_action_t, emb_t, subkey
            )

            new_carry = {"h": new_state["h"], "z": new_state["z"], "key": key}
            output = {
                "h": new_state["h"],
                "z": new_state["z"],
                "post_logits": post_logits,
                "prior_logits": prior_logits,
            }
            return new_carry, output

        init_carry = {
            "h": initial_state["h"],
            "z": initial_state["z"],
            "key": key,
        }
        xs = (emb_T, prev_act_T, done_prev_T)

        _, ys = jax.lax.scan(step_fn, init_carry, xs)
        # ys : pytree de shapes (T, B, ...) — il faut retransposer en (B, T, ...)
        return {
            "h":            jnp.transpose(ys["h"],            (1, 0, 2)),
            "z":            jnp.transpose(ys["z"],            (1, 0, 2)),
            "post_logits":  jnp.transpose(ys["post_logits"],  (1, 0, 2, 3)),
            "prior_logits": jnp.transpose(ys["prior_logits"], (1, 0, 2, 3)),
        }

    # --------------------------------------------------------- sequence (imagine)

    def imagine_sequence(
        self,
        initial_state: dict,
        actions_onehot: jax.Array,
        *,
        key: jax.Array,
    ) -> dict:
        """
        Génère une séquence imaginée depuis un état initial via jax.lax.scan.

        Args:
            initial_state  : dict {h, z} de shapes (B, h_dim), (B, z_dim)
            actions_onehot : (B, T, action_dim) — actions à exécuter dans l'imagination
            key            : PRNGKey

        Returns:
            dict avec :
                h            : (B, T, h_dim)
                z            : (B, T, z_dim)
                prior_logits : (B, T, z_cat, z_classes)
        """
        B, T, _ = actions_onehot.shape

        # Transposer (B, T, action_dim) → (T, B, action_dim) pour scan
        act_T = jnp.transpose(actions_onehot, (1, 0, 2))

        def step_fn(carry, action_t):
            h_prev, z_prev, key = carry["h"], carry["z"], carry["key"]
            key, subkey = jax.random.split(key)

            new_state, prior_logits = self.imagine_step(
                {"h": h_prev, "z": z_prev}, action_t, subkey
            )

            new_carry = {"h": new_state["h"], "z": new_state["z"], "key": key}
            output = {
                "h": new_state["h"],
                "z": new_state["z"],
                "prior_logits": prior_logits,
            }
            return new_carry, output

        init_carry = {
            "h": initial_state["h"],
            "z": initial_state["z"],
            "key": key,
        }
        _, ys = jax.lax.scan(step_fn, init_carry, act_T)

        return {
            "h":            jnp.transpose(ys["h"],            (1, 0, 2)),
            "z":            jnp.transpose(ys["z"],            (1, 0, 2)),
            "prior_logits": jnp.transpose(ys["prior_logits"], (1, 0, 2, 3)),
        }

    # --------------------------------------------------------- losses

    @staticmethod
    def kl_loss(
        post_logits: jax.Array,
        prior_logits: jax.Array,
        free_bits: float = 1.0,
        beta_dyn: float = 0.5,
        beta_rep: float = 0.1,
    ) -> jax.Array:
        """
        KL loss avec free bits GLOBAL et KL balancing CANONIQUE DreamerV3.

        Deux termes pondérés séparément :
            loss = β_dyn × KL(sg(post) || prior)    [le PRIOR apprend la dynamique]
                 + β_rep × KL(post || sg(prior))    [le POST s'aligne sur le prior]

        Coefs canoniques DreamerV3 : β_dyn=0.5, β_rep=0.1.
        Le free_bits (1 nat) s'applique à la KL TOTALE PAR STEP (somme sur
        les z_cat catégories de la factorized categorical), comme l'officiel.

        Args:
            post_logits  : (B, T, z_cat, z_classes)
            prior_logits : (B, T, z_cat, z_classes)
            free_bits    : seuil minimum PAR STEP (en nats, sur la KL sommée)
            beta_dyn     : poids du terme "prior apprend du posterior" (canonique 0.5)
            beta_rep     : poids du terme "posterior s'aligne sur prior" (canonique 0.1)

        Returns:
            loss : scalaire
        """
        # distrax.Categorical opère sur la dernière dim (z_classes)
        # Versions avec stop_gradient sur les logits
        post_logits_sg = jax.lax.stop_gradient(post_logits)
        prior_logits_sg = jax.lax.stop_gradient(prior_logits)

        post_dist = distrax.Categorical(logits=post_logits)
        prior_dist = distrax.Categorical(logits=prior_logits)
        post_dist_sg = distrax.Categorical(logits=post_logits_sg)
        prior_dist_sg = distrax.Categorical(logits=prior_logits_sg)

        # KL divergence : shape (B, T, z_cat) — la dim z_classes est consommée
        kl_prior_learn = post_dist_sg.kl_divergence(prior_dist)
        kl_post_learn = post_dist.kl_divergence(prior_dist_sg)

        # FIX AUDIT (free bits par STEP, pas par catégorie) :
        # z est une factorized categorical → KL jointe d'un step = SOMME des
        # KL des z_cat catégories. Le free bits du paper (1 nat) s'applique à
        # cette KL totale par step. L'ancienne version clampait CHAQUE
        # catégorie à 1 nat → free bits effectif = z_cat (24) nats/step →
        # quasi toute la KL sous le clamp → gradient prior ≈ 0 → le prior
        # (sur lequel roule l'imagination) n'était presque pas entraîné.
        kl_dyn_step = kl_prior_learn.sum(-1)   # (B, T) — KL totale par step
        kl_rep_step = kl_post_learn.sum(-1)    # (B, T)
        kl_dyn = jnp.maximum(kl_dyn_step, free_bits).mean()
        kl_rep = jnp.maximum(kl_rep_step, free_bits).mean()
        return beta_dyn * kl_dyn + beta_rep * kl_rep

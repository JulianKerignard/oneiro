"""
RSSM — Recurrent State Space Model.

Le cœur du World Model Dreamer-style. Apprend la dynamique latente
du jeu via 3 sous-réseaux :

  1. GRU      : h_t = f(h_{t-1}, z_{t-1}, a_{t-1})           — déterministe
  2. Prior    : z_t ~ p(z_t | h_t)                            — sans obs
  3. Posterior: z_t ~ q(z_t | h_t, embedding_t)               — avec obs

L'état complet du WM = (h, z) concaténés. C'est ce que les autres
composants (decoder, reward head, actor, critic) consomment.

Z est CATÉGORIQUE (DreamerV2/V3 style) :
    z = z_categories × z_classes (typiquement 16×16 = 256 valeurs one-hot)

Sampling via straight-through estimator :
    forward  : sample one-hot discret (no gradient)
    backward : passe par les probas (différentiable)

KL loss avec free bits (évite le posterior collapse).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================== sampling

def sample_categorical_straight_through(logits, uniform_mix=0.01):
    """
    Sample one-hot depuis une distribution catégorique avec
    straight-through estimator pour le gradient.

    Args:
        logits : (..., num_cat, num_classes)
        uniform_mix : proportion d'uniforme mélangée (anti-collapse, DreamerV3)

    Returns:
        sample : tensor de même shape que logits, one-hot dans la dim -1
    """
    probs = F.softmax(logits, dim=-1)
    # Mix avec uniforme pour éviter les distributions dégénérées
    uniform = torch.ones_like(probs) / probs.shape[-1]
    probs = (1.0 - uniform_mix) * probs + uniform_mix * uniform

    # PERF #6 : torch.multinomial direct au lieu de torch.distributions.Categorical
    # (évite l'overhead Python de création de l'objet Distribution à chaque call)
    num_classes = logits.shape[-1]
    flat_probs = probs.reshape(-1, num_classes)
    sample_idx_flat = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
    sample_idx = sample_idx_flat.view(probs.shape[:-1])
    sample_onehot = F.one_hot(sample_idx, num_classes=num_classes).to(probs.dtype)

    # Straight-through : forward = sample_onehot, backward = probs (différentiable)
    sample = probs + (sample_onehot - probs).detach()
    return sample


# ============================================================== RSSM

class RSSM(nn.Module):
    """Recurrent State Space Model avec z catégorique."""

    def __init__(
        self,
        embed_dim: int = 128,
        action_dim: int = 41,
        h_dim: int = 128,
        z_categories: int = 16,
        z_classes: int = 16,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.h_dim = h_dim
        self.z_categories = z_categories
        self.z_classes = z_classes
        self.z_dim = z_categories * z_classes
        self.hidden_dim = hidden_dim

        # Layer dense AVANT le GRU (DreamerV3 standard)
        # mappe (z, action) → hidden_dim avant la récurrence
        self.pre_gru = nn.Sequential(
            nn.Linear(self.z_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )
        self.gru = nn.GRUCell(hidden_dim, h_dim)

        # Prior network : h → logits z   (sert pour l'imagination)
        self.prior_net = nn.Sequential(
            nn.Linear(h_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, self.z_dim),
        )

        # Posterior network : (h, embedding) → logits z   (sert au training)
        self.posterior_net = nn.Sequential(
            nn.Linear(h_dim + embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, self.z_dim),
        )

    # --------------------------------------------------------- info

    @property
    def state_dim(self):
        """Taille totale du state complet [h ; z]."""
        return self.h_dim + self.z_dim

    def init_state(self, batch_size: int, device):
        """Initialise un état vide (h=0, z=0)."""
        return {
            "h": torch.zeros(batch_size, self.h_dim, device=device),
            "z": torch.zeros(batch_size, self.z_dim, device=device),
        }

    @staticmethod
    def get_state_vec(state):
        """Concatène h et z en un seul vecteur (utile pour les heads)."""
        return torch.cat([state["h"], state["z"]], dim=-1)

    # --------------------------------------------------------- step (observe)

    def observe_step(self, prev_state, prev_action, embedding):
        """
        Un step en mode OBSERVATION (training).
        Utilise le posterior (a accès à l'obs via l'embedding).

        Args:
            prev_state : dict {h, z} de shapes (B, h_dim), (B, z_dim)
            prev_action : (B, action_dim) one-hot
            embedding : (B, embed_dim) sortie de l'encoder pour obs_t

        Returns:
            new_state    : dict {h, z}
            post_logits  : (B, z_cat, z_classes)
            prior_logits : (B, z_cat, z_classes)
        """
        # 1. Update h via GRU
        gru_input = torch.cat([prev_state["z"], prev_action], dim=-1)
        gru_input = self.pre_gru(gru_input)
        h = self.gru(gru_input, prev_state["h"])

        # 2. Posterior : z sampled depuis (h, embedding)
        post_input = torch.cat([h, embedding], dim=-1)
        post_logits = self.posterior_net(post_input).view(
            -1, self.z_categories, self.z_classes
        )
        z_sample = sample_categorical_straight_through(post_logits)
        z = z_sample.view(-1, self.z_dim)

        # 3. Prior : logits depuis h seul (utilisé pour la KL loss)
        prior_logits = self.prior_net(h).view(
            -1, self.z_categories, self.z_classes
        )

        new_state = {"h": h, "z": z}
        return new_state, post_logits, prior_logits

    # --------------------------------------------------------- step (imagine)

    def imagine_step(self, prev_state, prev_action):
        """
        Un step en mode IMAGINATION (rêve).
        Utilise le prior (pas d'obs disponible).

        Args:
            prev_state : dict {h, z}
            prev_action : (B, action_dim) one-hot

        Returns:
            new_state    : dict {h, z}
            prior_logits : (B, z_cat, z_classes)
        """
        gru_input = torch.cat([prev_state["z"], prev_action], dim=-1)
        gru_input = self.pre_gru(gru_input)
        h = self.gru(gru_input, prev_state["h"])

        prior_logits = self.prior_net(h).view(
            -1, self.z_categories, self.z_classes
        )
        z_sample = sample_categorical_straight_through(prior_logits)
        z = z_sample.view(-1, self.z_dim)

        new_state = {"h": h, "z": z}
        return new_state, prior_logits

    # --------------------------------------------------------- sequence (observe)

    def observe_sequence(self, embeddings, actions_onehot, dones=None,
                          initial_state=None):
        """
        Process une séquence complète en mode observation (training).

        Args:
            embeddings : (B, T, embed_dim)
            actions_onehot : (B, T, action_dim)
            dones : (B, T) optional. Si fourni, on reset h et z après un done.
            initial_state : dict {h, z} optional, sinon zeros.

        Returns:
            dict avec :
                h            : (B, T, h_dim)
                z            : (B, T, z_dim)
                post_logits  : (B, T, z_cat, z_classes)
                prior_logits : (B, T, z_cat, z_classes)
        """
        B, T, _ = embeddings.shape
        device = embeddings.device

        state = initial_state if initial_state is not None else self.init_state(B, device)
        prev_action = torch.zeros(B, self.action_dim, device=device)

        h_seq, z_seq, post_seq, prior_seq = [], [], [], []

        for t in range(T):
            # Reset si done au step précédent (boundaries d'épisodes)
            if dones is not None and t > 0:
                done_mask = (1.0 - dones[:, t - 1].float()).unsqueeze(-1)  # (B, 1)
                state = {
                    "h": state["h"] * done_mask,
                    "z": state["z"] * done_mask,
                }
                prev_action = prev_action * done_mask

            new_state, post_logits, prior_logits = self.observe_step(
                state, prev_action, embeddings[:, t]
            )

            h_seq.append(new_state["h"])
            z_seq.append(new_state["z"])
            post_seq.append(post_logits)
            prior_seq.append(prior_logits)

            state = new_state
            prev_action = actions_onehot[:, t]

        return {
            "h":            torch.stack(h_seq, dim=1),
            "z":            torch.stack(z_seq, dim=1),
            "post_logits":  torch.stack(post_seq, dim=1),
            "prior_logits": torch.stack(prior_seq, dim=1),
        }

    # --------------------------------------------------------- sequence (imagine)

    def imagine_sequence(self, initial_state, actions_onehot):
        """
        Génère une séquence imaginée depuis un état initial.

        Args:
            initial_state : dict {h, z}
            actions_onehot : (B, T, action_dim)

        Returns:
            dict avec h, z, prior_logits de shapes (B, T, ...)
        """
        B, T, _ = actions_onehot.shape
        state = initial_state

        h_seq, z_seq, prior_seq = [], [], []

        for t in range(T):
            new_state, prior_logits = self.imagine_step(state, actions_onehot[:, t])
            h_seq.append(new_state["h"])
            z_seq.append(new_state["z"])
            prior_seq.append(prior_logits)
            state = new_state

        return {
            "h":            torch.stack(h_seq, dim=1),
            "z":            torch.stack(z_seq, dim=1),
            "prior_logits": torch.stack(prior_seq, dim=1),
        }

    # --------------------------------------------------------- losses

    @staticmethod
    def kl_loss(post_logits, prior_logits, free_bits=1.0, beta_dyn=0.5, beta_rep=0.1):
        """
        KL loss avec free bits GLOBAL et KL balancing CANONIQUE DreamerV3.

        Deux termes pondérés séparément (pas avec α/(1-α)) :
            loss = β_dyn × KL(sg(post) || prior)    [le PRIOR apprend la dynamique]
                 + β_rep × KL(post || sg(prior))    [le POST s'aligne sur le prior]

        Coefs canoniques DreamerV3 : β_dyn=0.5, β_rep=0.1.
        Avant on avait balance=0.8 ⇒ 0.8 + 0.2 = 1.0 (somme normalisée).
        Maintenant : 0.5 + 0.1 = 0.6 absolu (moins de pression KL globale,
        mais le ratio dyn/rep = 5 reste proche du précédent ratio=4).

        Le free_bits est appliqué GLOBALEMENT (sur la KL moyenne), pas par-catégorique.

        Args:
            post_logits  : (B, T, z_cat, z_classes)
            prior_logits : (B, T, z_cat, z_classes)
            free_bits    : seuil minimum GLOBAL (en nats, moyenne sur tout le batch)
            beta_dyn     : poids du terme "prior apprend du posterior" (canonique 0.5)
            beta_rep     : poids du terme "posterior s'aligne sur prior" (canonique 0.1)

        Returns:
            loss : scalaire
        """
        # Posterior et prior comme distributions
        post_dist = torch.distributions.Categorical(logits=post_logits)
        prior_dist = torch.distributions.Categorical(logits=prior_logits)
        post_dist_sg = torch.distributions.Categorical(logits=post_logits.detach())
        prior_dist_sg = torch.distributions.Categorical(logits=prior_logits.detach())

        # KL divergence par catégorique : (B, T, z_cat)
        kl_prior_learn = torch.distributions.kl.kl_divergence(post_dist_sg, prior_dist)
        kl_post_learn  = torch.distributions.kl.kl_divergence(post_dist, prior_dist_sg)

        # Moyenne d'abord (sur batch et catégoriques), puis free_bits GLOBAL
        kl_prior_mean = kl_prior_learn.mean()
        kl_post_mean  = kl_post_learn.mean()

        device = kl_prior_mean.device
        fb = torch.tensor(free_bits, device=device, dtype=kl_prior_mean.dtype)
        kl_prior_clamped = torch.maximum(kl_prior_mean, fb)
        kl_post_clamped  = torch.maximum(kl_post_mean,  fb)

        loss = beta_dyn * kl_prior_clamped + beta_rep * kl_post_clamped
        return loss

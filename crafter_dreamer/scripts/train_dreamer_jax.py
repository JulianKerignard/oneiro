"""
Training complet du mini-Dreamer pour Crafter — version JAX / Flax NNX.

Port de scripts/train_dreamer.py (PyTorch) vers JAX. Entraînement sur TPU v5e-8
(Kaggle) via crafter_dreamer/scripts/kaggle_train.py. Le gain vient de :
  - jit-compilation des train_steps (forward + backward + optimizer update fusionnés)
  - jax.lax.scan pour les séquences RSSM et l'imagination (pas de boucle Python)
  - dispatch GPU XLA optimisé

Pipeline (identique au PyTorch) :
    1. Warmup : 5k transitions random pour amorcer le buffer.
    2. Boucle principale (N itérations) :
       a. Collecte : actor joue dans l'env (Python pur, pas jit), buffer numpy.
       b. Train WM : encoder + RSSM + decoder + heads (jit).
       c. Train Actor + Critic : imagination 16 steps + lambda returns + PG (jit).
    3. Eval périodique : achievements moyens sur quelques épisodes (Python).
    4. Sauvegarde checkpoint + courbes.

PHASE 12 : Réintroduction des 3 mécanismes de DreamerV3 originaux :
    - adaptive_alpha : log_alpha trainable (SAC-style auto-tuning entropy).
    - auto_explore  : boost entropy multiplier si stagnation détectée à l'EVAL.
    - RND           : Random Network Distillation pour exploration dirigée.

Return normalization reste en simple EMA std (vs Percentile EMA).

Usage :
    .venv/bin/python crafter_dreamer/scripts/train_dreamer_jax.py
    .venv/bin/python crafter_dreamer/scripts/train_dreamer_jax.py \\
        --entropy_coef 0.005 --train_iter 30000 --n_envs 4 --batch_size 32 \\
        --run_name crafter_jax_v1
"""

import sys
import time
import argparse
import math
import json
import os
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.sharding import NamedSharding, PartitionSpec as P, AxisType
from flax import nnx
import optax

from crafter_dreamer.env import CrafterEnv, ACHIEVEMENTS, ACTION_NAMES
from src_jax.buffer import ImageReplayBufferJAX, ImageReplayBufferCPU
from src_jax.model import (
    CNNEncoder, CNNDecoder, RSSM, RewardHead, ContinueHead,
    Actor, Critic, RNDModule,
)


# ============================== Hyperparams (alignés sur le PyTorch)

SEED = 42

# Données — aligné config paper DreamerV3 Crafter (configs.yaml)
# 1M = paper. Couvre un run 30k iter ENTIER sans wrap FIFO (v21/H_312 : à
# 500k le buffer était plein à iter 15.6k → écrasement de la diversité early
# → perte de la capacité de récupération → dérive descendante après ~19k).
# ~12.3 GB en uint8 : tient en RAM host (--buffer_device cpu, le défaut en prod TPU)
# mais pas en VRAM sur un accélérateur modeste.
BUFFER_CAPACITY = 1_000_000
WARMUP_STEPS = 5_000

# Training principal.
# RÉALITÉ du ratio (audit) : collect_per_iter = COLLECT_PER_ITER // n_envs
# boucles × n_envs envs = 32 env_steps/iter. 1 update WM de batch 16×64 =
# 1024 timesteps rejoués / 32 collectés → replay_ratio = 32.
# Paper Crafter : train_ratio = 512 → on est 16× SOUS le paper.
# Levier pour s'en rapprocher : --wm_train_per_iter / --ac_train_per_iter.
TRAIN_ITERATIONS = 30_000
COLLECT_PER_ITER = 32        # 32 env_steps par iter (2 boucles × 16 envs)
WM_TRAIN_PER_ITER = 1
AC_TRAIN_PER_ITER = 1

# Sampling — aligné paper Crafter
BATCH_SIZE = 16              # paper utilise batch=16
SEQ_LEN = 64                 # paper utilise seq_len=64
IMAGINATION_HORIZON = 16

# Optimization (DreamerV3 canonique)
LR_WM = 1e-4   # aligné symoon11 (réf 17.65) : WM rapide
LR_AC = 1e-4   # RETENU sur preuve statistique (audit adversarial 2026-07-22) : sur 15 runs,
               # lr 1e-4 → 1.94% moyen (n=9, best 2.46% = record absolu) vs lr 3e-5 → 1.61%
               # moyen (n=6, best 1.89%). A/B propre : v26 (1e-4) 2.46% → v29 (3e-5) 1.89%.
               # Le passage à 3e-5 "façon symoon11" a été tenté puis ANNULÉ : leur LR va avec
               # un WM 3.35× plus gros (181M vs 54M), il ne se transpose pas isolément.
GRAD_CLIP_WM = 1000.0  # aligné symoon11 : clip quasi inactif (1.0 écrasait les gradients recon sommés sur 64x64x3 px)
GRAD_CLIP_AC = 100.0   # aligné symoon11
GRAD_CLIP = GRAD_CLIP_AC  # défaut générique (optim RND si activé)

# RL params
# GAMMA 0.997 (paper) : horizon de valeur ~330 steps (vs ~100 à 0.99).
# v18/v19b oscillaient car les returns imaginés mouraient une fois les
# quick-wins saturés (H_311) — les chaînes profondes (wood→table→pickaxe)
# étaient hors de l'horizon. 0.997 garde leur valeur visible.
GAMMA = 0.997
LAMBDA_GAE = 0.95
# ENTROPY_COEF : 3e-4 = paper et TOUTES les refs (danijar, symoon11 17.65 ach,
# NM512, sheeprl) — aucune n'utilise d'alpha adaptatif. L'audit a montré que
# le "breakthrough" v15 = alpha appris redescendant exp jusqu'à ~3e-4 à iter
# 3000. Coef fixe 3e-4 = démarrer directement au seuil (décollage prédit
# ~iter 700-1500, borné par la convergence WM/critic au lieu de l'érosion α).
# Le test v13 de 3e-4 datait d'AVANT le fix stop_gradient → non probant.
ENTROPY_COEF = 3e-4

# Adaptive entropy (SAC-style) — OFF par défaut depuis l'audit (aucune ref ne
# l'utilise). Si réactivé (--adaptive_alpha) :
# - Adam normalise le gradient constant de log_alpha → d(log α)/dt = -LR_ALPHA
#   exactement (sign-step). Le temps de décollage est ln(α_init/3e-4)/LR_ALPHA
#   iters. LR_ALPHA=1e-4 donnait ~28k iters (v16 plat par construction).
# - INIT_ALPHA=3e-4 : démarrer au seuil au lieu d'attendre la descente.
LR_ALPHA = 1e-3
INIT_ALPHA = 3e-4
H_TARGET = 2.0           # entropy cible (par défaut), action_dim=17 → log(17)=2.83
LOG_ALPHA_CLIP_MIN = -10.0   # alpha >= ~4.5e-5
LOG_ALPHA_CLIP_MAX = 0.0     # alpha <= 1.0

# Auto-explore : détection de stagnation sur les achievements EVAL
AUTO_EXPLORE_THRESHOLD = 0.05   # progrès relatif requis (5%)
AUTO_EXPLORE_PATIENCE = 2
AUTO_EXPLORE_BOOST = 1.5
AUTO_EXPLORE_DECAY = 0.85
AUTO_EXPLORE_MAX = 5.0

# RND : exploration dirigée par curiosité
LR_RND = 1e-4
RND_COEF = 0.5

# Return normalization (DreamerV3 canonique)
RETURN_EMA_DECAY = 0.99
RETURN_PERCENTILE_LOW = 0.05
RETURN_PERCENTILE_HIGH = 0.95

# Critic EMA target network
CRITIC_TARGET_TAU = 0.98

# Architecture — Étape 1 : scaling 14.4M → ~75M avec bonne répartition.
# AVANT (14.4M, AC famélique à 8%) : EMBED 512, H_DIM 1280, Z 32×16, HIDDEN 256, cnn 16,
# actor/critic = 2 couches × 256.
# Archi ~75M (deter 2048, stochastique 32×32, CNN_DEPTH 32, MLP 1024) + actor/critic
# profond (5×1280). On teste le FIX REWARD directement sur le 75M (cible) : les effets
# ne se transfèrent pas entre tailles (co-tuning taille-dépendant). Le H_collapse du
# gros actor est traité par LR_AC=3e-5 (valeur symoon11 pour gros actor).
EMBED_DIM = 1024         # sortie CNN
H_DIM = 2048             # deter GRU (taille officielle DreamerV3)
Z_CATEGORIES = 32        # 32 variables catégorielles
Z_CLASSES = 32           # × 32 classes → stochastique 32×32
HIDDEN_DIM = 1024        # MLP units RSSM + reward/continue heads
CNN_DEPTH = 32           # base channels CNN
# Actor-Critic : MLP 3×1024 (config danijar Crafter) = celle de nos 2 meilleurs runs 65M
# (v42 1.98%, v47 2.22%). Le passage à 5×1024 "façon symoon11" a été tenté puis ANNULÉ :
# le seul point de données sur l'élargissement de l'actor est NÉGATIF (v26 actor 2×256 →
# 2.46% record vs v41 actor 3×1024 → 1.61%), et H est piloté par la taille de l'actor
# (v25 actor 2×256 → H=1.01 vs v46 actor 3×1024 → H=0.40, à ratio identique).
AC_HIDDEN_DIM = 1024     # largeur MLP actor/critic (danijar)
AC_NUM_LAYERS = 3        # profondeur (danijar : 3 couches)

# KL loss DreamerV3
FREE_BITS = 1.0
BETA_DYN = 0.5
BETA_REP = 0.1

# WM loss weights
W_RECON = 1.0
W_KL = 1.0
W_REWARD = 1.0
W_CONTINUE = 1.0
# Poids de la CE reward head sur les ACHIEVEMENTS (|r| > 0.5) UNIQUEMENT — cf. heads.py.
# Contre le déséquilibre de classes (les +1 = ~1.4% des transitions) qui fait sous-prédire
# les achievements rares (mesuré v37 : pred~0.05-0.31 sur les +1). Calibrage (seuil 0.5) :
# w=10 → ~12% de la loss sur les achievements. HISTORIQUE v38-v43 : le seuil était 0.01 et
# pondérait AUSSI la santé (±0.1, 2.7× plus fréquente) → leakage rew@0=+0.015 → inflation
# critic +5 (γ=0.997) → scale 2.7→7.7 → PG écrasé ÷2.5. Fix v44 : seuil 0.5 (heads.py),
# la santé revient à poids 1. Critères de succès : rew@0 ≤ +0.005, scale ≤ 5.
REWARD_RARE_WEIGHT = 10.0

# Logging / eval
LOG_INTERVAL = 50
EVAL_INTERVAL = 2000
EVAL_EPISODES = 75   # 10 était trop bruité pour trancher un fix (écart-type ~0.5 ach sur 10 eps)


# ============================== Phase 13 : Safeguards auto-régulateurs


def get_rnd_coef(it: int, args) -> float:
    """
    SAFEGUARD 1 — RND coef warmup linéaire.

    Démarre rnd_coef à 0 et monte linéairement jusqu'à `args.rnd_coef` sur
    les `args.rnd_warmup_steps` premières itérations.
    Pendant le warmup, le WM apprend les bases sur reward extrinsèque pur,
    puis RND s'active progressivement.
    """
    if not args.use_rnd:
        return 0.0
    warmup = max(1, int(args.rnd_warmup_steps))
    if it < warmup:
        return float(args.rnd_coef) * (it / warmup)
    return float(args.rnd_coef)


def get_h_target(it: int, args) -> float:
    """
    SAFEGUARD 4 — Curriculum H_target (entropy decay).

    Schedule linéaire de `h_target_init` (exploration) à `h_target_final`
    (exploitation) sur `h_target_decay_steps` itérations.
    Si `h_target_schedule` est désactivé, retourne `args.h_target` constant.
    """
    if not args.h_target_schedule:
        return float(args.h_target)
    decay = max(1, int(args.h_target_decay_steps))
    progress = min(it / decay, 1.0)
    return float(args.h_target_init) * (1 - progress) + float(args.h_target_final) * progress


class TrainingHealthMonitor:
    """
    SAFEGUARD 3 — Health monitor + early stop.

    Détecte les pathologies en temps réel sur les metrics du training :
        - NaN/Inf  : fatal immédiat
        - H entropy collapse (< 0.3)
        - WM divergence (loss_wm > 5000.0, ajusté pour recon sum-over-pixels)
        - RND domination (bonus/extrinsic > 100)
        - Critic explosion (loss_critic > 10)

    Si une même catégorie de warning persiste `fatal_threshold_consec` itérations
    consécutives, le monitor signale `is_fatal=True` pour permettre un kill auto.
    """

    def __init__(self, args):
        self.args = args
        self.warnings_count = {}
        self.fatal_threshold_consec = int(args.health_consec_threshold)

    def check(self, metrics: dict, it: int):
        warnings = []

        # 1. NaN/Inf check (FATAL immédiat)
        for k, v in metrics.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(fv):
                return [f"FATAL_NAN: {k}={fv}"], True

        # 2. H entropy collapse
        H_val = float(metrics.get("H", metrics.get("entropy", 1.0)))
        if H_val < 0.3:
            warnings.append(f"H_collapse: H={H_val:.2f}")

        # 3. WM divergence
        # Note: après FIX recon-loss (sum sur pixels au lieu de mean),
        # loss_wm est typiquement ~50-2000 en early training. Seuil 5000 pour vraie divergence.
        loss_wm_val = float(metrics.get("loss_wm", 0.0))
        if loss_wm_val > 5000.0:
            warnings.append(f"WM_diverging: loss_wm={loss_wm_val:.2f}")

        # 4. RND domination (ratio bonus / extrinsic)
        bonus_mean = float(metrics.get("rnd_bonus_mean", 0.0))
        extr_mean = float(metrics.get("extrinsic_mean", 0.0))
        if extr_mean > 1e-6:
            ratio = bonus_mean / max(extr_mean, 1e-6)
            if ratio > 100.0:
                warnings.append(f"RND_dominates: ratio={ratio:.1f}")

        # 5. Critic explosion
        loss_critic_val = float(metrics.get("loss_critic", 0.0))
        if loss_critic_val > 10.0:
            warnings.append(f"Critic_exploding: loss_critic={loss_critic_val:.2f}")

        # Track consec warnings (par clé avant ':')
        active_keys = {w.split(":")[0] for w in warnings}
        for key in active_keys:
            self.warnings_count[key] = self.warnings_count.get(key, 0) + 1

        # Reset compteurs des catégories qui ne sont plus actives
        for key in list(self.warnings_count.keys()):
            if key not in active_keys:
                self.warnings_count[key] = 0

        # Check fatal : N consec sur la même catégorie
        is_fatal = False
        for key, count in self.warnings_count.items():
            if count >= self.fatal_threshold_consec:
                is_fatal = True
                warnings.append(f"FATAL: {key} persistent (consec={count})")

        return warnings, is_fatal


# ============================== AdaptiveAlpha (SAC-style trainable entropy coef)


class AdaptiveAlpha(nnx.Module):
    """
    Coefficient d'entropie appris (alpha = exp(log_alpha)).

    Loss : alpha_loss = -log_alpha * stop_gradient(H_target - H_obs)
        - Si H_obs > H_target  → log_alpha décroît → alpha baisse → moins de bonus
        - Si H_obs < H_target  → log_alpha augmente → alpha monte → plus de bonus

    log_alpha est clippé à [LOG_ALPHA_CLIP_MIN, LOG_ALPHA_CLIP_MAX] pour stabilité.
    """

    def __init__(self, init_log_alpha: float, *, rngs: nnx.Rngs = None):
        del rngs  # unused, kept for nnx interface symmetry
        self.log_alpha = nnx.Param(jnp.array(init_log_alpha, dtype=jnp.float32))

    def alpha(self) -> jax.Array:
        """Retourne alpha = exp(log_alpha) clippé."""
        clipped = jnp.clip(self.log_alpha[...], LOG_ALPHA_CLIP_MIN, LOG_ALPHA_CLIP_MAX)
        return jnp.exp(clipped)


# ============================== Helpers JAX


def compute_lambda_returns(
    rewards: jax.Array,
    values_next: jax.Array,
    continues: jax.Array,
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> jax.Array:
    """
    Lambda returns (TD-lambda / GAE) via jax.lax.scan en reverse.

    R_T = values_next[T]
    R_t = r_t + gamma * c_t * ((1 - lambda) * v_{t+1} + lambda * R_{t+1})

    Args:
        rewards     : (B, T)
        values_next : (B, T)  — value(s_{t+1}) du slow critic
        continues   : (B, T)  — proba(continue) prédite (déjà sigmoid)
        gamma       : facteur d'escompte
        lambda_     : trade-off Monte Carlo vs TD

    Returns:
        returns : (B, T)
    """
    # Transposer en (T, B) pour scan
    rewards_T = jnp.transpose(rewards, (1, 0))
    values_next_T = jnp.transpose(values_next, (1, 0))
    continues_T = jnp.transpose(continues, (1, 0))

    # R_final = values_next du dernier step
    R_final = values_next_T[-1]

    # Reverse scan : on parcourt t = T-1 → 0
    def step_fn(R_next, xs):
        r, v_next, c = xs
        R_t = r + gamma * c * ((1.0 - lambda_) * v_next + lambda_ * R_next)
        return R_t, R_t

    # Inverser les arrays pour le scan
    xs = (rewards_T[::-1], values_next_T[::-1], continues_T[::-1])
    _, returns_T_reversed = jax.lax.scan(step_fn, R_final, xs)

    # Un-reverse
    returns_T = returns_T_reversed[::-1]
    # Re-transposer en (B, T)
    return jnp.transpose(returns_T, (1, 0))


def imagine_trajectory(
    initial_state: dict,
    actor: Actor,
    rssm: RSSM,
    reward_head: RewardHead,
    continue_head: ContinueHead,
    horizon: int,
    action_dim: int,
    key: jax.Array,
) -> dict:
    """
    Rollout horizon steps imaginés dans le WM via jax.lax.scan.

    Args:
        initial_state : dict {h: (BT, h_dim), z: (BT, z_dim)}
        horizon       : nombre de steps imaginés (typiquement 16)
        key           : PRNGKey

    Returns:
        traj : dict avec
            states     : (BT, H, state_dim)  -- état avant le step
            next_states: (BT, H, state_dim)  -- état après le step
            actions    : (BT, H) int
            rewards    : (BT, H)             -- predicted reward
            continues  : (BT, H)             -- predicted P(continue)
            log_probs  : (BT, H)
            entropies  : (BT, H)
            last_state : (BT, state_dim)     -- dernier next_state
    """
    def step_fn(carry, _):
        state, key = carry
        state_vec = jnp.concatenate([state["h"], state["z"]], axis=-1)
        # FIX PHASE 2-ACTOR : stop_gradient sur state_vec avant l'actor.
        # Le gradient policy ne doit PAS remonter via le WM (sample STE × 16 steps
        # → bruit énorme dans le PG signal). Aligné avec officiel agent.py:196
        # (sg(imgfeat) qui enrobe toute l'imagination). C'est probablement LA
        # cause de l'actor stuck random observé jusqu'à v14.
        state_vec_sg = jax.lax.stop_gradient(state_vec)

        # Actor sample action (avec key séparée)
        key, subkey_a = jr.split(key)
        dist = actor.get_dist(state_vec_sg)  # sg : gradient seulement via params actor
        action_int = dist.sample(seed=subkey_a)
        log_prob = dist.log_prob(action_int)
        entropy = dist.entropy()
        action_oh = jax.nn.one_hot(action_int, action_dim)

        # Step RSSM (imagine)
        key, subkey_r = jr.split(key)
        new_state, _ = rssm.imagine_step(state, action_oh, subkey_r)
        new_state_vec = jnp.concatenate([new_state["h"], new_state["z"]], axis=-1)

        # REVERT de 058beff (vérifié verbatim sur danijar/dreamerv3@main, cf.
        # docs/HYPERPARAMS_COMPARISON.md) : la reward est prédite sur l'état d'ARRIVÉE
        # new_state_vec, celui que l'action vient de produire.
        #
        # danijar, rssm.py::imagine : `deter = self._core(carry['deter'], carry['stoch'],
        # actemb)` PUIS `feat = dict(deter=deter, ...)` → feat[t] est l'état APRÈS
        # action[t], et c'est sur feat que la reward head est appelée. La récompense
        # immédiate de l'advantage dépend donc de l'action créditée.
        #
        # 058beff avait déplacé ce calcul sur state_vec pour « cohérence » avec la
        # convention d'entraînement — mais c'est cette convention qui était fautive.
        # Les deux moitiés bougent ensemble : cible décalée (ci-dessus, dans
        # train_step_wm) + prédiction sur l'arrivée (ici). Avec les deux,
        # rewards[t] = r(s_t, a_t) et adv[t] = r(s_t,a_t) + γV(s_{t+1}) − V(s_t)
        # redevient le Bellman standard ; la λ-return d'Oneiro n'a besoin d'aucun
        # slicing (sa récurrence est déjà non décalée, là où danijar décale en interne
        # via `interm = rew[:, 1:]`).
        reward_pred = reward_head.predict(new_state_vec)
        # continue reste sur l'état de DÉPART : c_t = « continue après le step t »,
        # aligné sur la récurrence R_t = r_t + γ·c_t·(...). Cf. train_step_wm.
        continue_logit = continue_head(state_vec)
        continue_pred = jax.nn.sigmoid(continue_logit)

        out = {
            "state": state_vec,
            "next_state": new_state_vec,
            "action": action_int,
            "reward": reward_pred,
            "continue": continue_pred,
            "log_prob": log_prob,
            "entropy": entropy,
        }
        return (new_state, key), out

    init_carry = (initial_state, key)
    (_, _), traj = jax.lax.scan(step_fn, init_carry, None, length=horizon)

    # traj est en (H, BT, ...) — transposer en (BT, H, ...)
    # Pour les arrays 2D (H, BT) → (BT, H)
    # Pour les arrays 3D (H, BT, D) → (BT, H, D)
    def to_bt_first(x):
        if x.ndim == 1:
            return x  # scalaires globaux (rare ici)
        elif x.ndim == 2:
            return jnp.transpose(x, (1, 0))
        else:
            return jnp.transpose(x, (1, 0) + tuple(range(2, x.ndim)))

    traj = {k: to_bt_first(v) for k, v in traj.items()}
    last_state_vec = traj["next_state"][:, -1, :]
    traj["last_state"] = last_state_vec
    # Renommer pour cohérence avec PyTorch
    traj["states"] = traj.pop("state")
    traj["next_states"] = traj.pop("next_state")
    traj["actions"] = traj.pop("action")
    traj["rewards"] = traj.pop("reward")
    traj["continues"] = traj.pop("continue")
    traj["log_probs"] = traj.pop("log_prob")
    traj["entropies"] = traj.pop("entropy")
    return traj


# ============================== Train steps jit


def train_step_wm(
    encoder: CNNEncoder,
    rssm: RSSM,
    decoder: CNNDecoder,
    reward_head: RewardHead,
    continue_head: ContinueHead,
    optimizer: nnx.Optimizer,
    batch: dict,
    key: jax.Array,
) -> dict:
    """
    Un train step du World Model (encoder + RSSM + decoder + reward + continue).

    Loss : W_RECON * MSE(decoded, obs) + W_KL * KL + W_REWARD * twohot_CE + W_CONTINUE * BCE

    NOTE: mute en place les modules + l'optimizer (Flax NNX style).

    Returns:
        metrics : dict de scalaires
    """
    action_dim = rssm.action_dim

    def loss_fn(encoder, rssm, decoder, reward_head, continue_head):
        obs_seq = batch["obs"]                                 # (B, T, C, H, W)
        actions_int = batch["actions"]                         # (B, T)
        rewards = batch["rewards"]                             # (B, T)
        dones = batch["dones"]                                 # (B, T)

        actions_oh = jax.nn.one_hot(actions_int, action_dim)

        # Encoder + RSSM observe
        embeddings = encoder(obs_seq)                          # (B, T, embed_dim)
        rssm_out = rssm.observe_sequence(
            embeddings, actions_oh, dones=dones, key=key,
        )
        state_vec = jnp.concatenate(
            [rssm_out["h"], rssm_out["z"]], axis=-1,
        )                                                       # (B, T, state_dim)

        # Decoder
        decoded = decoder(state_vec)                           # (B, T, C, H, W)
        # Recon loss : sum sur pixels (C, H, W), mean sur batch + time
        # (aligné officiel dreamerv3/rssm.py:354-356 qui fait sum sur pixels)
        err = (decoded - obs_seq) ** 2  # (B, T, C, H, W)
        loss_recon = err.sum(axis=(-3, -2, -1)).mean()

        # KL
        loss_kl = RSSM.kl_loss(
            rssm_out["post_logits"], rssm_out["prior_logits"],
            free_bits=FREE_BITS, beta_dyn=BETA_DYN, beta_rep=BETA_REP,
        )

        # Reward — convention ENTRANTE (alignée danijar, cf. H_316/H_317).
        #
        # L'index t porte la récompense reçue en ARRIVANT sur obs_t, soit rewards[t-1].
        # POURQUOI : `state_vec[t]` est construit depuis obs_t et a_{t-1} (prev_action
        # décalé, rssm.py:349-352) — il ne contient JAMAIS a_t. Sous la convention
        # sortante (cible = r(obs_t, a_t)) la cible n'était donc pas une fonction de
        # l'entrée : l'optimum de Bayes était Σ_a π(a|s)·r(s,a), la *propension de la
        # politique courante*, quantité non stationnaire. Mesuré : la politique classait
        # `do` 17e/17 sur les états où `do` rapporte du bois (contrôle : 3e/17).
        # Avec la cible décalée, r(obs_{t-1}, a_{t-1}) est déterminée par (obs_t, a_{t-1}),
        # tous deux dans l'état → cible identifiable.
        #
        # Le masque annule la frontière d'épisode : si dones[t-1]=1, obs_t est une obs de
        # reset (l'env auto-reset à la collecte) et aucune récompense ne lui est imputable.
        # observe_sequence remet déjà h/z/prev_action à zéro au même endroit.
        def _shift_right(x):
            return jnp.concatenate([jnp.zeros_like(x[:, :1]), x[:, :-1]], axis=1)

        _not_episode_start = 1.0 - _shift_right(dones).astype(jnp.float32)
        rewards_in = _shift_right(rewards) * _not_episode_start

        loss_reward = reward_head.loss(state_vec, rewards_in, rare_weight=REWARD_RARE_WEIGHT)
        # Diagnostic (hors gradient) de la reward head, en 2 populations séparées.
        # Le seuil 0.01 mélangeait santé (±0.1, fréquente) et achievements (+1, rares) :
        # la moyenne noyait le signal. On isole donc les achievements à |r| > 0.5.
        #   rew_pred_ach : prédiction sur les ACHIEVEMENTS (true=+1). Sans fix ~0.05-0.3 ;
        #                  cible avec fix ~0.7-1.0. C'est LE critère (indépendant du bruit d'éval).
        #   rew_pred_zero: prédiction sur les états à reward NUL. Doit rester ~0.
        #                  S'il monte → la head hallucine du reward (rare_weight trop fort).
        # /!\ Les masques portent sur rewards_in (la cible réelle) : sur `rewards` ils
        # décriraient une autre population que celle qu'on entraîne → rew@ach mentirait.
        # NOTE : ce diagnostic reste IN-SAMPLE (calculé sur le batch qu'on fitte). Mesuré
        # hors échantillon, rare_weight=10 et =1 donnent la même valeur (0.40) — ne pas
        # lire rew@ach comme une capacité de généralisation.
        _rew_pred = reward_head.predict(state_vec)
        _ach = (jnp.abs(rewards_in) > 0.5).astype(jnp.float32)
        _zero = (jnp.abs(rewards_in) <= 0.01).astype(jnp.float32)
        rew_pred_ach = (_rew_pred * _ach).sum() / jnp.maximum(_ach.sum(), 1.0)
        rew_true_ach = (rewards_in * _ach).sum() / jnp.maximum(_ach.sum(), 1.0)
        rew_pred_zero = (_rew_pred * _zero).sum() / jnp.maximum(_zero.sum(), 1.0)
        rew_n_ach = _ach.sum()

        # Continue (BCE) — cible NON décalée, volontairement.
        # continue_head(s_t) ≈ 1-dones[t] signifie « l'épisode continue après le step t »,
        # ce qui est exactement le c_t de la récurrence λ-return
        # (R_t = r_t + γ·c_t·(...)). Décaler cette cible désalignerait c_t d'un cran et
        # casserait discount_cum. Seule la reward change de convention.
        continue_target = 1.0 - dones.astype(jnp.float32)
        loss_continue = continue_head.loss(state_vec, continue_target)

        loss_wm = (
            W_RECON * loss_recon + W_KL * loss_kl
            + W_REWARD * loss_reward + W_CONTINUE * loss_continue
        )
        aux = {
            "loss_wm": loss_wm,
            "loss_recon": loss_recon,
            "loss_kl": loss_kl,
            "loss_reward": loss_reward,
            "loss_continue": loss_continue,
            "rew_pred_ach": rew_pred_ach,
            "rew_true_ach": rew_true_ach,
            "rew_pred_zero": rew_pred_zero,
            "rew_n_ach": rew_n_ach,
        }
        return loss_wm, aux

    # value_and_grad with respect to all 5 modules
    grad_fn = nnx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4), has_aux=True)
    (loss_val, aux), grads = grad_fn(encoder, rssm, decoder, reward_head, continue_head)

    # Update via optimizer commun qui regroupe les 5 modules
    # Notre optimizer est un nnx.Optimizer attaché au "WM bundle"
    # On utilise le pattern : optimizer.update(model_tuple, grads_tuple)
    optimizer.update((encoder, rssm, decoder, reward_head, continue_head), grads)

    return aux


def train_step_ac(
    encoder: CNNEncoder,
    rssm: RSSM,
    reward_head: RewardHead,
    continue_head: ContinueHead,
    actor: Actor,
    critic: Critic,
    slow_critic: Critic,
    optimizer_ac: nnx.Optimizer,
    batch: dict,
    return_ema_std: jax.Array,
    key: jax.Array,
    entropy_coef: float,
    effective_alpha: jax.Array = None,
    actor_coef: jax.Array = None,
) -> tuple:
    """
    Un train step Actor + Critic via imagination dans le WM.

    actor_coef : scalaire jnp (défaut 1.0). Multiplie la loss actor (PG+entropy).
      0.0 = AC WARMUP : le CRITIC apprend les values (et l'EMA du scale chauffe)
      pendant que l'actor reste gelé (gradients nuls). Évite le double piège du
      démarrage : (a) l'actor se verrouille sur les returns bruités d'un WM nul
      (rich-get-richer, H 2.5→0.1 en <200 iters sur les actors >=1024) ; (b) au
      dégel, un critic à zéro rendrait tous les advantages positifs → re-collapse.

    Étapes :
      1. Encode batch + RSSM observe → initial states (stop_gradient, WM gelé).
      2. Imagine horizon=16 dans le WM avec actor courant.
      3. Lambda returns avec slow_critic comme bootstrap.
      4. Loss actor (PG + entropy) + loss critic (twohot CE).
      5. Polyak update slow_critic.

    Args:
        return_ema_std : jax.Array shape (2,) contenant [p5_ema, p95_ema]
                         (Percentile EMA aligné PyTorch baseline)

    Returns:
        (metrics dict, new_return_ema_std jax.Array shape (2,))
    """
    action_dim = rssm.action_dim
    key, subk_obs, subk_img = jr.split(key, 3)

    # ============== (1) Encode batch (stop_gradient, WM gelé pour l'AC train)
    obs_seq = batch["obs"]
    actions_int = batch["actions"]
    dones = batch["dones"]
    actions_oh = jax.nn.one_hot(actions_int, action_dim)

    embeddings = encoder(obs_seq)
    rssm_out_init = rssm.observe_sequence(
        embeddings, actions_oh, dones=dones, key=subk_obs,
    )
    # Detach via stop_gradient (le WM ne doit pas être trained pendant AC)
    h_init_flat = jax.lax.stop_gradient(rssm_out_init["h"]).reshape(-1, rssm.h_dim)
    z_init_flat = jax.lax.stop_gradient(rssm_out_init["z"]).reshape(-1, rssm.z_dim)
    initial_state = {"h": h_init_flat, "z": z_init_flat}

    # ============== (2) Imagine + (3) returns + (4) loss : tout dans value_and_grad
    def ac_loss_fn(actor, critic):
        # Imagine (avec gradient sur actor uniquement — rssm/heads gelés via stop_grad)
        # Note : on doit stopper le gradient sur rssm/heads à l'intérieur.
        # Pour simplifier, on prend le gradient sur actor+critic seulement,
        # et on stop_grad les outputs des heads/rssm dans la trajectoire.
        traj = imagine_trajectory(
            initial_state, actor, rssm, reward_head, continue_head,
            IMAGINATION_HORIZON, action_dim, subk_img,
        )

        states_traj = traj["states"]              # (BT, H, state_dim) — pour critic loss
        next_states_traj = traj["next_states"]    # (BT, H, state_dim) — pour values_next (target)

        # Stop gradient sur les rewards/continues/states issus du WM
        # (le gradient passe via log_probs/entropies de l'actor)
        rewards_pred = jax.lax.stop_gradient(traj["rewards"])
        continues_pred = jax.lax.stop_gradient(traj["continues"])
        states_traj_sg = jax.lax.stop_gradient(states_traj)
        next_states_sg = jax.lax.stop_gradient(next_states_traj)

        # Critic forward (vivant, gradient)
        values_pred = critic.predict(states_traj_sg)  # (BT, H)

        # FIX H_309 : bootstrap des lambda returns par le FAST critic (officiel),
        # pas le slow. Le slow (tau=0.98) retarde les values de ~50 iter → les
        # advantages d'un comportement déjà appris restent positifs longtemps →
        # sur-renforcement/verrouillage (pattern v18 : pic 4.0 puis oscillations).
        # Le slow critic ne sert que de régularisateur (slowreg, plus bas).
        # sg : c'est une target TD(λ), pas un chemin de gradient.
        values_next = jax.lax.stop_gradient(critic.predict(next_states_sg))  # (BT, H)

        # Lambda returns
        returns = compute_lambda_returns(
            rewards_pred, values_next, continues_pred,
            gamma=GAMMA, lambda_=LAMBDA_GAE,
        )
        returns = jax.lax.stop_gradient(returns)

        # Return scale (Percentile-EMA P5/P95, aligné PyTorch baseline)
        # return_ema_std est un array (2,) = [p5_ema, p95_ema]
        # On utilise les EMAs AVANT update pour le scale de cette iter
        p5_ema_prev = return_ema_std[0]
        p95_ema_prev = return_ema_std[1]
        # max(1, range) SANS cap supérieur (paper/officiel). Un cap (ex 5.0)
        # sur-amplifierait les advantages dès que les returns dépassent le cap.
        scale = jnp.maximum(p95_ema_prev - p5_ema_prev, 1.0)

        # Discount cumulatif
        gc = GAMMA * continues_pred  # (BT, H)
        # discount[t] = ∏_{k<t} gc[k]
        ones_col = jnp.ones_like(gc[:, :1])
        discount_cum = jnp.concatenate(
            [ones_col, jnp.cumprod(gc[:, :-1], axis=1)], axis=1,
        )
        discount_cum = jax.lax.stop_gradient(discount_cum)

        # Actor loss : -E[ discount * log_prob * advantage_normalisé ]
        advantages = (returns - jax.lax.stop_gradient(values_pred)) / scale
        loss_actor_pg = -jnp.mean(discount_cum * traj["log_probs"] * advantages)
        # effective_alpha : si fourni (mode adaptive), c'est jnp.array (alpha appris
        # × auto_explore_multiplier). Sinon on fallback sur entropy_coef constant.
        if effective_alpha is None:
            ent_coef_used = entropy_coef
        else:
            ent_coef_used = jax.lax.stop_gradient(effective_alpha)
        loss_actor_ent = -ent_coef_used * jnp.mean(discount_cum * traj["entropies"])
        loss_actor = loss_actor_pg + loss_actor_ent

        # Critic loss : twohot CE sur returns + slowreg paper DreamerV3
        # slowreg = CE(critic_logits, slow_critic_target) — régularise vers le slow critic
        # Coefficient 1.0 (paper default).
        loss_critic_main = critic.loss(states_traj_sg, returns)
        slow_critic_pred = slow_critic.predict(states_traj_sg)
        loss_critic_slowreg = critic.loss(
            states_traj_sg, jax.lax.stop_gradient(slow_critic_pred)
        )
        loss_critic = loss_critic_main + 1.0 * loss_critic_slowreg

        # actor_coef=0.0 (warmup) : seul le critic reçoit du gradient.
        ac_coef = 1.0 if actor_coef is None else actor_coef
        loss_ac = ac_coef * loss_actor + loss_critic

        mean_H = jnp.mean(traj["entropies"])
        # Percentiles du batch courant (pour update EMA hors gradient)
        flat_returns = returns.reshape(-1)
        p5_batch = jnp.quantile(flat_returns, RETURN_PERCENTILE_LOW)
        p95_batch = jnp.quantile(flat_returns, RETURN_PERCENTILE_HIGH)
        aux = {
            "loss_ac": loss_ac,
            "loss_actor": loss_actor,
            "loss_actor_pg": loss_actor_pg,
            "loss_actor_ent": loss_actor_ent,
            "loss_critic": loss_critic,
            "entropy": mean_H,
            "H": mean_H,
            "returns_mean": jnp.mean(returns),
            "returns_std": jnp.std(returns),
            "values_mean": jnp.mean(values_pred),
            "return_p5_batch": p5_batch,
            "return_p95_batch": p95_batch,
            "return_scale": scale,
            # Reward PRÉDIT dans l'imagination : c'est le SEUL signal que voit l'actor
            # (il ne voit jamais les vraies récompenses). Si img_rew_max reste ~0, aucune
            # trajectoire rêvée ne contient de récompense de taille achievement (+1) →
            # l'actor ne peut pas apprendre le craft, quel que soit rare_weight.
            "img_rew_mean": jnp.mean(rewards_pred),
            "img_rew_max": jnp.max(rewards_pred),
            "img_rew_p99": jnp.quantile(rewards_pred.reshape(-1), 0.99),
            "img_rew_frac_hi": jnp.mean((rewards_pred > 0.5).astype(jnp.float32)),
        }
        return loss_ac, aux

    grad_fn = nnx.value_and_grad(ac_loss_fn, argnums=(0, 1), has_aux=True)
    (loss_val, aux), grads = grad_fn(actor, critic)

    optimizer_ac.update((actor, critic), grads)

    # ============== (5) Polyak update slow_critic ← τ * slow + (1-τ) * critic
    critic_params = nnx.state(critic, nnx.Param)
    slow_params = nnx.state(slow_critic, nnx.Param)
    new_slow_params = jax.tree.map(
        lambda s, f: CRITIC_TARGET_TAU * s + (1.0 - CRITIC_TARGET_TAU) * f,
        slow_params, critic_params,
    )
    nnx.update(slow_critic, new_slow_params)

    # Update return Percentile-EMA P5/P95 (sortie de loss → carry).
    # Aligné PyTorch baseline (train_dreamer.py lignes 626-640) :
    #   p5_ema, p95_ema mis à jour avec EMA decay sur quantiles du batch.
    p5_batch = aux["return_p5_batch"]
    p95_batch = aux["return_p95_batch"]
    new_p5_ema = (
        RETURN_EMA_DECAY * return_ema_std[0] + (1.0 - RETURN_EMA_DECAY) * p5_batch
    )
    new_p95_ema = (
        RETURN_EMA_DECAY * return_ema_std[1] + (1.0 - RETURN_EMA_DECAY) * p95_batch
    )
    new_return_ema_std = jnp.stack([new_p5_ema, new_p95_ema])

    return aux, new_return_ema_std


# ============================== JIT wrappers


# nnx.jit avec modules + optimizer en arguments (legacy, gardés pour debug)
train_step_wm_jit = nnx.jit(train_step_wm)
train_step_ac_jit = nnx.jit(
    train_step_ac, static_argnames=("entropy_coef",),
)


# ============================== Functional training loop (FIX 1)
#
# Pour réduire l'overhead Python de nnx.jit (traversal du graph d'objets à
# chaque appel), on utilise le pattern recommandé par Flax :
#   1. nnx.split() une fois HORS de la boucle → (graphdef, state)
#   2. Décorer une fn pure (graphdef, state, batch, key) → (new_state, metrics)
#      avec jax.jit (pas nnx.jit). Le graphdef est capturé en closure → statique.
#   3. nnx.merge() À L'INTÉRIEUR du jit pour reconstruire les modules.
#   4. nnx.split() à la fin pour récupérer le NEW state.
# Voir : https://flax.readthedocs.io/en/stable/guides/performance.html


def make_functional_train_steps(
    wm_bundle: tuple,
    opt_wm: nnx.Optimizer,
    ac_bundle: tuple,
    opt_ac: nnx.Optimizer,
    slow_critic: Critic,
    alpha_module: "AdaptiveAlpha" = None,
    opt_alpha: nnx.Optimizer = None,
    rnd_module: "RNDModule" = None,
    opt_rnd: nnx.Optimizer = None,
    repl_sharding=None,
):
    """
    Factory : crée les versions fonctionnelles jit-compilées des train steps.

    Capture les graphdefs en closure (statiques) → le jit n'inclut que la
    logique de traversée XLA, pas le Python du graph d'objets.

    Args:
        wm_bundle    : tuple (encoder, rssm, decoder, reward_head, continue_head)
        opt_wm       : optimizer NNX pour le WM
        ac_bundle    : tuple (actor, critic)
        opt_ac       : optimizer NNX pour AC
        slow_critic  : module Critic slow (target network, Polyak)

    Returns:
        dict avec :
            wm_state         : state initial du WM (modules + opt)
            ac_state         : state initial de AC (modules + opt)
            slow_state       : state initial du slow_critic
            train_wm_fn      : (wm_state, batch, key) → (new_wm_state, metrics)
            train_ac_fn      : (wm_state_frozen, ac_state, slow_state, batch, ema_std, key, ent_coef)
                                → (new_ac_state, new_slow_state, new_ema_std, metrics)
            merge_wm         : (wm_state) → (modules, optimizer) pour eval/save
            merge_ac         : (ac_state) → (actor, critic) pour eval/save
            merge_slow       : (slow_state) → slow_critic pour Polyak inspect
    """
    # Split WM : on inclut l'optimizer car son state (Adam m/v) est aussi mutable.
    # Le filtre `...` capture TOUS les Variables (Param, OptState, constantes
    # comme les bins du twohot loss, etc.) — équivalent de Everything.
    wm_graphdef, wm_state = nnx.split((wm_bundle, opt_wm), ...)
    ac_graphdef, ac_state = nnx.split((ac_bundle, opt_ac), ...)
    slow_graphdef, slow_state = nnx.split(slow_critic, ...)

    # Bundles optionnels (alpha, rnd). On split seulement si activés ; le
    # state None indique au code appelant de ne pas appeler la fonction
    # correspondante.
    alpha_graphdef = None
    alpha_state = None
    if alpha_module is not None and opt_alpha is not None:
        alpha_graphdef, alpha_state = nnx.split((alpha_module, opt_alpha), ...)

    rnd_graphdef = None
    rnd_state = None
    if rnd_module is not None and opt_rnd is not None:
        rnd_graphdef, rnd_state = nnx.split((rnd_module, opt_rnd), ...)

    # ----------- Data-parallel : out_shardings pour pinner les sorties.
    # Le state (params + Adam m/v) reste RÉPLIQUÉ en entrée ET en sortie. Avec
    # mesh actif + entrées bien shardées, jit propage souvent seul ; annoter
    # out_shardings (state répliqué) évite des re-placements surprises entre
    # itérations. Les batchs (entrée) sont shardés côté buffer (device_put).
    # repl_sharding=None → out_shardings=None = comportement par défaut
    # (non-régression : aucun changement quand le data-parallel est inactif).
    repl = repl_sharding  # alias court (None si DP off)

    # train_wm_fn : (new_wm_state[repl], metrics[repl])
    out_sh_wm = None if repl is None else (repl, repl)
    # train_ac_fn* : (new_ac_state[repl], new_slow_state[repl],
    #                 new_ema_std[repl], metrics[repl])
    out_sh_ac = None if repl is None else (repl, repl, repl, repl)
    # act_fn_functional : (new_state[repl: h/z petits, répliqués], actions[repl])
    out_sh_act = None if repl is None else (repl, repl)

    @partial(jax.jit, out_shardings=out_sh_wm)
    def train_wm_fn(wm_state, batch, key):
        # Reconstruct modules + optimizer dans le scope du jit (pure)
        wm_bundle_local, opt_wm_local = nnx.merge(wm_graphdef, wm_state)
        encoder, rssm, decoder, reward_head, continue_head = wm_bundle_local

        # Réutilise la fn déjà définie (in-place sur les modules locaux)
        metrics = train_step_wm(
            encoder, rssm, decoder, reward_head, continue_head,
            opt_wm_local, batch, key,
        )

        # Re-split pour récupérer le NEW state (avec params mis à jour)
        new_wm_state = nnx.split(
            (wm_bundle_local, opt_wm_local), ...,
        )[1]
        return new_wm_state, metrics

    # entropy_coef est un float Python : JAX cache le trace par valeur tant
    # qu'elle ne change pas (pas de re-trace dans la hot loop si --entropy_coef
    # est constant pour tout le run).
    @partial(jax.jit, out_shardings=out_sh_ac)
    def train_ac_fn_inner(
        wm_state, ac_state, slow_state,
        batch, return_ema_std, key, entropy_coef, actor_coef=None,
    ):
        # Reconstruct AC + slow_critic + WM (read-only pour AC) dans le scope du jit
        ac_bundle_local, opt_ac_local = nnx.merge(ac_graphdef, ac_state)
        actor_l, critic_l = ac_bundle_local
        slow_local = nnx.merge(slow_graphdef, slow_state)
        wm_bundle_local, _opt_wm_unused = nnx.merge(wm_graphdef, wm_state)
        encoder_l, rssm_l, _dec_unused, reward_l, continue_l = wm_bundle_local

        metrics, new_ema_std = train_step_ac(
            encoder_l, rssm_l, reward_l, continue_l,
            actor_l, critic_l, slow_local,
            opt_ac_local, batch, return_ema_std, key, entropy_coef,
            actor_coef=actor_coef,
        )

        # Re-split AC et slow_critic
        new_ac_state = nnx.split(
            (ac_bundle_local, opt_ac_local), ...,
        )[1]
        new_slow_state = nnx.split(slow_local, ...)[1]
        return new_ac_state, new_slow_state, new_ema_std, metrics

    # Expose en alias public (pour rester compatible avec l'API précédente)
    train_ac_fn = train_ac_fn_inner

    # ---- Variante adaptive : prend effective_alpha (alpha appris * auto_explore_mult)
    # en jax.Array. On la trace via le même graphdef que train_ac_fn_inner.
    @partial(jax.jit, out_shardings=out_sh_ac)
    def train_ac_fn_adaptive(
        wm_state, ac_state, slow_state,
        batch, return_ema_std, key, effective_alpha, actor_coef=None,
    ):
        ac_bundle_local, opt_ac_local = nnx.merge(ac_graphdef, ac_state)
        actor_l, critic_l = ac_bundle_local
        slow_local = nnx.merge(slow_graphdef, slow_state)
        wm_bundle_local, _opt_wm_unused = nnx.merge(wm_graphdef, wm_state)
        encoder_l, rssm_l, _dec_unused, reward_l, continue_l = wm_bundle_local

        metrics, new_ema_std = train_step_ac(
            encoder_l, rssm_l, reward_l, continue_l,
            actor_l, critic_l, slow_local,
            opt_ac_local, batch, return_ema_std, key,
            entropy_coef=0.0,  # ignoré quand effective_alpha est fourni
            effective_alpha=effective_alpha,
            actor_coef=actor_coef,
        )

        new_ac_state = nnx.split((ac_bundle_local, opt_ac_local), ...)[1]
        new_slow_state = nnx.split(slow_local, ...)[1]
        return new_ac_state, new_slow_state, new_ema_std, metrics

    # ---- Alpha train step : mise à jour de log_alpha vers H_target.
    # alpha_loss = -log_alpha * stop_gradient(H_target - H_obs)
    # gradient : d/d log_alpha = -(H_target - H_obs)
    #   - H < target → grad < 0 (descente → log_alpha monte) → alpha augmente
    #   - H > target → grad > 0 (descente → log_alpha baisse) → alpha diminue
    if alpha_graphdef is not None:
        @jax.jit
        def train_alpha_fn(alpha_state, mean_H, h_target):
            # nnx.merge retourne ce qui avait été splitté → tuple (alpha_mod, opt)
            alpha_mod_local, opt_alpha_local = nnx.merge(alpha_graphdef, alpha_state)

            def alpha_loss_fn(am):
                log_a = am.log_alpha[...]
                # H_obs en stop_gradient, on backward via log_alpha uniquement
                return -log_a * jax.lax.stop_gradient(h_target - mean_H)

            grad_fn = nnx.value_and_grad(alpha_loss_fn)
            loss_val, grads = grad_fn(alpha_mod_local)
            opt_alpha_local.update(alpha_mod_local, grads)

            new_alpha_state = nnx.split((alpha_mod_local, opt_alpha_local), ...)[1]
            cur_alpha = alpha_mod_local.alpha()
            return new_alpha_state, {"alpha": cur_alpha, "alpha_loss": loss_val}
    else:
        train_alpha_fn = None

    # ---- RND train step : entraîne le predictor (target frozen via stop_gradient).
    if rnd_graphdef is not None:
        @jax.jit
        def train_rnd_fn(rnd_state, obs_batch):
            rnd_mod_local, opt_rnd_local = nnx.merge(rnd_graphdef, rnd_state)

            def rnd_loss_fn(rm):
                return rm.train_loss(obs_batch)

            grad_fn = nnx.value_and_grad(rnd_loss_fn)
            loss_val, grads = grad_fn(rnd_mod_local)
            opt_rnd_local.update(rnd_mod_local, grads)

            new_rnd_state = nnx.split((rnd_mod_local, opt_rnd_local), ...)[1]
            return new_rnd_state, {"loss_rnd": loss_val}

        # Bonus calculé via la version functional (mute running stats EMA)
        @jax.jit
        def rnd_bonus_fn(rnd_state, obs_batch):
            rnd_mod_local, opt_rnd_local = nnx.merge(rnd_graphdef, rnd_state)
            bonus_raw = rnd_mod_local.compute_bonus(obs_batch)
            bonus_norm = rnd_mod_local.normalize_bonus(bonus_raw)
            new_rnd_state = nnx.split((rnd_mod_local, opt_rnd_local), ...)[1]
            return new_rnd_state, bonus_norm
    else:
        train_rnd_fn = None
        rnd_bonus_fn = None

    def merge_wm(wm_state):
        return nnx.merge(wm_graphdef, wm_state)

    def merge_ac(ac_state):
        return nnx.merge(ac_graphdef, ac_state)

    def merge_slow(slow_state):
        return nnx.merge(slow_graphdef, slow_state)

    # Version functional de act_fn : prend les states au lieu des modules.
    # Permet d'utiliser les params à jour SANS muter les modules originaux.
    # Data-parallel : la COLLECTE n'est PAS shardée (env CPU séquentiel, batch =
    # n_envs souvent non divisible par device_count). On garde state + I/O
    # RÉPLIQUÉS (out_shardings=repl) : cohérent avec les states répliqués, pas
    # de split du petit batch n_envs.
    @partial(jax.jit, out_shardings=out_sh_act)
    def act_fn_functional(
        wm_state, ac_state,
        prev_state, prev_actions_oh, obs_batch, key, unimix=0.01,
    ):
        """unimix : plancher d'exploration de la COLLECTE (jnp scalaire → pas de re-trace).

        L'actor de DreamerV3 s'entraîne uniquement dans l'IMAGINATION, donc relever
        l'unimix ici ne touche AUCUN gradient — c'est purement la politique de
        comportement. Mesuré sur v50 : avec unimix=0.01 et une politique à 1.3 action
        effective /17, `place_table` n'est tentée que 0.10% des steps (57× moins qu'un
        agent random qui, lui, réussit 8% du temps) alors que l'agent A le bois requis
        dans 6.7% des épisodes → zéro table sur 6236 épisodes → aucun gradient ne peut
        jamais amorcer la chaîne de craft. Le schedule décroissant amorce tôt, puis
        rend la main à la politique apprise (retour au canonique 0.01).
        """
        wm_bundle_local, _opt_unused = nnx.merge(wm_graphdef, wm_state)
        ac_bundle_local, _opt_unused2 = nnx.merge(ac_graphdef, ac_state)
        encoder_l, rssm_l, _dec, _r, _c = wm_bundle_local
        actor_l, _critic_l = ac_bundle_local

        k_obs, k_act = jr.split(key)
        emb = encoder_l(obs_batch)
        new_state, _, _ = rssm_l.observe_step(prev_state, prev_actions_oh, emb, k_obs)
        state_vec = jnp.concatenate([new_state["h"], new_state["z"]], axis=-1)
        dist = actor_l.get_dist(state_vec, mask=None, unimix=unimix)
        actions_int = dist.sample(seed=k_act)
        return new_state, actions_int

    def merge_alpha(alpha_state_local):
        if alpha_graphdef is None:
            return None
        return nnx.merge(alpha_graphdef, alpha_state_local)

    def merge_rnd(rnd_state_local):
        if rnd_graphdef is None:
            return None
        return nnx.merge(rnd_graphdef, rnd_state_local)

    return {
        "wm_state": wm_state,
        "ac_state": ac_state,
        "slow_state": slow_state,
        "alpha_state": alpha_state,
        "rnd_state": rnd_state,
        "train_wm_fn": train_wm_fn,
        "train_ac_fn": train_ac_fn,
        "train_ac_fn_adaptive": train_ac_fn_adaptive,
        "train_alpha_fn": train_alpha_fn,
        "train_rnd_fn": train_rnd_fn,
        "rnd_bonus_fn": rnd_bonus_fn,
        "merge_wm": merge_wm,
        "merge_ac": merge_ac,
        "merge_slow": merge_slow,
        "merge_alpha": merge_alpha,
        "merge_rnd": merge_rnd,
        "act_fn_functional": act_fn_functional,
    }


# ============================== Action sampling (jit, pour la collecte)


def make_act_fn():
    """
    Factory : retourne une fonction jit-compilée pour sampling d'actions
    pendant la collecte (batch N_ENVS).
    """
    @nnx.jit
    def act_fn(
        encoder: CNNEncoder,
        rssm: RSSM,
        actor: Actor,
        prev_state: dict,
        prev_actions_oh: jax.Array,
        obs_batch: jax.Array,
        key: jax.Array,
        mask: jax.Array = None,
    ):
        """
        Single-step : encode obs + observe RSSM + sample action.

        Args:
            obs_batch       : (N, C, H, W)
            prev_state      : dict (h, z) de shapes (N, h_dim), (N, z_dim)
            prev_actions_oh : (N, action_dim)
            key             : PRNGKey
            mask            : (N, action_dim) bool ou None

        Returns:
            new_state    : dict h/z
            actions_int  : (N,) int32
        """
        k_obs, k_act = jr.split(key)
        emb = encoder(obs_batch)
        new_state, _, _ = rssm.observe_step(prev_state, prev_actions_oh, emb, k_obs)
        state_vec = jnp.concatenate([new_state["h"], new_state["z"]], axis=-1)
        dist = actor.get_dist(state_vec, mask=mask)
        actions_int = dist.sample(seed=k_act)
        return new_state, actions_int

    return act_fn


# ============================== Eval (Python, séquentiel)


def crafter_score(ach_counts: dict, n_episodes: int) -> float:
    """
    Score Crafter OFFICIEL (Hafner 2021) : moyenne géométrique des success
    rates (en %) sur les 22 achievements, calculée sur les épisodes de
    TRAINING. S = exp(mean(ln(1 + s_i))) - 1. C'est la métrique des papers.
    """
    if n_episodes == 0:
        return 0.0
    import math
    rates = [100.0 * ach_counts.get(name, 0) / n_episodes for name in ACHIEVEMENTS]
    return math.exp(sum(math.log(1.0 + r) for r in rates) / len(rates)) - 1.0


def _format_eta(seconds: float) -> str:
    """Formate un temps restant en h/m/s compact (ex: 3h45, 12m, 45s)."""
    seconds = int(max(0.0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}"
    if m > 0:
        return f"{m}m{s:02d}"
    return f"{s}s"


def eval_agent(
    eval_env: CrafterEnv,
    encoder: CNNEncoder,
    rssm: RSSM,
    actor: Actor,
    action_dim: int,
    n_episodes: int = 10,
    n_sample_episodes: int = 5,
) -> dict:
    """
    Eval séquentiel en Python (pas jit), en deux modes :
      - argmax (déterministe) : métrique primaire, comparable aux runs passés
      - sample (stochastique) : diagnostic — une policy quasi-uniforme donne
        un argmax arbitraire ; le mode sample révèle les progrès plus tôt
        (et c'est le mode d'éval du paper).
    """
    scores, lengths, achievements = [], [], []
    achievements_sample = []
    # Compteur par achievement : combien d'épisodes l'ont débloqué (success rate)
    ach_counts = {name: 0 for name in ACHIEVEMENTS}

    @nnx.jit
    def eval_step_argmax(encoder, rssm, actor, prev_state, prev_action_oh, obs_batch, key):
        emb = encoder(obs_batch)
        new_state, _, _ = rssm.observe_step(prev_state, prev_action_oh, emb, key)
        state_vec = jnp.concatenate([new_state["h"], new_state["z"]], axis=-1)
        logits = actor(state_vec)
        action_int = jnp.argmax(logits, axis=-1)
        return new_state, action_int

    @nnx.jit
    def eval_step_sample(encoder, rssm, actor, prev_state, prev_action_oh, obs_batch, key):
        emb = encoder(obs_batch)
        key_obs, key_act = jr.split(key)
        new_state, _, _ = rssm.observe_step(prev_state, prev_action_oh, emb, key_obs)
        state_vec = jnp.concatenate([new_state["h"], new_state["z"]], axis=-1)
        dist = actor.get_dist(state_vec)
        action_int = dist.sample(seed=key_act)
        return new_state, action_int

    def run_episode(ep_seed, key_seed, step_fn):
        obs = eval_env.reset(seed=ep_seed)
        state = rssm.init_state(1)
        prev_action_oh = jnp.zeros((1, action_dim))
        ep_reward, ep_len = 0.0, 0
        done = False
        key = jr.PRNGKey(key_seed)
        while not done and ep_len < 500:
            obs_jax = jnp.array(obs)[None, ...]  # (1, C, H, W)
            key, subk = jr.split(key)
            state, action_int = step_fn(
                encoder, rssm, actor, state, prev_action_oh, obs_jax, subk,
            )
            a_int = int(action_int[0])
            obs, r, done, info = eval_env.step(a_int)
            prev_action_oh = jax.nn.one_hot(action_int, action_dim)
            ep_reward += r
            ep_len += 1
        return ep_reward, ep_len

    # ---- Mode argmax (métrique primaire + détail par achievement)
    for ep in range(n_episodes):
        ep_reward, ep_len = run_episode(10_000 + ep, 20_000 + ep, eval_step_argmax)
        scores.append(ep_reward)
        lengths.append(ep_len)
        achievements.append(eval_env.n_unlocked_episode)
        for name in eval_env.unlocked_names:
            if name in ach_counts:
                ach_counts[name] += 1

    # ---- Mode sample (diagnostic décollage)
    for ep in range(n_sample_episodes):
        run_episode(10_000 + ep, 30_000 + ep, eval_step_sample)
        achievements_sample.append(eval_env.n_unlocked_episode)

    # Taux de réussite par achievement (fraction des épisodes argmax)
    ach_rates = {
        name: count / max(1, n_episodes)
        for name, count in ach_counts.items()
        if count > 0
    }

    return {
        "score": float(np.mean(scores)),
        "length": float(np.mean(lengths)),
        "achievements": float(np.mean(achievements)),
        "achievements_sample": float(np.mean(achievements_sample)) if achievements_sample else 0.0,
        "achievements_detail": ach_rates,
    }


# ============================== Multiprocessing env pool


def _env_worker(seed: int, conn):
    """
    Worker process : tient un CrafterEnv et obéit aux commandes du parent.

    Protocole :
        ("reset", _)        -> envoie obs (np.float32, C,H,W)
        ("step", action)    -> envoie (next_obs, reward, done, ep_unlocked)
                               où ep_unlocked = liste des achievements de
                               l'épisode si done (capturés AVANT l'auto-reset,
                               pour le score Crafter officiel), sinon None
        ("close", _)        -> termine la boucle
    """
    # Imports locaux pour éviter de polluer le parent (et JAX peut être lent à import)
    import numpy as np  # noqa
    from crafter_dreamer.env import CrafterEnv

    env = CrafterEnv(seed=seed)
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "reset":
                obs = env.reset()
                conn.send(obs)
            elif cmd == "step":
                a = int(payload)
                next_obs, r, done, _ = env.step(a)
                ep_unlocked = None
                if done:
                    ep_unlocked = sorted(env.unlocked_names)
                    next_obs = env.reset()
                conn.send((next_obs, float(r), bool(done), ep_unlocked))
            elif cmd == "close":
                conn.close()
                return
            else:
                # Unknown command : ignore
                conn.send(None)
    except (EOFError, KeyboardInterrupt):
        return


class MultiprocEnvPool:
    """
    Pool de N processus, 1 env Crafter par worker.

    Le parent envoie les actions en parallèle (chaque pipe non-bloquant),
    puis collecte les réponses. Permet de paralléliser step() qui est CPU-bound.

    NOTE: utilise spawn (pas fork) pour compat JAX/CUDA. L'env Crafter est
    pickle-safe (opensimplex géré dans son côté).
    """

    def __init__(self, n_envs: int, base_seed: int = 0):
        import multiprocessing as mp
        # Spawn obligatoire si JAX/CUDA dans le parent (sinon fork casse CUDA)
        ctx = mp.get_context("spawn")
        self._ctx = ctx
        self.n_envs = n_envs
        self._procs = []
        self._conns = []
        for i in range(n_envs):
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(
                target=_env_worker,
                args=(base_seed + i, child_conn),
                daemon=True,
            )
            p.start()
            self._procs.append(p)
            self._conns.append(parent_conn)
        self._closed = False

    def reset_all(self):
        """Reset tous les envs en parallèle, retourne list[obs]."""
        for c in self._conns:
            c.send(("reset", None))
        return [c.recv() for c in self._conns]

    def step_all(self, actions):
        """
        Step tous les envs en parallèle.

        Args:
            actions : array-like (N,) int

        Returns:
            list de tuples (next_obs, reward, done), len = N
        """
        # Phase 1 : envoyer toutes les actions (non-bloquant)
        for c, a in zip(self._conns, actions):
            c.send(("step", int(a)))
        # Phase 2 : collecter les réponses
        return [c.recv() for c in self._conns]

    def close(self):
        if self._closed:
            return
        for c in self._conns:
            try:
                c.send(("close", None))
            except Exception:
                pass
        for p in self._procs:
            try:
                p.join(timeout=2.0)
            except Exception:
                pass
            if p.is_alive():
                p.terminate()
        self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ============================== Main


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--entropy_coef", type=float, default=ENTROPY_COEF)
    p.add_argument("--train_iter", type=int, default=TRAIN_ITERATIONS)
    p.add_argument("--eval_interval", type=int, default=EVAL_INTERVAL)
    p.add_argument("--run_name", type=str, default="default")
    p.add_argument("--n_envs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--wm_train_per_iter", type=int, default=WM_TRAIN_PER_ITER)
    p.add_argument("--ac_train_per_iter", type=int, default=AC_TRAIN_PER_ITER)
    p.add_argument("--unimix_init", type=float, default=0.01,
                   help="Plancher d'exploration de la COLLECTE au début du run (défaut 0.01 "
                        "= canonique DreamerV3). Relever (0.2-0.3) amorce les actions de craft "
                        "jamais tentées : mesuré sur v50, place_table est tentée 0.10%% des "
                        "steps (57x moins qu'un random qui réussit 8%%) alors que le bois requis "
                        "est là dans 6.7%% des épisodes → 0 table sur 6236 ép. → rien à apprendre. "
                        "N'affecte AUCUN gradient (l'actor s'entraîne dans l'imagination).")
    p.add_argument("--unimix_final", type=float, default=0.01,
                   help="Plancher d'exploration en fin de schedule (retour au canonique).")
    p.add_argument("--unimix_decay_iters", type=int, default=0,
                   help="Durée de la décroissance linéaire unimix_init → unimix_final. 0 = off.")
    p.add_argument("--ac_warmup_iters", type=int, default=0,
                   help="Gèle l'actor-critic pendant N itérations (le WM apprend seul, "
                        "collecte via l'actor uniforme). Protège du verrouillage de politique "
                        "au démarrage (H 2.5→0.1 en <200 iters sur les actors >=1024 quand "
                        "l'AC apprend sur un WM encore nul). 0 = off (comportement historique).")
    p.add_argument("--rare_weight", type=float, default=None,
                   help="Override de REWARD_RARE_WEIGHT (poids CE des achievements). "
                        "None = constante du fichier.")
    p.add_argument("--replay_priority_frac", type=float, default=0.0,
                   help="Fraction du batch AC forcée à contenir un achievement (reward>0.5). "
                        "Prioritized replay reward, AC SEULEMENT (le WM garde une distribution "
                        "non-biaisée). 0.0 = off. Ne cible pas un achievement précis (buffer ne "
                        "stocke que reward=+1) → tous achievements confondus.")
    # ---- Architecture (défauts = constantes du fichier). Exposée en CLI pour pouvoir
    # tester la capacité sans commit. NOTE : le ranking des tailles mesuré avant le fix
    # de convention (H_316) est caduc — le 65M perdait contre le 14M parce qu'il
    # exploitait mieux l'optimum local créé par le crédit inversé, pas parce que la
    # capacité nuisait. Repères : symoon11 WM=181.6M → 17.65% ; danijar XL deter 8192.
    p.add_argument("--embed_dim", type=int, default=EMBED_DIM,
                   help=f"Sortie du CNN encoder (défaut {EMBED_DIM}).")
    p.add_argument("--h_dim", type=int, default=H_DIM,
                   help=f"Taille du deter GRU (défaut {H_DIM} ; danijar 4096-8192).")
    p.add_argument("--z_categories", type=int, default=Z_CATEGORIES,
                   help=f"Nombre de variables catégorielles z (défaut {Z_CATEGORIES}).")
    p.add_argument("--z_classes", type=int, default=Z_CLASSES,
                   help=f"Classes par variable z (défaut {Z_CLASSES}).")
    p.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM,
                   help=f"Largeur MLP du RSSM + reward/continue heads (défaut {HIDDEN_DIM}).")
    p.add_argument("--cnn_depth", type=int, default=CNN_DEPTH,
                   help=f"Canaux de base du CNN encoder/decoder (défaut {CNN_DEPTH} ; "
                        "c'est le levier du decoder, où symoon11 est 5.2x plus gros).")
    p.add_argument("--ac_hidden_dim", type=int, default=AC_HIDDEN_DIM,
                   help=f"Largeur MLP actor/critic (défaut {AC_HIDDEN_DIM}).")
    p.add_argument("--ac_num_layers", type=int, default=AC_NUM_LAYERS,
                   help=f"Profondeur MLP actor/critic (défaut {AC_NUM_LAYERS}).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS)
    p.add_argument("--resume_warmup_steps", type=int, default=50_000,
                   help="Transitions collectées pour re-remplir le buffer à la reprise "
                        "(--resume_from), avec la politique CHARGÉE et non au hasard. Le "
                        "buffer n'étant pas sauvé (12.3 GB à 1M), c'est ce qui évite de "
                        "réentraîner un agent compétent sur une poignée de données "
                        "aléatoires. 50k ≈ quelques minutes de collecte.")
    p.add_argument("--profile", action="store_true",
                   help="Active le profiling fin (breakdown collect/WM/AC/transfer).")
    p.add_argument("--buffer_capacity", type=int, default=BUFFER_CAPACITY,
                   help=f"Taille du replay buffer (défaut {BUFFER_CAPACITY} = paper ; "
                        "~12.3GB à 1M. Réduire pour smoketests locaux).")
    p.add_argument("--buffer_device", choices=["gpu", "cpu"], default="gpu",
                   help="gpu : buffer en VRAM (rapide, défaut, besoin ~12.3GB VRAM à 1M). "
                        "cpu : buffer en RAM host, transfert du batch au sample "
                        "(défaut en production TPU, et seule option si la VRAM ne tient "
                        "pas les 12.3 GB ; -20 à -40%% ips).")
    p.add_argument("--resume_from", type=str, default=None,
                   help="Checkpoint .npz à charger pour reprendre le training "
                        "(reprend à l'iter du .meta.json). Le buffer repart du "
                        "warmup (caveat : premières centaines d'iter sur données fraîches).")
    p.add_argument("--mp_collect", action="store_true",
                   help="Active la collecte multiprocessing (1 worker = 1 env).")
    # OFF par défaut (audit) : aucune implémentation de référence n'utilise
    # d'alpha adaptatif — toutes utilisent entropy_coef FIXE 3e-4. L'adaptive
    # ne faisait que redescendre lentement vers cette valeur.
    p.add_argument("--adaptive_alpha", action="store_true", default=False,
                   help="Active l'adaptation auto du coefficient d'entropie (OFF par défaut).")
    p.add_argument("--no_adaptive_alpha", dest="adaptive_alpha", action="store_false",
                   help="Désactive adaptive_alpha (défaut : entropy_coef fixe).")
    p.add_argument("--alpha_init", type=float, default=INIT_ALPHA,
                   help=f"Valeur initiale d'alpha (défaut: {INIT_ALPHA}).")
    p.add_argument("--h_target", type=float, default=H_TARGET,
                   help=f"Entropy cible pour adaptive_alpha (défaut: {H_TARGET}).")
    # OFF par défaut : adaptive_alpha gère déjà l'entropie. Superposer
    # auto_explore (multiplier sur stagnation) ajoute un 3e levier d'entropie
    # qui se multiplie aux deux autres → signal illisible. À réactiver
    # explicitement seulement pour une expérience exploration dédiée.
    p.add_argument("--auto_explore", action="store_true", default=False,
                   help="Active le boost entropy en cas de stagnation EVAL (OFF par défaut).")
    p.add_argument("--no_auto_explore", dest="auto_explore", action="store_false",
                   help="Désactive auto_explore (défaut).")
    p.add_argument("--use_rnd", action="store_true", default=True,
                   help="Active RND pour exploration dirigée (bonus intrinsèque).")
    p.add_argument("--no_use_rnd", dest="use_rnd", action="store_false",
                   help="Désactive RND.")
    p.add_argument("--rnd_coef", type=float, default=RND_COEF,
                   help=f"Coef du bonus intrinsèque RND (défaut: {RND_COEF}).")
    p.add_argument("--rnd_anneal", action="store_true", default=False,
                   help="Annealing lineaire du rnd_coef vers 0 apres le warmup "
                        "(exploration tot, exploitation ensuite ; style Burda).")
    # Phase 13 : Safeguards auto-régulateurs
    # SAFEGUARD 1 : RND warmup linéaire (0 → rnd_coef sur N iter)
    p.add_argument("--rnd_warmup_steps", type=int, default=5000,
                   help="Itérations pour rampe linéaire de rnd_coef (0 → rnd_coef).")
    # SAFEGUARD 2 : RND coef adaptatif (régulation runtime via ratio bonus/extrinsic)
    p.add_argument("--adaptive_rnd", action="store_true", default=True,
                   help="Régulation runtime de rnd_coef pour maintenir ratio bonus/extrinsic in [0.1, 5].")
    p.add_argument("--no_adaptive_rnd", dest="adaptive_rnd", action="store_false",
                   help="Désactive adaptive_rnd (rnd_coef purement statique post-warmup).")
    # SAFEGUARD 3 : Health monitor + auto-stop
    p.add_argument("--health_auto_stop", action="store_true", default=True,
                   help="Auto-stop si pathologie fatale détectée.")
    p.add_argument("--no_health_auto_stop", dest="health_auto_stop", action="store_false",
                   help="Désactive l'auto-stop (warnings seulement).")
    p.add_argument("--health_consec_threshold", type=int, default=5,
                   help="N warnings consec sur même catégorie → fatal.")
    # SAFEGUARD 4 : Curriculum H_target (entropy decay)
    p.add_argument("--h_target_schedule", action="store_true", default=True,
                   help="Schedule linéaire de H_target (init → final sur decay_steps).")
    p.add_argument("--no_h_target_schedule", dest="h_target_schedule", action="store_false",
                   help="Désactive le schedule (H_target constant = --h_target).")
    p.add_argument("--h_target_init", type=float, default=1.13,
                   help="H_target initial (1.13 = 0.4 × log(17), aligné PyTorch baseline).")
    p.add_argument("--h_target_final", type=float, default=1.13,
                   help="H_target final (constant 1.13, paper).")
    p.add_argument("--h_target_decay_steps", type=int, default=10000,
                   help="Itérations pour décroître (no-op si init=final).")
    return p.parse_args()


class Profiler:
    """
    Profiler minimaliste : accumule des durées par section.
    Sections trackées : act_fn (jit sample action), env_step (env.step Python),
    transfer (jnp <-> np), buffer_add, train_wm, train_ac, sample_batch, other.
    """

    SECTIONS = (
        "act_fn", "env_step", "transfer", "buffer_add",
        "sample_batch", "train_wm", "train_ac", "other",
    )

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.totals = {s: 0.0 for s in self.SECTIONS}
        self.counts = {s: 0 for s in self.SECTIONS}
        self._stack = []

    def tic(self, section: str):
        if not self.enabled:
            return
        self._stack.append((section, time.perf_counter()))

    def toc(self):
        if not self.enabled:
            return
        section, t0 = self._stack.pop()
        dt = time.perf_counter() - t0
        # NOTE: sur GPU, JAX async -> il faudrait block_until_ready
        # mais comme on mesure des grosses sections sur CPU, c'est acceptable
        self.totals[section] += dt
        self.counts[section] += 1

    def report(self, total_elapsed: float):
        if not self.enabled:
            return
        print()
        print("=" * 60)
        print(f"PROFILING REPORT (total elapsed: {total_elapsed:.2f}s)")
        print("=" * 60)
        accounted = sum(self.totals.values())
        for s in self.SECTIONS:
            t = self.totals[s]
            n = self.counts[s]
            pct = (t / total_elapsed * 100.0) if total_elapsed > 0 else 0.0
            avg_ms = (t / n * 1000.0) if n > 0 else 0.0
            print(f"  {s:<14} : {t:>7.2f}s  ({pct:>5.1f}%)  "
                  f"calls={n:>5}  avg={avg_ms:>7.2f}ms")
        unaccounted = total_elapsed - accounted
        pct_un = (unaccounted / total_elapsed * 100.0) if total_elapsed > 0 else 0.0
        print(f"  {'unaccounted':<14} : {unaccounted:>7.2f}s  ({pct_un:>5.1f}%)")
        print("=" * 60)


def main():
    args = parse_args()

    # Override runtime de REWARD_RARE_WEIGHT (--rare_weight). Doit se faire AVANT
    # le premier trace jit de train_step_wm (qui capture la globale au trace).
    if getattr(args, "rare_weight", None) is not None:
        globals()["REWARD_RARE_WEIGHT"] = float(args.rare_weight)
        print(f"[override] REWARD_RARE_WEIGHT = {REWARD_RARE_WEIGHT}")

    # Si mp_collect : forcer spawn (compat JAX/CUDA). Doit être fait avant
    # toute création de process. Idempotent : ne fail pas si déjà set.
    if args.mp_collect:
        import multiprocessing as mp
        try:
            mp.set_start_method("spawn", force=False)
        except RuntimeError:
            pass  # déjà set

    seed = args.seed
    np.random.seed(seed)

    # ----------- Setup JAX
    backend = jax.default_backend()
    # train_ratio effectif = tokens entraînés / tokens collectés par iter
    # = (batch × seq_len × wm_train_per_iter) / (collect_per_iter × n_envs)
    collected_per_iter_total = max(1, COLLECT_PER_ITER)
    train_ratio_eff = (args.batch_size * SEQ_LEN * args.wm_train_per_iter) / collected_per_iter_total
    print(f"JAX backend : {backend}")
    print(f"Device      : {jax.devices()}")

    # ----------- Data-parallel : mesh 1D sur tous les devices
    # Stratégie : params + état optimiseur RÉPLIQUÉS, batch SHARDÉ sur l'axe
    # `data` (split de la dim 0 = B). Activation automatique : 1 device → mesh
    # trivial (non-régression totale), N devices → vrai data-parallel.
    #
    # IMPORTANT : axis_types=Auto (mode GSPMD). Le défaut de jax.make_mesh est
    # Explicit, qui REJETTE le lax.scan du RSSM / de l'imagination (carry input
    # non-shardé vs output shardé → TypeError). En mode Auto, XLA propage le
    # sharding data-parallel à travers le scan et agrège les .mean()/.sum() des
    # losses cross-shard tout seul — aucun lax.pmean à écrire.
    n_devices = jax.device_count()
    mesh = jax.make_mesh((n_devices,), ('data',), axis_types=(AxisType.Auto,))
    data_sharding = NamedSharding(mesh, P('data'))   # batch : dim 0 (B) shardée
    repl_sharding = NamedSharding(mesh, P())         # state : répliqué
    if n_devices > 1:
        print(f"Sharding    : DATA-PARALLEL sur {n_devices} devices "
              f"(mesh {mesh.shape}, axis 'data', batch dim0 shardé, state répliqué)")
    else:
        print(f"Sharding    : 1 device → mesh trivial (data-parallel inactif, non-régression)")

    print(f"Run         : {args.run_name}")
    print(f"Config      : entropy={args.entropy_coef}  train_iter={args.train_iter}  "
          f"n_envs={args.n_envs}  batch={args.batch_size}  seq_len={SEQ_LEN}")
    print(f"Optim       : lr_wm={LR_WM:.1e}  lr_ac={LR_AC:.1e}  lr_alpha={LR_ALPHA:.1e}  "
          f"grad_clip={GRAD_CLIP}  horizon={IMAGINATION_HORIZON}")
    print(f"Replay      : collect/iter={COLLECT_PER_ITER}  wm_train/iter={args.wm_train_per_iter}  "
          f"ac_train/iter={args.ac_train_per_iter}  → train_ratio≈{train_ratio_eff:.0f}")
    print(f"Flags       : adaptive_alpha={args.adaptive_alpha}  auto_explore={args.auto_explore}  "
          f"use_rnd={args.use_rnd}")
    print()

    # ----------- Setup envs
    n_envs = args.n_envs
    envs = [CrafterEnv(seed=seed + i) for i in range(n_envs)]
    obs_list = [env.reset() for env in envs]
    obs_shape = obs_list[0].shape  # (3, 64, 64)
    action_dim = envs[0].action_dim  # 17

    eval_env = CrafterEnv(seed=seed + 9999)

    # Buffer per-env : les séquences RSSM doivent être des trajectoires d'UN
    # seul env (fix bug interleaving : avant, chaque séquence de 64 steps
    # changeait d'env à chaque step → dynamique fictive apprise par le prior).
    BufferClass = ImageReplayBufferJAX if args.buffer_device == "gpu" else ImageReplayBufferCPU
    buffer = BufferClass(
        capacity=args.buffer_capacity, obs_shape=obs_shape, n_envs=n_envs,
        data_sharding=data_sharding,
    )
    print(f"Buffer      : {args.buffer_device.upper()} "
          f"({args.buffer_capacity:,} cap, {buffer.memory_usage_mb():.0f} MB)")

    # ----------- Setup models
    # Dimensions prises depuis args (défauts = les constantes du fichier) : permet de
    # varier l'architecture sans commit, donc de garder la discipline « une variable
    # par run » sur les expériences de capacité.
    embed_dim = args.embed_dim
    h_dim = args.h_dim
    hidden_dim = args.hidden_dim
    cnn_depth = args.cnn_depth
    ac_hidden = args.ac_hidden_dim
    ac_layers = args.ac_num_layers

    rngs = nnx.Rngs(seed)
    encoder = CNNEncoder(in_channels=3, embed_dim=embed_dim, base_channels=cnn_depth, rngs=rngs)
    rssm = RSSM(
        embed_dim=embed_dim, action_dim=action_dim,
        h_dim=h_dim, z_categories=args.z_categories, z_classes=args.z_classes,
        hidden_dim=hidden_dim, rngs=rngs,
    )
    decoder = CNNDecoder(state_dim=rssm.state_dim, out_channels=3, base_channels=cnn_depth, rngs=rngs)
    reward_head = RewardHead(state_dim=rssm.state_dim, hidden_dim=hidden_dim, rngs=rngs)
    continue_head = ContinueHead(state_dim=rssm.state_dim, hidden_dim=hidden_dim, rngs=rngs)
    actor = Actor(
        state_dim=rssm.state_dim, hidden_dim=ac_hidden, action_dim=action_dim,
        num_layers=ac_layers, rngs=rngs,
    )
    critic = Critic(
        state_dim=rssm.state_dim, hidden_dim=ac_hidden,
        num_layers=ac_layers, rngs=rngs,
    )

    # Slow critic : copie initiale du critic
    slow_critic = Critic(
        state_dim=rssm.state_dim, hidden_dim=ac_hidden,
        num_layers=ac_layers, rngs=nnx.Rngs(seed + 100),
    )
    nnx.update(slow_critic, nnx.state(critic, nnx.Param))

    # ----------- Adaptive alpha + RND (Phase 12)
    if args.adaptive_alpha:
        alpha_module = AdaptiveAlpha(init_log_alpha=math.log(args.alpha_init))
        opt_alpha_tx = optax.adam(LR_ALPHA)
        opt_alpha = nnx.Optimizer(alpha_module, opt_alpha_tx, wrt=nnx.Param)
        print(f"Adaptive α : ON   H_target={args.h_target:.3f}   init α={args.alpha_init:.4f}   lr_α={LR_ALPHA}")
    else:
        alpha_module = None
        opt_alpha = None
        print(f"Adaptive α : OFF (entropy_coef fixe = {args.entropy_coef})")

    if args.use_rnd:
        rnd_module = RNDModule(
            in_channels=3, embed_dim=EMBED_DIM, base_channels=32,
            input_resolution=64,
            rngs=nnx.Rngs(seed + 200),
        )
        tx_rnd = optax.chain(
            optax.clip_by_global_norm(GRAD_CLIP),
            optax.adam(LR_RND),
        )
        opt_rnd = nnx.Optimizer(rnd_module, tx_rnd, wrt=nnx.Param)
        print(f"RND        : ON   rnd_coef={args.rnd_coef}   lr_rnd={LR_RND}")
    else:
        rnd_module = None
        opt_rnd = None
        print(f"RND        : OFF")

    if args.auto_explore:
        print(f"Auto-explore : ON  (threshold={AUTO_EXPLORE_THRESHOLD}, patience={AUTO_EXPLORE_PATIENCE}, max={AUTO_EXPLORE_MAX})")
    else:
        print(f"Auto-explore : OFF")
    print()

    # Param counts
    def count_params(m):
        params = nnx.state(m, nnx.Param)
        return int(sum(p.size for p in jax.tree.leaves(params)))

    n_wm = sum(count_params(m) for m in [encoder, rssm, decoder, reward_head, continue_head])
    n_ac = sum(count_params(m) for m in [actor, critic])

    print("=" * 60)
    print("Architecture")
    print("=" * 60)
    print(f"  WM modules (CNNEnc+RSSM+CNNDec+heads) : {n_wm:>9,} params")
    print(f"  Actor + Critic                        : {n_ac:>9,} params")
    print(f"  TOTAL                                 : {n_wm + n_ac:>9,} params (~{(n_wm + n_ac)/1e6:.2f}M)")
    print()

    # ----------- Optimizers : on group les modules en tuples pour partager un opt
    # WM bundle : encoder + rssm + decoder + reward_head + continue_head
    wm_bundle = (encoder, rssm, decoder, reward_head, continue_head)
    tx_wm = optax.apply_if_finite(optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP_WM),
        optax.adam(LR_WM),
    ), max_consecutive_errors=10)
    opt_wm = nnx.Optimizer(wm_bundle, tx_wm, wrt=nnx.Param)

    # AC bundle : actor + critic
    ac_bundle = (actor, critic)
    tx_ac = optax.apply_if_finite(optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP_AC),
        optax.adam(LR_AC),
    ), max_consecutive_errors=10)
    opt_ac = nnx.Optimizer(ac_bundle, tx_ac, wrt=nnx.Param)

    # Return normalization : Percentile-EMA P5/P95 (DreamerV3 canonique)
    # Aligné PyTorch baseline. Stocké sous forme array (2,) = [p5_ema, p95_ema].
    # Init étroit (-0.05, 0.05) → scale initial clamped à 1.0 (no scaling early,
    # normal car les returns sont petits au début) → EMA converge sur premiers batches.
    return_ema_std = jnp.array([-0.05, 0.05], dtype=jnp.float32)

    # ----------- Resume : charger un checkpoint AVANT le split functional
    # (les Variables nnx sont partagées : muter les modules ici suffit,
    # le nnx.split de make_functional_train_steps capturera les poids chargés).
    start_iter = 0
    if args.resume_from:
        start_iter = load_checkpoint_into_modules(args.resume_from, {
            "encoder": encoder, "rssm": rssm, "decoder": decoder,
            "reward_head": reward_head, "continue_head": continue_head,
            "actor": actor, "critic": critic, "slow_critic": slow_critic,
        })
        if start_iter >= args.train_iter:
            print(f"[resume] iter {start_iter} >= train_iter {args.train_iter} : rien à faire.")
            return
        # Restaure le scale EMA (P5/P95) s'il est présent dans le checkpoint. C'était LE
        # piège de la reprise : sans lui, scale = max(p95-p5, 1.0) repart à ~1.0 alors
        # qu'un run mature tourne à 5-8 → advantages gonflés d'autant sur plusieurs
        # centaines d'itérations (EMA decay 0.99), juste après la reprise.
        # Etat Adam (mu/nu/count) des deux optimizers.
        load_optimizer_state(args.resume_from, {"__opt_wm": opt_wm, "__opt_ac": opt_ac})

        _ck = np.load(args.resume_from, allow_pickle=False)
        if "__return_ema_std" in _ck.files:
            return_ema_std = jnp.asarray(_ck["__return_ema_std"])
            _p5, _p95 = float(return_ema_std[0]), float(return_ema_std[1])
            print(f"[resume] scale EMA restauré : p5={_p5:.2f} p95={_p95:.2f} "
                  f"→ scale={max(_p95 - _p5, 1.0):.2f}")
        else:
            print("[resume] /!\\ pas de __return_ema_std dans ce checkpoint (format ancien) : "
                  "le scale repart de son init → advantages gonflés au démarrage.")
        # Caveat restant : la PRNG key repart de zéro (sans effet sur la qualité, seule
        # la reproductibilité bit-a-bit est perdue). Le buffer, lui, est re-rempli par un
        # warmup ON-POLICY — cf. Phase 0 plus bas.

    # ----------- FIX 1 : Functional training loop
    # Split modules + optimizers en (graphdef, state). Le graphdef est capturé
    # en closure dans les fonctions jit, ce qui supprime l'overhead Python du
    # graph traversal NNX à chaque appel (gain estimé 30-50% sur GPU).
    functional = make_functional_train_steps(
        wm_bundle, opt_wm, ac_bundle, opt_ac, slow_critic,
        alpha_module=alpha_module, opt_alpha=opt_alpha,
        rnd_module=rnd_module, opt_rnd=opt_rnd,
        repl_sharding=repl_sharding,
    )
    wm_state = functional["wm_state"]
    ac_state = functional["ac_state"]
    slow_state = functional["slow_state"]
    alpha_state = functional["alpha_state"]
    rnd_state = functional["rnd_state"]

    # ----------- Data-parallel : RÉPLIQUER les states sur tous les devices.
    # Params + état optimiseur (Adam m/v) sont répliqués (le 75M tient large sur
    # un core → pas de FSDP). Pas besoin d'init-sous-jit : un device_put du state
    # déjà construit suffit. Sur 1 device, c'est un no-op fonctionnel.
    wm_state = jax.device_put(wm_state, repl_sharding)
    ac_state = jax.device_put(ac_state, repl_sharding)
    slow_state = jax.device_put(slow_state, repl_sharding)
    if alpha_state is not None:
        alpha_state = jax.device_put(alpha_state, repl_sharding)
    if rnd_state is not None:
        rnd_state = jax.device_put(rnd_state, repl_sharding)
    # return_ema_std (array (2,)) : scalaire d'état → répliqué.
    return_ema_std = jax.device_put(return_ema_std, repl_sharding)

    train_wm_fn = functional["train_wm_fn"]
    train_ac_fn = functional["train_ac_fn"]
    train_ac_fn_adaptive = functional["train_ac_fn_adaptive"]
    train_alpha_fn = functional["train_alpha_fn"]
    train_rnd_fn = functional["train_rnd_fn"]
    rnd_bonus_fn = functional["rnd_bonus_fn"]
    merge_wm_fn = functional["merge_wm"]
    merge_ac_fn = functional["merge_ac"]
    merge_slow_fn = functional["merge_slow"]
    act_fn_func = functional["act_fn_functional"]

    # ----------- Output dirs
    output_root = Path(os.environ.get("WORLDMODEL_OUTPUT_DIR", "."))
    runs_dir = output_root / "runs"
    ckpt_dir = output_root / "checkpoints"
    runs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dirs : runs={runs_dir.resolve()}  checkpoints={ckpt_dir.resolve()}")
    print()

    # ----------- Setup act_fn (jit compilé)
    act_fn = make_act_fn()

    # ----------- Data-parallel : active le mesh pour toute la suite (warmup +
    # hot loop). Sur 1 device, mesh trivial → aucun effet. La correctness du DP
    # repose surtout sur les shardings d'entrée (portés par les arrays
    # device_put) + out_shardings des jit ; ce set_mesh global permet en plus
    # aux specs P() brutes de résoudre et fixe le contexte d'exécution.
    jax.set_mesh(mesh)

    # ----------- Phase 0 : Warmup — remplissage initial du buffer
    #
    # Le buffer n'est PAS sauvegardé dans les checkpoints (1M transitions uint8 =
    # 12.3 GB). A la reprise il repart donc vide, et c'était la principale source de
    # perte de qualité : l'ancienne version le remplissait avec 5000 transitions de
    # politique ALEATOIRE, puis relançait l'entraînement dessus à replay ratio 128-256.
    # Un agent à 10 achievements/épisode voyait ainsi son world model réentraîné sur des
    # données de marche au hasard, et chaque transition rejouée des dizaines de fois.
    #
    # A la reprise on remplit donc avec la politique CHARGEE, et sur bien plus de pas
    # (--resume_warmup_steps) : la distribution du buffer redevient celle que l'agent
    # produit réellement, et le sur-apprentissage sur un échantillon minuscule disparaît.
    resuming = bool(args.resume_from)
    n_warmup = args.resume_warmup_steps if resuming else args.warmup_steps
    mode = "ON-POLICY (reprise)" if resuming else "random"
    print("=" * 60)
    print(f"Phase 0 : Warmup {mode} ({n_warmup} steps, {n_envs} envs)")
    print("=" * 60)
    t_start = time.time()
    steps_per_env = max(1, n_warmup // n_envs)

    if resuming:
        # Même boucle que la collecte de la phase 1 : état RSSM porté d'un step à
        # l'autre, remis à zéro sur `done`, actions tirées de la politique chargée.
        warm_state = jax.device_put(rssm.init_state(n_envs), repl_sharding)
        warm_prev_a = jax.device_put(
            jnp.zeros((n_envs, action_dim), dtype=jnp.float32), repl_sharding)
        # Clé dédiée dérivée du seed : `main_key` n'est créée qu'après cette phase,
        # et on ne veut pas consommer son flux (la collecte de la phase 1 doit rester
        # reproductible indépendamment de la longueur du warmup).
        warm_key = jr.PRNGKey(seed + 777)
        for _ in range(steps_per_env):
            obs_batch = jax.device_put(np.stack(obs_list), repl_sharding)
            warm_key, subk = jr.split(warm_key)
            warm_state, actions_int = act_fn_func(
                wm_state, ac_state, warm_state, warm_prev_a, obs_batch, subk, 0.01)
            actions_np = np.asarray(actions_int)
            h_arr, z_arr = np.array(warm_state["h"]), np.array(warm_state["z"])
            next_prev_a = np.zeros((n_envs, action_dim), dtype=np.float32)
            for i, env in enumerate(envs):
                a = int(actions_np[i])
                next_obs, r, done, _ = env.step(a)
                buffer.add(obs_list[i], a, r, next_obs, done, env_id=i)
                if done:
                    h_arr[i] = 0.0        # reset de l'état latent sur fin d'épisode
                    z_arr[i] = 0.0
                    obs_list[i] = env.reset()
                else:
                    next_prev_a[i, a] = 1.0
                    obs_list[i] = next_obs
            warm_state = {"h": jax.device_put(h_arr, repl_sharding),
                          "z": jax.device_put(z_arr, repl_sharding)}
            warm_prev_a = jax.device_put(next_prev_a, repl_sharding)
    else:
        for _ in range(steps_per_env):
            for i, env in enumerate(envs):
                action = np.random.randint(0, action_dim)
                next_obs, r, done, _ = env.step(action)
                buffer.add(obs_list[i], action, r, next_obs, done, env_id=i)
                obs_list[i] = next_obs if not done else env.reset()
    # Densité de récompense du buffer initial : c'est LE diagnostic de la reprise.
    # Un warmup aléatoire produit ~0.012 reward/step (≈2.3 achievements/épisode) ; un
    # agent competent ~0.035-0.045. Si ce chiffre s'effondre après une reprise, le
    # world model va être réentraîné sur des données non représentatives.
    _wr = np.asarray(buffer.rewards)
    _n = min(len(buffer), _wr.size)
    _rate = float(np.abs(_wr).sum() / max(_n, 1))
    _ach = float((np.abs(_wr) > 0.5).sum() / max(_n, 1))
    print(f"Buffer : {len(buffer)} transitions en {time.time() - t_start:.1f}s")
    print(f"         reward/step = {_rate:.4f}   transitions à |r|>0.5 = {100 * _ach:.2f}%")
    print(f"         Mémoire buffer : {buffer.memory_usage_mb():.1f} MB")
    print()

    # ----------- Phase 1 : Main loop
    print("=" * 60)
    print(f"Phase 1 : Training Dreamer JAX ({args.train_iter} itérations)")
    print("=" * 60)

    history = {
        "iter": [],
        "loss_wm": [], "loss_recon": [], "loss_kl": [], "loss_reward": [], "loss_continue": [],
        "loss_actor": [], "loss_critic": [], "entropy": [],
        "env_reward_per_step": [],
        # Métriques fines pour les figures du paper (par LOG_INTERVAL)
        "loss_pg": [], "returns_mean": [], "values_mean": [],
        "return_scale": [], "return_p5": [], "return_p95": [], "ips": [],
        # Diagnostics reward head + imagination. Ils n'existaient QUE dans le texte du
        # log, or Kaggle tronque le début des gros logs (sur v52/v53, les lignes d'iter
        # ne commencent qu'à ~8000) → les critères de succès du fix reward étaient
        # illisibles sur les 40% initiaux du run. Ici ils atterrissent dans le summary
        # JSON, qui couvre lui l'intégralité du run.
        #   rew_pred_ach / rew_pred_zero : prédiction de la head sur les achievements
        #     (cible ~1.0) et sur les états à reward nul (doit rester ~0). Restent
        #     IN-SAMPLE (calculés sur le batch fitté) — ne pas y lire une capacité de
        #     généralisation, cf. docs/HYPOTHESES.md H_313.
        #   img_rew_max / img_rew_frac_hi : le reward que l'actor voit réellement dans
        #     l'imagination. Si img_rew_max reste ~0, aucune trajectoire rêvée ne
        #     contient de récompense de taille achievement.
        "rew_pred_ach": [], "rew_pred_zero": [], "rew_n_ach": [],
        "img_rew_mean": [], "img_rew_max": [], "img_rew_frac_hi": [],
        # Par EVAL
        "eval_iter": [], "eval_score": [], "eval_length": [], "eval_achievements": [],
        "eval_sample": [], "eval_detail": [], "eval_crafter_score": [],
        "eval_train_episodes": [],
    }

    # Protocole Crafter officiel : compteurs sur les épisodes de TRAINING.
    # Repris du checkpoint si disponible : le crafter_score etant cumule depuis
    # l'iteration 0, repartir de zero apres une reprise casserait la comparabilite
    # avec la premiere moitie du run.
    train_episode_count = 0
    train_ach_counts = {}
    _resumed_counters = {}
    if args.resume_from:
        _mp = Path(args.resume_from).with_suffix(".meta.json")
        if _mp.exists():
            _resumed_counters = json.loads(_mp.read_text()).get("counters", {}) or {}
        if _resumed_counters:
            train_ach_counts = dict(_resumed_counters.get("train_ach_counts", {}))
            train_episode_count = int(_resumed_counters.get("train_episode_count", 0))
            print(f"[resume] compteurs restaurés : {train_episode_count} épisodes, "
                  f"{len(train_ach_counts)} achievements déjà vus")
        else:
            print("[resume] /!\\ pas de compteurs dans le meta (format ancien) : "
                  "le crafter_score repart de zéro et n'est pas comparable au pré-reprise.")
    # DIAGNOSTIC comportemental (cumulé sur les épisodes de train) :
    #   diag_wood_hist  : distribution du max de bois atteint par épisode (0..5+).
    #     place_table coûte 2 bois, la pioche +1 → si la masse est sur 0-1, tout le
    #     craft est ARITHMÉTIQUEMENT hors de portée, quel que soit le modèle.
    #   diag_action_counts : histogramme des 17 actions réellement jouées → distingue
    #     "n'essaie jamais l'action" de "l'essaie mais échoue".
    diag_wood_hist = np.array(_resumed_counters.get("diag_wood_hist") or [0] * 6,
                              dtype=np.int64)
    diag_action_counts = np.array(
        _resumed_counters.get("diag_action_counts") or [0] * action_dim, dtype=np.int64)
    diag_reached_stone = int(_resumed_counters.get("diag_reached_stone", 0))

    collected_rewards = []
    # RSSM state multi-env pour la collecte.
    # Data-parallel : la collecte reste RÉPLIQUÉE (non shardée) — batch=n_envs,
    # env CPU séquentiel. On place l'état initial répliqué pour rester cohérent
    # avec act_fn_functional (out_shardings=repl).
    rssm_state_multi = jax.device_put(rssm.init_state(n_envs), repl_sharding)
    prev_actions_oh_multi = jax.device_put(
        jnp.zeros((n_envs, action_dim)), repl_sharding)

    # CHOIX RNG (data-parallel) : clé RÉPLIQUÉE sur tous les shards.
    # jax.lax.axis_index('data') n'est PAS dispo en jit auto/GSPMD (contrairement
    # à pmap/shard_map), donc pas de bruit distinct par shard sans pré-splitter
    # une clé shardée. On choisit volontairement le simple : la même clé sur
    # chaque shard. C'est acceptable car le BATCH diffère déjà par shard (split
    # de la dim B) → les gradients diffèrent par shard, puis sont moyennés
    # cross-shard automatiquement sous GSPMD. Le seul effet d'une clé répliquée
    # est que le bruit stochastique interne (sample z du RSSM, sample actions en
    # imagination) est corrélé entre shards — impact négligeable sur la
    # dynamique d'apprentissage, et la non-régression 1-device est exacte.
    main_key = jr.PRNGKey(seed + 1)
    collect_per_iter = max(1, COLLECT_PER_ITER // n_envs)
    t_start = time.time()

    # ----------- Auto-explore state (Python locals, mute par EVAL)
    auto_explore_multiplier = 1.0
    auto_explore_best = 0.0
    auto_explore_consec_stag = 0
    # current_alpha_val : valeur float Python rafraîchie après chaque train_alpha
    # (utilisée pour calculer effective_alpha = alpha * multiplier passé au train_ac)
    current_alpha_val = float(args.alpha_init) if args.adaptive_alpha else float(args.entropy_coef)
    last_loss_rnd_val = 0.0

    # ----------- Phase 13 : Safeguards state
    # SAFEGUARD 2 : runtime rnd_coef adaptatif. Démarre à 0 (warmup),
    # SAFEGUARD 1 le fait monter linéairement, puis SAFEGUARD 2 régule.
    rnd_coef_runtime = 0.0
    RND_RATIO_TARGET_MAX = 5.0
    RND_RATIO_TARGET_MIN = 0.1
    RUNNING_DECAY = 0.99
    running_bonus_mean = 0.0
    running_extrinsic_mean = 0.0
    # SAFEGUARD 3 : Health monitor
    health = TrainingHealthMonitor(args)
    health_fatal_triggered = False
    print(f"[Phase 13] Safeguards : warmup={args.rnd_warmup_steps}  "
          f"adaptive_rnd={args.adaptive_rnd}  health_auto_stop={args.health_auto_stop}  "
          f"h_target_schedule={args.h_target_schedule} "
          f"[{args.h_target_init:.2f}→{args.h_target_final:.2f} over {args.h_target_decay_steps}]")

    last_metrics = {}

    # Best EVAL tracking (indépendant de auto_explore, qui peut être off)
    eval_best_ach = 0.0
    eval_best_iter = 0

    # Profiler (no-op si --profile pas activé)
    prof = Profiler(enabled=args.profile)
    if args.profile:
        print(f"[PROFILE] Mode profile activé pour {args.train_iter} itérations")

    # Setup pool multiprocessing si demandé
    env_pool = None
    if args.mp_collect:
        env_pool = MultiprocEnvPool(n_envs=n_envs, base_seed=seed)
        # Sync les obs initiales (override obs_list avec celles du pool)
        obs_list = env_pool.reset_all()
        print(f"[MP] Pool multiprocessing initialisé ({n_envs} workers)")

    # PHASE 11B : double-buffering des samples WM + AC.
    # On prefetch les batches du iter+1 pendant que train_step du iter courant
    # tourne sur GPU. JAX async dispatch permet à XLA d'overlap gather et train
    # kernels indépendants. Le sync explicite n'arrive qu'à la lecture des
    # metrics (toutes les LOG_INTERVAL iter).
    # Pré-conditions : buffer warmupé (>= SEQ_LEN transitions), ce qui est
    # garanti par la Phase 0 plus haut (warmup_steps >= SEQ_LEN).
    main_key, k_wm0, k_ac0 = jr.split(main_key, 3)
    batch_wm_next = buffer.sample_sequences(k_wm0, args.batch_size, SEQ_LEN)
    batch_ac_next = buffer.sample_sequences(k_ac0, args.batch_size, SEQ_LEN,
                                            priority_frac=args.replay_priority_frac)

    if start_iter > 0:
        print(f"[resume] Boucle reprise à iter {start_iter} (jusqu'à {args.train_iter})")
    for it in range(start_iter, args.train_iter):

        # ============ Phase 13 : rnd_coef effectif pour CET iter
        # SAFEGUARD 1 (warmup) : rampe 0 → args.rnd_coef sur rnd_warmup_steps.
        # SAFEGUARD 2 (adaptive) : régule par dessus le warmup. Pendant le warmup,
        # on suit la rampe linéaire ; après, le rnd_coef_runtime est ajusté tous
        # les LOG_INTERVAL iter selon le ratio bonus/extrinsic.
        if it < int(args.rnd_warmup_steps):
            # Pendant le warmup, override directement (warmup pilote).
            rnd_coef_runtime = get_rnd_coef(it, args)
        elif it == int(args.rnd_warmup_steps):
            # Fin de warmup : initialise le runtime à la valeur cible.
            rnd_coef_runtime = float(args.rnd_coef)
        # Sinon : rnd_coef_runtime est piloté par SAFEGUARD 2 (ajusté plus bas).
        rnd_coef_effective = rnd_coef_runtime
        # ANNEALING (style Burda) : décroissance linéaire du coef vers 0 après le
        # warmup → le bonus sert à explorer tôt puis s'efface pour laisser
        # l'exploitation (extrinsèque) prendre le relais. Facteur 1.0 → 0.0.
        if getattr(args, "rnd_anneal", False):
            _w = int(args.rnd_warmup_steps)
            if it >= _w:
                _span = max(1, args.train_iter - _w)
                rnd_coef_effective = rnd_coef_runtime * max(0.0, 1.0 - (it - _w) / _span)

        # ============ (a) Collecte
        # UNIMIX décroissant (exploration d'amorçage) : linéaire init → final sur
        # unimix_decay_iters, puis constant. jnp scalaire → un seul trace jit.
        if args.unimix_decay_iters > 0:
            _frac = min(1.0, it / float(args.unimix_decay_iters))
            _u = args.unimix_init + _frac * (args.unimix_final - args.unimix_init)
        else:
            _u = args.unimix_final
        unimix_now = jnp.array(_u, dtype=jnp.float32)

        for _ in range(collect_per_iter):
            # --- act_fn (jit) : encode + observe RSSM + sample action
            # FIX 1 : version functional → utilise les states (params à jour)
            # sans muter les modules originaux. FIX 3 : device_put explicite.
            prof.tic("act_fn")
            obs_batch_np = np.stack(obs_list)  # (N, C, H, W)
            # Collecte répliquée (non shardée) : obs sur tous les devices.
            obs_batch_jax = jax.device_put(obs_batch_np, repl_sharding)
            main_key, subk = jr.split(main_key)
            new_state, actions_int = act_fn_func(
                wm_state, ac_state,
                rssm_state_multi, prev_actions_oh_multi,
                obs_batch_jax, subk, unimix_now,
            )
            # Force materialize en numpy avant env.step (sync nécessaire car
            # actions_int doit être lu pour driver l'env Python).
            actions_int_np = np.asarray(actions_int)
            prof.toc()

            # --- env.step : Python pur, séquentiel ou multiproc
            prof.tic("env_step")
            if env_pool is not None:
                # Multiprocessing : send actions, receive (next_obs, r, done)
                results = env_pool.step_all(actions_int_np)
            else:
                # Sequentiel (baseline)
                results = []
                for i, env in enumerate(envs):
                    a = int(actions_int_np[i])
                    next_obs, r, done, _ = env.step(a)
                    ep_unlocked = None
                    if done:
                        # Achievements de l'épisode AVANT le reset (score Crafter officiel)
                        ep_unlocked = sorted(env.unlocked_names)
                        # DIAGNOSTIC comportemental : inventaire max atteint + actions
                        # tentées, relevés AVANT le reset (sinon perdus).
                        _inv = env.inv_max
                        diag_wood_hist[min(_inv.get("wood", 0), 5)] += 1
                        if _inv.get("stone", 0) > 0:
                            diag_reached_stone += 1
                        diag_action_counts += env.action_counts
                        next_obs = env.reset()
                    results.append((next_obs, r, done, ep_unlocked))
            prof.toc()

            # --- transfer + buffer_add : copies CPU <-> GPU + insertion buffer
            prof.tic("transfer")
            new_h_arr = np.array(new_state["h"])
            new_z_arr = np.array(new_state["z"])
            new_prev_actions = np.zeros((n_envs, action_dim), dtype=np.float32)
            prof.toc()

            # ---- RND bonus (batched sur next_obs)
            # Calcul du bonus intrinsèque AVANT le buffer.add, pour ajouter au reward.
            # FIX PHASE 17 : guard sur rnd_coef_effective > 0 pour éviter saturation des
            # running stats avec des bonus du predictor random pendant le warmup.
            # Sans cette garde, les running stats convergeraient vers les bonus très élevés
            # du predictor random, et quand RND s'active après warmup, tous les bonus
            # normalisés seraient quasi-zéro → RND mort.
            rnd_bonuses = None
            if args.use_rnd and rnd_bonus_fn is not None and rnd_coef_effective > 0.0:
                # CrafterEnv retourne déjà [0,1], pas de /255 (double-normalisation bug fixé)
                next_obs_np = np.stack([results[i][0] for i in range(n_envs)]).astype(np.float32)
                # Collecte répliquée (cohérent avec rnd_state répliqué).
                next_obs_jax = jax.device_put(next_obs_np, repl_sharding)
                rnd_state, bonus_jax = rnd_bonus_fn(rnd_state, next_obs_jax)
                rnd_bonuses = np.array(bonus_jax)  # (n_envs,)

            prof.tic("buffer_add")
            new_obs_list = []
            # Phase 13 : tracking séparé extrinsic vs bonus pour SAFEGUARD 2
            step_extr_sum = 0.0
            step_bonus_sum = 0.0
            for i in range(n_envs):
                next_obs, r, done, ep_unlocked = results[i]
                a = int(actions_int_np[i])
                # Reward extrinsèque + bonus intrinsèque RND (si activé)
                bonus_i = float(rnd_bonuses[i]) if rnd_bonuses is not None else 0.0
                r_extr = float(r)
                r_final = r_extr + rnd_coef_effective * bonus_i
                buffer.add(obs_list[i], a, r_final, next_obs, done, env_id=i)
                collected_rewards.append(r_final)
                step_extr_sum += r_extr
                step_bonus_sum += bonus_i
                if done:
                    new_h_arr[i] = 0.0
                    new_z_arr[i] = 0.0
                    # Protocole Crafter officiel : success rates sur les
                    # épisodes de TRAINING (pas l'eval) → score = geom mean
                    train_episode_count += 1
                    for name in (ep_unlocked or []):
                        train_ach_counts[name] = train_ach_counts.get(name, 0) + 1
                else:
                    new_prev_actions[i, a] = 1.0
                new_obs_list.append(next_obs)
            obs_list = new_obs_list
            # SAFEGUARD 2 : running means (avant ajout au reward, bonus brut)
            step_extr_mean = step_extr_sum / max(1, n_envs)
            step_bonus_mean = step_bonus_sum / max(1, n_envs)
            running_extrinsic_mean = (
                RUNNING_DECAY * running_extrinsic_mean
                + (1.0 - RUNNING_DECAY) * step_extr_mean
            )
            running_bonus_mean = (
                RUNNING_DECAY * running_bonus_mean
                + (1.0 - RUNNING_DECAY) * step_bonus_mean
            )
            prof.toc()

            prof.tic("transfer")
            # FIX 3 : device_put au lieu de jnp.array.
            # Data-parallel : collecte répliquée (cohérent avec act_fn_functional).
            rssm_state_multi = {
                "h": jax.device_put(new_h_arr, repl_sharding),
                "z": jax.device_put(new_z_arr, repl_sharding),
            }
            prev_actions_oh_multi = jax.device_put(new_prev_actions, repl_sharding)
            prof.toc()

        # ============ (b) Train WM (avec double-buffering du sample)
        # PHASE 11B : on swap batch_wm_next → batch_wm_current (instant),
        # puis on prefetch le batch du PROCHAIN iter en async dispatch. XLA
        # overlap le gather kernel suivant avec le train_wm courant.
        prof.tic("sample_batch")
        batch_wm_current = batch_wm_next
        main_key, subk = jr.split(main_key)
        batch_wm_next = buffer.sample_sequences(subk, args.batch_size, SEQ_LEN)
        prof.toc()

        prof.tic("train_wm")
        main_key, subk = jr.split(main_key)
        # FIX 1 : functional train step (jax.jit + nnx.split/merge en closure)
        # → supprime overhead Python du graph traversal de nnx.jit.
        # FIX 2 : pas de block_until_ready dans la hot loop → JAX dispatch
        # async, le Python continue pendant que XLA calcule sur GPU.
        wm_state, wm_metrics = train_wm_fn(wm_state, batch_wm_current, subk)
        last_metrics.update(wm_metrics)
        prof.toc()

        # ---- Train RND predictor (en parallèle, sur les obs du batch WM)
        if args.use_rnd and train_rnd_fn is not None:
            # batch_wm_current["obs"] shape : (B, T, C, H, W) float32 [0,1]
            obs_seq_wm = batch_wm_current["obs"]
            B_rnd, T_rnd = obs_seq_wm.shape[:2]
            obs_flat_rnd = obs_seq_wm.reshape(B_rnd * T_rnd, *obs_seq_wm.shape[2:])
            rnd_state, rnd_metrics = train_rnd_fn(rnd_state, obs_flat_rnd)
            last_metrics.update(rnd_metrics)

        # Train steps additionnels (cas wm_train_per_iter > 1) : pas de
        # prefetch supplémentaire, on sample en série (cas rare, default = 1).
        for _ in range(args.wm_train_per_iter - 1):
            prof.tic("sample_batch")
            main_key, subk = jr.split(main_key)
            batch_extra = buffer.sample_sequences(subk, args.batch_size, SEQ_LEN)
            prof.toc()
            prof.tic("train_wm")
            main_key, subk = jr.split(main_key)
            wm_state, wm_metrics = train_wm_fn(wm_state, batch_extra, subk)
            last_metrics.update(wm_metrics)
            prof.toc()

        # ============ (c) Train AC (avec double-buffering du sample)
        # AC WARMUP : geler l'actor-critic pendant les ac_warmup_iters premières
        # itérations — le WM apprend seul (collecte via l'actor uniforme zero-init
        # = exploration random). Sans ce gel, l'AC apprend sur un WM encore nul :
        # returns imaginés = bruit biaisé positif → chaque action échantillonnée
        # est renforcée (rich-get-richer) → verrouillage de la politique. Mesuré :
        # H 2.5→0.08-0.18 en <200 iters sur TOUS les runs 75M (actor >=1024),
        # alors que le 14M (actor 2x256) résiste (H 2.59 -> 1.73 en douceur).
        ac_frozen = it < args.ac_warmup_iters
        # jnp.array (dynamique) → un seul trace jit pour les deux phases.
        actor_coef = jnp.array(0.0 if ac_frozen else 1.0, dtype=jnp.float32)
        prof.tic("sample_batch")
        batch_ac_current = batch_ac_next
        main_key, subk = jr.split(main_key)
        batch_ac_next = buffer.sample_sequences(subk, args.batch_size, SEQ_LEN,
                                                priority_frac=args.replay_priority_frac)
        prof.toc()

        prof.tic("train_ac")
        main_key, subk = jr.split(main_key)
        # FIX 1 + 2 : functional + no block_until_ready
        if args.adaptive_alpha:
            # effective_alpha = alpha_appris × multiplier auto_explore (float Py)
            effective_alpha = jnp.array(
                current_alpha_val * auto_explore_multiplier, dtype=jnp.float32,
            )
            ac_state, slow_state, return_ema_std, ac_metrics = train_ac_fn_adaptive(
                wm_state, ac_state, slow_state,
                batch_ac_current, return_ema_std, subk,
                effective_alpha, actor_coef,
            )
        else:
            ent_coef_eff = float(args.entropy_coef) * auto_explore_multiplier
            ac_state, slow_state, return_ema_std, ac_metrics = train_ac_fn(
                wm_state, ac_state, slow_state,
                batch_ac_current, return_ema_std, subk,
                ent_coef_eff, actor_coef,
            )
        last_metrics.update(ac_metrics)
        prof.toc()

        # ---- Train adaptive alpha (après le train AC, utilise mean_H observé)
        if not ac_frozen and args.adaptive_alpha and train_alpha_fn is not None:
            mean_H = ac_metrics["H"]
            # SAFEGUARD 4 : H_target curriculum (linear schedule)
            h_target_cur = get_h_target(it, args)
            alpha_state, alpha_metrics = train_alpha_fn(
                alpha_state, mean_H, jnp.array(h_target_cur, dtype=jnp.float32),
            )
            # Récupère la valeur Python à jour (pour calculer effective_alpha next iter)
            current_alpha_val = float(alpha_metrics["alpha"])
            last_metrics["alpha"] = alpha_metrics["alpha"]

        # Train steps additionnels (cas ac_train_per_iter > 1). Pendant le warmup
        # (actor_coef=0), ces steps continuent d'entraîner le CRITIC.
        for _ in range(args.ac_train_per_iter - 1):
            prof.tic("sample_batch")
            main_key, subk = jr.split(main_key)
            batch_extra = buffer.sample_sequences(subk, args.batch_size, SEQ_LEN,
                                                  priority_frac=args.replay_priority_frac)
            prof.toc()
            prof.tic("train_ac")
            main_key, subk = jr.split(main_key)
            if args.adaptive_alpha:
                effective_alpha = jnp.array(
                    current_alpha_val * auto_explore_multiplier, dtype=jnp.float32,
                )
                ac_state, slow_state, return_ema_std, ac_metrics = train_ac_fn_adaptive(
                    wm_state, ac_state, slow_state,
                    batch_extra, return_ema_std, subk,
                    effective_alpha, actor_coef,
                )
            else:
                ent_coef_eff = float(args.entropy_coef) * auto_explore_multiplier
                ac_state, slow_state, return_ema_std, ac_metrics = train_ac_fn(
                    wm_state, ac_state, slow_state,
                    batch_extra, return_ema_std, subk,
                    ent_coef_eff, actor_coef,
                )
            last_metrics.update(ac_metrics)
            prof.toc()
            if not ac_frozen and args.adaptive_alpha and train_alpha_fn is not None:
                mean_H = ac_metrics["H"]
                # FIX : utiliser le schedule (comme la boucle principale),
                # pas args.h_target brut (=2.0) qui ignorait h_target_schedule.
                h_target_cur = get_h_target(it, args)
                alpha_state, alpha_metrics = train_alpha_fn(
                    alpha_state, mean_H, jnp.array(h_target_cur, dtype=jnp.float32),
                )
                current_alpha_val = float(alpha_metrics["alpha"])
                last_metrics["alpha"] = alpha_metrics["alpha"]

        # ============ Logs
        if (it + 1) % LOG_INTERVAL == 0:
            elapsed = time.time() - t_start
            ips = (it + 1 - start_iter) / elapsed
            vals = {k: float(v) for k, v in last_metrics.items()}
            history["iter"].append(it)
            history["loss_wm"].append(vals.get("loss_wm", 0.0))
            history["loss_recon"].append(vals.get("loss_recon", 0.0))
            history["loss_kl"].append(vals.get("loss_kl", 0.0))
            history["loss_reward"].append(vals.get("loss_reward", 0.0))
            history["loss_continue"].append(vals.get("loss_continue", 0.0))
            history["loss_actor"].append(vals.get("loss_actor", 0.0))
            history["loss_critic"].append(vals.get("loss_critic", 0.0))
            history["entropy"].append(vals.get("entropy", 0.0))
            history["env_reward_per_step"].append(
                float(np.mean(collected_rewards[-1000:])) if collected_rewards else 0.0
            )
            history["loss_pg"].append(vals.get("loss_actor_pg", 0.0))
            history["returns_mean"].append(vals.get("returns_mean", 0.0))
            history["values_mean"].append(vals.get("values_mean", 0.0))
            history["return_scale"].append(vals.get("return_scale", 1.0))
            history["return_p5"].append(float(return_ema_std[0]))
            history["return_p95"].append(float(return_ema_std[1]))
            history["ips"].append(float(ips))
            # Diagnostics reward head + imagination (cf. commentaire à l'init de history)
            history["rew_pred_ach"].append(vals.get("rew_pred_ach", 0.0))
            history["rew_pred_zero"].append(vals.get("rew_pred_zero", 0.0))
            history["rew_n_ach"].append(vals.get("rew_n_ach", 0.0))
            history["img_rew_mean"].append(vals.get("img_rew_mean", 0.0))
            history["img_rew_max"].append(vals.get("img_rew_max", 0.0))
            history["img_rew_frac_hi"].append(vals.get("img_rew_frac_hi", 0.0))

            # ETA : iters restants × temps moyen par iter écoulé
            iters_left = args.train_iter - (it + 1)
            eta_s = iters_left / ips if ips > 1e-6 else 0.0
            eta_tag = _format_eta(eta_s)

            pct = 100.0 * (it + 1) / args.train_iter
            alpha_tag = f" α={current_alpha_val:.4f}" if args.adaptive_alpha else ""
            ax_tag = f" ax={auto_explore_multiplier:.2f}" if args.auto_explore and auto_explore_multiplier > 1.01 else ""
            rnd_tag = f" rnd_c={rnd_coef_effective:.3f} rnd_l={vals.get('loss_rnd', 0):.3f}" if args.use_rnd else ""
            h_tgt_tag = f" H*={get_h_target(it, args):.2f}" if args.h_target_schedule and args.adaptive_alpha else ""
            # Sections : [progress] | WM | AC | imagination(returns vs values) | collecte | throughput
            print(
                f"  iter {it+1:5d}/{args.train_iter} [{pct:4.1f}%] | "
                f"WM wm={vals.get('loss_wm', 0):.2f} rec={vals.get('loss_recon', 0):.2f} "
                f"kl={vals.get('loss_kl', 0):.2f} rew={vals.get('loss_reward', 0):.3f} con={vals.get('loss_continue', 0):.3f} "
                f"rew@ach={vals.get('rew_pred_ach', 0):.2f}/{vals.get('rew_true_ach', 0):.2f}"
                f"(n={vals.get('rew_n_ach', 0):.0f}) rew@0={vals.get('rew_pred_zero', 0):+.3f} | "
                f"AC act={vals.get('loss_actor', 0):.3f} crit={vals.get('loss_critic', 0):.3f} "
                f"pg={vals.get('loss_actor_pg', 0):.3f} H={vals.get('entropy', 0):.2f} | "
                f"img ret={vals.get('returns_mean', 0):.2f} val={vals.get('values_mean', 0):.2f} "
                f"imgR(mu={vals.get('img_rew_mean', 0):.3f} max={vals.get('img_rew_max', 0):.2f} "
                f"p99={vals.get('img_rew_p99', 0):.2f} hi={100*vals.get('img_rew_frac_hi', 0):.2f}%) "
                f"scale={vals.get('return_scale', 1.0):.2f} p5={float(return_ema_std[0]):.2f} p95={float(return_ema_std[1]):.2f}"
                f"{alpha_tag}{ax_tag}{rnd_tag}{h_tgt_tag} | "
                f"umix={float(_u):.3f} | "
                f"r/step={history['env_reward_per_step'][-1]:.4f} | {ips:.1f} ips ETA {eta_tag}"
            )

            # ============ SAFEGUARD 2 : adaptive_rnd runtime adjustment
            # Post-warmup, on régule rnd_coef_runtime pour maintenir le ratio
            # bonus_RND / extrinsic dans [RND_RATIO_TARGET_MIN, MAX].
            if (args.use_rnd and args.adaptive_rnd
                    and it >= int(args.rnd_warmup_steps)
                    and running_extrinsic_mean > 1e-6):
                ratio = (running_bonus_mean * rnd_coef_runtime) / max(running_extrinsic_mean, 1e-6)
                if ratio > RND_RATIO_TARGET_MAX:
                    new_coef = rnd_coef_runtime * 0.9
                    print(f"  [adaptive_rnd] ratio={ratio:.2f} > {RND_RATIO_TARGET_MAX} "
                          f"→ rnd_coef {rnd_coef_runtime:.4f} → {new_coef:.4f}")
                    rnd_coef_runtime = new_coef
                elif ratio < RND_RATIO_TARGET_MIN:
                    new_coef = rnd_coef_runtime * 1.05
                    print(f"  [adaptive_rnd] ratio={ratio:.3f} < {RND_RATIO_TARGET_MIN} "
                          f"→ rnd_coef {rnd_coef_runtime:.4f} → {new_coef:.4f}")
                    rnd_coef_runtime = new_coef

            # ============ SAFEGUARD 3 : Health monitor
            # Enrichit les metrics avec les running means RND pour le check.
            health_metrics = dict(vals)
            health_metrics["rnd_bonus_mean"] = float(running_bonus_mean)
            health_metrics["extrinsic_mean"] = float(running_extrinsic_mean)
            health_metrics["H"] = float(vals.get("entropy", vals.get("H", 1.0)))
            warnings_list, is_fatal = health.check(health_metrics, it)
            if warnings_list:
                for w in warnings_list:
                    print(f"  [health] WARN {w}")
            if is_fatal and args.health_auto_stop:
                print(f"  [health] FATAL pathology detected at iter {it+1}, stopping training.")
                health_fatal_triggered = True
                break

        # ============ Eval périodique
        if (it + 1) % args.eval_interval == 0:
            # FIX 1 : merge les states pour récupérer des modules à jour avant
            # l'eval / save. Les modules originaux (encoder, rssm, ...) sont
            # restés "stale" car les params vivent dans wm_state / ac_state.
            (enc_eval, rssm_eval, dec_eval, rew_eval, cont_eval), _opt = merge_wm_fn(wm_state)
            (actor_eval, critic_eval), _opt_ac = merge_ac_fn(ac_state)
            slow_eval = merge_slow_fn(slow_state)

            eval_res = eval_agent(
                eval_env, enc_eval, rssm_eval, actor_eval, action_dim,
                n_episodes=EVAL_EPISODES,
            )
            history["eval_iter"].append(it + 1)
            history["eval_score"].append(eval_res["score"])
            history["eval_length"].append(eval_res["length"])
            history["eval_achievements"].append(eval_res["achievements"])
            history["eval_sample"].append(eval_res.get("achievements_sample", 0.0))
            history["eval_detail"].append(eval_res.get("achievements_detail", {}))
            # Score Crafter OFFICIEL (succès cumulés sur les épisodes de training)
            cscore = crafter_score(train_ach_counts, train_episode_count)
            history["eval_crafter_score"].append(cscore)
            history["eval_train_episodes"].append(train_episode_count)
            ach_now = eval_res["achievements"]

            # Best-so-far (indépendant de auto_explore)
            if ach_now > eval_best_ach:
                eval_best_ach = ach_now
                eval_best_iter = it + 1
            trend = "↑" if (len(history["eval_achievements"]) >= 2
                            and ach_now > history["eval_achievements"][-2]) else (
                    "↓" if (len(history["eval_achievements"]) >= 2
                            and ach_now < history["eval_achievements"][-2]) else "=")
            print(
                f"  >>> EVAL @ iter {it+1} : "
                f"score={eval_res['score']:.2f}  length={eval_res['length']:.0f}  "
                f"achievements={ach_now:.2f} {trend}  "
                f"sample={eval_res.get('achievements_sample', 0.0):.2f}  "
                f"(best={eval_best_ach:.2f} @{eval_best_iter})  "
                f"crafter_score={cscore:.2f}% [{train_episode_count} train eps]"
            )
            # Détail par achievement : taux de réussite trié décroissant.
            # Crucial pour voir la progression hiérarchique (collect_wood →
            # place_table → make_wood_pickaxe → collect_stone → ...).
            ach_detail = eval_res.get("achievements_detail", {})
            if ach_detail:
                ranked = sorted(ach_detail.items(), key=lambda kv: -kv[1])
                detail_str = "  ".join(f"{name}={rate*100:.0f}%" for name, rate in ranked)
                print(f"      unlocked ({len(ranked)}/{len(ACHIEVEMENTS)}): {detail_str}")
            else:
                print(f"      unlocked (0/{len(ACHIEVEMENTS)}): — aucun achievement débloqué")

            # Détail sur les épisodes de TRAIN (n = centaines) : 40× moins bruité que
            # l'éval (n=EVAL_EPISODES) et c'est la base du crafter_score. Surtout :
            # un achievement à 0 occurrence ici = ZÉRO exemple dans le buffer, donc la
            # reward head ne peut PAS l'apprendre (problème de donnée, pas de head).
            if train_episode_count > 0:
                train_ranked = sorted(train_ach_counts.items(), key=lambda kv: -kv[1])
                train_str = "  ".join(
                    f"{n}={c}({100.0*c/train_episode_count:.1f}%)" for n, c in train_ranked
                ) or "— aucun"
                never = [a for a in ACHIEVEMENTS if train_ach_counts.get(a, 0) == 0]
                print(f"      TRAIN ({len(train_ranked)}/{len(ACHIEVEMENTS)} sur {train_episode_count} eps): {train_str}")
                print(f"      JAMAIS vus en train ({len(never)}) : {'  '.join(never) if never else '—'}")

            # ---- DIAGNOSTIC COMPORTEMENTAL (le "pourquoi" du plafond)
            # 1) BOIS : place_table coûte 2 bois, +1 pour la pioche → sans épisodes
            #    à wood>=2, tout le craft est hors de portée par ARITHMÉTIQUE.
            # 2) ACTIONS : une politique effondrée n'essaie que 2-3 actions sur 17 ;
            #    les actions de craft ne sont alors JAMAIS tentées (≠ "mal apprises").
            _nw = int(diag_wood_hist.sum())
            if _nw > 0:
                _pct = 100.0 / _nw
                print(f"      BOIS/épisode (n={_nw}): "
                      + "  ".join(f"{k}{'+' if k == 5 else ''}={diag_wood_hist[k] * _pct:.1f}%"
                                  for k in range(6))
                      + f"  | >=2 bois (table possible): {diag_wood_hist[2:].sum() * _pct:.1f}%"
                      + f"  | >=3 (pioche): {diag_wood_hist[3:].sum() * _pct:.1f}%"
                      + f"  | a eu de la pierre: {100.0 * diag_reached_stone / _nw:.2f}%")
            _na = int(diag_action_counts.sum())
            if _na > 0:
                _order = np.argsort(-diag_action_counts)
                _top = "  ".join(f"{ACTION_NAMES[i]}={100.0 * diag_action_counts[i] / _na:.1f}%"
                                 for i in _order[:5])
                _used = int((diag_action_counts > 0.01 * _na).sum())
                _pt = diag_action_counts[ACTION_NAMES.index("place_table")]
                _mp = diag_action_counts[ACTION_NAMES.index("make_wood_pickaxe")]
                print(f"      ACTIONS top5: {_top}  | {_used}/17 actions >1%"
                      f"  | place_table tenté {100.0 * _pt / _na:.2f}%"
                      f"  | make_wood_pickaxe tenté {100.0 * _mp / _na:.2f}%")

            # ---- Auto-explore : détection de stagnation
            if args.auto_explore:
                # Progrès = ach_now > best × (1 + threshold)
                if ach_now > auto_explore_best * (1.0 + AUTO_EXPLORE_THRESHOLD):
                    new_mult = max(1.0, auto_explore_multiplier * AUTO_EXPLORE_DECAY)
                    if auto_explore_multiplier > 1.01:
                        print(
                            f"  [auto_explore] Progress (+{ach_now - auto_explore_best:.2f}) : "
                            f"multiplier {auto_explore_multiplier:.2f} → {new_mult:.2f}"
                        )
                    auto_explore_multiplier = new_mult
                    auto_explore_best = ach_now
                    auto_explore_consec_stag = 0
                else:
                    auto_explore_consec_stag += 1
                    if auto_explore_consec_stag >= AUTO_EXPLORE_PATIENCE:
                        new_mult = min(AUTO_EXPLORE_MAX, auto_explore_multiplier * AUTO_EXPLORE_BOOST)
                        print(
                            f"  [auto_explore] STAGNATION (best={auto_explore_best:.2f}, "
                            f"now={ach_now:.2f}) : boost multiplier "
                            f"{auto_explore_multiplier:.2f} → {new_mult:.2f}"
                        )
                        auto_explore_multiplier = new_mult
                        auto_explore_consec_stag = 0

            # Save checkpoint léger (state dicts) + extras de reprise (Adam, scale EMA)
            ckpt_path = ckpt_dir / f"dreamer_crafter_jax_{args.run_name}_iter{it+1:06d}.npz"
            save_checkpoint(
                ckpt_path, enc_eval, rssm_eval, dec_eval, rew_eval, cont_eval,
                actor_eval, critic_eval, slow_eval, it + 1, args, history,
                opt_wm=_opt, opt_ac=_opt_ac, return_ema_std=return_ema_std,
                counters={"train_ach_counts": train_ach_counts,
                          "train_episode_count": train_episode_count,
                          "diag_wood_hist": diag_wood_hist,
                          "diag_action_counts": diag_action_counts,
                          "diag_reached_stone": diag_reached_stone},
            )
            print(f"  Checkpoint saved : {ckpt_path.name}")

            # Save summary JSON — LE fichier de données du run (figures paper).
            # Contient TOUT : config, history complet des métriques fines,
            # toutes les EVAL avec détail par achievement, et le protocole
            # Crafter officiel (success rates training + score).
            train_success_rates = {
                name: train_ach_counts.get(name, 0) / max(1, train_episode_count)
                for name in ACHIEVEMENTS
            }
            summary_path = runs_dir / f"dreamer_crafter_jax_summary_{args.run_name}.json"
            with open(summary_path, "w") as f:
                json.dump({
                    "run_name": args.run_name,
                    "iter": it + 1,
                    "total_iter": args.train_iter,
                    "env_steps_per_iter": COLLECT_PER_ITER,
                    "config": {k: v for k, v in vars(args).items()
                               if isinstance(v, (int, float, str, bool, type(None)))},
                    "constants": {
                        "GAMMA": GAMMA, "LAMBDA_GAE": LAMBDA_GAE,
                        "ENTROPY_COEF": ENTROPY_COEF, "LR_WM": LR_WM, "LR_AC": LR_AC,
                        "BUFFER_CAPACITY": args.buffer_capacity, "SEQ_LEN": SEQ_LEN,
                        "IMAGINATION_HORIZON": IMAGINATION_HORIZON,
                        "H_DIM": H_DIM, "EMBED_DIM": EMBED_DIM, "HIDDEN_DIM": HIDDEN_DIM,
                    },
                    "last_eval_score": float(eval_res["score"]),
                    "last_eval_length": float(eval_res["length"]),
                    "last_eval_achievements": float(eval_res["achievements"]),
                    "last_eval_achievements_sample": float(eval_res.get("achievements_sample", 0.0)),
                    "last_eval_achievements_detail": eval_res.get("achievements_detail", {}),
                    "best_eval_achievements": float(eval_best_ach),
                    "best_eval_iter": int(eval_best_iter),
                    # Protocole Crafter OFFICIEL (épisodes de training)
                    "crafter_score": float(cscore),
                    "train_episode_count": int(train_episode_count),
                    "train_success_rates": train_success_rates,
                    # History complet (métriques fines + toutes les EVAL)
                    "history": history,
                }, f, indent=2)

            # Nettoyer les anciens checkpoints (garder 3 derniers)
            old_ckpts = sorted(ckpt_dir.glob(f"dreamer_crafter_jax_{args.run_name}_iter*.npz"))
            if len(old_ckpts) > 3:
                for old in old_ckpts[:-3]:
                    old.unlink()

    total_elapsed = time.time() - t_start
    print(f"\nTraining fini en {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"Best EVAL achievements : {eval_best_ach:.2f} @ iter {eval_best_iter}"
          f"  (dernier : {history['eval_achievements'][-1]:.2f})"
          if history["eval_achievements"] else "Pas d'EVAL effectuée.")
    print()

    # Report profile (no-op si --profile pas activé)
    prof.report(total_elapsed)

    # Shutdown pool multiprocessing
    if env_pool is not None:
        env_pool.close()

    # En mode profile : pas de save final, on veut juste mesurer
    if args.profile:
        print("\n[PROFILE] Mode profile : skip checkpoint final.")
        print("=" * 60)
        print("DONE (profile)")
        print("=" * 60)
        return

    # ----------- Final save (merge des states pour récupérer les params finaux)
    (enc_f, rssm_f, dec_f, rew_f, cont_f), _opt = merge_wm_fn(wm_state)
    (actor_f, critic_f), _opt_ac = merge_ac_fn(ac_state)
    slow_f = merge_slow_fn(slow_state)

    final_ckpt = ckpt_dir / f"dreamer_crafter_jax_{args.run_name}.npz"
    save_checkpoint(
        final_ckpt, enc_f, rssm_f, dec_f, rew_f, cont_f,
        actor_f, critic_f, slow_f, args.train_iter, args, history,
        opt_wm=opt_wm, opt_ac=opt_ac, return_ema_std=return_ema_std,
        counters={"train_ach_counts": train_ach_counts,
                  "train_episode_count": train_episode_count,
                  "diag_wood_hist": diag_wood_hist,
                  "diag_action_counts": diag_action_counts,
                  "diag_reached_stone": diag_reached_stone},
    )
    print(f"Final checkpoint : {final_ckpt}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


def save_checkpoint(path, encoder, rssm, decoder, reward_head, continue_head,
                    actor, critic, slow_critic, it, args, history,
                    opt_wm=None, opt_ac=None, return_ema_std=None, counters=None):
    """
    Sauvegarde via numpy npz : convertit chaque state nnx.Param en numpy.
    Format simple (pas orbax) pour rester portable et minimal.

    Les clés `<module>.<path>` sont le format historique (lu par visualize_jax.py et
    experiments/credit_assignment_probe.py) — ne pas les renommer.

    Extras optionnels, préfixés `__` pour ne pas collisionner avec un nom de module :
      __opt_wm.*, __opt_ac.*  : état Adam (m/v). Sans eux, un --resume_from redémarre
                                l'optimiseur à froid.
      __return_ema_std        : les percentiles EMA P5/P95 du return. Sans eux, le
                                scale repart de son init (≈1.0) alors qu'un run mature
                                tourne à 5-8 → advantages gonflés d'autant pendant
                                plusieurs centaines d'itérations après la reprise.
    Le buffer et la PRNG key restent non sauvés (volumineux / peu utiles) : la reprise
    reste donc « soft », mais sans le saut de scale qui était le vrai danger.
    """
    def state_to_numpy(m, filt=nnx.Param):
        params = nnx.state(m, filt) if filt is not None else nnx.state(m)
        return {k: np.array(v) for k, v in flatten_state(params).items()}

    payload = {}
    for name, m in [
        ("encoder", encoder), ("rssm", rssm), ("decoder", decoder),
        ("reward_head", reward_head), ("continue_head", continue_head),
        ("actor", actor), ("critic", critic), ("slow_critic", slow_critic),
    ]:
        for k, v in state_to_numpy(m).items():
            payload[f"{name}.{k}"] = v

    for name, opt in (("__opt_wm", opt_wm), ("__opt_ac", opt_ac)):
        if opt is None:
            continue
        try:
            for k, v in state_to_numpy(opt, filt=None).items():
                payload[f"{name}.{k}"] = v
        except Exception as e:            # non fatal : le checkpoint reste utilisable
            print(f"  [ckpt] état optimizer {name} non sauvé ({type(e).__name__}: {e})")

    if return_ema_std is not None:
        payload["__return_ema_std"] = np.array(return_ema_std)

    np.savez_compressed(path, **payload)
    # Metadata side file (JSON). `counters` permet a une reprise de continuer le
    # crafter_score sur la meme base : il est cumule depuis l'iteration 0 (protocole
    # officiel), donc repartir de zero rendrait la metrique incomparable de part et
    # d'autre de la reprise.
    meta_path = path.with_suffix(".meta.json")
    meta = {"iter": int(it), "args": vars(args)}
    if counters is not None:
        meta["counters"] = {k: (dict(v) if isinstance(v, dict) else
                                (v.tolist() if hasattr(v, "tolist") else v))
                            for k, v in counters.items()}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def _set_at_path(state, path_parts, value):
    """Affecte récursivement une valeur dans un state nnx (cf. visualize_jax)."""
    def _resolve_key(container, key):
        try:
            key_int = int(key)
            if isinstance(container, (list, tuple)):
                return key_int
            try:
                container[key_int]
                return key_int
            except (KeyError, TypeError):
                return key
        except ValueError:
            return key

    node = state
    for p in path_parts[:-1]:
        node = node[_resolve_key(node, p)]
    leaf_key = _resolve_key(node, path_parts[-1])
    leaf = node[leaf_key]
    if hasattr(leaf, "value"):
        leaf.value = jnp.asarray(value)
    else:
        node[leaf_key] = jnp.asarray(value)


def load_optimizer_state(path, named_optimizers) -> int:
    """Recharge l'etat interne des optimizers (Adam mu/nu, count) depuis un checkpoint.

    Sans ca, une reprise redemarre Adam a froid : les moments valent 0, donc les
    premiers pas sont mal calibres (le pas effectif d'Adam vaut ~lr quel que soit le
    gradient tant que mu/nu n'ont pas chauffe) — exactement au moment ou le modele
    est le plus fragile.

    named_optimizers : {prefixe: nnx.Optimizer}, memes prefixes qu'a la sauvegarde
    (`__opt_wm`, `__opt_ac`). Retourne le nombre de tenseurs restaures.
    """
    ckpt = np.load(path, allow_pickle=False)
    total = 0
    for prefix, opt in named_optimizers.items():
        state = nnx.state(opt)
        expected = flatten_state(state)
        found = 0
        for key in expected:
            full = f"{prefix}.{key}"
            if full in ckpt.files:
                _set_at_path(state, key.split("."), ckpt[full])
                found += 1
        if found:
            nnx.update(opt, state)
            total += found
        print(f"[resume]   {prefix} : {found}/{len(expected)} tenseurs"
              + ("" if found else "  (absent du checkpoint — Adam repart a froid)"))
    return total


def load_checkpoint_into_modules(path, named_modules):
    """
    Charge un checkpoint .npz (format save_checkpoint) dans les modules.

    Les Variables nnx du state étant partagées avec le module, la mutation
    leaf.value se propage directement — pas de nnx.update nécessaire.

    Args:
        path : chemin du .npz
        named_modules : dict {prefix: module} (mêmes prefixes que save_checkpoint)

    Returns:
        start_iter (int) : l'itération du checkpoint (0 si meta absent).
    """
    path = Path(path)
    ckpt = np.load(path, allow_pickle=False)
    print(f"[resume] {len(ckpt.files)} clés chargées depuis {path.name}")

    for prefix, module in named_modules.items():
        state = nnx.state(module, nnx.Param)
        expected = flatten_state(state)
        missing = 0
        for key in expected:
            full_key = f"{prefix}.{key}"
            if full_key in ckpt.files:
                _set_at_path(state, key.split("."), ckpt[full_key])
            else:
                missing += 1
        tag = f" ({missing} clés manquantes !)" if missing else ""
        print(f"[resume]   {prefix} OK{tag}")

    start_iter = 0
    meta_path = path.with_suffix(".meta.json")
    if meta_path.exists():
        with open(meta_path) as f:
            start_iter = int(json.load(f).get("iter", 0))
    print(f"[resume] reprise à iter {start_iter}")
    return start_iter


def _variable_to_numpy(v):
    """Extrait le contenu d'une nnx.Variable ou d'un array en numpy."""
    # API Flax NNX moderne : variable[...] retourne l'array sous-jacent
    try:
        return np.array(v[...])
    except (TypeError, IndexError):
        pass
    if hasattr(v, "get_value"):
        return np.array(v.get_value())
    return np.array(v)


def flatten_state(state, prefix=""):
    """Flatten un pytree de state nnx.Param en {dotted_path: array}."""
    out = {}
    if hasattr(state, "items"):
        items = state.items()
    elif isinstance(state, (list, tuple)):
        items = enumerate(state)
    else:
        return {prefix.rstrip("."): _variable_to_numpy(state)}

    for k, v in items:
        new_prefix = f"{prefix}{k}."
        # Feuille = Variable (a un attribut .value) ou ndarray
        if isinstance(v, (jnp.ndarray, np.ndarray)):
            out[new_prefix.rstrip(".")] = np.array(v)
        elif hasattr(v, "items") or isinstance(v, (list, tuple)):
            out.update(flatten_state(v, new_prefix))
        else:
            # Variable nnx
            out[new_prefix.rstrip(".")] = _variable_to_numpy(v)
    return out


if __name__ == "__main__":
    main()

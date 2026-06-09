"""
Training complet du mini-Dreamer pour Tetris.

Pipeline :
    1. Warmup : 10k transitions random pour amorcer le buffer.
    2. Boucle principale (N itérations) :
       a. Collecte : l'actor joue dans l'env, transitions ajoutées au buffer.
       b. Train WM : encoder + RSSM + decoder + heads sur séquences du buffer.
       c. Train Actor + Critic : imagination 16 steps dans le WM, policy gradient.
    3. Eval périodique : score moyen sur quelques épisodes.
    4. Sauvegarde checkpoint + courbes.

Usage :
    python scripts/train_dreamer.py
    python scripts/train_dreamer.py --entropy_coef 0.05 --max_episode_steps 200 \\
                                     --invalid_penalty -0.5 --train_iter 1500 \\
                                     --run_name configE
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from tetris.env import TetrisEnv
from tetris.env.heuristic import select_heuristic_action
from src.model import (
    ReplayBuffer, Encoder, Decoder, RSSM,
    RewardHead, ContinueHead, Actor, Critic,
)


# ============================== Hyperparams

SEED = 42

# Données
BUFFER_CAPACITY = 100_000
WARMUP_STEPS = 10_000

# Training principal
TRAIN_ITERATIONS = 5000
COLLECT_PER_ITER = 10
WM_TRAIN_PER_ITER = 1          # 1 update par iter (rapide). Passer à 4-8 si compute dispo.
AC_TRAIN_PER_ITER = 1          # idem.

# Sampling
BATCH_SIZE = 64                # passé de 16 à 64 pour mieux saturer le GPU MPS
SEQ_LEN = 50
IMAGINATION_HORIZON = 16

# Optimization
LR_WM = 3e-4
LR_AC = 3e-4
GRAD_CLIP = 100.0

# RL params
GAMMA = 0.99
LAMBDA_GAE = 0.95
ENTROPY_COEF = 0.003

# Adaptive entropy (SAC-style automatic temperature tuning)
#   Si activé (--adaptive_alpha), entropy_coef devient un paramètre appris
#   qui s'ajuste pour maintenir H(π) proche de H_target.
#   H_target = H_TARGET_FRAC × log(action_dim)
#   Mécanisme : alpha_loss = -log_alpha * (H_target - H_current)
#     - si H < H_target → log_alpha monte → plus d'exploration
#     - si H > H_target → log_alpha descend → exploite
LR_ALPHA = 3e-4
INIT_ALPHA = 0.01
H_TARGET_FRAC = 0.4              # 0.4 × log(41) ≈ 1.49 nats par défaut

# Return normalization (DreamerV3 canonique)
RETURN_EMA_DECAY = 0.99      # vitesse de mise à jour des percentiles EMA
RETURN_PERCENTILE_LOW = 0.05
RETURN_PERCENTILE_HIGH = 0.95

# Critic EMA target network (DreamerV3 canonique)
CRITIC_TARGET_TAU = 0.98     # poids du slow_critic dans le Polyak update
                              # τ=0.98 → 50% mix après ~34 steps

# Architecture (mini-Dreamer)
EMBED_DIM = 128                 # garder, suffisant pour Tetris
H_DIM = 256                      # 128 → 256 (mémoire RNN x2)
Z_CATEGORIES = 16                # garder
Z_CLASSES = 16                   # garder (z_dim total = 256)
HIDDEN_DIM = 512                 # 256 → 512 (capacity MLPs x2)
FREE_BITS = 1.0
# Total params : ~4.87M (vs 1.54M avant) — modèle "big"
# state_dim = h+z = 512

# WM loss weights
W_RECON = 1.0
W_KL = 1.0
W_REWARD = 1.0
W_CONTINUE = 1.0

# Logging / eval
LOG_INTERVAL = 50
EVAL_INTERVAL = 500
EVAL_EPISODES = 20         # passé de 5 à 20 : variance d'estimation beaucoup plus faible
                            # coût marginal (~5 sec par eval au lieu de 1)


# ============================== Helpers

def symlog_np(x):
    """
    Symlog : sign(x) * log(1 + |x|). Compresse les extrêmes, préserve l'ordre.
    Linéaire pour |x| < 1, log pour |x| grand.
        symlog(0.1) = 0.095  (presque identique)
        symlog(5.0) = 1.79   (compressé)
        symlog(25)  = 3.26   (très compressé)
    DreamerV3 paper, stabilise critic + actor quand rewards ont des pics.
    """
    import numpy as np
    return np.sign(x) * np.log1p(np.abs(x))


def compute_schedule(iter_t, total_iter, args):
    """
    Schedule BC→RL canonique (DreamerV3 style).

    Phase 1 (0% - 20%)   : BC PUR              pg=0.0,           bcλ=args.bc_lambda
    Phase 2 (20% - 50%)  : TRANSITION linéaire pg=0→args.pg_coef, bcλ=bc→0.1*bc
    Phase 3 (50% - 100%) : RL avec BC résiduel pg=args.pg_coef,   bcλ=0.1*args.bc_lambda

    Sans --schedule : retourne les valeurs constantes (args.pg_coef, args.bc_lambda).
    """
    if not args.schedule:
        return args.pg_coef, args.bc_lambda

    p1_end = total_iter * 0.20
    p2_end = total_iter * 0.50

    if iter_t < p1_end:
        return 0.0, args.bc_lambda
    elif iter_t < p2_end:
        progress = (iter_t - p1_end) / (p2_end - p1_end)
        pg = progress * args.pg_coef
        bc = args.bc_lambda * (1.0 - 0.9 * progress)
        return pg, bc
    else:
        return args.pg_coef, args.bc_lambda * 0.1


def compute_lambda_returns(rewards, values_next, continues, gamma=0.99, lambda_=0.95):
    """
    Calcule les lambda-returns (TD-lambda, à la GAE).

    Args:
        rewards     : (B, T)   rewards imaginés
        values_next : (B, T)   value V(s_{t+1}) prédite par le critic
        continues   : (B, T)   1.0 si continue, 0.0 si done (P(continue))
        gamma       : facteur d'escompte
        lambda_     : trade-off Monte Carlo vs TD

    Returns:
        returns : (B, T) lambda-returns servant de target au critic
                  et de signal pour l'actor (via advantage = returns - value(s_t))
    """
    T = rewards.shape[1]
    returns = torch.zeros_like(rewards)
    # Boostrap final = dernière value
    next_return = values_next[:, -1]

    for t in reversed(range(T)):
        # G_t = r_t + γ × continue_t × [(1-λ) × V_{t+1}  +  λ × G_{t+1}]
        returns[:, t] = rewards[:, t] + gamma * continues[:, t] * (
            (1.0 - lambda_) * values_next[:, t] + lambda_ * next_return
        )
        next_return = returns[:, t]
    return returns


def imagine_trajectory(initial_state, actor, rssm, reward_head, continue_head, horizon):
    """
    Roll out une trajectoire imaginée dans le WM.

    Args:
        initial_state : dict {h, z} de shape (B, h_dim), (B, z_dim)
        ...

    Returns:
        dict avec :
            states     : (B, T, state_dim)   états à chaque step (avant l'action)
            actions    : (B, T)              actions prises
            rewards    : (B, T)              rewards prédits (depuis next state)
            continues  : (B, T)              P(continue) prédits (depuis next state)
            log_probs  : (B, T)              log π(a | s)  ← gradient flow
            entropies  : (B, T)              H(π)          ← gradient flow
            last_state : (B, state_dim)      pour le bootstrap final
    """
    state = {"h": initial_state["h"], "z": initial_state["z"]}
    action_dim = actor.action_dim

    # PERF #3 : pendant la boucle on collectionne SEULEMENT ce qui dépend du step-by-step
    # (states, actions, log_probs, entropies pour le gradient flow PG).
    # Les rewards et continues sont calculés EN BATCH après la boucle (1 forward au lieu de T).
    states, next_states, actions = [], [], []
    log_probs, entropies = [], []

    for _ in range(horizon):
        state_vec = torch.cat([state["h"], state["z"]], dim=-1)
        dist = actor.get_dist(state_vec)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        action_oh = F.one_hot(action, num_classes=action_dim).float()
        new_state, _ = rssm.imagine_step(state, action_oh)
        new_state_vec = torch.cat([new_state["h"], new_state["z"]], dim=-1)

        states.append(state_vec)
        next_states.append(new_state_vec)
        actions.append(action)
        log_probs.append(log_prob)
        entropies.append(entropy)

        state = new_state

    last_state_vec = next_states[-1]   # dernier next_state = état final

    # Stack tout
    states_traj = torch.stack(states, dim=1)            # (B, T, state_dim)
    next_states_traj = torch.stack(next_states, dim=1)  # (B, T, state_dim)

    # PERF #3 : reward et continue batchés sur TOUS les next_states en 1 forward
    rewards_pred = reward_head.predict(next_states_traj)              # (B, T)
    continue_pred = torch.sigmoid(continue_head(next_states_traj))    # (B, T)

    return {
        "states":     states_traj,                       # (B, T, state_dim) — avant action
        "actions":    torch.stack(actions, dim=1),       # (B, T)
        "rewards":    rewards_pred,                      # (B, T)
        "continues":  continue_pred,                     # (B, T)
        "log_probs":  torch.stack(log_probs, dim=1),     # (B, T) — gradient flow PG préservé
        "entropies":  torch.stack(entropies, dim=1),     # (B, T) — gradient flow entropy préservé
        "last_state": last_state_vec,                    # (B, state_dim)
    }


def eval_agent(env, encoder, rssm, actor, action_dim, device, n_episodes=5, use_mask=True):
    """
    Joue n_episodes complets avec l'actor en mode greedy. Retourne le score moyen.
    Si use_mask=True, applique le mask d'actions valides (recommandé).

    Returns:
        dict avec : score, length, lines (moyens), + n_invalid (debug, devrait être ~0 avec mask)
    """
    scores, lengths, lines, n_invalids = [], [], [], []
    encoder.eval(); rssm.eval(); actor.eval()

    with torch.no_grad():
        for ep in range(n_episodes):
            obs = env.reset(seed=10_000 + ep)  # seeds eval distincts du training
            state = rssm.init_state(1, device)
            prev_action = torch.zeros(1, action_dim, device=device)
            ep_reward, ep_len, ep_invalid = 0.0, 0, 0
            done = False

            while not done and ep_len < 500:
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                emb = encoder(obs_t)
                new_state, _, _ = rssm.observe_step(state, prev_action, emb)
                state_vec = torch.cat([new_state["h"], new_state["z"]], dim=-1)

                mask = None
                if use_mask:
                    mask_np = env.get_action_mask()
                    mask = torch.from_numpy(mask_np).unsqueeze(0).to(device)

                action = actor.act(state_vec, deterministic=True, mask=mask)
                action_int = action.item()
                obs, r, done, info = env.step(action_int)
                if info.get("invalid", False):
                    ep_invalid += 1
                state = new_state
                prev_action = F.one_hot(action, num_classes=action_dim).float()
                ep_reward += r
                ep_len += 1

            scores.append(ep_reward)
            lengths.append(ep_len)
            lines.append(env.game.lines_total)
            n_invalids.append(ep_invalid)

    encoder.train(); rssm.train(); actor.train()
    return {
        "score":      float(np.mean(scores)),
        "length":     float(np.mean(lengths)),
        "lines":      float(np.mean(lines)),
        "n_invalid":  float(np.mean(n_invalids)),
    }


# ============================== Main

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--entropy_coef", type=float, default=ENTROPY_COEF)
    p.add_argument("--invalid_penalty", type=float, default=-0.1,
                   help="Pénalité par action invalide (défaut: -0.1)")
    p.add_argument("--max_episode_steps", type=int, default=None,
                   help="Truncate l'épisode après N steps (défaut: None = pas de limite)")
    p.add_argument("--train_iter", type=int, default=TRAIN_ITERATIONS)
    p.add_argument("--eval_interval", type=int, default=EVAL_INTERVAL)
    p.add_argument("--run_name", type=str, default="default",
                   help="Suffix pour les fichiers de sortie (checkpoint, viz)")
    p.add_argument("--n_envs", type=int, default=4,
                   help="Nombre d'envs en parallèle pour la collecte (défaut: 4)")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                   help="Batch size pour le training (défaut: 64)")
    p.add_argument("--reward_shaping", action="store_true",
                   help="Active le reward shaping (dense reward sur hauteur/trous/bumpiness)")
    p.add_argument("--wm_train_per_iter", type=int, default=WM_TRAIN_PER_ITER,
                   help="Nombre d'updates du World Model par itération (défaut: 4)")
    p.add_argument("--ac_train_per_iter", type=int, default=AC_TRAIN_PER_ITER,
                   help="Nombre d'updates Actor+Critic par itération (défaut: 4)")
    p.add_argument("--adaptive_alpha", action="store_true",
                   help="Active l'adaptation auto du coefficient d'entropie (SAC-style)")
    p.add_argument("--h_target_frac", type=float, default=H_TARGET_FRAC,
                   help="Fraction du max entropy log(action_dim) à cibler (défaut: 0.4)")
    p.add_argument("--init_alpha", type=float, default=INIT_ALPHA,
                   help="Valeur initiale du coefficient d'entropie pour adaptive_alpha (défaut: 0.01)")
    p.add_argument("--heuristic_warmup", action="store_true",
                   help="Remplace le warmup random par l'heuristique Dellacherie "
                        "(injecte ~1500 line clears dans le buffer initial)")
    p.add_argument("--bc_lambda", type=float, default=0.0,
                   help="Coef Behavior Cloning auxiliaire (force l'actor à imiter "
                        "les actions du buffer). Défaut 0.0 = désactivé. "
                        "Recommandé : 0.3-0.5 avec --heuristic_warmup.")
    p.add_argument("--pg_coef", type=float, default=1.0,
                   help="Coef Policy Gradient. Défaut 1.0. Si --schedule, c'est la "
                        "valeur finale (départ à 0). Mettre 0.0 pour désactiver PG.")
    p.add_argument("--schedule", action="store_true",
                   help="Schedule BC->RL canonique (DreamerV3). "
                        "Phase 1 (0-20pct) BC pur, Phase 2 (20-50pct) transition, "
                        "Phase 3 (50-100pct) RL+BC résiduel.")
    p.add_argument("--symlog_reward", action="store_true",
                   help="Applique symlog(r) = sign(r) * log(1+|r|) sur les rewards "
                        "avant stockage. Réduit l'impact des pics (clear bonus +5, "
                        "Tetris +25) sans écraser les petits rewards (shaping). "
                        "Canonical DreamerV3 stabilization.")
    return p.parse_args()


def main():
    args = parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Run     : {args.run_name}")
    print(f"Config  : entropy={args.entropy_coef}  invalid_pen={args.invalid_penalty}  "
          f"max_ep_steps={args.max_episode_steps}  train_iter={args.train_iter}  "
          f"reward_shaping={args.reward_shaping}  n_envs={args.n_envs}  batch={args.batch_size}")
    print()

    # ------------------------- Setup envs (multi-env pour collecte parallèle)
    N_ENVS = args.n_envs
    envs = [
        TetrisEnv(
            seed=SEED + i,
            invalid_penalty=args.invalid_penalty,
            max_episode_steps=args.max_episode_steps,
            reward_shaping=args.reward_shaping,
        )
        for i in range(N_ENVS)
    ]
    obs_list = [env.reset() for env in envs]   # N obs (numpy)
    obs_dim = obs_list[0].shape[0]
    action_dim = envs[0].action_dim
    # Env de référence pour eval
    eval_env = TetrisEnv(
        seed=SEED + 9999,
        invalid_penalty=args.invalid_penalty,
        max_episode_steps=args.max_episode_steps,
        reward_shaping=args.reward_shaping,
    )

    buffer = ReplayBuffer(capacity=BUFFER_CAPACITY, obs_dim=obs_dim)
    # Buffer EXPERT séparé : contient SEULEMENT les transitions du warmup heuristique.
    # Utilisé exclusivement pour BC, jamais pollué par les transitions agent.
    # Active si --heuristic_warmup ET --bc_lambda > 0.
    use_expert_buffer = args.heuristic_warmup and args.bc_lambda > 0.0
    expert_buffer = (
        ReplayBuffer(capacity=WARMUP_STEPS, obs_dim=obs_dim) if use_expert_buffer else None
    )

    # ------------------------- Setup models
    encoder = Encoder(obs_dim=obs_dim, hidden_dim=HIDDEN_DIM, embed_dim=EMBED_DIM).to(device)
    rssm = RSSM(
        embed_dim=EMBED_DIM, action_dim=action_dim,
        h_dim=H_DIM, z_categories=Z_CATEGORIES, z_classes=Z_CLASSES,
        hidden_dim=HIDDEN_DIM,
    ).to(device)
    decoder = Decoder(embed_dim=rssm.state_dim, hidden_dim=HIDDEN_DIM, obs_dim=obs_dim).to(device)
    reward_head = RewardHead(state_dim=rssm.state_dim, hidden_dim=HIDDEN_DIM).to(device)
    continue_head = ContinueHead(state_dim=rssm.state_dim, hidden_dim=HIDDEN_DIM).to(device)
    actor = Actor(state_dim=rssm.state_dim, hidden_dim=HIDDEN_DIM, action_dim=action_dim).to(device)
    critic = Critic(state_dim=rssm.state_dim, hidden_dim=HIDDEN_DIM).to(device)

    # Slow critic = copie EMA du critic, utilisé comme target stable pour le bootstrap
    # Brise le feedback toxique "critic apprend contre ses propres prédictions décalées".
    slow_critic = Critic(state_dim=rssm.state_dim, hidden_dim=HIDDEN_DIM).to(device)
    slow_critic.load_state_dict(critic.state_dict())
    for p in slow_critic.parameters():
        p.requires_grad = False    # le slow ne reçoit pas de gradient direct

    wm_modules = [encoder, rssm, decoder, reward_head, continue_head]
    ac_modules = [actor, critic]    # slow_critic n'est PAS dans ac_modules (pas d'optim)

    n_wm = sum(sum(p.numel() for p in m.parameters()) for m in wm_modules)
    n_ac = sum(sum(p.numel() for p in m.parameters()) for m in ac_modules)
    print("=" * 60)
    print("Architecture")
    print("=" * 60)
    print(f"  WM modules (encoder+RSSM+decoder+heads) : {n_wm:>9,} params")
    print(f"  Actor + Critic                          : {n_ac:>9,} params")
    print(f"  TOTAL                                   : {n_wm + n_ac:>9,} params (~{(n_wm + n_ac)/1e6:.2f}M)")
    print()

    # Optimizers séparés
    wm_params = []
    for m in wm_modules:
        wm_params.extend(list(m.parameters()))
    ac_params = []
    for m in ac_modules:
        ac_params.extend(list(m.parameters()))

    optim_wm = torch.optim.Adam(wm_params, lr=LR_WM)
    optim_ac = torch.optim.Adam(ac_params, lr=LR_AC)

    # Adaptive entropy temperature (SAC-style)
    # log_alpha est appris : exp(log_alpha) = coefficient d'entropie effectif
    import math
    H_max = math.log(action_dim)
    H_target = args.h_target_frac * H_max
    log_alpha = torch.tensor(
        math.log(args.init_alpha), device=device, dtype=torch.float32, requires_grad=True,
    )
    optim_alpha = torch.optim.Adam([log_alpha], lr=LR_ALPHA)
    if args.adaptive_alpha:
        print(f"Adaptive α : ON   H_target={H_target:.3f}  (= {args.h_target_frac:.2f} × log({action_dim}))")
        print(f"             init α={args.init_alpha:.4f}   lr_α={LR_ALPHA}")
    else:
        print(f"Adaptive α : OFF (entropy_coef fixe = {args.entropy_coef})")
    print()

    # ------------------------- Phase 0 : Warmup
    print("=" * 60)
    if args.heuristic_warmup:
        print(f"Phase 0 : Warmup HEURISTIQUE Dellacherie ({WARMUP_STEPS} steps, {N_ENVS} envs)")
    else:
        print(f"Phase 0 : Warmup random ({WARMUP_STEPS} steps, {N_ENVS} envs en parallèle)")
    print("=" * 60)
    start = time.time()
    steps_per_env = WARMUP_STEPS // N_ENVS
    n_clears_warmup = 0
    for _ in range(steps_per_env):
        for i, env in enumerate(envs):
            prev_lines = env.game.lines_total
            if args.heuristic_warmup:
                action = select_heuristic_action(env)
            else:
                action = np.random.randint(0, action_dim)
            next_obs, r, done, _ = env.step(action)
            r_stored = symlog_np(r) if args.symlog_reward else r
            buffer.add(obs_list[i], action, r_stored, next_obs, done)
            # Si buffer expert actif : duplique dans le buffer expert (heuristique uniquement)
            if use_expert_buffer:
                expert_buffer.add(obs_list[i], action, r_stored, next_obs, done)
            n_clears_warmup += (env.game.lines_total - prev_lines)
            obs_list[i] = next_obs if not done else env.reset()
    elapsed = time.time() - start
    print(f"Buffer : {len(buffer)} transitions en {elapsed:.1f}s")
    if args.heuristic_warmup:
        # BUG FIX : reset lifetime_lines_cleared après warmup heuristique.
        # Sinon le bonus clear (5/(1+0.1*lifetime)) est déjà épuisé au démarrage RL
        # car le warmup a accumulé ~200 clears/env, ramenant le bonus de 5.0 à 0.24.
        # Conséquence : clear net devient marginal (+0.79 au lieu de +5.55)
        # et l'agent peut apprendre à ÉVITER le clear (car delta_potential = -0.45).
        total_lifetime_before = sum(e.lifetime_lines_cleared for e in envs)
        for e in envs:
            e.lifetime_lines_cleared = 0
        print(f"         {n_clears_warmup} line clears injectés ({n_clears_warmup/len(buffer)*100:.1f}% du buffer)")
        print(f"         lifetime_lines_cleared reset à 0 (était ~{total_lifetime_before/len(envs):.0f}/env)")
        if use_expert_buffer:
            print(f"         Buffer EXPERT séparé activé : {len(expert_buffer)} transitions pures (pour BC)")
    print()

    # ------------------------- Phase 1 : Main loop
    print("=" * 60)
    print(f"Phase 1 : Training Dreamer ({TRAIN_ITERATIONS} itérations)")
    print("=" * 60)

    history = {
        "iter": [],
        "loss_wm": [], "loss_recon": [], "loss_kl": [], "loss_reward": [], "loss_continue": [],
        "loss_actor": [], "loss_critic": [], "entropy": [],
        "env_reward_per_step": [],
        "eval_iter": [], "eval_score": [], "eval_length": [], "eval_lines": [],
    }

    collected_rewards = []
    # Multi-env states : (N_ENVS, ...) batched
    rssm_state_multi = rssm.init_state(N_ENVS, device)
    prev_actions_oh_multi = torch.zeros(N_ENVS, action_dim, device=device)

    # Return normalization EMA (DreamerV3 canonique)
    # Maintient des EMA de p5 et p95 des returns observés.
    # Le scale = max(1, p95_ema - p5_ema) sert à normaliser les advantages pour le PG.
    # Permet à entropy_coef d'être constant indépendamment de la magnitude des rewards.
    return_p5_ema = None
    return_p95_ema = None

    start = time.time()
    collect_per_iter_envs = max(1, COLLECT_PER_ITER // N_ENVS)
    for it in range(args.train_iter):
        # ----------------- (a) Collecte avec l'actor courant (N envs batched + MASK)
        for _ in range(collect_per_iter_envs):
            with torch.no_grad():
                # Batched forward sur les N obs + masks
                obs_batch = torch.from_numpy(np.stack(obs_list)).to(device)  # (N, obs_dim)
                masks_batch = torch.from_numpy(
                    np.stack([env.get_action_mask() for env in envs])
                ).to(device)  # (N, action_dim)

                emb = encoder(obs_batch)
                new_rssm_state, _, _ = rssm.observe_step(
                    rssm_state_multi, prev_actions_oh_multi, emb
                )
                state_vec = torch.cat([new_rssm_state["h"], new_rssm_state["z"]], dim=-1)
                # MASK appliqué : actions invalides ont proba = 0
                actions_t = actor.act(state_vec, mask=masks_batch)
                actions_int = actions_t.cpu().tolist()

            # Step chaque env séquentiellement (la step est CPU/rapide)
            new_obs_list = []
            new_prev_actions = torch.zeros(N_ENVS, action_dim, device=device)
            for i, env in enumerate(envs):
                next_obs, r, done, _ = env.step(actions_int[i])
                r_stored = symlog_np(r) if args.symlog_reward else r
                buffer.add(obs_list[i], actions_int[i], r_stored, next_obs, done)
                collected_rewards.append(r)   # raw pour le log r/step (interprétable)
                if done:
                    new_obs_list.append(env.reset())
                    # Reset le rssm state de cet env
                    new_rssm_state["h"][i].zero_()
                    new_rssm_state["z"][i].zero_()
                    # prev_action déjà zero pour cet env
                else:
                    new_obs_list.append(next_obs)
                    new_prev_actions[i] = F.one_hot(
                        actions_t[i:i+1], num_classes=action_dim
                    ).float().squeeze(0)
            obs_list = new_obs_list
            rssm_state_multi = new_rssm_state
            prev_actions_oh_multi = new_prev_actions

        # ----------------- (b) Train WM
        for _ in range(args.wm_train_per_iter):
            batch = buffer.sample_sequences(batch_size=args.batch_size, seq_len=SEQ_LEN)
            obs_seq = torch.from_numpy(batch["obs"]).to(device)
            actions_int = torch.from_numpy(batch["actions"]).long().to(device)
            rewards = torch.from_numpy(batch["rewards"]).to(device)
            dones = torch.from_numpy(batch["dones"]).to(device)
            actions_oh = F.one_hot(actions_int, num_classes=action_dim).float()

            embeddings = encoder(obs_seq)
            rssm_out = rssm.observe_sequence(embeddings, actions_oh, dones=dones)
            state_vec = torch.cat([rssm_out["h"], rssm_out["z"]], dim=-1)

            recon_logits = decoder(state_vec)
            continue_logit = continue_head(state_vec)

            loss_recon = F.binary_cross_entropy_with_logits(recon_logits, obs_seq)
            loss_kl = RSSM.kl_loss(rssm_out["post_logits"], rssm_out["prior_logits"], free_bits=FREE_BITS)
            # Reward head twohot symlog : cross-entropy au lieu de MSE
            loss_reward = reward_head.loss(state_vec, rewards)
            continue_target = 1.0 - dones.float()
            loss_continue = F.binary_cross_entropy_with_logits(continue_logit, continue_target)

            loss_wm = (
                W_RECON * loss_recon + W_KL * loss_kl
                + W_REWARD * loss_reward + W_CONTINUE * loss_continue
            )

            optim_wm.zero_grad()
            loss_wm.backward()
            torch.nn.utils.clip_grad_norm_(wm_params, max_norm=GRAD_CLIP)
            optim_wm.step()

        # ----------------- (c) Train Actor + Critic via imagination
        for _ in range(args.ac_train_per_iter):
            # Encode + RSSM observe pour avoir des états réels initiaux.
            # Si BC actif SANS expert buffer : on garde le gradient ici (BC partage le batch).
            # Sinon (BC avec expert_buffer OU pas de BC) : torch.no_grad() suffit (BC fait
            # son propre forward depuis expert_buffer).
            bc_uses_main_batch = args.bc_lambda > 0.0 and not use_expert_buffer
            with torch.no_grad():
                batch = buffer.sample_sequences(batch_size=args.batch_size, seq_len=SEQ_LEN)
                obs_seq = torch.from_numpy(batch["obs"]).to(device)
                actions_int = torch.from_numpy(batch["actions"]).long().to(device)
                dones = torch.from_numpy(batch["dones"]).to(device)
                actions_oh = F.one_hot(actions_int, num_classes=action_dim).float()
            if bc_uses_main_batch:
                # AVEC gradient : BC backprop dans encoder + RSSM via ce batch
                embeddings = encoder(obs_seq)
                rssm_out_init = rssm.observe_sequence(embeddings, actions_oh, dones=dones)
            else:
                with torch.no_grad():
                    embeddings = encoder(obs_seq)
                    rssm_out_init = rssm.observe_sequence(embeddings, actions_oh, dones=dones)

            # Flatten (B, T) → (B*T) pour avoir plein d'états initiaux
            B, T = rssm_out_init["h"].shape[:2]
            if bc_uses_main_batch:
                # BC partage ce batch (pas d'expert buffer) : non détaché pour gradient BC
                h_bc_main = rssm_out_init["h"].reshape(B * T, -1)
                z_bc_main = rssm_out_init["z"].reshape(B * T, -1)
                h_init = h_bc_main.detach()
                z_init = z_bc_main.detach()
            else:
                # Pas de BC OU BC sample depuis expert_buffer : main batch sans gradient
                h_init = rssm_out_init["h"].reshape(B * T, -1)
                z_init = rssm_out_init["z"].reshape(B * T, -1)
            initial_state = {"h": h_init, "z": z_init}

            # Imagine
            traj = imagine_trajectory(
                initial_state, actor, rssm, reward_head, continue_head, IMAGINATION_HORIZON
            )

            # Critic prediction VIVANT (gradient pour la loss critic)
            # .predict() decode les logits twohot en scalaire (espace original).
            states_traj = traj["states"]                 # (BT, H, state_dim)
            values_pred = critic.predict(states_traj)    # (BT, H) — scalaires decoded

            # SLOW critic pour le bootstrap des targets (no gradient, stable, decoded en scalaire)
            with torch.no_grad():
                slow_values = slow_critic.predict(states_traj)               # (BT, H)
                last_value_slow = slow_critic.predict(traj["last_state"])    # (BT,)
                values_next_slow = torch.cat(
                    [slow_values[:, 1:], last_value_slow.unsqueeze(-1)], dim=1
                )  # (BT, H)

                returns = compute_lambda_returns(
                    traj["rewards"].detach(), values_next_slow,
                    traj["continues"].detach(), gamma=GAMMA, lambda_=LAMBDA_GAE,
                )

                # --- Return normalization Percentile-EMA (DreamerV3 canonique) ---
                # Maintient p5 et p95 EMA, normalise l'advantage par leur range.
                # Critique : permet à entropy_coef d'être constant sans risque de noyade.
                flat_returns = returns.reshape(-1)
                p5 = torch.quantile(flat_returns, RETURN_PERCENTILE_LOW)
                p95 = torch.quantile(flat_returns, RETURN_PERCENTILE_HIGH)
                if return_p5_ema is None:
                    return_p5_ema = p5.detach().clone()
                    return_p95_ema = p95.detach().clone()
                else:
                    return_p5_ema  = RETURN_EMA_DECAY * return_p5_ema  + (1.0 - RETURN_EMA_DECAY) * p5
                    return_p95_ema = RETURN_EMA_DECAY * return_p95_ema + (1.0 - RETURN_EMA_DECAY) * p95
                # Scale clampé : MIN=1.0 (évite explosion sur petite range)
                #                MAX=5.0 (évite l'écrasement du signal quand le buffer
                #                contient des distributions très hétérogènes, ex:
                #                heuristic warmup avec returns +30 + agent débutant -10).
                #                Sans ce cap, scale peut monter à 50+ et advantage/scale ≈ 0.
                return_scale = torch.clamp(
                    return_p95_ema - return_p5_ema,
                    min=1.0,
                    max=5.0,
                )

            # Discount cumulatif : discount[t] = ∏_{k<t} (γ × continue_pred[k])
            # Pondère les steps post-terminal imaginé pour qu'ils ne contribuent pas à plein.
            with torch.no_grad():
                cont_det = traj["continues"].detach()
                gc = GAMMA * cont_det           # (BT, H)
                # discount[0] = 1 (présent), discount[t>0] = produit cumulatif
                discount = torch.cat([
                    torch.ones_like(gc[:, :1]),
                    torch.cumprod(gc[:, :-1], dim=1),
                ], dim=1)

            # Loss actor : advantage NORMALISÉ par scale + PONDÉRÉE par discount cumulatif
            advantages = ((returns - values_pred) / return_scale).detach()
            loss_actor_pg = -(discount * traj["log_probs"] * advantages).mean()

            # Coefficient d'entropie : appris si --adaptive_alpha, sinon constante CLI
            if args.adaptive_alpha:
                current_alpha = log_alpha.exp().detach()
            else:
                current_alpha = args.entropy_coef
            loss_actor_ent = -current_alpha * (discount * traj["entropies"]).mean()

            # BEHAVIOR CLONING auxiliaire (DreamerV3 paper, section "hard exploration").
            # Force l'actor à imiter les actions du buffer sur les ÉTATS RÉELS.
            # Particulièrement utile avec --heuristic_warmup : sans BC, l'imagination
            # horizon=16 ne contient jamais de clears donc l'actor n'apprend pas à
            # imiter l'heuristique. Avec BC, on injecte un signal direct supervisé.
            if args.bc_lambda > 0.0:
                if use_expert_buffer:
                    # BC : forward SÉPARÉ depuis expert_buffer (100pct heuristique pur)
                    with torch.no_grad():
                        batch_bc = expert_buffer.sample_sequences(
                            batch_size=args.batch_size, seq_len=SEQ_LEN
                        )
                        obs_bc = torch.from_numpy(batch_bc["obs"]).to(device)
                        actions_int_bc = torch.from_numpy(batch_bc["actions"]).long().to(device)
                        dones_bc = torch.from_numpy(batch_bc["dones"]).to(device)
                        actions_oh_bc = F.one_hot(actions_int_bc, num_classes=action_dim).float()
                    # Forward AVEC gradient pour BC
                    embeddings_bc = encoder(obs_bc)
                    rssm_out_bc = rssm.observe_sequence(embeddings_bc, actions_oh_bc, dones=dones_bc)
                    B_bc, T_bc = rssm_out_bc["h"].shape[:2]
                    h_bc = rssm_out_bc["h"].reshape(B_bc * T_bc, -1)
                    z_bc = rssm_out_bc["z"].reshape(B_bc * T_bc, -1)
                    actions_bc = actions_int_bc.reshape(-1)
                else:
                    # Pas d'expert buffer : utilise les états du main batch (avec grad)
                    h_bc = h_bc_main
                    z_bc = z_bc_main
                    actions_bc = actions_int.reshape(-1)

                # h_bc, z_bc NON détachés : le gradient BC remonte dans encoder + RSSM
                state_vec_bc = torch.cat([h_bc, z_bc], dim=-1)       # (B*T, state_dim)
                logits_bc = actor(state_vec_bc)                       # (B*T, action_dim)
                log_probs_bc = F.log_softmax(logits_bc, dim=-1)
                bc_loss = -log_probs_bc.gather(
                    1, actions_bc.unsqueeze(-1)
                ).squeeze(-1).mean()
            else:
                bc_loss = torch.tensor(0.0, device=device)

            # Schedule BC→RL : pg_coef monte de 0 à args.pg_coef, bc_lambda descend
            pg_coef_t, bc_lambda_t = compute_schedule(it, args.train_iter, args)
            loss_actor = pg_coef_t * loss_actor_pg + loss_actor_ent + bc_lambda_t * bc_loss

            # Loss critic : cross-entropy twohot symlog (DreamerV3 canonique)
            # ► robuste aux returns de grande magnitude, gradient stable
            loss_critic = critic.loss(states_traj, returns)

            loss_ac = loss_actor + loss_critic

            # Si BC actif avec gradient non-détaché, on doit aussi appliquer le grad
            # BC sur encoder + RSSM via optim_wm. Sinon le grad est calculé mais perdu
            # quand optim_wm.zero_grad() est appelé à la prochaine WM iter.
            bc_propagates_to_wm = args.bc_lambda > 0.0
            if bc_propagates_to_wm:
                optim_wm.zero_grad()    # clear encoder/RSSM grads avant BC backward
            optim_ac.zero_grad()
            loss_ac.backward()
            torch.nn.utils.clip_grad_norm_(ac_params, max_norm=GRAD_CLIP)
            if bc_propagates_to_wm:
                torch.nn.utils.clip_grad_norm_(wm_params, max_norm=GRAD_CLIP)
                optim_wm.step()         # apply BC gradient sur encoder + RSSM
            optim_ac.step()

            # Polyak update du slow_critic : θ_slow = τ × θ_slow + (1-τ) × θ_fast
            # τ=0.98 → le slow suit lentement le critic vivant, sert de target stable.
            with torch.no_grad():
                for p_slow, p_fast in zip(slow_critic.parameters(), critic.parameters()):
                    p_slow.data.mul_(CRITIC_TARGET_TAU).add_(
                        p_fast.data, alpha=1.0 - CRITIC_TARGET_TAU
                    )

            # Adaptive alpha update (SAC-style)
            #   alpha_loss = -log_alpha × (H_target - H_current.detach())
            #   ► si H_current < H_target → grad de log_alpha négatif → log_alpha monte → α monte
            #   ► si H_current > H_target → grad de log_alpha positif → log_alpha descend → α descend
            if args.adaptive_alpha:
                with torch.no_grad():
                    H_current = traj["entropies"].mean()
                alpha_loss = -(log_alpha * (H_target - H_current))
                optim_alpha.zero_grad()
                alpha_loss.backward()
                optim_alpha.step()

        # PERF #4 : on stocke les loss tenseurs sans .item() à chaque iter.
        # Les .item() (syncs MPS→CPU) sont concentrés au moment du log uniquement.
        last_loss_tensors = {
            "wm": loss_wm.detach(), "recon": loss_recon.detach(),
            "kl": loss_kl.detach(), "reward": loss_reward.detach(),
            "continue": loss_continue.detach(), "actor": loss_actor.detach(),
            "critic": loss_critic.detach(), "entropy": traj["entropies"].mean().detach(),
            "bc": bc_loss.detach(),
        }

        if (it + 1) % LOG_INTERVAL == 0:
            elapsed = time.time() - start
            ips = (it + 1) / elapsed
            # Flush des loss values vers history (8 .item() concentrés ici)
            vals = {k: t.item() for k, t in last_loss_tensors.items()}
            history["iter"].append(it)
            history["loss_wm"].append(vals["wm"])
            history["loss_recon"].append(vals["recon"])
            history["loss_kl"].append(vals["kl"])
            history["loss_reward"].append(vals["reward"])
            history["loss_continue"].append(vals["continue"])
            history["loss_actor"].append(vals["actor"])
            history["loss_critic"].append(vals["critic"])
            history["entropy"].append(vals["entropy"])
            history["env_reward_per_step"].append(
                float(np.mean(collected_rewards[-1000:])) if collected_rewards else 0.0
            )

            scale_val = return_scale.item() if return_scale is not None else 1.0
            alpha_val = log_alpha.exp().item() if args.adaptive_alpha else args.entropy_coef
            alpha_tag = f" α={alpha_val:.4f}" if args.adaptive_alpha else ""
            bc_tag = f" bc={vals['bc']:.2f}" if args.bc_lambda > 0 else ""
            if args.schedule:
                pg_now, bc_now = compute_schedule(it, args.train_iter, args)
                sched_tag = f" pg={pg_now:.2f} bcλ={bc_now:.2f}"
            else:
                sched_tag = ""
            print(
                f"  iter {it+1:5d}/{args.train_iter} | "
                f"wm={vals['wm']:.3f}  recon={vals['recon']:.3f}  "
                f"actor={vals['actor']:.3f}  critic={vals['critic']:.3f}  "
                f"H={vals['entropy']:.2f} scale={scale_val:.2f}{alpha_tag}{bc_tag}{sched_tag} | "
                f"r/step={history['env_reward_per_step'][-1]:.4f} | {ips:.1f} ips"
            )

        if (it + 1) % args.eval_interval == 0:
            eval_res = eval_agent(eval_env, encoder, rssm, actor, action_dim, device, n_episodes=EVAL_EPISODES)
            history["eval_iter"].append(it + 1)
            history["eval_score"].append(eval_res["score"])
            history["eval_length"].append(eval_res["length"])
            history["eval_lines"].append(eval_res["lines"])
            print(
                f"  >>> EVAL @ iter {it+1} : "
                f"score={eval_res['score']:.2f}  length={eval_res['length']:.0f}  "
                f"lines={eval_res['lines']:.2f}  invalid={eval_res['n_invalid']:.1f}"
            )

    total_elapsed = time.time() - start
    print(f"\nTraining fini en {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)\n")

    # ------------------------- Sauvegarde
    print("=" * 60)
    print("Sauvegarde + visualisations")
    print("=" * 60)
    runs_dir = Path("runs")
    ckpt_dir = Path("checkpoints")
    runs_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    # Courbes
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    axes = axes.flatten()

    def plot_curve(ax, key, title, log=False):
        h = history[key]
        ax.plot(h, linewidth=0.5, alpha=0.4, color="lightblue")
        window = 50
        if len(h) > window:
            smoothed = np.convolve(h, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(h)), smoothed, color="steelblue", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("iter")
        if log:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    plot_curve(axes[0], "loss_recon", "loss_recon (BCE)", log=True)
    plot_curve(axes[1], "loss_kl", "loss_kl (free bits)")
    plot_curve(axes[2], "loss_actor", "loss_actor (policy gradient)")
    plot_curve(axes[3], "loss_critic", "loss_critic (MSE lambda-returns)", log=True)
    plot_curve(axes[4], "entropy", "policy entropy (4=uniform, 0=déterministe)")
    plot_curve(axes[5], "env_reward_per_step", "env reward / step (rolling 1000)")

    if history["eval_iter"]:
        axes[5].plot(history["eval_iter"], history["eval_score"], "ro-",
                     label="eval score / ep")
        axes[5].legend()

    fig.suptitle(f"Dreamer training — {args.run_name}", fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    losses_png = runs_dir / f"dreamer_training_{args.run_name}.png"
    plt.savefig(losses_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Courbes : {losses_png}")

    # Checkpoint
    ckpt_path = ckpt_dir / f"dreamer_{args.run_name}.pt"
    torch.save({
        "encoder":       encoder.state_dict(),
        "rssm":          rssm.state_dict(),
        "decoder":       decoder.state_dict(),
        "reward_head":   reward_head.state_dict(),
        "continue_head": continue_head.state_dict(),
        "actor":         actor.state_dict(),
        "critic":        critic.state_dict(),
        "config": {
            "obs_dim": obs_dim, "action_dim": action_dim,
            "embed_dim": EMBED_DIM, "h_dim": H_DIM,
            "z_categories": Z_CATEGORIES, "z_classes": Z_CLASSES,
            "hidden_dim": HIDDEN_DIM,
        },
        "history": {k: v for k, v in history.items()},
    }, ckpt_path)
    print(f"Checkpoint : {ckpt_path}")

    # Final eval propre
    print("\n=== Final eval (10 episodes deterministic) ===")
    final_eval = eval_agent(eval_env, encoder, rssm, actor, action_dim, device, n_episodes=10)
    print(f"  Score moyen   : {final_eval['score']:.2f}")
    print(f"  Length moyen  : {final_eval['length']:.0f} steps")
    print(f"  Lines moyennes: {final_eval['lines']:.2f}")

    # Sauvegarde JSON résumé pour le sweep
    import json
    summary = {
        "run_name": args.run_name,
        "config": {
            "entropy_coef": args.entropy_coef,
            "invalid_penalty": args.invalid_penalty,
            "max_episode_steps": args.max_episode_steps,
            "train_iter": args.train_iter,
        },
        "final_eval": final_eval,
        "training_time_sec": total_elapsed,
    }
    summary_path = runs_dir / f"dreamer_summary_{args.run_name}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary       : {summary_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()

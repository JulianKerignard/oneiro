"""
Training complet du mini-Dreamer pour Crafter.

Pipeline :
    1. Warmup : 5k transitions random pour amorcer le buffer.
    2. Boucle principale (N itérations) :
       a. Collecte : l'actor joue dans l'env, transitions ajoutées au buffer.
       b. Train WM : CNN encoder + RSSM + CNN decoder + heads sur séquences du buffer.
       c. Train Actor + Critic : imagination 16 steps dans le WM, policy gradient.
    3. Eval périodique : achievements moyens sur quelques épisodes.
    4. Sauvegarde checkpoint + courbes.

Usage :
    python crafter_dreamer/scripts/train_dreamer.py
    python crafter_dreamer/scripts/train_dreamer.py --entropy_coef 0.005 \\
        --train_iter 30000 --n_envs 4 --batch_size 32 --run_name crafter_baseline_v1
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

from crafter_dreamer.env import CrafterEnv
from src.model import (
    ImageReplayBuffer, CNNEncoder, CNNDecoder, RSSM,
    RewardHead, ContinueHead, Actor, Critic, RNDModule,
)


# ============================== Hyperparams

SEED = 42

# Données — image plus lourde que vecteur Tetris, on réduit le buffer
BUFFER_CAPACITY = 50_000
WARMUP_STEPS = 5_000

# Training principal
TRAIN_ITERATIONS = 30000
COLLECT_PER_ITER = 10
WM_TRAIN_PER_ITER = 1
AC_TRAIN_PER_ITER = 1

# Sampling — batch plus petit (image)
BATCH_SIZE = 32
SEQ_LEN = 50
IMAGINATION_HORIZON = 16

# Optimization
LR_WM = 3e-4
LR_AC = 3e-4
GRAD_CLIP = 100.0

# RL params
GAMMA = 0.99
LAMBDA_GAE = 0.95
ENTROPY_COEF = 0.005

# Adaptive entropy (SAC-style automatic temperature tuning)
#   Si activé (--adaptive_alpha), entropy_coef devient un paramètre appris
#   qui s'ajuste pour maintenir H(π) proche de H_target.
#   H_target = H_TARGET_FRAC × log(action_dim)
LR_ALPHA = 3e-4
INIT_ALPHA = 0.01
H_TARGET_FRAC = 0.4              # 0.4 × log(17) ≈ 1.13 nats par défaut

# Return normalization (DreamerV3 canonique)
RETURN_EMA_DECAY = 0.99
RETURN_PERCENTILE_LOW = 0.05
RETURN_PERCENTILE_HIGH = 0.95

# Critic EMA target network (DreamerV3 canonique)
CRITIC_TARGET_TAU = 0.98

# Architecture (mini-Dreamer Palier 1 — identique à Tetris)
EMBED_DIM = 192          # Palier 2 (vs 128 palier 1)
H_DIM = 384              # Palier 2 (vs 256)
Z_CATEGORIES = 24        # Palier 2 (vs 16)
Z_CLASSES = 24           # Palier 2 (vs 16)
HIDDEN_DIM = 768         # Palier 2 (vs 512)
FREE_BITS = 1.0
# state_dim = h + z = 256 + 256 = 512

# WM loss weights
W_RECON = 1.0
W_KL = 1.0
W_REWARD = 1.0
W_CONTINUE = 1.0

# Logging / eval
LOG_INTERVAL = 50
EVAL_INTERVAL = 2000
EVAL_EPISODES = 10


# ============================== Helpers


def compute_lambda_returns(rewards, values_next, continues, gamma=0.99, lambda_=0.95):
    """
    Calcule les lambda-returns (TD-lambda, à la GAE).

    Args:
        rewards     : (B, T)
        values_next : (B, T)
        continues   : (B, T)
        gamma       : facteur d'escompte
        lambda_     : trade-off Monte Carlo vs TD

    Returns:
        returns : (B, T) lambda-returns servant de target au critic
                  et de signal pour l'actor (via advantage = returns - value(s_t))
    """
    T = rewards.shape[1]
    returns = torch.zeros_like(rewards)
    next_return = values_next[:, -1]

    for t in reversed(range(T)):
        returns[:, t] = rewards[:, t] + gamma * continues[:, t] * (
            (1.0 - lambda_) * values_next[:, t] + lambda_ * next_return
        )
        next_return = returns[:, t]
    return returns


def imagine_trajectory(initial_state, actor, rssm, reward_head, continue_head, horizon):
    """
    Roll out une trajectoire imaginée dans le WM.

    Returns:
        dict avec states, actions, rewards, continues, log_probs, entropies, last_state.
    """
    state = {"h": initial_state["h"], "z": initial_state["z"]}
    action_dim = actor.action_dim

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

    last_state_vec = next_states[-1]

    states_traj = torch.stack(states, dim=1)
    next_states_traj = torch.stack(next_states, dim=1)

    rewards_pred = reward_head.predict(next_states_traj)
    continue_pred = torch.sigmoid(continue_head(next_states_traj))

    return {
        "states":     states_traj,
        "actions":    torch.stack(actions, dim=1),
        "rewards":    rewards_pred,
        "continues":  continue_pred,
        "log_probs":  torch.stack(log_probs, dim=1),
        "entropies":  torch.stack(entropies, dim=1),
        "last_state": last_state_vec,
    }


def eval_agent(env, encoder, rssm, actor, action_dim, device, n_episodes=10, use_mask=True):
    """
    Joue n_episodes complets avec l'actor en mode greedy.

    Returns:
        dict avec : score, length, achievements (moyens), + n_invalid (debug).
        Pour Crafter, n_invalid est toujours 0 (toutes actions valides).
    """
    scores, lengths, achievements, n_invalids = [], [], [], []
    encoder.eval(); rssm.eval(); actor.eval()

    with torch.no_grad():
        for ep in range(n_episodes):
            obs = env.reset(seed=10_000 + ep)
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
            achievements.append(env.n_unlocked_episode)
            n_invalids.append(ep_invalid)

    encoder.train(); rssm.train(); actor.train()
    return {
        "score":          float(np.mean(scores)),
        "length":         float(np.mean(lengths)),
        "achievements":   float(np.mean(achievements)),
        "n_invalid":      float(np.mean(n_invalids)),
    }


# ============================== Main

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--entropy_coef", type=float, default=ENTROPY_COEF)
    p.add_argument("--train_iter", type=int, default=TRAIN_ITERATIONS)
    p.add_argument("--eval_interval", type=int, default=EVAL_INTERVAL)
    p.add_argument("--run_name", type=str, default="default",
                   help="Suffix pour les fichiers de sortie (checkpoint, viz)")
    p.add_argument("--n_envs", type=int, default=4,
                   help="Nombre d'envs en parallèle pour la collecte (défaut: 4)")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                   help="Batch size pour le training (défaut: 32)")
    p.add_argument("--wm_train_per_iter", type=int, default=WM_TRAIN_PER_ITER,
                   help="Nombre d'updates du World Model par itération (défaut: 1)")
    p.add_argument("--ac_train_per_iter", type=int, default=AC_TRAIN_PER_ITER,
                   help="Nombre d'updates Actor+Critic par itération (défaut: 1)")
    p.add_argument("--adaptive_alpha", action="store_true",
                   help="Active l'adaptation auto du coefficient d'entropie (SAC-style)")
    p.add_argument("--h_target_frac", type=float, default=H_TARGET_FRAC,
                   help="Fraction du max entropy log(action_dim) à cibler (défaut: 0.4)")
    p.add_argument("--init_alpha", type=float, default=INIT_ALPHA,
                   help="Valeur initiale du coefficient d'entropie pour adaptive_alpha (défaut: 0.01)")
    p.add_argument("--auto_explore", action="store_true",
                   help="Active la détection de stagnation : si 2 EVAL consécutifs sans progrès "
                        "(achievements n'augmente pas de +0.1), multiplie entropy_coef par 1.5. "
                        "Décay vers 1.0 quand progrès. Cap à 5.0.")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed pour reproductibilité (et multistart)")
    p.add_argument("--use_rnd", action="store_true",
                   help="Active RND (Random Network Distillation) pour exploration dirigée. "
                        "Ajoute un bonus intrinsèque par nouveauté d'état au reward.")
    p.add_argument("--rnd_coef", type=float, default=0.5,
                   help="Coef du bonus intrinsèque RND (défaut: 0.5). "
                        "Trop bas = pas d'effet, trop haut = ignore reward extrinsèque.")
    return p.parse_args()


def main():
    args = parse_args()

    # Seed depuis CLI (pour multistart) ou défaut
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Device auto-detect : cuda (cloud GPU) > mps (M4 Max) > cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device  : {device}")

    # OPTIMISATIONS GPU (safe, gain ~15-30pct sans risque sur les perfs train)
    if device.type == "cuda":
        # cuDNN benchmark : optimise les algos conv automatiquement
        # (recherche les meilleurs kernels à la 1re iter, puis cache)
        torch.backends.cudnn.benchmark = True
        # TF32 sur Ampere+ (A100, H100, RTX 30/40/Ada) : 10-20pct sur matmul
        # Précision réduite mais imperceptible sur training RL
        torch.set_float32_matmul_precision("high")
        print(f"   GPU optims : cudnn.benchmark=ON, float32_matmul='high' (TF32)")
    print(f"Run     : {args.run_name}")
    print(f"Config  : entropy={args.entropy_coef}  train_iter={args.train_iter}  "
          f"n_envs={args.n_envs}  batch={args.batch_size}")
    print()

    # ------------------------- Setup envs (multi-env pour collecte parallèle)
    N_ENVS = args.n_envs
    envs = [CrafterEnv(seed=SEED + i) for i in range(N_ENVS)]
    obs_list = [env.reset() for env in envs]   # N obs (numpy)
    obs_shape = obs_list[0].shape              # (3, 64, 64)
    action_dim = envs[0].action_dim            # 17

    # Env de référence pour eval
    eval_env = CrafterEnv(seed=SEED + 9999)

    buffer = ImageReplayBuffer(capacity=BUFFER_CAPACITY, obs_shape=obs_shape)

    # ------------------------- Setup models
    encoder = CNNEncoder(in_channels=3, embed_dim=EMBED_DIM, base_channels=32).to(device)
    rssm = RSSM(
        embed_dim=EMBED_DIM, action_dim=action_dim,
        h_dim=H_DIM, z_categories=Z_CATEGORIES, z_classes=Z_CLASSES,
        hidden_dim=HIDDEN_DIM,
    ).to(device)
    decoder = CNNDecoder(state_dim=rssm.state_dim, out_channels=3, base_channels=32).to(device)
    reward_head = RewardHead(state_dim=rssm.state_dim, hidden_dim=HIDDEN_DIM).to(device)
    continue_head = ContinueHead(state_dim=rssm.state_dim, hidden_dim=HIDDEN_DIM).to(device)
    actor = Actor(state_dim=rssm.state_dim, hidden_dim=HIDDEN_DIM, action_dim=action_dim).to(device)
    critic = Critic(state_dim=rssm.state_dim, hidden_dim=HIDDEN_DIM).to(device)

    # Slow critic = copie EMA du critic, target stable pour le bootstrap
    slow_critic = Critic(state_dim=rssm.state_dim, hidden_dim=HIDDEN_DIM).to(device)
    slow_critic.load_state_dict(critic.state_dict())
    for p in slow_critic.parameters():
        p.requires_grad = False

    # torch.compile DÉSACTIVÉ : a causé OOM en mode "reduce-overhead" et
    # ralentissement (0.5 ips) en mode "default". On garde juste cudnn + TF32
    # comme optims GPU (safe, +15-25pct sans risque).
    # Pour réactiver : décommenter et tester.
    # if device.type == "cuda":
    #     try:
    #         encoder = torch.compile(encoder)
    #         ...
    #     except Exception as e:
    #         pass

    wm_modules = [encoder, rssm, decoder, reward_head, continue_head]
    ac_modules = [actor, critic]

    # RND : exploration dirigée par curiosité (optionnel via --use_rnd)
    if args.use_rnd:
        rnd_module = RNDModule(
            in_channels=3, embed_dim=EMBED_DIM, base_channels=32,
        ).to(device)
        n_rnd = sum(p.numel() for p in rnd_module.parameters())
        n_rnd_train = sum(p.numel() for p in rnd_module.parameters() if p.requires_grad)
    else:
        rnd_module = None
        n_rnd = 0
        n_rnd_train = 0

    n_wm = sum(sum(p.numel() for p in m.parameters()) for m in wm_modules)
    n_ac = sum(sum(p.numel() for p in m.parameters()) for m in ac_modules)
    print("=" * 60)
    print("Architecture")
    print("=" * 60)
    print(f"  WM modules (CNNEnc+RSSM+CNNDec+heads) : {n_wm:>9,} params")
    print(f"  Actor + Critic                        : {n_ac:>9,} params")
    if args.use_rnd:
        print(f"  RND (target frozen + predictor)       : {n_rnd:>9,} params ({n_rnd_train:,} trainable)")
    print(f"  TOTAL                                 : {n_wm + n_ac + n_rnd:>9,} params (~{(n_wm + n_ac + n_rnd)/1e6:.2f}M)")
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
    # Optimizer RND séparé (entraîne SEULEMENT le predictor)
    if args.use_rnd:
        optim_rnd = torch.optim.Adam(
            [p for p in rnd_module.parameters() if p.requires_grad],
            lr=LR_WM,  # même LR que le WM
        )
    else:
        optim_rnd = None

    # Adaptive entropy temperature (SAC-style)
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

    # ------------------------- Phase 0 : Warmup random
    print("=" * 60)
    print(f"Phase 0 : Warmup random ({WARMUP_STEPS} steps, {N_ENVS} envs en parallèle)")
    print("=" * 60)
    start = time.time()
    steps_per_env = WARMUP_STEPS // N_ENVS
    for _ in range(steps_per_env):
        for i, env in enumerate(envs):
            action = np.random.randint(0, action_dim)
            next_obs, r, done, _ = env.step(action)
            buffer.add(obs_list[i], action, r, next_obs, done)
            obs_list[i] = next_obs if not done else env.reset()
    elapsed = time.time() - start
    print(f"Buffer : {len(buffer)} transitions en {elapsed:.1f}s")
    print(f"         Mémoire buffer : {buffer.memory_usage_mb():.1f} MB")
    print()

    # ------------------------- Phase 1 : Main loop
    print("=" * 60)
    print(f"Phase 1 : Training Dreamer ({args.train_iter} itérations)")
    print("=" * 60)

    history = {
        "iter": [],
        "loss_wm": [], "loss_recon": [], "loss_kl": [], "loss_reward": [], "loss_continue": [],
        "loss_actor": [], "loss_critic": [], "entropy": [],
        "env_reward_per_step": [],
        "eval_iter": [], "eval_score": [], "eval_length": [], "eval_achievements": [],
    }

    collected_rewards = []
    # Multi-env states : (N_ENVS, ...) batched
    rssm_state_multi = rssm.init_state(N_ENVS, device)
    prev_actions_oh_multi = torch.zeros(N_ENVS, action_dim, device=device)

    # Output dirs : configurables via env var pour Modal (vol persistant)
    # DÉFINIS AVANT la boucle pour permettre save intermédiaire à chaque EVAL.
    import os as _os
    output_root = Path(_os.environ.get("WORLDMODEL_OUTPUT_DIR", "."))
    runs_dir = output_root / "runs"
    ckpt_dir = output_root / "checkpoints"
    runs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dirs : runs={runs_dir.resolve()}  checkpoints={ckpt_dir.resolve()}")
    print()

    # Return normalization EMA (DreamerV3 canonique)
    return_p5_ema = None
    return_p95_ema = None

    # Auto-explore : détection de stagnation + boost entropy
    # Surveille les EVAL achievements et booste entropy_coef si pas de progrès.
    auto_explore_multiplier = 1.0   # × entropy_coef effective
    auto_explore_best = 0.0          # meilleur achievements observé
    auto_explore_consec_stag = 0     # nb EVAL sans progrès
    AUTO_EXPLORE_THRESHOLD = 0.1     # achievements amélioration min pour compter
    AUTO_EXPLORE_PATIENCE = 2         # EVAL sans progrès avant boost
    AUTO_EXPLORE_BOOST = 1.5          # multiplicateur du boost
    AUTO_EXPLORE_DECAY = 0.85         # facteur de décroissance après progrès
    AUTO_EXPLORE_MAX = 5.0            # cap maximum

    start = time.time()
    collect_per_iter_envs = max(1, COLLECT_PER_ITER // N_ENVS)
    for it in range(args.train_iter):
        # ----------------- (a) Collecte avec l'actor courant (N envs batched + MASK)
        for _ in range(collect_per_iter_envs):
            with torch.no_grad():
                # Batched forward sur les N obs + masks
                obs_batch = torch.from_numpy(np.stack(obs_list)).to(device)  # (N, 3, 64, 64)
                masks_batch = torch.from_numpy(
                    np.stack([env.get_action_mask() for env in envs])
                ).to(device)  # (N, action_dim) — all True pour Crafter

                emb = encoder(obs_batch)
                new_rssm_state, _, _ = rssm.observe_step(
                    rssm_state_multi, prev_actions_oh_multi, emb
                )
                state_vec = torch.cat([new_rssm_state["h"], new_rssm_state["z"]], dim=-1)
                actions_t = actor.act(state_vec, mask=masks_batch)
                actions_int = actions_t.cpu().tolist()

            # Step chaque env séquentiellement
            new_obs_list = []
            new_prev_actions = torch.zeros(N_ENVS, action_dim, device=device)

            # Pré-calcule les bonus RND sur les next_obs pour TOUS les envs en parallèle
            # (seulement si RND actif). On les ajoutera au reward de chaque env.
            rnd_bonuses = None
            if args.use_rnd:
                # Step d'abord tous les envs pour avoir next_obs, puis batched RND
                step_results = [env.step(actions_int[i]) for i, env in enumerate(envs)]
                next_obs_arr = np.stack([res[0] for res in step_results])  # (N_ENVS, 3, 64, 64)
                with torch.no_grad():
                    obs_t_rnd = torch.from_numpy(next_obs_arr).to(device)
                    bonus_raw = rnd_module.compute_bonus(obs_t_rnd)
                    bonus_norm = rnd_module.normalize_bonus(bonus_raw)
                rnd_bonuses = bonus_norm.cpu().numpy()  # (N_ENVS,)

            for i, env in enumerate(envs):
                if args.use_rnd:
                    next_obs, r_extr, done, _ = step_results[i]
                    r_intr = float(rnd_bonuses[i])
                    r = r_extr + args.rnd_coef * r_intr
                else:
                    next_obs, r, done, _ = env.step(actions_int[i])
                buffer.add(obs_list[i], actions_int[i], r, next_obs, done)
                collected_rewards.append(r)
                if done:
                    new_obs_list.append(env.reset())
                    new_rssm_state["h"][i].zero_()
                    new_rssm_state["z"][i].zero_()
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

            decoded_obs = decoder(state_vec)
            continue_logit = continue_head(state_vec)

            # Recon loss : MSE pour images (CNNDecoder applique sigmoid → [0,1])
            loss_recon = F.mse_loss(decoded_obs, obs_seq)
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

            # ----------------- (b.2) Train RND predictor (en parallèle du WM)
            # Le predictor apprend à imiter le target sur les obs DÉJÀ VUES.
            # Plus une obs est vue, plus l'erreur (= bonus) baisse → exploration auto-dirigée.
            if args.use_rnd:
                # Reshape obs_seq pour CNN (B*T, C, H, W)
                B_rnd, T_rnd = obs_seq.shape[:2]
                obs_flat = obs_seq.reshape(B_rnd * T_rnd, *obs_seq.shape[2:])
                loss_rnd = rnd_module.train_loss(obs_flat)
                optim_rnd.zero_grad()
                loss_rnd.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in rnd_module.parameters() if p.requires_grad],
                    max_norm=GRAD_CLIP,
                )
                optim_rnd.step()
                last_loss_rnd = loss_rnd.detach()
            else:
                last_loss_rnd = None

        # ----------------- (c) Train Actor + Critic via imagination
        for _ in range(args.ac_train_per_iter):
            # Encode + RSSM observe pour avoir des états réels initiaux (sans gradient).
            with torch.no_grad():
                batch = buffer.sample_sequences(batch_size=args.batch_size, seq_len=SEQ_LEN)
                obs_seq = torch.from_numpy(batch["obs"]).to(device)
                actions_int = torch.from_numpy(batch["actions"]).long().to(device)
                dones = torch.from_numpy(batch["dones"]).to(device)
                actions_oh = F.one_hot(actions_int, num_classes=action_dim).float()
                embeddings = encoder(obs_seq)
                rssm_out_init = rssm.observe_sequence(embeddings, actions_oh, dones=dones)

            # Flatten (B, T) → (B*T) pour avoir plein d'états initiaux
            B, T = rssm_out_init["h"].shape[:2]
            h_init = rssm_out_init["h"].reshape(B * T, -1)
            z_init = rssm_out_init["z"].reshape(B * T, -1)
            initial_state = {"h": h_init, "z": z_init}

            # Imagine
            traj = imagine_trajectory(
                initial_state, actor, rssm, reward_head, continue_head, IMAGINATION_HORIZON
            )

            # Critic prediction VIVANT (gradient pour la loss critic)
            states_traj = traj["states"]
            values_pred = critic.predict(states_traj)

            # SLOW critic pour le bootstrap des targets (no gradient)
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
                flat_returns = returns.reshape(-1)
                p5 = torch.quantile(flat_returns, RETURN_PERCENTILE_LOW)
                p95 = torch.quantile(flat_returns, RETURN_PERCENTILE_HIGH)
                if return_p5_ema is None:
                    return_p5_ema = p5.detach().clone()
                    return_p95_ema = p95.detach().clone()
                else:
                    return_p5_ema  = RETURN_EMA_DECAY * return_p5_ema  + (1.0 - RETURN_EMA_DECAY) * p5
                    return_p95_ema = RETURN_EMA_DECAY * return_p95_ema + (1.0 - RETURN_EMA_DECAY) * p95
                return_scale = torch.clamp(
                    return_p95_ema - return_p5_ema,
                    min=1.0,
                    max=5.0,
                )

            # Discount cumulatif : discount[t] = ∏_{k<t} (γ × continue_pred[k])
            with torch.no_grad():
                cont_det = traj["continues"].detach()
                gc = GAMMA * cont_det           # (BT, H)
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
            # Auto-explore : multiplier appliqué si stagnation détectée
            if args.auto_explore:
                current_alpha = current_alpha * auto_explore_multiplier
            loss_actor_ent = -current_alpha * (discount * traj["entropies"]).mean()

            loss_actor = loss_actor_pg + loss_actor_ent

            # Loss critic : cross-entropy twohot symlog (DreamerV3 canonique)
            loss_critic = critic.loss(states_traj, returns)

            loss_ac = loss_actor + loss_critic

            optim_ac.zero_grad()
            loss_ac.backward()
            torch.nn.utils.clip_grad_norm_(ac_params, max_norm=GRAD_CLIP)
            optim_ac.step()

            # Polyak update du slow_critic
            with torch.no_grad():
                for p_slow, p_fast in zip(slow_critic.parameters(), critic.parameters()):
                    p_slow.data.mul_(CRITIC_TARGET_TAU).add_(
                        p_fast.data, alpha=1.0 - CRITIC_TARGET_TAU
                    )

            # Adaptive alpha update (SAC-style)
            if args.adaptive_alpha:
                with torch.no_grad():
                    H_current = traj["entropies"].mean()
                alpha_loss = -(log_alpha * (H_target - H_current))
                optim_alpha.zero_grad()
                alpha_loss.backward()
                optim_alpha.step()

        # Stockage des tenseurs loss (on .item() seulement au moment du log)
        last_loss_tensors = {
            "wm": loss_wm.detach(), "recon": loss_recon.detach(),
            "kl": loss_kl.detach(), "reward": loss_reward.detach(),
            "continue": loss_continue.detach(), "actor": loss_actor.detach(),
            "critic": loss_critic.detach(), "entropy": traj["entropies"].mean().detach(),
        }

        if (it + 1) % LOG_INTERVAL == 0:
            elapsed = time.time() - start
            ips = (it + 1) / elapsed
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
            ax_tag = f" ax={auto_explore_multiplier:.2f}" if args.auto_explore and auto_explore_multiplier > 1.01 else ""
            rnd_tag = f" rnd={last_loss_rnd.item():.3f}" if args.use_rnd and last_loss_rnd is not None else ""
            print(
                f"  iter {it+1:5d}/{args.train_iter} | "
                f"wm={vals['wm']:.3f}  recon={vals['recon']:.4f}  "
                f"actor={vals['actor']:.3f}  critic={vals['critic']:.3f}  "
                f"H={vals['entropy']:.2f} scale={scale_val:.2f}{alpha_tag}{ax_tag}{rnd_tag} | "
                f"r/step={history['env_reward_per_step'][-1]:.4f} | {ips:.1f} ips"
            )

        if (it + 1) % args.eval_interval == 0:
            eval_res = eval_agent(eval_env, encoder, rssm, actor, action_dim, device, n_episodes=EVAL_EPISODES)
            history["eval_iter"].append(it + 1)
            history["eval_score"].append(eval_res["score"])
            history["eval_length"].append(eval_res["length"])
            history["eval_achievements"].append(eval_res["achievements"])
            ach_now = eval_res["achievements"]
            print(
                f"  >>> EVAL @ iter {it+1} : "
                f"score={eval_res['score']:.2f}  length={eval_res['length']:.0f}  "
                f"achievements={ach_now:.2f}"
            )

            # Auto-explore : détection de stagnation
            if args.auto_explore:
                if ach_now > auto_explore_best + AUTO_EXPLORE_THRESHOLD:
                    # Vrai progrès : décay multiplier vers 1.0
                    new_mult = max(1.0, auto_explore_multiplier * AUTO_EXPLORE_DECAY)
                    print(
                        f"  ✓ Auto-explore : progrès (+{ach_now - auto_explore_best:.2f}), "
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
                            f"  ⚠️  Auto-explore : STAGNATION (best={auto_explore_best:.2f}, "
                            f"now={ach_now:.2f}), boost multiplier "
                            f"{auto_explore_multiplier:.2f} → {new_mult:.2f}"
                        )
                        auto_explore_multiplier = new_mult
                        auto_explore_consec_stag = 0

            # SAVE CHECKPOINT intermédiaire (à chaque EVAL = tous les eval_interval iter)
            # Permet de récupérer en cas d'interruption (Lightning interruptible, OOM, etc.).
            ckpt_intermediate = ckpt_dir / f"dreamer_crafter_{args.run_name}_iter{it+1:06d}.pt"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "iter": it + 1,
                "encoder": encoder.state_dict(),
                "rssm": rssm.state_dict(),
                "decoder": decoder.state_dict(),
                "reward_head": reward_head.state_dict(),
                "continue_head": continue_head.state_dict(),
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "slow_critic": slow_critic.state_dict(),
                "history": history,
                "args": vars(args),
                "auto_explore_multiplier": auto_explore_multiplier,
                "auto_explore_best": auto_explore_best,
                "auto_explore_consec_stag": auto_explore_consec_stag,
            }, ckpt_intermediate)
            print(f"  💾 Checkpoint saved : {ckpt_intermediate.name}")

            # Sauvegarder aussi le summary JSON intermédiaire (achievements actuels)
            import json
            summary_path = runs_dir / f"dreamer_crafter_summary_{args.run_name}.json"
            runs_dir.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w") as f:
                json.dump({
                    "run_name": args.run_name,
                    "iter": it + 1,
                    "total_iter": args.train_iter,
                    "last_eval_score": float(eval_res["score"]),
                    "last_eval_length": float(eval_res["length"]),
                    "last_eval_achievements": float(eval_res["achievements"]),
                    "history_eval_iter": history["eval_iter"],
                    "history_eval_achievements": history["eval_achievements"],
                    "history_eval_score": history["eval_score"],
                }, f, indent=2)

            # Nettoyer les anciens checkpoints (garder seulement les 3 derniers)
            old_ckpts = sorted(ckpt_dir.glob(f"dreamer_crafter_{args.run_name}_iter*.pt"))
            if len(old_ckpts) > 3:
                for old in old_ckpts[:-3]:
                    old.unlink()

    total_elapsed = time.time() - start
    print(f"\nTraining fini en {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)\n")

    # ------------------------- Sauvegarde
    print("=" * 60)
    print("Sauvegarde + visualisations")
    print("=" * 60)
    # runs_dir et ckpt_dir déjà définis plus haut (avant la boucle de training)

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

    plot_curve(axes[0], "loss_recon", "loss_recon (MSE)", log=True)
    plot_curve(axes[1], "loss_kl", "loss_kl (free bits)")
    plot_curve(axes[2], "loss_actor", "loss_actor (policy gradient)")
    plot_curve(axes[3], "loss_critic", "loss_critic (twohot CE)", log=True)
    plot_curve(axes[4], "entropy", f"policy entropy (max={np.log(action_dim):.2f})")

    # Subplot achievements (eval) sur axes[5]
    if history["eval_iter"]:
        axes[5].plot(history["eval_iter"], history["eval_achievements"], "go-",
                     label="eval achievements / ep", linewidth=1.2)
        axes[5].set_title("eval achievements (mean per episode)")
        axes[5].set_xlabel("iter")
        axes[5].grid(True, alpha=0.3)
        axes[5].legend()
    else:
        axes[5].set_title("achievements (no eval yet)")
        axes[5].grid(True, alpha=0.3)

    fig.suptitle(f"Dreamer Crafter training — {args.run_name}", fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    losses_png = runs_dir / f"dreamer_crafter_{args.run_name}.png"
    plt.savefig(losses_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Courbes : {losses_png}")

    # Checkpoint
    ckpt_path = ckpt_dir / f"dreamer_crafter_{args.run_name}.pt"
    torch.save({
        "encoder":       encoder.state_dict(),
        "rssm":          rssm.state_dict(),
        "decoder":       decoder.state_dict(),
        "reward_head":   reward_head.state_dict(),
        "continue_head": continue_head.state_dict(),
        "actor":         actor.state_dict(),
        "critic":        critic.state_dict(),
        "config": {
            "obs_shape": obs_shape, "action_dim": action_dim,
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
    print(f"  Score moyen        : {final_eval['score']:.2f}")
    print(f"  Length moyen       : {final_eval['length']:.0f} steps")
    print(f"  Achievements moyen : {final_eval['achievements']:.2f}")

    # Sauvegarde JSON résumé
    import json
    summary = {
        "run_name": args.run_name,
        "config": {
            "entropy_coef": args.entropy_coef,
            "train_iter": args.train_iter,
            "n_envs": args.n_envs,
            "batch_size": args.batch_size,
        },
        "final_eval": final_eval,
        "training_time_sec": total_elapsed,
    }
    summary_path = runs_dir / f"dreamer_crafter_summary_{args.run_name}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary            : {summary_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()

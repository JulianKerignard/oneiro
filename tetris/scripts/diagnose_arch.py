"""
DIAGNOSTIC 2 : l'architecture Dreamer (encoder + RSSM + actor) peut-elle
                apprendre Dellacherie en BC-only ?

Différence avec diagnose_bc.py :
    diagnose_bc  : MLP simple sur obs brute → bc_loss → 0.0006 (marche)
    diagnose_arch: Dreamer arch complète sur obs → encoder → RSSM → (h,z) → actor

Pas de decoder, pas de reward_head, pas de critic, pas d'imagination, pas de PG.
Train tout via la SEULE loss BC : cross_entropy(actor(h,z), action_heuristic).

Verdict :
    ✓ bc_loss → 0.0  → l'architecture peut apprendre, PG sabote dans Dreamer normal
    ✗ bc_loss bloque > 0.5 → l'architecture est intrinsèquement limitée
       → suspects : encoder lossy, z catégorique bottleneck, RSSM dilue
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tetris.env import TetrisEnv, OBS_DIM, ACTION_DIM, HOLD_ACTION
from tetris.env.heuristic import select_heuristic_action
from src.model import Encoder, RSSM, Actor


# ============================== Config (matchs Dreamer)
SEED = 42
N_TRANSITIONS = 10_000          # buffer expert
N_TRAIN_STEPS = 3_000           # iter de BC training (3000 suffit pour voir convergence ou plateau)
BATCH_SIZE = 32                  # plus petit que train_dreamer (128) pour MPS
SEQ_LEN = 50                     # même que Dreamer
LR = 3e-4
LOG_EVERY = 50
EVAL_EVERY = 500
EVAL_EPISODES = 5

# Hyperparams architecture (matchs scripts/train_dreamer.py)
EMBED_DIM = 128
H_DIM = 256
Z_CATEGORIES = 16
Z_CLASSES = 16
HIDDEN_DIM = 512


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"OBS_DIM={OBS_DIM}, ACTION_DIM={ACTION_DIM}, state_dim={H_DIM + Z_CATEGORIES*Z_CLASSES}")
    print()

    # ============================== 1. Build expert buffer (sequences avec dones)
    print("=" * 60)
    print(f"Phase 1 : Génération de {N_TRANSITIONS} transitions heuristiques")
    print("=" * 60)
    env = TetrisEnv(seed=SEED)
    obs_list, action_list, done_list = [], [], []
    obs = env.reset()
    start = time.time()
    n_clears = 0
    n_eps = 0
    while len(obs_list) < N_TRANSITIONS:
        action = select_heuristic_action(env)
        obs_list.append(obs.copy())
        action_list.append(action)
        prev_lines = env.game.lines_total
        obs, _, done, _ = env.step(action)
        done_list.append(done)
        n_clears += (env.game.lines_total - prev_lines)
        if done:
            obs = env.reset()
            n_eps += 1
    obs_arr = np.array(obs_list, dtype=np.float32)
    action_arr = np.array(action_list, dtype=np.int64)
    done_arr = np.array(done_list, dtype=np.float32)
    print(f"  {N_TRANSITIONS} transitions, {n_clears} clears, {n_eps} episodes")

    # ============================== 2. Setup model (encoder + RSSM + actor)
    print()
    print("=" * 60)
    print(f"Phase 2 : Training BC-only sur architecture Dreamer")
    print("=" * 60)
    encoder = Encoder(obs_dim=OBS_DIM, hidden_dim=HIDDEN_DIM, embed_dim=EMBED_DIM).to(device)
    rssm = RSSM(
        embed_dim=EMBED_DIM, action_dim=ACTION_DIM,
        h_dim=H_DIM, z_categories=Z_CATEGORIES, z_classes=Z_CLASSES,
        hidden_dim=HIDDEN_DIM,
    ).to(device)
    actor = Actor(state_dim=rssm.state_dim, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM).to(device)

    n_params = sum(p.numel() for p in encoder.parameters())
    n_params += sum(p.numel() for p in rssm.parameters())
    n_params += sum(p.numel() for p in actor.parameters())
    print(f"  encoder + RSSM + actor : {n_params:,} params ({n_params/1e6:.2f}M)")

    optim = torch.optim.Adam(
        list(encoder.parameters()) + list(rssm.parameters()) + list(actor.parameters()),
        lr=LR,
    )

    # Tensors GPU
    obs_t = torch.from_numpy(obs_arr).to(device)
    action_t = torch.from_numpy(action_arr).to(device)
    done_t = torch.from_numpy(done_arr).to(device)
    actions_oh_all = F.one_hot(action_t, num_classes=ACTION_DIM).float()

    # ============================== 3. Train BC sur séquences
    print()
    eval_env = TetrisEnv(seed=SEED + 999)
    start = time.time()
    history_loss = []
    history_eval = []

    for it in range(N_TRAIN_STEPS):
        # Sample BATCH_SIZE séquences de longueur SEQ_LEN
        # Indices de départ : entre 0 et N_TRANSITIONS - SEQ_LEN
        start_idx = torch.randint(0, N_TRANSITIONS - SEQ_LEN, (BATCH_SIZE,), device=device)
        # Indices (B, T) = start_idx[:, None] + arange(T)
        offsets = torch.arange(SEQ_LEN, device=device).unsqueeze(0)
        idx = start_idx.unsqueeze(1) + offsets   # (B, T)

        obs_seq = obs_t[idx]                          # (B, T, obs_dim)
        actions_seq = action_t[idx]                   # (B, T)
        actions_oh_seq = actions_oh_all[idx]          # (B, T, action_dim)
        dones_seq = done_t[idx]                       # (B, T)

        # Forward : encoder → RSSM observe → actor
        # Reshape pour encoder : (B*T, obs_dim)
        B, T = obs_seq.shape[:2]
        emb = encoder(obs_seq.reshape(B * T, OBS_DIM)).reshape(B, T, EMBED_DIM)

        # Action one-hot DÉCALÉE de 1 : on passe a_{t-1} pour générer state à t
        # actions_oh_seq[:, t] = action prise AU state t
        # Pour observe_step on a besoin de prev_action = a_{t-1}
        # observe_sequence gère ça en interne avec prev_action = 0 au début
        rssm_out = rssm.observe_sequence(emb, actions_oh_seq, dones=dones_seq)

        # state_vec à chaque step : (B, T, state_dim)
        state_vec = torch.cat([rssm_out["h"], rssm_out["z"]], dim=-1)

        # Actor predit l'action AU step t depuis state à t
        # state_vec[:, t] correspond à AVANT l'action a_t car observe_step utilise
        # prev_action = a_{t-1} et embedding[t] (= obs_t) pour calculer state à t
        logits = actor(state_vec.reshape(B * T, -1)).reshape(B, T, ACTION_DIM)

        # BC loss : actor doit prédire action a_t depuis state à t
        log_probs = F.log_softmax(logits, dim=-1)
        bc_loss = -log_probs.gather(2, actions_seq.unsqueeze(-1)).squeeze(-1).mean()

        # Accuracy : % du temps où argmax = action heuristique
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == actions_seq).float().mean().item()

        optim.zero_grad()
        bc_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(rssm.parameters()) + list(actor.parameters()),
            max_norm=10.0,
        )
        optim.step()

        if (it + 1) % LOG_EVERY == 0:
            ips = (it + 1) / (time.time() - start)
            print(f"  iter {it+1:5d}/{N_TRAIN_STEPS} | bc_loss={bc_loss.item():.4f}  acc={acc*100:.1f}%  ({ips:.1f} it/s)")
            history_loss.append(bc_loss.item())

        # ============================== EVAL périodique
        if (it + 1) % EVAL_EVERY == 0:
            encoder.eval(); rssm.eval(); actor.eval()
            with torch.no_grad():
                scores, lengths, lines_list = [], [], []
                for ep in range(EVAL_EPISODES):
                    obs = eval_env.reset(seed=10000 + ep)
                    # Init state
                    state = rssm.init_state(batch_size=1, device=device)
                    prev_action_oh = torch.zeros(1, ACTION_DIM, device=device)
                    ep_score, ep_len = 0.0, 0
                    done = False
                    while not done and ep_len < 500:
                        mask = eval_env.get_action_mask()
                        obs_t_eval = torch.from_numpy(obs).unsqueeze(0).to(device)
                        # Encoder + RSSM observe_step
                        emb_eval = encoder(obs_t_eval)
                        state, _, _ = rssm.observe_step(state, prev_action_oh, emb_eval)
                        state_vec_eval = torch.cat([state["h"], state["z"]], dim=-1)
                        logits_e = actor(state_vec_eval)
                        # Apply mask
                        mask_e = torch.from_numpy(mask).unsqueeze(0).to(device)
                        logits_e = torch.where(mask_e, logits_e, torch.tensor(-1e9, device=device))
                        action = logits_e.argmax(dim=-1).item()
                        prev_action_oh = F.one_hot(
                            torch.tensor([action], device=device), num_classes=ACTION_DIM
                        ).float()
                        obs, r, done, _ = eval_env.step(action)
                        ep_score += r
                        ep_len += 1
                    scores.append(ep_score)
                    lengths.append(ep_len)
                    lines_list.append(eval_env.game.lines_total)
                avg_score = np.mean(scores)
                avg_len = np.mean(lengths)
                avg_lines = np.mean(lines_list)
                print(f"  >>> EVAL : score={avg_score:.2f}  length={avg_len:.0f}  lines={avg_lines:.2f}")
                history_eval.append((it+1, avg_score, avg_len, avg_lines))
            encoder.train(); rssm.train(); actor.train()

    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    final_loss = history_loss[-1] if history_loss else None
    last_eval = history_eval[-1] if history_eval else None
    print(f"  bc_loss final : {final_loss:.4f}")
    if last_eval:
        print(f"  EVAL final    : score={last_eval[1]:.2f}  length={last_eval[2]:.0f}  lines={last_eval[3]:.2f}")
    print()
    if final_loss is not None and final_loss < 0.3:
        print("  ✅ bc_loss < 0.3 → l'architecture Dreamer PEUT apprendre l'heuristique.")
        print("     → Dans Dreamer normal, c'est PG qui sabote le BC.")
        print("     → Test suivant : désactiver PG dans Dreamer.")
    elif final_loss is not None and final_loss < 0.7:
        print("  ⚠️  bc_loss intermédiaire → architecture limitée mais pas complètement.")
    else:
        print("  ❌ bc_loss bloque > 0.7 → l'architecture Dreamer NE PEUT PAS apprendre Dellacherie.")
        print("     → Suspect : encoder lossy, z catégorique bottleneck, ou RSSM dilue.")
        print("     → Test suivant : encoder seul (sans RSSM).")


if __name__ == "__main__":
    main()

"""
DIAGNOSTIC : peut-on apprendre l'heuristique Dellacherie via BC PUR ?

But : isoler la capacité de l'actor à apprendre, sans WM, sans RL.

Pipeline :
    1. Génère un buffer 10k transitions via l'heuristique Dellacherie
    2. Train un Actor SEUL en supervised : minimize -log P(a_heuristic | obs)
    3. Logger bc_loss au fil du temps
    4. Eval périodique : l'actor seul joue, on compte les lines

Verdict :
    - bc_loss < 0.2 ET lines > 1 en eval
      → l'actor PEUT apprendre, problème dans le pipeline RL/WM
    - bc_loss bloque > 0.7
      → capacité actor insuffisante OU bruit dans le buffer
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


# ============================== Config

SEED = 42
N_TRANSITIONS = 10_000          # taille du buffer expert
N_TRAIN_STEPS = 5_000           # iterations de BC training
BATCH_SIZE = 256
LR = 3e-4
HIDDEN_DIM = 512
LOG_EVERY = 100
EVAL_EVERY = 500
EVAL_EPISODES = 5


# ============================== Simple Actor (obs directe → action, pas de WM)

class SimpleActor(nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"OBS_DIM = {OBS_DIM}, ACTION_DIM = {ACTION_DIM}")
    print()

    # ----------------------------------------------------- 1. Build expert buffer
    print("=" * 60)
    print(f"Phase 1 : Génération de {N_TRANSITIONS} transitions heuristiques")
    print("=" * 60)
    env = TetrisEnv(seed=SEED)
    obs_list, action_list, mask_list = [], [], []
    obs = env.reset()
    start = time.time()
    n_clears = 0
    n_eps = 0
    while len(obs_list) < N_TRANSITIONS:
        mask = env.get_action_mask()
        action = select_heuristic_action(env)
        obs_list.append(obs.copy())
        action_list.append(action)
        mask_list.append(mask.copy())
        prev_lines = env.game.lines_total
        obs, _, done, _ = env.step(action)
        n_clears += (env.game.lines_total - prev_lines)
        if done:
            obs = env.reset()
            n_eps += 1
    obs_arr = np.array(obs_list, dtype=np.float32)
    action_arr = np.array(action_list, dtype=np.int64)
    mask_arr = np.array(mask_list, dtype=bool)
    print(f"  {N_TRANSITIONS} transitions générées en {time.time()-start:.1f}s")
    print(f"  {n_clears} line clears injectés ({n_clears/N_TRANSITIONS*100:.1f}%)")
    print(f"  {n_eps} episodes complets ({N_TRANSITIONS/n_eps:.0f} steps/episode)")

    # ----------------------------------------------------- 2. Setup model
    print()
    print("=" * 60)
    print(f"Phase 2 : Training BC pur ({N_TRAIN_STEPS} iter, batch={BATCH_SIZE})")
    print("=" * 60)
    actor = SimpleActor(OBS_DIM, HIDDEN_DIM, ACTION_DIM).to(device)
    optim = torch.optim.Adam(actor.parameters(), lr=LR)
    n_params = sum(p.numel() for p in actor.parameters())
    print(f"  Actor : {n_params:,} params ({n_params/1e6:.2f}M)")
    print()

    # Tensors GPU
    obs_t = torch.from_numpy(obs_arr).to(device)
    action_t = torch.from_numpy(action_arr).to(device)

    # ----------------------------------------------------- 3. Train BC
    eval_env = TetrisEnv(seed=SEED + 999)

    history_loss = []
    history_eval = []
    start = time.time()
    for it in range(N_TRAIN_STEPS):
        idx = torch.randint(0, N_TRANSITIONS, (BATCH_SIZE,), device=device)
        obs_batch = obs_t[idx]
        action_batch = action_t[idx]

        logits = actor(obs_batch)
        log_probs = F.log_softmax(logits, dim=-1)
        bc_loss = -log_probs.gather(1, action_batch.unsqueeze(-1)).squeeze(-1).mean()

        # Accuracy = % du batch où argmax == action heuristique
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == action_batch).float().mean().item()

        optim.zero_grad()
        bc_loss.backward()
        optim.step()

        if (it + 1) % LOG_EVERY == 0:
            ips = (it + 1) / (time.time() - start)
            print(f"  iter {it+1:5d}/{N_TRAIN_STEPS} | bc_loss={bc_loss.item():.4f}  acc={acc*100:.1f}%  ({ips:.0f} it/s)")
            history_loss.append(bc_loss.item())

        # ----------------------------------------------- Eval périodique
        if (it + 1) % EVAL_EVERY == 0:
            actor.eval()
            with torch.no_grad():
                scores, lengths, lines_list = [], [], []
                for ep in range(EVAL_EPISODES):
                    obs = eval_env.reset(seed=10000 + ep)
                    ep_score, ep_len = 0.0, 0
                    done = False
                    while not done and ep_len < 500:
                        mask = eval_env.get_action_mask()
                        obs_e = torch.from_numpy(obs).unsqueeze(0).to(device)
                        logits_e = actor(obs_e)
                        # Applique le mask : actions invalides → -inf
                        mask_e = torch.from_numpy(mask).unsqueeze(0).to(device)
                        logits_e = torch.where(mask_e, logits_e, torch.tensor(-1e9, device=device))
                        action_e = logits_e.argmax(dim=-1).item()
                        obs, r, done, _ = eval_env.step(action_e)
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
            actor.train()

    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    final_loss = history_loss[-1] if history_loss else None
    last_eval = history_eval[-1] if history_eval else None
    print(f"  bc_loss final  : {final_loss:.4f}")
    if last_eval:
        print(f"  EVAL final     : score={last_eval[1]:.2f}  length={last_eval[2]:.0f}  lines={last_eval[3]:.2f}")

    print()
    if final_loss is not None and final_loss < 0.3:
        print("  ✅ bc_loss < 0.3 → l'actor PEUT apprendre l'heuristique.")
        print("     Le bottleneck dans Dreamer est ailleurs : WM/imagination/PG.")
    elif final_loss is not None and final_loss < 0.7:
        print("  ⚠️  bc_loss intermédiaire → apprentissage partiel, peut s'améliorer.")
    else:
        print("  ❌ bc_loss bloque > 0.7 → problème de capacité ou bruit dans le buffer.")
        print("     Impossible pour l'actor de mémoriser l'heuristique même en pur supervised.")


if __name__ == "__main__":
    main()

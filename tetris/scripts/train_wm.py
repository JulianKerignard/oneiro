"""
Training du World Model complet (encoder + RSSM + decoder + heads).

Pipeline :
    1. Collecte ~10 000 transitions random dans le buffer
    2. Boucle d'entraînement :
       - Sample séquences (B=16, T=50)
       - Forward : encoder → RSSM → (decoder + reward_head + continue_head)
       - Multi-loss : reconstruction + KL + reward + continue
       - Backprop + step
    3. Sauvegarde checkpoint + courbes de loss + viz reconstructions

Usage : python scripts/train_wm.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from tetris.env import TetrisEnv
from tetris.env.constants import PIECES
from src.model import (
    ReplayBuffer, Encoder, Decoder, RSSM, RewardHead, ContinueHead,
)


# ============================== Hyperparams

SEED = 42

# Données
BUFFER_CAPACITY = 50_000
COLLECT_STEPS = 10_000

# Training
TRAIN_STEPS = 5000
BATCH_SIZE = 16
SEQ_LEN = 50
LR = 3e-4
GRAD_CLIP = 1000.0
LOG_INTERVAL = 100

# Architecture (mini-DreamerV3)
EMBED_DIM = 128
H_DIM = 128
Z_CATEGORIES = 16
Z_CLASSES = 16
HIDDEN_DIM = 256

# Loss weights et options
W_RECON = 1.0
W_KL = 1.0
W_REWARD = 1.0
W_CONTINUE = 1.0
FREE_BITS = 1.0


# ============================== Main

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device : {device}")
    print()

    # ------------------------- Setup env + buffer
    env = TetrisEnv(seed=SEED)
    obs = env.reset()
    obs_dim = obs.shape[0]
    action_dim = env.action_dim

    buffer = ReplayBuffer(capacity=BUFFER_CAPACITY, obs_dim=obs_dim)

    # ------------------------- Setup models
    encoder = Encoder(obs_dim=obs_dim, hidden_dim=HIDDEN_DIM, embed_dim=EMBED_DIM).to(device)
    rssm = RSSM(
        embed_dim=EMBED_DIM,
        action_dim=action_dim,
        h_dim=H_DIM,
        z_categories=Z_CATEGORIES,
        z_classes=Z_CLASSES,
        hidden_dim=HIDDEN_DIM,
    ).to(device)
    # Decoder prend en input le STATE complet (h + z), pas juste l'embedding
    decoder = Decoder(embed_dim=rssm.state_dim, hidden_dim=HIDDEN_DIM, obs_dim=obs_dim).to(device)
    reward_head = RewardHead(state_dim=rssm.state_dim, hidden_dim=HIDDEN_DIM).to(device)
    continue_head = ContinueHead(state_dim=rssm.state_dim, hidden_dim=HIDDEN_DIM).to(device)

    modules = [encoder, rssm, decoder, reward_head, continue_head]
    module_names = ["Encoder", "RSSM", "Decoder", "RewardHead", "ContinueHead"]

    print("=" * 60)
    print("Architecture du World Model")
    print("=" * 60)
    total = 0
    for name, mod in zip(module_names, modules):
        n = sum(p.numel() for p in mod.parameters())
        print(f"  {name:14s} : {n:>10,} params")
        total += n
    print(f"  {'TOTAL':14s} : {total:>10,} params (~{total/1e6:.2f}M)")
    print()

    # Optimizer unique sur tous les params
    all_params = []
    for m in modules:
        all_params.extend(list(m.parameters()))
    optim = torch.optim.Adam(all_params, lr=LR)

    # ------------------------- Phase 1 : Collecte
    print("=" * 60)
    print(f"Phase 1 : Collecte de {COLLECT_STEPS} transitions random")
    print("=" * 60)
    start = time.time()
    for _ in range(COLLECT_STEPS):
        action = np.random.randint(0, action_dim)
        next_obs, r, done, _ = env.step(action)
        buffer.add(obs, action, r, next_obs, done)
        obs = next_obs if not done else env.reset()
    print(f"Buffer rempli : {len(buffer)} transitions en {time.time()-start:.1f}s")
    n_dones = int(buffer.dones[: buffer.size].sum())
    print(f"   Episodes terminés : {n_dones}")
    print(f"   Memoire buffer    : {buffer.memory_usage_mb():.1f} MB")
    print()

    # ------------------------- Phase 2 : Training
    print("=" * 60)
    print(f"Phase 2 : Training WM ({TRAIN_STEPS} steps)")
    print("=" * 60)
    for m in modules:
        m.train()

    loss_history = {"total": [], "recon": [], "kl": [], "reward": [], "continue": []}

    start = time.time()
    for step in range(TRAIN_STEPS):
        batch = buffer.sample_sequences(batch_size=BATCH_SIZE, seq_len=SEQ_LEN)

        # → tensors sur device
        obs_seq = torch.from_numpy(batch["obs"]).to(device)            # (B, T, 276)
        actions_int = torch.from_numpy(batch["actions"]).long().to(device)  # (B, T)
        rewards = torch.from_numpy(batch["rewards"]).to(device)        # (B, T)
        dones = torch.from_numpy(batch["dones"]).to(device)             # (B, T) bool

        # One-hot actions
        actions_oh = F.one_hot(actions_int, num_classes=action_dim).float()  # (B, T, A)

        # Forward
        embeddings = encoder(obs_seq)                                        # (B, T, E)
        rssm_out = rssm.observe_sequence(embeddings, actions_oh, dones=dones)
        state_vec = torch.cat([rssm_out["h"], rssm_out["z"]], dim=-1)        # (B, T, state_dim)

        recon_logits = decoder(state_vec)                                    # (B, T, 276)
        reward_pred = reward_head(state_vec)                                  # (B, T)
        continue_logit = continue_head(state_vec)                             # (B, T)

        # Losses
        loss_recon = F.binary_cross_entropy_with_logits(recon_logits, obs_seq)
        loss_kl = RSSM.kl_loss(
            rssm_out["post_logits"], rssm_out["prior_logits"], free_bits=FREE_BITS
        )
        loss_reward = F.mse_loss(reward_pred, rewards)
        continue_target = 1.0 - dones.float()
        loss_continue = F.binary_cross_entropy_with_logits(continue_logit, continue_target)

        loss_total = (
            W_RECON * loss_recon
            + W_KL * loss_kl
            + W_REWARD * loss_reward
            + W_CONTINUE * loss_continue
        )

        # Backward
        optim.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=GRAD_CLIP)
        optim.step()

        # Log
        loss_history["total"].append(loss_total.item())
        loss_history["recon"].append(loss_recon.item())
        loss_history["kl"].append(loss_kl.item())
        loss_history["reward"].append(loss_reward.item())
        loss_history["continue"].append(loss_continue.item())

        if (step + 1) % LOG_INTERVAL == 0:
            avg = {k: float(np.mean(v[-LOG_INTERVAL:])) for k, v in loss_history.items()}
            elapsed = time.time() - start
            sps = (step + 1) / elapsed
            print(
                f"step {step+1:5d}/{TRAIN_STEPS}  |  "
                f"tot={avg['total']:.4f}  recon={avg['recon']:.4f}  "
                f"kl={avg['kl']:.3f}  rew={avg['reward']:.4f}  cont={avg['continue']:.4f}  "
                f"|  {sps:.0f} sps"
            )

    total_elapsed = time.time() - start
    print()
    print(f"Training fini en {total_elapsed:.1f}s")
    print()

    # ------------------------- Phase 3 : Sauvegarde + viz
    print("=" * 60)
    print("Phase 3 : Sauvegarde + visualisations")
    print("=" * 60)

    for m in modules:
        m.eval()

    runs_dir = Path("runs")
    ckpt_dir = Path("checkpoints")
    runs_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    # Courbes de losses
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.flatten()
    keys = ["recon", "kl", "reward", "continue"]
    titles = {
        "recon": "loss_recon (BCE, log)", "kl": "loss_kl (free bits clamp)",
        "reward": "loss_reward (MSE, log)", "continue": "loss_continue (BCE, log)",
    }
    for ax, key in zip(axes, keys):
        hist = loss_history[key]
        ax.plot(hist, linewidth=0.5, alpha=0.4, color="lightblue")
        # Smoothed
        window = 100
        if len(hist) > window:
            smoothed = np.convolve(hist, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(hist)), smoothed, color="steelblue", linewidth=1.2)
        ax.set_title(titles[key])
        ax.set_xlabel("step")
        if key in ("recon", "reward", "continue"):
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    fig.suptitle("World Model training losses", fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    losses_png = runs_dir / "wm_losses.png"
    plt.savefig(losses_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Courbes losses     : {losses_png}")

    # Reconstructions WM sur quelques exemples
    with torch.no_grad():
        batch = buffer.sample_sequences(batch_size=4, seq_len=20)
        obs_seq = torch.from_numpy(batch["obs"]).to(device)
        actions_int = torch.from_numpy(batch["actions"]).long().to(device)
        dones = torch.from_numpy(batch["dones"]).to(device)
        actions_oh = F.one_hot(actions_int, num_classes=action_dim).float()

        embeddings = encoder(obs_seq)
        rssm_out = rssm.observe_sequence(embeddings, actions_oh, dones=dones)
        state_vec = torch.cat([rssm_out["h"], rssm_out["z"]], dim=-1)
        recon_logits = decoder(state_vec)
        recon = (torch.sigmoid(recon_logits) > 0.5).float()

    # On prend la 1ère séquence, 6 timesteps espacés
    obs_np = obs_seq[0].cpu().numpy()
    recon_np = recon[0].cpu().numpy()
    indices = np.linspace(0, 19, 6).astype(int)

    fig, axes = plt.subplots(6, 2, figsize=(6, 12))
    fig.suptitle("WM reconstructions sur 1 séquence (target vs recon)", fontsize=13, y=0.995)
    for i, t in enumerate(indices):
        grid_t = obs_np[t, :240].reshape(24, 10)
        grid_r = recon_np[t, :240].reshape(24, 10)
        piece_t = PIECES[int(np.argmax(obs_np[t, 240:247]))]
        piece_r = PIECES[int(np.argmax(recon_np[t, 240:247]))]
        n_correct = int((obs_np[t] == recon_np[t]).sum())

        axes[i, 0].imshow(grid_t[4:], cmap="Blues", vmin=0, vmax=1, aspect="equal")
        axes[i, 0].set_title(f"t={t}  target (piece={piece_t})", fontsize=9)
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])

        axes[i, 1].imshow(grid_r[4:], cmap="Blues", vmin=0, vmax=1, aspect="equal")
        axes[i, 1].set_title(
            f"t={t}  recon (piece={piece_r}, {n_correct}/276 OK)", fontsize=9,
            color=("green" if piece_t == piece_r else "red"),
        )
        axes[i, 1].set_xticks([]); axes[i, 1].set_yticks([])

    plt.tight_layout(rect=(0, 0, 1, 0.985))
    recon_png = runs_dir / "wm_reconstructions.png"
    plt.savefig(recon_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Reconstructions WM : {recon_png}")

    # Checkpoint
    ckpt_path = ckpt_dir / "wm.pt"
    torch.save({
        "encoder": encoder.state_dict(),
        "rssm": rssm.state_dict(),
        "decoder": decoder.state_dict(),
        "reward_head": reward_head.state_dict(),
        "continue_head": continue_head.state_dict(),
        "config": {
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "embed_dim": EMBED_DIM,
            "h_dim": H_DIM,
            "z_categories": Z_CATEGORIES,
            "z_classes": Z_CLASSES,
            "hidden_dim": HIDDEN_DIM,
        },
        "final_losses": {k: v[-1] for k, v in loss_history.items()},
    }, ckpt_path)
    print(f"Checkpoint         : {ckpt_path}")

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()

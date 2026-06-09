"""
Mini-training d'un autoencoder sur les observations Tetris.

But pédagogique : valider que l'encoder + decoder peuvent compresser
et reconstruire les obs (276 dim → 128 dim → 276 dim) avant d'attaquer
le RSSM.

Pipeline :
    1. Collecte ~10 000 transitions avec un agent random
    2. Training : sample batch, encode → decode, loss BCE, backprop
    3. Eval visuelle : compare obs vs reconstruction

Usage :
    python scripts/train_autoencoder.py
"""

import sys
import time
from pathlib import Path

# Ajoute la racine du projet au PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # backend non-interactif (juste save PNG)

from tetris.env import TetrisEnv
from tetris.env.constants import PIECES
from src.model import ReplayBuffer, Encoder, Decoder


# ============================== Hyperparams

SEED = 42
BUFFER_CAPACITY = 10_000
COLLECT_STEPS = 10_000
TRAIN_STEPS = 5000
BATCH_SIZE = 64
LR = 3e-4
LOG_INTERVAL = 250


# ============================== Helpers

def render_grid(arr_276, label=""):
    """Affichage console d'une obs Tetris (les 240 premières dim = grille 24×10)."""
    grid = arr_276[:240].reshape(24, 10)
    print(f"  {label}")
    for r in range(4, 24):  # skip buffer
        row = "    " + "".join("██" if grid[r, c] else "··" for c in range(10))
        print(row)


def visualize_reconstructions(obs_np, recon_np, save_path, n_examples=6):
    """Génère un PNG comparant target vs reconstruction pour n exemples."""
    fig, axes = plt.subplots(n_examples, 2, figsize=(6, n_examples * 2.0))
    fig.suptitle("Autoencoder reconstructions (target vs recon)", fontsize=13, y=0.995)

    for i in range(n_examples):
        # Grille (240 premières dim → 24×10)
        grid_t = obs_np[i, :240].reshape(24, 10)
        grid_r = recon_np[i, :240].reshape(24, 10)

        # Pièce active (one-hot 240:247)
        piece_t = PIECES[int(np.argmax(obs_np[i, 240:247]))]
        piece_r = PIECES[int(np.argmax(recon_np[i, 240:247]))]

        n_correct = int((obs_np[i] == recon_np[i]).sum())

        # On skip le buffer (4 premières rangées) pour la viz
        axes[i, 0].imshow(grid_t[4:], cmap="Blues", vmin=0, vmax=1, aspect="equal")
        axes[i, 0].set_title(f"target  (piece={piece_t})", fontsize=10)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        axes[i, 1].imshow(grid_r[4:], cmap="Blues", vmin=0, vmax=1, aspect="equal")
        axes[i, 1].set_title(
            f"recon  (piece={piece_r}, {n_correct}/276 OK)", fontsize=10,
            color=("green" if piece_t == piece_r else "red"),
        )
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

    plt.tight_layout(rect=(0, 0, 1, 0.985))
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(losses, save_path):
    """Génère un PNG de la courbe de loss."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, linewidth=0.8, color="steelblue")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("BCE loss (log scale)")
    ax.set_title("Autoencoder training loss")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ============================== Main

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device : {device}")
    print()

    # ------------------------- Setup
    env = TetrisEnv(seed=SEED)
    obs = env.reset()
    obs_dim = obs.shape[0]

    buffer = ReplayBuffer(capacity=BUFFER_CAPACITY, obs_dim=obs_dim)

    encoder = Encoder(obs_dim=obs_dim).to(device)
    decoder = Decoder(obs_dim=obs_dim).to(device)

    n_params = (
        sum(p.numel() for p in encoder.parameters())
        + sum(p.numel() for p in decoder.parameters())
    )
    print(f"Autoencoder params : {n_params:,} (~{n_params/1e6:.3f}M)")
    print()

    optim = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=LR,
    )

    # ------------------------- Phase 1 : Collecte
    print("=" * 60)
    print(f"Phase 1 : Collecte de {COLLECT_STEPS} transitions random")
    print("=" * 60)
    start = time.time()

    for _ in range(COLLECT_STEPS):
        action = np.random.randint(0, env.action_dim)
        next_obs, reward, done, _ = env.step(action)
        buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs if not done else env.reset()

    elapsed = time.time() - start
    n_done = int(buffer.dones[: buffer.size].sum())
    print(f"Buffer remplis : {len(buffer)} transitions en {elapsed:.1f}s")
    print(f"   Episodes terminés : {n_done}")
    print(f"   Memoire buffer    : {buffer.memory_usage_mb():.1f} MB")
    print()

    # ------------------------- Phase 2 : Training
    print("=" * 60)
    print(f"Phase 2 : Training autoencoder ({TRAIN_STEPS} steps)")
    print("=" * 60)

    encoder.train()
    decoder.train()

    losses = []
    start = time.time()

    for step in range(TRAIN_STEPS):
        batch = buffer.sample(BATCH_SIZE)
        obs_batch = torch.from_numpy(batch["obs"]).to(device)

        # Forward
        emb = encoder(obs_batch)
        logits = decoder(emb)

        # Loss BCE (l'obs est binaire : grille + one-hots)
        loss = F.binary_cross_entropy_with_logits(logits, obs_batch)

        # Backward
        optim.zero_grad()
        loss.backward()
        optim.step()

        losses.append(loss.item())

        if (step + 1) % LOG_INTERVAL == 0:
            avg_loss = float(np.mean(losses[-LOG_INTERVAL:]))
            elapsed_now = time.time() - start
            sps = (step + 1) / elapsed_now
            print(f"  step {step+1:5d}/{TRAIN_STEPS}  |  loss = {avg_loss:.4f}  |  {sps:.0f} steps/sec")

    total_elapsed = time.time() - start
    print()
    print(f"Training fini en {total_elapsed:.1f}s "
          f"(loss : {losses[0]:.3f} → {losses[-1]:.4f})")
    print()

    # ------------------------- Phase 3 : Eval visuelle + sauvegarde
    print("=" * 60)
    print("Phase 3 : Eval visuelle + sauvegarde")
    print("=" * 60)

    encoder.eval()
    decoder.eval()

    # 6 exemples pour la viz
    with torch.no_grad():
        batch = buffer.sample(6)
        obs_eval = torch.from_numpy(batch["obs"]).to(device)
        emb = encoder(obs_eval)
        logits = decoder(emb)
        probs = torch.sigmoid(logits)
        recon = (probs > 0.5).float()

    obs_np = obs_eval.cpu().numpy()
    recon_np = recon.cpu().numpy()

    # Console output rapide
    for i in range(3):
        n_correct = int((obs_np[i] == recon_np[i]).sum())
        piece_t = int(np.argmax(obs_np[i, 240:247]))
        piece_r = int(np.argmax(recon_np[i, 240:247]))
        ok = "OK" if piece_t == piece_r else "BAD"
        print(f"  Exemple {i+1} : {n_correct}/276 corrects, current piece {ok}")

    # Création des dossiers de sortie
    runs_dir = Path("runs")
    ckpt_dir = Path("checkpoints")
    runs_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    # Sauvegarde PNG des reconstructions
    recon_png = runs_dir / "autoencoder_reconstructions.png"
    visualize_reconstructions(obs_np, recon_np, recon_png, n_examples=6)
    print(f"\nReconstructions PNG  : {recon_png}")

    # Sauvegarde de la courbe de loss
    loss_png = runs_dir / "autoencoder_loss.png"
    plot_loss_curve(losses, loss_png)
    print(f"Courbe de loss PNG    : {loss_png}")

    # Sauvegarde des poids
    ckpt_path = ckpt_dir / "autoencoder.pt"
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "obs_dim": obs_dim,
            "final_loss": float(losses[-1]),
            "config": {
                "obs_dim": obs_dim,
                "hidden_dim": 256,
                "embed_dim": 128,
            },
        },
        ckpt_path,
    )
    print(f"Checkpoint            : {ckpt_path}")

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()

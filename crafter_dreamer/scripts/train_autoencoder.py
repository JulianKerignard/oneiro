"""
Pré-train AUTOENCODER pour Crafter — SANITY CHECK de l'architecture CNN.

But : vérifier visuellement que CNNEncoder + CNNDecoder peuvent apprendre
      à compresser et reconstruire les images 64×64×3 de Crafter.

Pipeline :
    1. Collecte 5000 obs random depuis Crafter
    2. Train encoder + decoder ensemble sur MSE loss (2000 iter)
    3. Sauvegarde grid de reconstructions PNG
    4. Verdict visuel : recon ressemble-t-elle à l'original ?

⚠️ Les poids appris ICI sont JETÉS après. On RESTART Dreamer avec encoder+decoder
   vierges (les features apprises en autoencoder pur ne sont pas optimales pour le RL).

Lance :
    .venv/bin/python crafter_dreamer/scripts/train_autoencoder.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from crafter_dreamer.env import CrafterEnv, ACTION_DIM
from src.model import CNNEncoder, CNNDecoder


# ============================== Config

SEED = 42
N_OBS = 5000                # transitions collectées (random play)
N_TRAIN_ITER = 10000         # iters d'entraînement autoencoder (test stabilité long terme)
BATCH_SIZE = 32
LR = 3e-4
EMBED_DIM = 128
BASE_CHANNELS = 32

LOG_EVERY = 200
EVAL_EVERY = 1000             # sauvegarde recon PNG (10 PNG sur 10k iter)
N_VIZ = 8                     # nb d'obs à visualiser dans la grid

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "runs"


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device : {device}")
    print()

    # ===================================================== 1. Collecte obs random
    print("=" * 60)
    print(f"Phase 1 : Collecte {N_OBS} obs random depuis Crafter")
    print("=" * 60)
    env = CrafterEnv(seed=SEED)
    obs = env.reset()
    obs_list = []
    start = time.time()
    while len(obs_list) < N_OBS:
        obs_list.append(obs.copy())
        action = np.random.randint(0, ACTION_DIM)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
    obs_arr = np.array(obs_list, dtype=np.float32)   # (N, 3, 64, 64)
    print(f"  Collected {N_OBS} obs en {time.time()-start:.1f}s")
    print(f"  obs_arr shape : {obs_arr.shape}, range [{obs_arr.min():.3f}, {obs_arr.max():.3f}]")
    print()

    # ===================================================== 2. Build models
    print("=" * 60)
    print(f"Phase 2 : Training autoencoder ({N_TRAIN_ITER} iter, batch={BATCH_SIZE})")
    print("=" * 60)
    encoder = CNNEncoder(in_channels=3, embed_dim=EMBED_DIM, base_channels=BASE_CHANNELS).to(device)
    # Note : pour l'autoencoder, le decoder prend l'embed DIRECTEMENT (pas le state h+z).
    decoder = CNNDecoder(state_dim=EMBED_DIM, out_channels=3, base_channels=BASE_CHANNELS).to(device)

    params = list(encoder.parameters()) + list(decoder.parameters())
    n_params = sum(p.numel() for p in params)
    print(f"  Total params : {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"  Encoder : {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  Decoder : {sum(p.numel() for p in decoder.parameters()):,}")
    print()

    optim = torch.optim.Adam(params, lr=LR)
    obs_t = torch.from_numpy(obs_arr).to(device)     # GPU

    # ===================================================== 3. Train loop
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history_loss = []
    start = time.time()

    for it in range(N_TRAIN_ITER):
        idx = torch.randint(0, N_OBS, (BATCH_SIZE,), device=device)
        obs_batch = obs_t[idx]                        # (B, 3, 64, 64)

        emb = encoder(obs_batch)                      # (B, embed_dim)
        recon = decoder(emb)                          # (B, 3, 64, 64) sigmoid [0,1]
        loss = F.mse_loss(recon, obs_batch)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
        optim.step()

        history_loss.append(loss.item())

        if (it + 1) % LOG_EVERY == 0:
            ips = (it + 1) / (time.time() - start)
            # Rolling stats sur 100 dernières iter (détection anomalies)
            recent = history_loss[-100:]
            mean_100 = float(np.mean(recent))
            max_100 = float(np.max(recent))
            spike_ratio = max_100 / mean_100 if mean_100 > 0 else 1.0
            spike_tag = " ⚠️" if spike_ratio > 3.0 else ""
            print(
                f"  iter {it+1:5d}/{N_TRAIN_ITER} | "
                f"loss={loss.item():.5f}  "
                f"mean100={mean_100:.5f}  "
                f"max100={max_100:.5f}  "
                f"spike={spike_ratio:.1f}x{spike_tag}  "
                f"({ips:.1f} it/s)"
            )

        # Visualisation périodique
        if (it + 1) % EVAL_EVERY == 0 or (it + 1) == N_TRAIN_ITER:
            save_recon_grid(
                encoder, decoder, obs_t,
                save_path=OUTPUT_DIR / f"autoencoder_recon_iter{it+1:05d}.png",
                n_viz=N_VIZ,
                title=f"iter {it+1} | loss={loss.item():.5f}",
            )

    elapsed = time.time() - start
    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    # Stats globales
    arr_loss = np.array(history_loss)
    print(f"  Total training time   : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print()
    print("  --- Statistiques loss ---")
    print(f"  Final loss            : {arr_loss[-1]:.5f}")
    print(f"  Min loss              : {arr_loss.min():.5f}  (iter {int(arr_loss.argmin())})")
    print(f"  Max loss              : {arr_loss.max():.5f}  (iter {int(arr_loss.argmax())})")
    print(f"  Mean loss (all)       : {arr_loss.mean():.5f}")
    print(f"  Std loss (all)        : {arr_loss.std():.5f}")
    print(f"  Mean loss (last 10pct): {arr_loss[-len(arr_loss)//10:].mean():.5f}")
    print(f"  Std loss (last 10pct) : {arr_loss[-len(arr_loss)//10:].std():.5f}")

    # Détection pics (anomalies)
    window = 100
    if len(arr_loss) >= window * 2:
        running_mean = np.convolve(arr_loss, np.ones(window) / window, mode="valid")
        # Pour chaque point après la fenêtre, vérifier s'il est >3× la moyenne mobile précédente
        spikes = []
        for i in range(window, len(arr_loss)):
            local_mean = running_mean[i - window] if i - window < len(running_mean) else running_mean[-1]
            if local_mean > 0 and arr_loss[i] > 3.0 * local_mean:
                spikes.append((i, arr_loss[i], local_mean))

        print()
        print("  --- Détection anomalies ---")
        if spikes:
            print(f"  ⚠️  {len(spikes)} pics détectés (loss > 3× moyenne mobile 100)")
            # Affiche les 5 plus gros
            spikes_sorted = sorted(spikes, key=lambda x: x[1] / x[2], reverse=True)[:5]
            for it_idx, l, m in spikes_sorted:
                print(f"     iter {it_idx} : loss={l:.5f} ({l/m:.1f}× mean={m:.5f})")
        else:
            print(f"  ✅ Aucun pic détecté (training stable)")

    print()
    print(f"  Reconstructions       : {OUTPUT_DIR}/autoencoder_recon_*.png")

    # Sauvegarde courbe loss
    plt.figure(figsize=(10, 4))
    plt.plot(history_loss, alpha=0.5, label="loss raw")
    if len(history_loss) > 50:
        smooth = np.convolve(history_loss, np.ones(50) / 50, mode="valid")
        plt.plot(range(49, len(history_loss)), smooth, label="loss smoothed (50)", linewidth=2)
    plt.xlabel("iter")
    plt.ylabel("MSE loss")
    plt.yscale("log")
    plt.title("Autoencoder pre-training (sanity check)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = OUTPUT_DIR / "autoencoder_loss.png"
    plt.savefig(loss_path, dpi=80)
    plt.close()
    print(f"  Loss curve          : {loss_path}")
    print()

    # Verdict heuristique
    final_loss = history_loss[-1]
    if final_loss < 0.005:
        print("  ✅ Loss < 0.005 → reconstruction probablement excellente.")
        print("     CNN architecture validée. Tu peux lancer Dreamer en confiance.")
    elif final_loss < 0.02:
        print("  ⚠️  Loss < 0.02 → reconstruction correcte mais pas parfaite.")
        print("     L'architecture marche. Visualise les PNG pour décider.")
    else:
        print("  ❌ Loss > 0.02 → reconstruction probablement floue.")
        print("     Vérifie les PNG : si visuellement HS, bug d'archi.")
        print("     Sinon, peut être OK pour démarrer Dreamer (qui ajoute pression de gradient).")


def save_recon_grid(encoder, decoder, obs_t, save_path, n_viz=8, title=""):
    """Sauve une grid PNG comparant N obs originales vs leurs reconstructions."""
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        # Prend les N premières obs
        sample = obs_t[:n_viz]                        # (N, 3, 64, 64)
        emb = encoder(sample)
        recon = decoder(emb)                          # (N, 3, 64, 64)

    # Convert pour matplotlib : (N, 3, 64, 64) → (N, 64, 64, 3)
    orig_np = sample.cpu().numpy().transpose(0, 2, 3, 1)
    recon_np = recon.cpu().numpy().transpose(0, 2, 3, 1)

    fig, axes = plt.subplots(2, n_viz, figsize=(n_viz * 2, 5))
    for i in range(n_viz):
        axes[0, i].imshow(orig_np[i])
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("ORIGINAL", fontsize=10, loc="left")

        axes[1, i].imshow(recon_np[i].clip(0, 1))
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("RECON", fontsize=10, loc="left")

    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=80, bbox_inches="tight")
    plt.close()

    encoder.train()
    decoder.train()


if __name__ == "__main__":
    main()

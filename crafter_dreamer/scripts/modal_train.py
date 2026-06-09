"""
Wrapper Modal pour entraîner Dreamer sur Crafter en cloud GPU.

Usage :
    modal token new                            # Auth (une seule fois)
    modal run crafter_dreamer/scripts/modal_train.py

Coût estimé :
    A100 40GB : $1.10/h
    Run 30k iter ≈ 2-3h = ~$3 (sur 30 credits disponibles → 10+ runs possibles)

Configuration cloud :
    - GPU : A100 40GB (meilleur rapport qualité/prix pour 7M params)
    - Volume Modal : /vol → persiste runs/ et checkpoints/ entre runs
    - Code monté en local : crafter_dreamer/ + src/

Pour récupérer les outputs après le run :
    modal volume get worldmodel-outputs / ./modal_outputs
"""

import sys
from pathlib import Path

import modal


# ============================== Setup Modal app

APP_NAME = "crafter-dreamer"
VOLUME_NAME = "worldmodel-outputs"

app = modal.App(APP_NAME)

# Volume persistant pour outputs (runs/, checkpoints/)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# ============================== Image Docker

# Image avec PyTorch CUDA + dépendances Crafter
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "numpy==1.26.4",       # compatible torch 2.5
        "matplotlib==3.9.2",
        "Pillow==10.4.0",
        "crafter==1.8.3",
        "opensimplex==0.4.5",  # dependency de crafter
        "ruamel.yaml==0.18.6",
    )
    # Copy le code local dans le container (rebuilt si code change).
    # Paths RELATIFS au CWD : il faut lancer modal depuis le root du projet
    # (sinon les paths ne seront pas trouvés). __file__ ne marche pas ici
    # car Modal ré-évalue le code dans le container où __file__ = /root/modal_train.py.
    .add_local_dir(
        local_path="crafter_dreamer",
        remote_path="/code/crafter_dreamer",
    )
    .add_local_dir(
        local_path="src",
        remote_path="/code/src",
    )
)


# ============================== Training function

@app.function(
    image=image,
    gpu="L4",                  # L4 = $0.80/h, GPU suffisant pour 7M params
    cpu=16.0,                  # 16 vCPUs pour saturer multi-env CrafterEnv (CPU-bound)
    memory=32768,              # 32 GB RAM pour buffer image (uint8)
    timeout=10 * 3600,         # 10h max par run (sécurité)
    volumes={"/vol": volume},  # Outputs persistants
)
def train(
    train_iter: int = 30000,
    eval_interval: int = 2000,
    n_envs: int = 16,           # 16 envs en parallèle (saturer 16 CPUs)
    batch_size: int = 64,        # 64 safe pour L4 22GB (128 → OOM avec torch.compile)
    wm_train_per_iter: int = 1,
    ac_train_per_iter: int = 1,
    entropy_coef: float = 0.005,
    auto_explore: bool = True,
    use_rnd: bool = False,
    rnd_coef: float = 0.5,
    seed: int = 42,
    run_name: str = "crafter_modal_v1",
):
    """Lance le training Crafter Dreamer sur A100."""
    import os
    import subprocess
    import sys

    # Configure les chemins
    os.environ["WORLDMODEL_OUTPUT_DIR"] = "/vol"
    sys.path.insert(0, "/code")

    print("=" * 60)
    print(f"Modal training : {run_name}")
    print("=" * 60)

    # GPU info
    import torch
    print(f"CUDA disponible : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU            : {torch.cuda.get_device_name(0)}")
        print(f"GPU memory     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Lance le script de training avec -u (unbuffered stdout pour logs en streaming)
    cmd = [
        sys.executable,
        "-u",
        "/code/crafter_dreamer/scripts/train_dreamer.py",
        "--train_iter", str(train_iter),
        "--eval_interval", str(eval_interval),
        "--n_envs", str(n_envs),
        "--batch_size", str(batch_size),
        "--wm_train_per_iter", str(wm_train_per_iter),
        "--ac_train_per_iter", str(ac_train_per_iter),
        "--entropy_coef", str(entropy_coef),
        "--run_name", run_name,
        "--seed", str(seed),
        "--rnd_coef", str(rnd_coef),
    ]
    if auto_explore:
        cmd.append("--auto_explore")
    if use_rnd:
        cmd.append("--use_rnd")

    print("Command :", " ".join(cmd))
    print()

    # Run inheriting stdout/stderr → logs visibles dans Modal
    result = subprocess.run(cmd, cwd="/vol", check=False)

    # Commit volume pour persister les outputs
    volume.commit()

    print()
    print("=" * 60)
    if result.returncode == 0:
        print("✓ Training terminé avec succès")
    else:
        print(f"✗ Training échoué (return code {result.returncode})")
    print(f"Outputs sauvegardés sur volume Modal : {VOLUME_NAME}/runs et /checkpoints")
    print("Récupérer avec : modal volume get worldmodel-outputs / ./modal_outputs")
    print("=" * 60)


# ============================== Local entrypoint

@app.local_entrypoint()
def main(
    train_iter: int = 30000,
    eval_interval: int = 2000,
    n_envs: int = 16,
    batch_size: int = 64,
    wm_train_per_iter: int = 1,
    ac_train_per_iter: int = 1,
    entropy_coef: float = 0.005,
    auto_explore: bool = True,
    use_rnd: bool = False,
    rnd_coef: float = 0.5,
    run_name: str = "crafter_modal_v1",
):
    """
    Lance le training en cloud Modal.

    Usage :
        modal run crafter_dreamer/scripts/modal_train.py
        modal run crafter_dreamer/scripts/modal_train.py --train-iter 50000 --run-name big_run

    Le script local appelle .remote() qui dispatch vers le cloud.
    """
    print(f"Lancement du training Crafter sur Modal A100-40GB")
    print(f"  Run name      : {run_name}")
    print(f"  Iterations    : {train_iter}")
    print(f"  N envs        : {n_envs}")
    print(f"  Batch size    : {batch_size}")
    print(f"  Auto explore  : {auto_explore}")
    print()
    print("Logs streaming en temps réel ci-dessous (Ctrl+C pour interrompre l'observation,")
    print("le job continue tourner côté Modal).")
    print()
    train.remote(
        train_iter=train_iter,
        eval_interval=eval_interval,
        n_envs=n_envs,
        batch_size=batch_size,
        wm_train_per_iter=wm_train_per_iter,
        ac_train_per_iter=ac_train_per_iter,
        entropy_coef=entropy_coef,
        auto_explore=auto_explore,
        use_rnd=use_rnd,
        rnd_coef=rnd_coef,
        run_name=run_name,
    )

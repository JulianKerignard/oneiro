"""
Wrapper Modal pour entraîner mini-DreamerV3 JAX/Flax sur Crafter.

Setup :
    - GPU L4 (ou A100 selon budget)
    - Image Docker : jax[cuda12] + flax + optax + distrax + crafter
    - Volume Modal : /vol persistant pour runs et checkpoints

Usage :
    modal token new                          # une seule fois
    modal run crafter_dreamer/scripts/modal_train_jax.py
    modal run --detach crafter_dreamer/scripts/modal_train_jax.py --train-iter 30000

Différences avec modal_train.py (PyTorch) :
    - Image JAX CUDA (au lieu de torch CUDA)
    - Script de training : train_dreamer_jax.py (à créer en Phase 4)
    - Imports source : src_jax/ (au lieu de src/)
"""

import sys
from pathlib import Path

import modal


# ============================== Setup Modal app

APP_NAME = "crafter-dreamer-jax"
VOLUME_NAME = "worldmodel-outputs"

app = modal.App(APP_NAME)

# Volume persistant partagé avec runs PyTorch
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# ============================== Image Docker JAX CUDA

image = (
    modal.Image.debian_slim(python_version="3.11")
    # JAX CUDA12 (embarque CUDA + cuDNN via pip wheels)
    .pip_install(
        "jax[cuda12]==0.10.1",
    )
    # Écosystème JAX
    .pip_install(
        "flax==0.12.7",
        "optax==0.2.8",
        "distrax==0.1.8",
        "chex==0.1.91",
        "orbax-checkpoint==0.12.0",
    )
    # Dépendances Crafter + utils
    # numpy >= 2.0 obligatoire pour JAX 0.10.1 (np.dtypes.StringDType)
    .pip_install(
        "numpy>=2.0",
        "matplotlib==3.9.2",
        "Pillow==10.4.0",
        "crafter==1.8.3",
        "opensimplex==0.4.5",
        "ruamel.yaml==0.18.6",
    )
    # Code monté depuis le local (paths relatifs au CWD au lancement modal)
    .add_local_dir(
        local_path="crafter_dreamer",
        remote_path="/code/crafter_dreamer",
    )
    .add_local_dir(
        local_path="src_jax",
        remote_path="/code/src_jax",
    )
    # On copie aussi src/ pour pouvoir comparer avec la version PyTorch (parity tests)
    .add_local_dir(
        local_path="src",
        remote_path="/code/src",
    )
)


# ============================== Training function

@app.function(
    image=image,
    gpu="L4",                  # L4 24GB : meilleur rapport qualité/prix pour JAX
    cpu=8.0,                   # CrafterEnv (CPU) collecté en parallèle
    memory=16384,              # 16GB RAM (JAX prend moins que PyTorch buffer uint8)
    timeout=10 * 3600,         # 10h max safety
    volumes={"/vol": volume},
)
def train_jax(
    train_iter: int = 30000,
    eval_interval: int = 2000,
    n_envs: int = 16,
    batch_size: int = 16,
    seed: int = 42,
    run_name: str = "crafter_jax_v1",
    mp_collect: bool = False,
    profile: bool = False,
    entropy_coef: float = 3e-4,
    wm_train_per_iter: int = 1,
    ac_train_per_iter: int = 1,
    adaptive_alpha: bool = False,
    auto_explore: bool = False,
    use_rnd: bool = True,
    alpha_init: float = 3e-4,
    h_target: float = 2.0,
    rnd_coef: float = 0.5,
    # Phase 13 : safeguards
    rnd_warmup_steps: int = 5000,
    adaptive_rnd: bool = True,
    health_auto_stop: bool = True,
    health_consec_threshold: int = 5,
    h_target_schedule: bool = True,
    h_target_init: float = 1.13,
    h_target_final: float = 1.13,
    h_target_decay_steps: int = 10000,
):
    """Lance le training Crafter Dreamer JAX sur L4."""
    import os
    import subprocess
    import sys

    # Path setup pour les imports src_jax
    os.environ["WORLDMODEL_OUTPUT_DIR"] = "/vol"
    sys.path.insert(0, "/code")

    print("=" * 60)
    print(f"Modal training JAX : {run_name}")
    print("=" * 60)

    # GPU + JAX info
    import jax
    print(f"JAX version       : {jax.__version__}")
    print(f"JAX devices       : {jax.devices()}")
    print(f"Default backend   : {jax.default_backend()}")
    print()

    # Vérifier que CUDA est dispo
    if jax.default_backend() != "gpu":
        print(f"⚠️  WARNING : JAX backend = {jax.default_backend()} (attendu : gpu)")
        print("   Les variables d'environnement CUDA peuvent être mal configurées.")
    print()

    # Lance le script de training JAX (créé en Phase 4)
    cmd = [
        sys.executable,
        "-u",                  # unbuffered stdout pour streaming logs
        "/code/crafter_dreamer/scripts/train_dreamer_jax.py",
        "--train_iter", str(train_iter),
        "--eval_interval", str(eval_interval),
        "--n_envs", str(n_envs),
        "--batch_size", str(batch_size),
        "--seed", str(seed),
        "--run_name", run_name,
    ]
    if mp_collect:
        cmd.append("--mp_collect")
    if profile:
        cmd.append("--profile")
    # Entropy + replay ratio : explicites pour ne pas dépendre des défauts du script
    cmd += ["--entropy_coef", str(entropy_coef),
            "--wm_train_per_iter", str(wm_train_per_iter),
            "--ac_train_per_iter", str(ac_train_per_iter)]
    # Flags booléens explicites dans les DEUX sens : le param du wrapper dit
    # toujours la vérité, quel que soit le défaut du script.
    cmd.append("--adaptive_alpha" if adaptive_alpha else "--no_adaptive_alpha")
    cmd.append("--auto_explore" if auto_explore else "--no_auto_explore")
    if not use_rnd:
        cmd.append("--no_use_rnd")
    cmd += ["--alpha_init", str(alpha_init),
            "--h_target", str(h_target),
            "--rnd_coef", str(rnd_coef)]
    # Phase 13 : safeguards
    cmd += ["--rnd_warmup_steps", str(rnd_warmup_steps),
            "--health_consec_threshold", str(health_consec_threshold),
            "--h_target_init", str(h_target_init),
            "--h_target_final", str(h_target_final),
            "--h_target_decay_steps", str(h_target_decay_steps)]
    if not adaptive_rnd:
        cmd.append("--no_adaptive_rnd")
    if not health_auto_stop:
        cmd.append("--no_health_auto_stop")
    if not h_target_schedule:
        cmd.append("--no_h_target_schedule")
    # Note: seq_len et imagination_horizon sont des constantes côté script
    # (SEQ_LEN, IMAGINATION_HORIZON) — pas configurables via CLI pour l'instant.

    print("Command :", " ".join(cmd))
    print()

    result = subprocess.run(cmd, cwd="/vol", check=False)

    # Commit volume pour persister les outputs
    volume.commit()

    print()
    print("=" * 60)
    if result.returncode == 0:
        print("✓ Training JAX terminé avec succès")
    else:
        print(f"✗ Training JAX échoué (return code {result.returncode})")
    print(f"Outputs sauvegardés sur volume Modal : {VOLUME_NAME}/runs et /checkpoints")
    print("Récupérer avec : modal volume get worldmodel-outputs / ./modal_outputs")
    print("=" * 60)


@app.function(
    image=image,
    gpu="L4",
    cpu=4.0,
    memory=8192,
    timeout=30 * 60,           # 30 min max pour smoketest
    volumes={"/vol": volume},
)
def smoketest_jax():
    """
    Smoketest rapide : vérifie que JAX + Flax + Optax + Distrax fonctionnent
    sur GPU L4 cloud. Run le hello_world_jax.py.
    """
    import os
    import subprocess
    import sys

    os.environ["WORLDMODEL_OUTPUT_DIR"] = "/vol"
    sys.path.insert(0, "/code")

    print("=" * 60)
    print("SMOKETEST JAX sur Modal L4")
    print("=" * 60)

    import jax
    print(f"JAX version : {jax.__version__}")
    print(f"Devices     : {jax.devices()}")
    print(f"Backend     : {jax.default_backend()}")
    print()

    cmd = [sys.executable, "-u", "/code/crafter_dreamer/scripts/hello_world_jax.py"]
    result = subprocess.run(cmd, check=False)
    print()
    print(f"Smoketest exit code : {result.returncode}")
    return result.returncode


# ============================== Local entrypoints

@app.local_entrypoint()
def main(
    train_iter: int = 30000,
    eval_interval: int = 2000,
    n_envs: int = 16,
    batch_size: int = 16,
    seed: int = 42,
    run_name: str = "crafter_jax_v1",
    mp_collect: bool = False,
    profile: bool = False,
    entropy_coef: float = 3e-4,
    wm_train_per_iter: int = 1,
    ac_train_per_iter: int = 1,
    adaptive_alpha: bool = False,
    auto_explore: bool = False,
    use_rnd: bool = True,
    alpha_init: float = 3e-4,
    h_target: float = 2.0,
    rnd_coef: float = 0.5,
    # Phase 13 : safeguards
    rnd_warmup_steps: int = 5000,
    adaptive_rnd: bool = True,
    health_auto_stop: bool = True,
    health_consec_threshold: int = 5,
    h_target_schedule: bool = True,
    h_target_init: float = 1.13,
    h_target_final: float = 1.13,
    h_target_decay_steps: int = 10000,
):
    """
    Lance le training Crafter Dreamer JAX sur Modal L4.

    Usage :
        modal run crafter_dreamer/scripts/modal_train_jax.py
        modal run --detach crafter_dreamer/scripts/modal_train_jax.py --train-iter 30000
    """
    print(f"Lancement training Crafter JAX sur Modal L4")
    print(f"  Run name           : {run_name}")
    print(f"  Iterations         : {train_iter}")
    print(f"  N envs             : {n_envs}")
    print(f"  Batch size         : {batch_size}")
    print(f"  Entropy coef       : {entropy_coef}  (adaptive_alpha={adaptive_alpha})")
    print(f"  Train/iter         : wm={wm_train_per_iter}  ac={ac_train_per_iter}")
    print(f"  Seed               : {seed}")
    print()

    train_jax.remote(
        train_iter=train_iter,
        eval_interval=eval_interval,
        n_envs=n_envs,
        batch_size=batch_size,
        seed=seed,
        run_name=run_name,
        mp_collect=mp_collect,
        profile=profile,
        entropy_coef=entropy_coef,
        wm_train_per_iter=wm_train_per_iter,
        ac_train_per_iter=ac_train_per_iter,
        adaptive_alpha=adaptive_alpha,
        auto_explore=auto_explore,
        use_rnd=use_rnd,
        alpha_init=alpha_init,
        h_target=h_target,
        rnd_coef=rnd_coef,
        # Phase 13 : safeguards
        rnd_warmup_steps=rnd_warmup_steps,
        adaptive_rnd=adaptive_rnd,
        health_auto_stop=health_auto_stop,
        health_consec_threshold=health_consec_threshold,
        h_target_schedule=h_target_schedule,
        h_target_init=h_target_init,
        h_target_final=h_target_final,
        h_target_decay_steps=h_target_decay_steps,
    )


@app.local_entrypoint()
def smoketest():
    """
    Smoketest rapide : vérifie l'environnement JAX cloud sans rien entraîner.

    Usage :
        modal run crafter_dreamer/scripts/modal_train_jax.py::smoketest
    """
    print("Lancement smoketest JAX cloud (vérifie devices + imports)")
    exit_code = smoketest_jax.remote()
    print(f"Smoketest terminé (exit={exit_code})")

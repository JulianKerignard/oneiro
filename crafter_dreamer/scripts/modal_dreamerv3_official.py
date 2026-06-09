"""
Wrapper Modal pour lancer DreamerV3 OFFICIEL (danijar/dreamerv3) sur Crafter.

Source of truth : https://github.com/danijar/dreamerv3
Objectif : valider que le code officiel atteint 6-8 achievements sur Crafter
en ~1M env_steps (paper target = 8.9 ach a 1M).

Si oui : confirme bug systemique dans notre implementation JAX (stagne 2-3 ach).
Si non : probleme cote env (crafter lib) ou hardware.

USAGE
-----
Smoketest (verifie que le setup tourne, ~5min, ~$0.20) :
    modal run crafter_dreamer/scripts/modal_dreamerv3_official.py::smoketest_run

Validation rapide 100k steps (~1h, ~$1.50) :
    modal run --detach crafter_dreamer/scripts/modal_dreamerv3_official.py --steps 100000

Run complet paper budget 1M steps (~6-10h, ~$8-15) :
    modal run --detach crafter_dreamer/scripts/modal_dreamerv3_official.py --steps 1000000

OUTPUT
------
Les logs et checkpoints sont sauves dans le volume Modal `worldmodel-outputs` :
    /vol/dreamerv3_official/<run_name>/
        - metrics.jsonl
        - scores/
        - checkpoints/
        - stdout.log

Pour recuperer en local :
    modal volume get worldmodel-outputs dreamerv3_official/<run_name> ./

NOTES IMPLEMENTATION
--------------------
- Le repo officiel est clone dans /opt/dreamerv3 a la build de l'image.
- Toutes les deps viennent du requirements.txt officiel (source of truth).
- JAX version pinned a 0.4.33 (cf requirements.txt officiel).
- Python 3.11 (requis par le repo).
- Default Crafter config : run.steps=1.1e6, envs=1, train_ratio=512 (cf configs.yaml).
- Pour overrider steps : --run.steps <N> (notation Dreamer/embodied flags).

CONFIG CRAFTER (extrait de dreamerv3/configs.yaml officiel) :
    crafter:
      task: crafter_reward
      run: {steps: 1.1e6, envs: 1, train_ratio: 512}

ESTIMATION COUT (GPU L4 ~$0.80/h sur Modal au 2026-06) :
- Smoketest 1000 steps    : ~5 min   ~= $0.10
- 100k steps              : ~1-2h    ~= $1-2
- 1M steps (paper budget) : ~8-12h   ~= $6-10
"""

import modal


# ============================== Setup Modal app

APP_NAME = "dreamerv3-official-test"
VOLUME_NAME = "worldmodel-outputs"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# ============================== Image Docker

# Reproduit le setup officiel :
#   1. Python 3.11+
#   2. Install JAX (deja inclus dans requirements.txt : jax[cuda12]==0.4.33)
#   3. pip install -U -r requirements.txt
#   4. pip install -e . pour rendre dreamerv3 importable

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "ffmpeg")
    # Clone du repo officiel (depth=1 pour vitesse)
    .run_commands(
        "git clone --depth 1 https://github.com/danijar/dreamerv3.git /opt/dreamerv3",
    )
    # Install des deps exactes du requirements.txt officiel
    # (jax[cuda12]==0.4.33 est inclus dedans)
    .run_commands(
        "cd /opt/dreamerv3 && pip install -U -r requirements.txt",
    )
    # Install Crafter (env benchmark)
    # crafter 1.8.3 fonctionne avec gym, dependances minimales
    .pip_install(
        "crafter==1.8.3",
        "opensimplex==0.4.5",
        "imageio==2.36.0",
    )
    # Make dreamerv3 importable comme package
    .run_commands(
        "cd /opt/dreamerv3 && pip install -e . || true",
    )
    # Verification rapide de l'install
    .run_commands(
        "python -c 'import jax; print(\"JAX:\", jax.__version__)'",
        "python -c 'import crafter; print(\"Crafter OK\")'",
    )
)


# ============================== Training function

@app.function(
    image=image,
    gpu="L4",
    cpu=8.0,
    memory=16384,
    timeout=12 * 3600,   # 12h max (couvre run 1M steps)
    volumes={"/vol": volume},
)
def train_official(
    steps: int = 1_000_000,
    run_name: str = "dreamerv3_official_crafter",
    train_ratio: int = 512,
    size_config: str = "size12m",
):
    """Lance le training DreamerV3 OFFICIEL sur Crafter.

    Args:
        steps: nombre de env steps. Paper budget = 1M (1e6). Valid rapide = 100k.
        run_name: nom du run (logdir sera /vol/dreamerv3_official/<run_name>).
        train_ratio: ratio update/env_step. Default Crafter officiel = 512.
        size_config: taille du modèle. Options : size1m, size12m, size25m, size50m, size100m, size200m, size400m.
                     Default size12m (~25M params, plus proche de notre archi 15M).
                     size200m = XL = config Crafter default du paper (165M, mais lourd à entraîner).
    """
    import os
    import subprocess
    import sys
    import time

    print("=" * 70)
    print("DreamerV3 OFFICIEL (danijar/dreamerv3) on Crafter")
    print("=" * 70)
    print(f"Steps        : {steps:,}")
    print(f"Train ratio  : {train_ratio}")
    print(f"Run name     : {run_name}")
    print(f"GPU          : L4")

    logdir = f"/vol/dreamerv3_official/{run_name}"
    os.makedirs(logdir, exist_ok=True)
    print(f"Logdir       : {logdir}")
    print("=" * 70)

    # Verifie GPU dispo
    try:
        import jax
        print(f"JAX devices  : {jax.devices()}")
    except Exception as e:
        print(f"WARN: JAX check failed: {e}")

    # Construction commande officielle (cf README + configs.yaml)
    # --configs accepte plusieurs noms séparés : le dernier override le précédent
    cmd = [
        sys.executable, "dreamerv3/main.py",
        "--logdir", logdir,
        "--configs", "crafter", size_config,
        "--run.steps", str(steps),
        "--run.train_ratio", str(train_ratio),
    ]

    print("Command :")
    print("  " + " ".join(cmd))
    print("=" * 70)

    # Log stdout/stderr dans un fichier ET vers la console
    log_file = os.path.join(logdir, "stdout.log")
    start = time.time()

    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            cmd,
            cwd="/opt/dreamerv3",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
            f.flush()
            # Commit volume periodiquement pour ne pas perdre les logs
            # (toutes les ~5000 lignes)
        proc.wait()

    elapsed = time.time() - start
    print("=" * 70)
    print(f"Elapsed      : {elapsed/60:.1f} min ({elapsed/3600:.2f}h)")
    print(f"Exit code    : {proc.returncode}")

    # Commit final pour persister logs + checkpoints
    volume.commit()

    if proc.returncode == 0:
        print("OK : Training officiel termine avec succes")
        print(f"Logs : {logdir}")
        print("Pour recuperer en local :")
        print(f"  modal volume get {VOLUME_NAME} dreamerv3_official/{run_name} ./")
    else:
        print(f"FAIL : Training echoue (code {proc.returncode})")
        print(f"Check logs : {log_file}")

    return proc.returncode


# ============================== Smoketest function

@app.function(
    image=image,
    gpu="L4",
    cpu=4.0,
    memory=8192,
    timeout=30 * 60,
    volumes={"/vol": volume},
)
def smoketest():
    """Validation rapide : 1000 steps pour verifier que le repo officiel tourne.

    Verifie :
    - Image Modal build correctement (clone + deps)
    - JAX detecte le GPU L4
    - Le script dreamerv3/main.py demarre sans crash
    - Crafter env tourne quelques steps
    """
    import os
    import subprocess
    import sys

    print("=" * 70)
    print("SMOKETEST DreamerV3 officiel sur Modal L4")
    print("=" * 70)

    # Check GPU
    try:
        import jax
        print(f"JAX version  : {jax.__version__}")
        print(f"JAX devices  : {jax.devices()}")
    except Exception as e:
        print(f"JAX check failed: {e}")
        return 1

    # Check crafter import
    try:
        import crafter  # noqa: F401
        print("Crafter      : OK")
    except Exception as e:
        print(f"Crafter import failed: {e}")
        return 1

    logdir = "/vol/dreamerv3_official/_smoketest"
    os.makedirs(logdir, exist_ok=True)

    cmd = [
        sys.executable, "dreamerv3/main.py",
        "--logdir", logdir,
        "--configs", "crafter",
        "--run.steps", "1000",
        "--run.train_ratio", "32",   # ratio plus bas pour aller vite au smoketest
    ]

    print("Command :")
    print("  " + " ".join(cmd))
    print("=" * 70)

    result = subprocess.run(cmd, cwd="/opt/dreamerv3", check=False)

    volume.commit()

    print("=" * 70)
    print(f"Smoketest exit code : {result.returncode}")
    if result.returncode == 0:
        print("OK : Setup officiel fonctionne. Lance maintenant `main` pour le vrai run.")
    else:
        print("FAIL : Probleme dans l'image ou la commande. Voir logs ci-dessus.")
    return result.returncode


# ============================== Local entrypoints

@app.local_entrypoint()
def main(
    steps: int = 1_000_000,
    run_name: str = "dreamerv3_official_crafter",
    train_ratio: int = 512,
    size_config: str = "size12m",
):
    """Launch the official DreamerV3 on Crafter via Modal.

    Defaults to 1M steps (paper budget ~8.9 ach).
    Default size_config = size12m (~25M params, comparable à notre archi 15M).

    Examples :
        modal run modal_dreamerv3_official.py                    # 1M steps
        modal run modal_dreamerv3_official.py --steps 100000     # quick check
        modal run --detach modal_dreamerv3_official.py           # background
    """
    print(f"Launching official DreamerV3 (steps={steps:,}, size={size_config}, run_name={run_name})")
    code = train_official.remote(
        steps=steps,
        run_name=run_name,
        train_ratio=train_ratio,
        size_config=size_config,
    )
    print(f"Exit code : {code}")


@app.local_entrypoint()
def smoketest_run():
    """Smoketest : verify setup works (~5min, ~$0.10)."""
    print("Smoketest DreamerV3 official...")
    code = smoketest.remote()
    print(f"Exit code : {code}")
    if code == 0:
        print("OK. Lance le run complet avec :")
        print("  modal run --detach crafter_dreamer/scripts/modal_dreamerv3_official.py")

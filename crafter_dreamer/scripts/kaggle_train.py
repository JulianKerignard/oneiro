"""
Launcher Kaggle pour le training Oneiro (notebook batch sur GPU P100 gratuit).

Génère un notebook .ipynb (git clone branche + pip + run), le pousse via
l'API Kaggle en mode "save & run all" sur P100, puis poll / récupère l'output.

Même esprit que lightning_train_jax.py, adapté aux contraintes Kaggle :
  - pas de logs en direct (output dispo à la FIN du kernel) → on poll le statut
  - session batch max ~12h → run 30k (~7-10h) tient, sinon resume via re-push
  - buffer 1M en RAM host (32GB Kaggle) via --buffer_device cpu

Auth (CLI kaggle) : ~/.kaggle/kaggle.json {"username","key"} OU env
KAGGLE_USERNAME / KAGGLE_KEY. Le username sert aussi à nommer le kernel.

Usage :
    python crafter_dreamer/scripts/kaggle_train.py launch \\
        --run-name v23-p100-buffer1m --train-iter 30000
    python crafter_dreamer/scripts/kaggle_train.py status --run-name v23-p100-buffer1m
    python crafter_dreamer/scripts/kaggle_train.py pull   --run-name v23-p100-buffer1m
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_URL = "https://github.com/JulianKerignard/oneiro.git"
BRANCH = "feat/tpu"  # archi rééquilibrée + buffer CPU + support TPU (--tpu)

# Le CLI `kaggle` vit à côté du python courant (venv), pas forcément dans le PATH.
import shutil
KAGGLE_BIN = (
    str(Path(sys.executable).parent / "kaggle")
    if (Path(sys.executable).parent / "kaggle").exists()
    else (shutil.which("kaggle") or "kaggle")
)


def kaggle_user():
    user = os.environ.get("KAGGLE_USERNAME")
    if not user:
        kj = Path.home() / ".kaggle" / "kaggle.json"
        if kj.exists():
            user = json.loads(kj.read_text()).get("username")
    if not user:
        sys.exit("ERREUR : username Kaggle introuvable (KAGGLE_USERNAME ou ~/.kaggle/kaggle.json).")
    return user


def build_notebook(args) -> dict:
    """Construit un notebook nbformat minimal : clone + install + train."""
    flags = [
        f"--train_iter {args.train_iter}",
        f"--eval_interval {args.eval_interval}",
        f"--wm_train_per_iter {args.wm_train_per_iter}",
        f"--n_envs {args.n_envs}",
        f"--batch_size {args.batch_size}",
        f"--buffer_device {args.buffer_device}",
        f"--buffer_capacity {args.buffer_capacity}",
        f"--run_name {args.run_name}",
    ]
    if args.no_use_rnd:
        flags.append("--no_use_rnd")
    if args.no_health_auto_stop:
        flags.append("--no_health_auto_stop")
    if args.extra_args:
        flags.append(args.extra_args)
    train_cmd = "python -u crafter_dreamer/scripts/train_dreamer_jax.py " + " ".join(flags)

    # JAX backend selon l'accélérateur : cuda12 pour GPU, tpu pour TPU v5e-8.
    # Sur TPU, le code JAX tourne tel quel sur 1 core (single-device, pas de
    # pmap) ; le buffer CPU est idéal (host TPU-VM a beaucoup de RAM).
    # /!\ Risque connu : matching jax[tpu] x.y.z <-> libtpu de l'image Kaggle.
    jax_pkg = '"jax[tpu]==0.10.1" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html' if args.tpu else '"jax[cuda12]==0.10.1"'
    # Note : %cd oneiro (cellule 1) fixe le cwd du notebook → les cellules
    # suivantes y sont déjà, ne PAS refaire cd. Checkpoints/summaries dans
    # /kaggle/working (récupérés en output du kernel par `pull`).
    # /!\ NE PAS importer jax dans le kernel notebook avant le training :
    # sur TPU le device est EXCLUSIF (vfio, 1 process). Si le notebook fait
    # `import jax`, il réserve le TPU et le sous-process training (`!python`)
    # ne peut plus l'ouvrir → "Device busy". Le training affiche lui-même le
    # backend dans son header, donc pas de cellule de vérif séparée.
    cells = [
        f"!rm -rf oneiro && git clone -b {BRANCH} {REPO_URL}",
        f'%cd oneiro\n!pip install -q -r requirements.txt && pip install -q -U {jax_pkg}',
        f"!WORLDMODEL_OUTPUT_DIR=/kaggle/working {train_cmd}",
    ]
    return {
        "cells": [
            {"cell_type": "code", "source": c, "metadata": {}, "outputs": [], "execution_count": None}
            for c in cells
        ],
        "metadata": {"kernelspec": {"language": "python", "name": "python3", "display_name": "Python 3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def kernel_metadata(user, slug, tpu=False) -> dict:
    meta = {
        "id": f"{user}/{slug}",
        "title": slug,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_internet": True,    # requis pour git clone + pip
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    if tpu:
        # /!\ champ accélérateur TPU Kaggle à confirmer au 1er push (v5e-8).
        # Sinon : régler l'accélérateur = TPU VM v5e-8 dans l'UI du notebook.
        meta["enable_tpu"] = True
    else:
        meta["enable_gpu"] = True   # P100 (ou régler dans l'UI)
    return meta


def cmd_launch(args):
    user = kaggle_user()
    slug = args.run_name.lower().replace("_", "-")[:50]
    work = Path(tempfile.mkdtemp(prefix="kaggle_oneiro_"))
    (work / "notebook.ipynb").write_text(json.dumps(build_notebook(args)))
    (work / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(user, slug, args.tpu), indent=2))
    accel = "TPU v5e-8" if args.tpu else "GPU P100"
    print(f"Kernel   : {user}/{slug}")
    print(f"Config   : {args.train_iter} iter, buffer {args.buffer_device} {args.buffer_capacity:,}")
    print(f"Staging  : {work}")
    print(f"Push (save & run all sur {accel})...\n")
    r = subprocess.run([KAGGLE_BIN, "kernels", "push", "-p", str(work)], capture_output=True, text=True)
    print(r.stdout + r.stderr)
    if r.returncode == 0:
        print(f"\nSuivi : python {sys.argv[0]} status --run-name {args.run_name}")
        print(f"Dashboard : https://www.kaggle.com/code/{user}/{slug}")


def cmd_status(args):
    user = kaggle_user()
    slug = args.run_name.lower().replace("_", "-")[:50]
    r = subprocess.run([KAGGLE_BIN, "kernels", "status", f"{user}/{slug}"], capture_output=True, text=True)
    print(r.stdout + r.stderr)


def cmd_pull(args):
    user = kaggle_user()
    slug = args.run_name.lower().replace("_", "-")[:50]
    out = Path(args.target) / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([KAGGLE_BIN, "kernels", "output", f"{user}/{slug}", "-p", str(out)],
                       capture_output=True, text=True)
    print(r.stdout + r.stderr)
    print(f"Output → {out}/")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)

    pl = sub.add_parser("launch")
    pl.add_argument("--run-name", required=True)
    pl.add_argument("--train-iter", type=int, default=30000)
    pl.add_argument("--eval-interval", type=int, default=1000)
    pl.add_argument("--wm-train-per-iter", type=int, default=4)
    pl.add_argument("--n-envs", type=int, default=16)
    pl.add_argument("--batch-size", type=int, default=16)
    pl.add_argument("--tpu", action="store_true", default=False,
                    help="Cible TPU v5e-8 (jax[tpu], 1 core) au lieu de GPU P100.")
    pl.add_argument("--buffer-device", choices=["gpu", "cpu"], default="cpu")
    pl.add_argument("--buffer-capacity", type=int, default=1_000_000)
    pl.add_argument("--no-use-rnd", action="store_true", default=True)
    pl.add_argument("--no-health-auto-stop", action="store_true", default=True)
    pl.add_argument("--extra-args", type=str, default="")
    pl.set_defaults(func=cmd_launch)

    for name, fn in (("status", cmd_status), ("pull", cmd_pull)):
        ps = sub.add_parser(name)
        ps.add_argument("--run-name", required=True)
        if name == "pull":
            ps.add_argument("--target", default="kaggle_outputs")
        ps.set_defaults(func=fn)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

"""
Launcher Kaggle pour le training Oneiro (notebook batch, quota gratuit).

Génère un notebook .ipynb (git clone branche + pip + run), le pousse via l'API
Kaggle en mode "save & run all", puis poll / récupère l'output.
Accélérateur : TPU v5e-8 avec --tpu (ce que la prod utilise), GPU P100 sinon.

Contraintes Kaggle prises en compte :
  - pas de logs en direct (output dispo à la FIN du kernel) → on poll le statut
  - session batch max ~12h → viser < 10h (mesuré : un run coupé à 9.7h), sinon
    reprendre via --resume-from-kernel
  - buffer 1M en RAM host (32GB Kaggle) via --buffer_device cpu

Auth : le CLI kaggle >= 2.2 veut un TOKEN, pas l'ancien couple username/key —
exporter KAGGLE_API_TOKEN (+ KAGGLE_USERNAME, qui sert à nommer le kernel).
Un ~/.kaggle/kaggle.json seul ne suffit PLUS : le CLI répond "Authentication
required" sans expliquer que le format a changé.
Dans ce repo : `source ./.env.kaggle` (fichier local, gitignored).

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
DEFAULT_BRANCH = "feat/tpu"   # branche de travail ; surchargeable via --branch

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
    if getattr(args, "use_rnd", False):
        flags.append("--use_rnd")
    else:
        flags.append("--no_use_rnd")
    if not args.health_auto_stop:
        flags.append("--no_health_auto_stop")
    if args.extra_args:
        flags.append(args.extra_args)
    train_cmd = "python -u crafter_dreamer/scripts/train_dreamer_jax.py " + " ".join(flags)
    # RESUME : le kernel précédent (kernel_sources) est monté sous /kaggle/input/<slug>/.
    # On reprend le checkpoint iterXXXXXX le plus avancé (zero-padded → tri lexical OK).
    # Fail-fast : si aucun checkpoint trouvé, le && court-circuite et le train ne part pas.
    train_prefix = ""
    if getattr(args, "resume_from_kernel", None):
        train_prefix = ('CKPT=$(ls -1 /kaggle/input/*/checkpoints/*iter*.npz 2>/dev/null | sort | tail -1) && '
                        'echo "RESUME depuis $CKPT" && ')
        train_cmd += ' --resume_from "$CKPT"'

    # JAX backend selon l'accélérateur : cuda12 pour GPU, tpu pour TPU v5e-8.
    # Sur TPU v5e-8, JAX voit les 8 cores ; le training s'auto-active en
    # data-parallel (mesh GSPMD, batch shardé sur 'data', state répliqué) —
    # batch_size doit être divisible par 8 (16 OK). buffer CPU idéal (host RAM).
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
        f"!rm -rf oneiro && git clone -b {args.branch} {REPO_URL}",
        # FIX env (juil. 2026) : l'image Kaggle est passée à NumPy 2.x, mais le numba
        # tiré par opensimplex 0.4.5 référence np.row_stack (retiré en NumPy 2.0) → crash
        # à env.reset(). On force numba récent (compatible NumPy 2, sans row_stack).
        f'%cd oneiro\n!pip install -q -r requirements.txt && pip install -q -U {jax_pkg} && pip install -q -U "numba>=0.60"',
        f"!{train_prefix}WORLDMODEL_OUTPUT_DIR=/kaggle/working {train_cmd}",
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


def kernel_metadata(user, slug, tpu=False, resume_from_kernel=None) -> dict:
    # kernel_sources : monte l'OUTPUT du kernel cité sous /kaggle/input/<slug>/
    # (checkpoints .npz + .meta.json inclus) → utilisé par --resume-from-kernel.
    sources = [f"{user}/{resume_from_kernel}"] if resume_from_kernel else []
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
        "kernel_sources": sources,
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
    resume_slug = getattr(args, "resume_from_kernel", None)
    if resume_slug:
        resume_slug = resume_slug.lower().replace("_", "-")[:50]
    (work / "kernel-metadata.json").write_text(
        json.dumps(kernel_metadata(user, slug, args.tpu, resume_from_kernel=resume_slug), indent=2))
    accel = "TPU v5e-8" if args.tpu else "GPU P100"
    print(f"Kernel   : {user}/{slug}")
    print(f"Branche  : {args.branch}")
    if resume_slug:
        print(f"Resume   : depuis l'output de {user}/{resume_slug} (checkpoint iter le plus avancé)")
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
    pl.add_argument("--branch", default=DEFAULT_BRANCH,
                    help=f"Branche clonee par le kernel (defaut {DEFAULT_BRANCH}). "
                         "Le launcher clone depuis GitHub, donc POUSSER avant de lancer : "
                         "un commit local non pousse ne sera pas dans le run.")
    pl.add_argument("--tpu", action="store_true", default=False,
                    help="Cible TPU v5e-8 (jax[tpu], 1 core) au lieu de GPU P100.")
    pl.add_argument("--buffer-device", choices=["gpu", "cpu"], default="cpu")
    pl.add_argument("--buffer-capacity", type=int, default=1_000_000)
    pl.add_argument("--use-rnd", action="store_true", default=False,
                    help="Active RND (bonus d'exploration intrinseque). Defaut off.")
    pl.add_argument("--no-use-rnd", dest="use_rnd", action="store_false",
                    help="Desactive RND (defaut).")
    pl.add_argument("--health-auto-stop", action="store_true", default=False,
                    help="Active l'auto-stop du health monitor. Defaut off : les runs vont "
                         "au bout meme en cas de warning persistant (on veut la trajectoire "
                         "complete pour l'analyse).")
    pl.add_argument("--extra-args", type=str, default="")
    pl.add_argument("--resume-from-kernel", type=str, default=None,
                    help="Slug d'un kernel précédent : monte son output (kernel_sources) et "
                         "reprend son checkpoint iterXXXXXX le plus avancé via --resume_from.")
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

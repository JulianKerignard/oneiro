"""
Sync des artefacts d'un run vers/depuis le storage du Teamspace Lightning.

Utilisé À L'INTÉRIEUR des jobs Lightning (lancés par lightning_train_jax.py) :
  - `up`   : pousse checkpoints/ et runs/ vers oneiro/<run_name>/ (périodique + fin de run)
  - `down` : récupère les checkpoints d'un run précédent (pour --resume_from)

Auth : variables d'env LIGHTNING_API_KEY / LIGHTNING_USER_ID / LIGHTNING_TEAMSPACE
(injectées par le launcher via Job.run(env=...)).
"""

import argparse
from pathlib import Path


def get_teamspace():
    from lightning_sdk import Teamspace
    return Teamspace()  # résolution via env LIGHTNING_*


def cmd_up(args):
    import time
    ts = get_teamspace()
    pushed, failed = [], []
    # runs/ d'abord : petit (summary JSON) et prioritaire pour le monitoring —
    # un échec sur les gros checkpoints ne doit pas le bloquer.
    for sub in ("runs", "checkpoints"):
        local = Path(args.workdir) / sub
        if not (local.is_dir() and any(local.iterdir())):
            continue
        for attempt in range(3):
            try:
                ts.upload_folder(
                    str(local),
                    remote_path=f"oneiro/{args.run_name}/{sub}",
                    progress_bar=False,
                )
                pushed.append(sub)
                break
            except Exception as e:
                print(f"[sync] {sub} tentative {attempt+1}/3 échouée : {type(e).__name__}: {e}")
                time.sleep(5)
        else:
            failed.append(sub)
    print(f"[sync] up : OK={pushed or '-'} FAILED={failed or '-'} -> oneiro/{args.run_name}/")


def cmd_down(args):
    ts = get_teamspace()
    target = Path(args.workdir) / "resume"
    try:
        ts.download_folder(
            f"oneiro/{args.run_name}/checkpoints", target_path=str(target),
        )
    except Exception as e:
        print(f"[sync] pas de checkpoints pour {args.run_name} ({e})")
        return
    # Le checkpoint le plus avancé (iterXXXXXX zero-paddé → tri lexical OK)
    ckpts = sorted(target.glob("*iter*.npz"))
    if ckpts:
        print(f"[sync] latest checkpoint : {ckpts[-1]}")
    else:
        print("[sync] dossier téléchargé mais aucun *iter*.npz")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    for name, fn in (("up", cmd_up), ("down", cmd_down)):
        ps = sub.add_parser(name)
        ps.add_argument("--run-name", required=True)
        ps.add_argument("--workdir", default=".")
        ps.set_defaults(func=fn)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

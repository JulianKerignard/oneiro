"""
Launcher Lightning AI pour le training Oneiro (Crafter DreamerV3 JAX).

Lance un Job Lightning (mode image Docker, pas besoin de Studio) qui clone le
repo public GitHub, installe les deps et lance train_dreamer_jax.py.

Auth (à mettre dans le shell, JAMAIS dans le code) :
    export LIGHTNING_API_KEY=...      # lightning.ai -> Profile -> Keys
    export LIGHTNING_USER_ID=...      # affiché au même endroit

Usage :
    # Lancer v19 sur RTXP 6000 spot
    python crafter_dreamer/scripts/lightning_train_jax.py launch \\
        --run-name v19_fast_critic_bootstrap \\
        --machine rtxp6000 --interruptible \\
        --train-iter 4000 --eval-interval 500 --wm-train-per-iter 4

    # Suivre les logs / le statut / arrêter
    python crafter_dreamer/scripts/lightning_train_jax.py logs   --run-name v19_fast_critic_bootstrap
    python crafter_dreamer/scripts/lightning_train_jax.py status --run-name v19_fast_critic_bootstrap
    python crafter_dreamer/scripts/lightning_train_jax.py stop   --run-name v19_fast_critic_bootstrap

Note : les checkpoints du job sont éphémères pour l'instant (les gates se
jugent aux logs EVAL). Persistance via path_mappings prévue pour le run 30k.
"""

import argparse
import os
import sys

REPO_URL = "https://github.com/JulianKerignard/oneiro.git"
IMAGE = "python:3.11-slim"

MACHINES = {
    "t4": "T4",
    "l4": "L4",
    "l40s": "L40S",
    "rtxp6000": "RTXP_6000",
    "a100": "A100",
    "h100": "H100",
}

# Machines avec assez de CPUs pour que --mp_collect paie (collecte Python parallèle)
MP_COLLECT_MACHINES = {"rtxp6000", "l40s", "a100", "h100"}


def check_auth():
    missing = [v for v in ("LIGHTNING_API_KEY", "LIGHTNING_USER_ID") if not os.environ.get(v)]
    if missing:
        print("ERREUR : variables d'auth manquantes :", ", ".join(missing))
        print("  -> lightning.ai -> Profile -> Keys, puis :")
        for v in missing:
            print(f"     export {v}=...")
        sys.exit(1)


def build_command(args) -> str:
    """Commande shell exécutée dans le conteneur du job."""
    train_flags = [
        f"--train_iter {args.train_iter}",
        f"--eval_interval {args.eval_interval}",
        f"--wm_train_per_iter {args.wm_train_per_iter}",
        f"--ac_train_per_iter {args.ac_train_per_iter}",
        f"--n_envs {args.n_envs}",
        f"--batch_size {args.batch_size}",
        f"--seed {args.seed}",
        f"--run_name {args.run_name}",
    ]
    if args.no_use_rnd:
        train_flags.append("--no_use_rnd")
    if args.machine in MP_COLLECT_MACHINES and not args.no_mp_collect:
        train_flags.append("--mp_collect")
    if args.extra_args:
        train_flags.append(args.extra_args)

    return " && ".join([
        "set -e",
        "apt-get update -qq",
        "apt-get install -y -qq git > /dev/null",
        f"git clone --depth 1 {REPO_URL} /work",
        "cd /work",
        "pip install -q -r requirements.txt",
        'pip install -q -U "jax[cuda12]==0.10.1"',
        # -u : stdout unbuffered pour le streaming des logs
        "python -u crafter_dreamer/scripts/train_dreamer_jax.py " + " ".join(train_flags),
    ])


def cmd_launch(args):
    check_auth()
    from lightning_sdk import Job, Machine

    machine = getattr(Machine, MACHINES[args.machine])
    command = build_command(args)

    print(f"Job        : {args.run_name}")
    print(f"Machine    : {MACHINES[args.machine]}  (interruptible={args.interruptible})")
    print(f"Max runtime: {args.max_hours}h")
    print(f"Command    : {command[:120]}...")
    print()

    job = Job.run(
        name=args.run_name,
        machine=machine,
        image=IMAGE,
        command=command,
        interruptible=args.interruptible,
        max_runtime=int(args.max_hours * 3600),
    )
    print(f"Job lancé : {job.name}")
    print(f"Dashboard : {job.link}")
    print(f"\nSuivi : python {sys.argv[0]} logs --run-name {args.run_name}")


def _get_job(name):
    check_auth()
    from lightning_sdk import Job
    return Job(name=name)


def cmd_logs(args):
    job = _get_job(args.run_name)
    # job.logs est une propriété (str). N'est disponible qu'une fois le job
    # terminé/arrêté (limitation SDK : pas de streaming pendant le run —
    # utiliser le dashboard web pour le temps réel).
    print(job.logs)


def cmd_status(args):
    job = _get_job(args.run_name)
    print(f"{job.name} : {job.status}")
    try:
        print(f"Coût : ${job.total_cost:.2f}")
    except Exception:
        pass
    print(f"Dashboard : {job.link}")


def cmd_stop(args):
    job = _get_job(args.run_name)
    job.stop()
    print(f"{job.name} : stop demandé.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)

    pl = sub.add_parser("launch", help="Lancer un job de training")
    pl.add_argument("--run-name", required=True)
    pl.add_argument("--machine", choices=sorted(MACHINES), default="rtxp6000")
    pl.add_argument("--interruptible", action="store_true", default=False,
                    help="Spot : 50-80%% moins cher mais préemptible (pas de resume pour l'instant).")
    pl.add_argument("--max-hours", type=float, default=3.0,
                    help="Durée max du job en heures (défaut 3h ; monter pour le run 30k).")
    pl.add_argument("--train-iter", type=int, default=4000)
    pl.add_argument("--eval-interval", type=int, default=500)
    pl.add_argument("--wm-train-per-iter", type=int, default=4)
    pl.add_argument("--ac-train-per-iter", type=int, default=1)
    pl.add_argument("--n-envs", type=int, default=16)
    pl.add_argument("--batch-size", type=int, default=16)
    pl.add_argument("--seed", type=int, default=42)
    pl.add_argument("--no-use-rnd", action="store_true", default=True,
                    help="RND off (défaut, comme v17-v19).")
    pl.add_argument("--no-mp-collect", action="store_true", default=False,
                    help="Désactive mp_collect même sur les grosses machines.")
    pl.add_argument("--extra-args", type=str, default="",
                    help="Flags bruts supplémentaires passés à train_dreamer_jax.py.")
    pl.set_defaults(func=cmd_launch)

    for name, fn in (("logs", cmd_logs), ("status", cmd_status), ("stop", cmd_stop)):
        ps = sub.add_parser(name)
        ps.add_argument("--run-name", required=True)
        ps.set_defaults(func=fn)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

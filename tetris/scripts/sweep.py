"""
Hyperparameter sweep pour le mini-Dreamer Tetris.

Lance N configs séquentiellement, sauve un résumé final.
Plot comparatif des eval scores.

Usage : python scripts/sweep.py
"""

import sys
import json
import subprocess
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


# ============================== Configs

# Chaque config = un dict d'args à passer à train_dreamer.py
CONFIGS = [
    {
        "run_name": "A_baseline",
        "entropy_coef": 0.003,
        "invalid_penalty": -0.1,
        "max_episode_steps": None,
    },
    {
        "run_name": "B_more_entropy",
        "entropy_coef": 0.05,
        "invalid_penalty": -0.1,
        "max_episode_steps": None,
    },
    {
        "run_name": "C_truncate",
        "entropy_coef": 0.003,
        "invalid_penalty": -0.1,
        "max_episode_steps": 200,
    },
    {
        "run_name": "D_strong_pen",
        "entropy_coef": 0.003,
        "invalid_penalty": -0.5,
        "max_episode_steps": None,
    },
    {
        "run_name": "E_all_soft_fixes",
        "entropy_coef": 0.05,
        "invalid_penalty": -0.5,
        "max_episode_steps": 200,
    },
]

# Hyperparams partagés (réduits pour le sweep)
TRAIN_ITER = 1500
EVAL_INTERVAL = 250


def run_config(config):
    """Lance une config via subprocess. Retourne le résumé."""
    cmd = [
        ".venv/bin/python", "scripts/train_dreamer.py",
        "--entropy_coef", str(config["entropy_coef"]),
        "--invalid_penalty", str(config["invalid_penalty"]),
        "--train_iter", str(TRAIN_ITER),
        "--eval_interval", str(EVAL_INTERVAL),
        "--run_name", config["run_name"],
    ]
    if config["max_episode_steps"] is not None:
        cmd.extend(["--max_episode_steps", str(config["max_episode_steps"])])

    print("\n" + "#" * 70)
    print(f"# RUN : {config['run_name']}")
    print(f"# CMD : {' '.join(cmd)}")
    print("#" * 70 + "\n")

    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"!!! Run {config['run_name']} a échoué (returncode={result.returncode})")
        return None

    # Charger le summary JSON
    summary_path = Path("runs") / f"dreamer_summary_{config['run_name']}.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return None


def main():
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print(f"Sweep de {len(CONFIGS)} configs, {TRAIN_ITER} iter chacune")
    print("Durée estimée : ~5 min × N = ~25 min")
    print("=" * 70)

    summaries = []
    overall_start = time.time()
    for cfg in CONFIGS:
        s = run_config(cfg)
        if s is not None:
            summaries.append(s)

    overall_elapsed = time.time() - overall_start
    print(f"\n\nSweep terminé en {overall_elapsed:.1f}s ({overall_elapsed/60:.1f}min)")

    # ------------------------- Comparatif

    if not summaries:
        print("Aucun run n'a réussi.")
        return

    print("\n" + "=" * 70)
    print("RÉCAP DES RUNS")
    print("=" * 70)
    print(f"{'run_name':<22s} {'score':>10s} {'length':>10s} {'lines':>8s}  config")
    print("-" * 70)
    for s in summaries:
        cfg = s["config"]
        print(
            f"{s['run_name']:<22s} "
            f"{s['final_eval']['score']:>10.2f} "
            f"{s['final_eval']['length']:>10.1f} "
            f"{s['final_eval']['lines']:>8.2f}  "
            f"H={cfg['entropy_coef']} pen={cfg['invalid_penalty']} maxep={cfg['max_episode_steps']}"
        )

    # Identifier le meilleur
    best = max(summaries, key=lambda s: s["final_eval"]["score"])
    print(f"\nMeilleur : {best['run_name']} (score={best['final_eval']['score']:.2f}, "
          f"lines={best['final_eval']['lines']:.2f})")

    # Plot comparatif bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    names = [s["run_name"] for s in summaries]
    scores = [s["final_eval"]["score"] for s in summaries]
    lengths = [s["final_eval"]["length"] for s in summaries]
    lines = [s["final_eval"]["lines"] for s in summaries]

    colors = plt.cm.tab10(np.arange(len(names)))

    axes[0].bar(names, scores, color=colors)
    axes[0].set_title("Eval score moyen (10 ep)")
    axes[0].set_ylabel("score")
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].axhline(0, color="black", linewidth=0.5)

    axes[1].bar(names, lengths, color=colors)
    axes[1].set_title("Eval episode length moyen")
    axes[1].set_ylabel("steps")
    axes[1].grid(True, alpha=0.3, axis="y")

    axes[2].bar(names, lines, color=colors)
    axes[2].set_title("Eval lignes clearées moyennes")
    axes[2].set_ylabel("lines")
    axes[2].grid(True, alpha=0.3, axis="y")

    for ax in axes:
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Hyperparameter sweep mini-Dreamer Tetris", fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    sweep_png = runs_dir / "sweep_comparison.png"
    plt.savefig(sweep_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComparatif PNG : {sweep_png}")

    # Sauvegarde JSON global
    with open(runs_dir / "sweep_results.json", "w") as f:
        json.dump({
            "total_time_sec": overall_elapsed,
            "configs": [s for s in summaries],
            "best_run_name": best["run_name"],
        }, f, indent=2)
    print(f"Sweep results : runs/sweep_results.json")


if __name__ == "__main__":
    main()

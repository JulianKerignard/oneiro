"""
Génère les figures du paper depuis experiments/data/*.csv (runs passés, via
parse_logs.py) et les summaries JSON enrichis (runs futurs).

Figures :
  1. learning_curves.png   — achievements vs env_steps, tous les runs
  2. internals_<run>.png   — métriques internes (H, ret/val, p95, kl, rec, scale)
  3. spectrum_<run>.png    — heatmap taux par achievement × EVAL

Usage :
    python experiments/plot_runs.py                  # toutes les figures
    python experiments/plot_runs.py --run v21-benchmark-1m
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DATA = HERE / "data"
FIGS = HERE / "figures"

RUN_LABELS = {
    "v19b-fast-critic-nohealthstop": "v19b (γ=0.99, fast critic)",
    "v20-gamma0997": "v20 (γ=0.997)",
    "v20b-gamma0997-8k": "v20b (γ=0.997, 8k)",
    "v21-benchmark-1m": "v21 (γ=0.997, buffer 500k, 1M steps)",
    "v22-buffer1m": "v22 (buffer 1M)",
}


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def fnum(x, default=None):
    try:
        return float(x)
    except (ValueError, TypeError):
        return default


def fig_learning_curves(eval_files):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for path in eval_files:
        run = path.stem.replace("_evals", "")
        rows = read_csv(path)
        steps = [fnum(r["env_steps"]) for r in rows]
        ach = [fnum(r["achievements"]) for r in rows]
        label = RUN_LABELS.get(run, run)
        line, = ax.plot(steps, ach, marker="o", ms=3, lw=1.5, label=label)
        sample = [fnum(r.get("sample")) for r in rows]
        if any(s is not None for s in sample):
            ax.plot(steps, sample, ls="--", lw=0.8, alpha=0.5, color=line.get_color())
    ax.axhline(4.3, color="gray", ls=":", lw=1)
    ax.text(0.99, 4.35, "Rainbow @ 1M", ha="right", fontsize=8, color="gray",
            transform=ax.get_yaxis_transform())
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Achievements / episode (eval argmax ; -- sample)")
    ax.set_title("Oneiro — Crafter learning curves")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIGS / "learning_curves.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


def fig_internals(metrics_file):
    run = metrics_file.stem.replace("_metrics", "")
    rows = read_csv(metrics_file)
    it = [fnum(r["iter"]) for r in rows]
    panels = [
        ("entropy", "Policy entropy H", None),
        (("returns_mean", "values_mean"), "Imagined returns vs values", None),
        (("p5", "p95"), "Return percentiles (EMA)", None),
        ("loss_kl", "KL (per-step, summed)", None),
        ("loss_recon", "Reconstruction loss", None),
        ("return_scale", "Advantage scale", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), dpi=150)
    for ax, (keys, title, _) in zip(axes.flat, panels):
        for key in ([keys] if isinstance(keys, str) else keys):
            ax.plot(it, [fnum(r[key]) for r in rows], lw=0.9, label=key)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
        if not isinstance(keys, str):
            ax.legend(fontsize=7)
    fig.suptitle(f"Internal metrics — {RUN_LABELS.get(run, run)}", fontsize=11)
    fig.tight_layout()
    out = FIGS / f"internals_{run}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


def fig_spectrum(evals_file):
    run = evals_file.stem.replace("_evals", "")
    rows = read_csv(evals_file)
    # Achievements rencontrés, ordonnés par 1ère apparition
    names = []
    for r in rows:
        for pair in (r.get("unlocked_detail") or "").split(";"):
            if ":" in pair:
                n = pair.split(":")[0]
                if n not in names:
                    names.append(n)
    if not names:
        return
    grid = [[0.0] * len(rows) for _ in names]
    for j, r in enumerate(rows):
        d = dict(p.split(":") for p in (r.get("unlocked_detail") or "").split(";") if ":" in p)
        for i, n in enumerate(names):
            grid[i][j] = float(d.get(n, 0))
    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 0.4), 0.35 * len(names) + 1.5), dpi=150)
    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=100)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_xticks(range(len(rows)), [r["iter"] for r in rows], fontsize=6, rotation=45)
    ax.set_xlabel("Eval @ iter")
    ax.set_title(f"Achievement success rates (%) — {RUN_LABELS.get(run, run)}", fontsize=10)
    fig.colorbar(im, ax=ax, label="%")
    fig.tight_layout()
    out = FIGS / f"spectrum_{run}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None, help="Limiter à un run (sinon tous)")
    args = p.parse_args()

    FIGS.mkdir(parents=True, exist_ok=True)
    pattern = f"{args.run}_evals.csv" if args.run else "*_evals.csv"
    eval_files = sorted(DATA.glob(pattern))
    fig_learning_curves(eval_files)
    for f in eval_files:
        fig_spectrum(f)
    m_pattern = f"{args.run}_metrics.csv" if args.run else "*_metrics.csv"
    for f in sorted(DATA.glob(m_pattern)):
        fig_internals(f)


if __name__ == "__main__":
    main()

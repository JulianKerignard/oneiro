"""
Rapport PDF multi-pages comparant des runs, depuis leurs summaries JSON.

Chaque page repond a une question :
  1. courbes d'apprentissage      — ou en est-on vs les baselines publiees ?
  2. arbre du craft               — quels paliers se debloquent, et quand ?
  3. spectre par achievement      — heatmap taux x eval, un panneau par run
  4. metriques internes           — le WM et l'actor-critic sont-ils sains ?

L'axe x est en ENV STEPS (pas en iterations) : c'est l'unite du benchmark Crafter,
et c'est ce qui rend deux runs d'architectures differentes comparables.

Usage :
    python experiments/make_report.py                      # tous les runs recents
    python experiments/make_report.py --runs v54-... v55-... --output rapport.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).parent
KAGGLE = HERE.parent / "kaggle_outputs"

# Scoreboard Crafter officiel (github.com/danijar/crafter), budget 1M env steps.
# Uniquement des gris (+ un rouge pour la cible DreamerV3) : les baselines doivent
# rester visuellement distinctes des courbes de runs, qui elles sont en couleurs pleines.
BASELINES = [
    ("DreamerV3  14.5%", 14.5, "#c0392b"),
    ("DreamerV2  10.0%", 10.0, "#5d6d7e"),
    ("PPO         4.6%", 4.6, "#7f8c8d"),
    ("Rainbow     4.3%", 4.3, "#95a5a6"),
    ("Random      1.6%", 1.6, "#bdc3c7"),
]
BUDGET = 1_000_000          # budget benchmark officiel

# La chaine du craft, dans l'ordre des prerequis (crafter/data.yaml).
CHAIN = ["collect_wood", "place_table", "make_wood_pickaxe", "collect_stone",
         "place_stone", "collect_coal", "place_furnace", "make_stone_pickaxe"]

PALETTE = ["#2c3e50", "#e67e22", "#16a085", "#8e44ad", "#c0392b"]


def describe_arch(run: str, config: dict) -> str:
    """Etiquette de run : nombre exact de params + dimensions cles.

    Le summary JSON ne stocke pas le compte de params, et les runs anterieurs a
    l'exposition CLI de l'architecture n'ont meme pas h_dim/cnn_depth dans leur
    config. On lit donc le dernier checkpoint : les formes des tenseurs donnent
    les deux, sans dependre de ce que la config a bien voulu enregistrer.
    """
    h_dim, cnn = config.get("h_dim"), config.get("cnn_depth")
    n_params = None
    ckpts = sorted((KAGGLE / run / "checkpoints").glob("*.npz")) if (KAGGLE / run).exists() else []
    if ckpts:
        try:
            import numpy as np
            z = np.load(ckpts[-1], allow_pickle=False)
            # On compte le MODELE : encoder + rssm + decoder + heads + actor + critic.
            # Exclus : les extras de reprise (prefixe __, etat Adam / scale EMA) et le
            # slow_critic, qui est une copie EMA du critic servant de cible de
            # regularisation — le compter doublerait le critic et gonflerait le total
            # (70.5M au lieu de 64.9M sur v54).
            n_params = sum(int(np.prod(z[k].shape)) for k in z.files
                           if not k.startswith(("__", "slow_critic.")))
            if h_dim is None and "rssm.gru.dense_h.kernel" in z.files:
                h_dim = int(z["rssm.gru.dense_h.kernel"].shape[0])
            if cnn is None:
                for k in ("encoder.conv1.kernel", "encoder.conv_1.kernel"):
                    if k in z.files:
                        cnn = int(z[k].shape[-1])
                        break
        except Exception:
            pass
    size = f"{n_params / 1e6:.1f}M" if n_params else "?"
    return f"{run}  —  {size} params  (deter {h_dim or '?'}, cnn {cnn or '?'})"


def load(run: str) -> dict:
    hits = sorted(KAGGLE.glob(f"{run}/runs/dreamer_crafter_jax_summary_*.json"))
    if not hits:
        raise SystemExit(f"pas de summary JSON pour '{run}' sous {KAGGLE}/{run}/runs/")
    d = json.loads(hits[0].read_text())
    d["_steps"] = d.get("env_steps_per_iter", 32)
    d["_label"] = describe_arch(run, d.get("config", {}))
    return d


def ev_steps(d):
    return [i * d["_steps"] / 1e6 for i in d["history"]["eval_iter"]]


def it_steps(d):
    return [i * d["_steps"] / 1e6 for i in d["history"]["iter"]]


def mark_budget(ax):
    ax.axvline(BUDGET / 1e6, color="k", ls=":", lw=1, alpha=0.6)
    ax.text(BUDGET / 1e6, ax.get_ylim()[1], " budget 1M ", fontsize=7,
            va="top", ha="left", color="k", alpha=0.7)


def page_curves(pdf, runs):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)

    ax = axes[0]
    for d, col in zip(runs, PALETTE):
        ax.plot(ev_steps(d), d["history"]["eval_achievements"], marker="o", ms=3.5,
                lw=1.8, color=col, label=d["_label"])
        if any(d["history"].get("eval_sample", [])):
            ax.plot(ev_steps(d), d["history"]["eval_sample"], ls="--", lw=0.8,
                    alpha=0.45, color=col)
    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel("Achievements / épisode")
    ax.set_title("Achievements par épisode  (trait plein : argmax — pointillé : sample)",
                 fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    mark_budget(ax)

    ax = axes[1]
    for d, col in zip(runs, PALETTE):
        ax.plot(ev_steps(d), d["history"]["eval_crafter_score"], marker="o", ms=3.5,
                lw=1.8, color=col, label=d["_label"])
    for i, (name, val, col) in enumerate(BASELINES):
        ax.axhline(val, color=col, ls="--", lw=1, alpha=0.8)
        # PPO (4.6%) et Rainbow (4.3%) sont trop proches pour deux etiquettes alignees :
        # on alterne gauche/droite pour qu'elles ne se recouvrent pas.
        right = i % 2 == 0
        ax.text(0.995 if right else 0.005, val, (name + " ") if right else (" " + name),
                fontsize=7, color=col, ha="right" if right else "left", va="bottom",
                transform=ax.get_yaxis_transform())
    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel("Crafter score (%)")
    ax.set_title("Crafter score vs baselines publiées  (protocole officiel, cumulé)",
                 fontsize=10)
    ax.grid(alpha=0.25)
    mark_budget(ax)

    fig.suptitle("Courbes d'apprentissage — Oneiro sur Crafter", fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def page_chain(pdf, runs):
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2), dpi=150, sharex=True)
    for ax, ach in zip(axes.flat, CHAIN):
        for d, col in zip(runs, PALETTE):
            ys = [100 * det.get(ach, 0.0) for det in d["history"]["eval_detail"]]
            ax.plot(ev_steps(d), ys, lw=1.6, color=col, marker="o", ms=2.5)
        ax.set_title(ach, fontsize=9)
        ax.set_ylim(-4, 104)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
    for ax in axes[1]:
        ax.set_xlabel("env steps (M)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("taux de réussite (%)", fontsize=8)
    handles = [plt.Line2D([], [], color=c, lw=2, label=d["_label"])
               for d, c in zip(runs, PALETTE)]
    fig.legend(handles=handles, fontsize=8, loc="lower center", ncol=len(runs),
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Arbre du craft — chaque palier ouvre le suivant "
                 "(ordre des prérequis de data.yaml)", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def page_spectrum(pdf, runs):
    for d in runs:
        dets = d["history"]["eval_detail"]
        names = sorted({k for det in dets for k in det},
                       key=lambda n: -max(det.get(n, 0) for det in dets))
        if not names:
            continue
        grid = [[100 * det.get(n, 0.0) for det in dets] for n in names]
        fig, ax = plt.subplots(figsize=(max(7, len(dets) * 0.55),
                                        0.32 * len(names) + 2.2), dpi=150)
        im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=100)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.set_xticks(range(len(dets)),
                      [f"{s:.2f}" for s in ev_steps(d)], fontsize=7, rotation=45)
        ax.set_xlabel("Environment steps (millions)", fontsize=9)
        ax.set_title(f"Taux de réussite par achievement — {d['_label']}", fontsize=11)
        fig.colorbar(im, ax=ax, label="%")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def page_internals(pdf, runs):
    panels = [
        (("loss_recon",), "Reconstruction (WM)", True),
        (("loss_kl",), "KL par step (free bits = 1)", False),
        (("loss_reward",), "Reward head (CE twohot)", True),
        (("entropy",), "Entropie de la politique H", False),
        (("returns_mean", "values_mean"), "Returns imaginés vs values", False),
        (("return_scale",), "Scale des advantages (P95−P5)", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2), dpi=150)
    for ax, (keys, title, logy) in zip(axes.flat, panels):
        for d, col in zip(runs, PALETTE):
            x = it_steps(d)
            for k, ls in zip(keys, ["-", "--"]):
                if k in d["history"]:
                    ax.plot(x, d["history"][k], lw=0.9, color=col, ls=ls, alpha=0.9)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("env steps (M)", fontsize=8)
        if logy:
            ax.set_yscale("log")
    handles = [plt.Line2D([], [], color=c, lw=2, label=d["_label"])
               for d, c in zip(runs, PALETTE)]
    fig.legend(handles=handles, fontsize=8, loc="lower center", ncol=len(runs),
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Métriques internes — santé du world model et de l'actor-critic",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=["v54-rewardin-40k", "v55-small14m-40k"])
    p.add_argument("--output", default="experiments/figures/rapport_oneiro.pdf")
    args = p.parse_args()

    runs = [load(r) for r in args.runs]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out) as pdf:
        page_curves(pdf, runs)
        page_chain(pdf, runs)
        page_spectrum(pdf, runs)
        page_internals(pdf, runs)
        meta = pdf.infodict()
        meta["Title"] = "Oneiro — courbes d'apprentissage sur Crafter"
        meta["Subject"] = ", ".join(d["run_name"] for d in runs)

    print(f"PDF écrit : {out}  ({out.stat().st_size / 1024:.0f} Ko)")
    for d in runs:
        h = d["history"]
        print(f"  {d['run_name']:<22} {d['iter']} iter = {d['iter'] * d['_steps'] / 1e6:.2f}M steps"
              f"   best {max(h['eval_achievements']):.2f} ach"
              f"   score final {h['eval_crafter_score'][-1]:.2f}%")


if __name__ == "__main__":
    main()

"""
Rapport PDF multi-pages comparant des runs, depuis leurs summaries JSON.

Chaque page repond a une question :
  1. courbes d'apprentissage      — ou en est-on vs les baselines publiees ?
  2. arbre du craft               — quels paliers se debloquent, et quand ?
  3. spectre par achievement      — heatmap taux x eval, un panneau par run
  4. metriques internes           — le WM et l'actor-critic sont-ils sains ?
  5. tendance                     — le run apprend-il ENCORE, ou a-t-il plafonne ?

L'axe x est en ENV STEPS (pas en iterations) : c'est l'unite du benchmark Crafter,
et c'est ce qui rend deux runs d'architectures differentes comparables.

Usage :
    python experiments/make_report.py                      # tous les runs recents
    python experiments/make_report.py --runs v54-... v55-... --output rapport.pdf
"""

import argparse
import json
import math
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
# Les 22 achievements de Crafter, pour recomposer le crafter_score d'une chaine de runs.
ALL22 = ["collect_wood", "place_table", "eat_cow", "collect_sapling", "collect_drink",
         "make_wood_pickaxe", "make_wood_sword", "place_plant", "defeat_zombie",
         "collect_stone", "place_stone", "eat_plant", "defeat_skeleton", "collect_coal",
         "make_stone_pickaxe", "make_stone_sword", "wake_up", "place_furnace",
         "collect_iron", "make_iron_pickaxe", "make_iron_sword", "collect_diamond"]

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
    # Le replay ratio est souvent LA variable entre deux runs de meme archi : sans lui,
    # deux etiquettes identiques rendraient la legende inutile.
    # Formule de train_dreamer_jax.py:1633 : batch * seq_len * wm_train / collected.
    wm = config.get("wm_train_per_iter")
    bs = config.get("batch_size", 16)
    ratio = f", ratio {int(bs * 64 * wm / 32)}" if wm else ""
    return f"{run}  —  {size} params  (deter {h_dim or '?'}, cnn {cnn or '?'}{ratio})"


def load(run: str) -> dict:
    hits = sorted(KAGGLE.glob(f"{run}/runs/dreamer_crafter_jax_summary_*.json"))
    if not hits:
        raise SystemExit(f"pas de summary JSON pour '{run}' sous {KAGGLE}/{run}/runs/")
    d = json.loads(hits[0].read_text())
    d["_steps"] = d.get("env_steps_per_iter", 32)
    d["_label"] = describe_arch(run, d.get("config", {}))
    return d


def is_resume(d) -> bool:
    """Un run repris commence ses evals loin de zero : son crafter_score cumule ne
    couvre que la portion post-reprise, donc il n'est PAS comparable a un run complet."""
    ev = d["history"]["eval_iter"]
    if len(ev) < 2:
        return False
    return ev[0] > (ev[1] - ev[0])


def crafter_score(counts: dict, n_eps: int) -> float:
    """Moyenne geometrique des taux de reussite, protocole Crafter officiel."""
    return math.exp(sum(math.log(1 + 100 * counts.get(a, 0) / n_eps) for a in ALL22) / 22) - 1


def chain_score(child: dict, parent: dict) -> tuple:
    """Recompose le crafter_score d'une chaine parent -> enfant repris.

    Les compteurs d'achievements ne sont pas restaures a la reprise (checkpoints
    anterieurs au commit 322bf6e), donc on additionne les compteurs finaux des deux
    runs. Seul le point FINAL est reconstituable : les compteurs par eval ne sont pas
    dans les summaries, donc la courbe intermediaire reste incalculable.
    """
    out = {}
    for d in (parent, child):
        n = d["train_episode_count"]
        for a in ALL22:
            out[a] = out.get(a, 0) + d["train_success_rates"].get(a, 0) * n
    n_tot = parent["train_episode_count"] + child["train_episode_count"]
    return crafter_score(out, n_tot), n_tot


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
    warned = False
    for d, col in zip(runs, PALETTE):
        resumed = is_resume(d)
        # Un cumul post-reprise ne contient aucun episode de debut d'entrainement :
        # il surestime largement. On le degrade visuellement pour qu'il ne se lise
        # pas comme une courbe comparable aux autres.
        ax.plot(ev_steps(d), d["history"]["eval_crafter_score"], marker="o", ms=3.5,
                lw=0.9 if resumed else 1.8, ls=":" if resumed else "-",
                alpha=0.45 if resumed else 1.0, color=col, label=d["_label"])
        if resumed:
            warned = True
            ax.annotate("cumul depuis la reprise\nNON comparable",
                        xy=(ev_steps(d)[-1], d["history"]["eval_crafter_score"][-1]),
                        xytext=(-6, 4), textcoords="offset points", fontsize=6.5,
                        color=col, ha="right", va="bottom", style="italic")
        star = d.get("_chain_score")
        if star:
            ax.plot([ev_steps(d)[-1]], [star[0]], marker="*", ms=16, color=col,
                    mec="k", mew=0.6, zorder=5)
            ax.annotate(f"chaine complete : {star[0]:.2f}%\n({star[1]} eps depuis 0)",
                        xy=(ev_steps(d)[-1], star[0]), xytext=(-8, -14),
                        textcoords="offset points", fontsize=7, color=col,
                        ha="right", va="top", fontweight="bold")
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
    title = "Crafter score vs baselines publiées  (protocole officiel, cumulé)"
    if warned:
        title += "\n★ = score recomposé sur la chaîne complète depuis 0 env step"
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    mark_budget(ax)

    fig.suptitle("Courbes d'apprentissage — Oneiro sur Crafter", fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def _linfit(xs, ys):
    """Regression lineaire simple -> (pente, erreur-type de la pente, ordonnee)."""
    n = len(xs)
    if n < 3:
        return 0.0, float("inf"), ys[-1] if ys else 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return 0.0, float("inf"), my
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    se = (sum(r * r for r in resid) / (n - 2) / sxx) ** 0.5
    return b, se, a


def _smooth(ys, w):
    """Moyenne mobile centree, fenetre w (impaire de fait via les bornes)."""
    out = []
    for i in range(len(ys)):
        lo, hi = max(0, i - w // 2), min(len(ys), i + w // 2 + 1)
        out.append(sum(ys[lo:hi]) / (hi - lo))
    return out


def page_trend(pdf, runs, fit_pts=4):
    """Le run apprend-il encore ?

    Le crafter_score cumule ne peut par construction que monter : il ne repond pas a
    la question. Trois signaux qui, eux, peuvent redescendre :
      - achievements par eval, avec la pente ajustee sur les derniers points ;
      - env_reward_per_step, dense (1 point / 50 iter) donc bien moins bruite ;
      - efficacite (achievements pour 100 steps), qui separe "mieux jouer" de
        "survivre plus longtemps".
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=150)

    # --- 1. achievements par eval + extrapolation de la pente finale
    ax = axes[0][0]
    verdicts = []
    for d, col in zip(runs, PALETTE):
        xs, ys = ev_steps(d), d["history"]["eval_achievements"]
        ax.plot(xs, ys, marker="o", ms=3.5, lw=1.8, color=col, label=d["_label"].split("  —")[0])
        if len(xs) >= fit_pts:
            fx, fy = xs[-fit_pts:], ys[-fit_pts:]
            b, se, a = _linfit(fx, fy)
            ext = [fx[0], fx[-1] + 0.25]
            ax.plot(ext, [a + b * x for x in ext], ls="--", lw=1.2, color=col, alpha=0.65)
            sig = "monte" if b - 2 * se > 0 else ("descend" if b + 2 * se < 0 else "plat")
            verdicts.append((d, col, b, se, sig))
    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel("Achievements / épisode")
    ax.set_title(f"Tendance — droite ajustée sur les {fit_pts} dernières évals", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="upper left")

    # --- 2. verdict chiffre
    ax = axes[0][1]
    ax.axis("off")
    ax.text(0, 1.0, "Apprend-il encore ?", fontsize=11, fontweight="bold", va="top")
    ax.text(0, 0.93, f"pente sur les {fit_pts} dernières évals, ± erreur-type\n"
                     "significatif = la pente s'écarte de 0 de plus de 2 σ",
            fontsize=7.5, va="top", style="italic", color="#555")
    y = 0.80
    for d, col, b, se, sig in verdicts:
        mark = {"monte": "OUI", "descend": "REGRESSE", "plat": "indécis"}[sig]
        ax.text(0, y, d["_label"].split("  —")[0], fontsize=9, color=col, fontweight="bold", va="top")
        ax.text(0.02, y - 0.06, f"{b:+.2f} ± {se:.2f} ach / Mstep      →  {mark}",
                fontsize=9, color=col, va="top", family="monospace")
        y -= 0.16
    ax.text(0, max(y, 0.04),
            "Un « indécis » ne veut pas dire plateau : sur 75 épisodes par éval,\n"
            "le bruit d'échantillonnage suffit à masquer une pente réelle.",
            fontsize=7.5, va="top", style="italic", color="#555")

    # --- 3. signal dense : reward par step
    ax = axes[1][0]
    for d, col in zip(runs, PALETTE):
        xs = it_steps(d)
        ys = d["history"]["env_reward_per_step"]
        ax.plot(xs, ys, lw=0.5, color=col, alpha=0.18)
        ax.plot(xs, _smooth(ys, 21), lw=1.8, color=col)
    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel("Reward / step (collecte)")
    ax.set_title("Signal dense — reward par step, lissé sur 21 points\n"
                 "(politique de collecte, pas l'éval argmax)", fontsize=10)
    ax.grid(alpha=0.25)

    # --- 4. efficacite : distinguer "mieux jouer" de "survivre plus longtemps"
    ax = axes[1][1]
    for d, col in zip(runs, PALETTE):
        xs = ev_steps(d)
        eff = [100 * a / l if l else 0
               for a, l in zip(d["history"]["eval_achievements"], d["history"]["eval_length"])]
        ax.plot(xs, eff, marker="o", ms=3.5, lw=1.8, color=col)
    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel("Achievements pour 100 steps")
    ax.set_title("Efficacité — un agent qui survit plus longtemps gagne des\n"
                 "achievements sans mieux jouer ; cette courbe neutralise l'effet", fontsize=10)
    ax.grid(alpha=0.25)

    fig.suptitle("Le run apprend-il encore ?", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
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


def page_reward_diagnostics(pdf, runs):
    """Diagnostics reward head + imagination. Absents des runs anterieurs a da09328."""
    have = [d for d in runs if "rew_pred_ach" in d["history"]]
    if not have:
        return
    panels = [
        ("rew_pred_ach", "Prediction sur les ACHIEVEMENTS (cible ~1.0)", None),
        ("rew_pred_zero", "Prediction sur les etats a reward NUL (doit rester ~0)", 0.005),
        ("rew_n_ach", "Achievements par batch (densite du replay)", None),
        ("img_rew_max", "Reward MAX dans l'imagination", None),
        ("img_rew_frac_hi", "Fraction des rewards imagines > 0.5", None),
        ("img_rew_mean", "Reward moyen dans l'imagination", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2), dpi=150)
    for ax, (key, title, target) in zip(axes.flat, panels):
        for d, col in zip(have, PALETTE):
            if key in d["history"]:
                ax.plot(it_steps(d), d["history"][key], lw=0.9, color=col, alpha=0.9)
        if target is not None:
            ax.axhline(target, color="k", ls=":", lw=1, alpha=0.6)
            ax.text(0.99, target, f" cible {target} ", fontsize=6, ha="right", va="bottom",
                    transform=ax.get_yaxis_transform())
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("env steps (M)", fontsize=8)
    handles = [plt.Line2D([], [], color=c, lw=2, label=d["_label"])
               for d, c in zip(have, PALETTE)]
    fig.legend(handles=handles, fontsize=8, loc="lower center", ncol=len(have),
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Diagnostics reward head et imagination — le signal que voit l'actor",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=["v54-rewardin-40k", "v55-small14m-40k"])
    p.add_argument("--output", default="experiments/figures/rapport_oneiro.pdf")
    p.add_argument("--resume-parent", nargs="+", default=[], metavar="ENFANT=PARENT",
                   help="recompose le crafter_score d'un run repris en y ajoutant les "
                        "episodes de son run parent (ex: v58-...=v56-ratio256)")
    args = p.parse_args()

    runs = [load(r) for r in args.runs]
    for pair in args.resume_parent:
        child_name, _, parent_name = pair.partition("=")
        child = next((d for d in runs if d["run_name"] == child_name), None)
        if child is None:
            raise SystemExit(f"--resume-parent : '{child_name}' n'est pas dans --runs")
        child["_chain_score"] = chain_score(child, load(parent_name))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out) as pdf:
        page_curves(pdf, runs)
        page_trend(pdf, runs)
        page_chain(pdf, runs)
        page_spectrum(pdf, runs)
        page_internals(pdf, runs)
        page_reward_diagnostics(pdf, runs)
        meta = pdf.infodict()
        meta["Title"] = "Oneiro — courbes d'apprentissage sur Crafter"
        meta["Subject"] = ", ".join(d["run_name"] for d in runs)

    print(f"PDF écrit : {out}  ({out.stat().st_size / 1024:.0f} Ko)")
    for d in runs:
        h = d["history"]
        print(f"  {d['run_name']:<22} {d['iter']} iter = {d['iter'] * d['_steps'] / 1e6:.2f}M steps"
              f"   best {max(h['eval_achievements']):.2f} ach"
              f"   score final {h['eval_crafter_score'][-1]:.2f}%"
              + (f"  (reprise — chaine complete : {d['_chain_score'][0]:.2f}%)"
                 if d.get("_chain_score") else
                 "  (reprise — NON comparable)" if is_resume(d) else ""))


if __name__ == "__main__":
    main()

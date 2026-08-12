"""
Produit les CSV exploités par plot_runs.py pour les figures du paper.

Deux sources, dans cet ordre de préférence :

  1. Les summaries JSON  (runs Kaggle/TPU, v23+)  ← LA source à utiliser
     kaggle_outputs/<run>/runs/dreamer_crafter_jax_summary_<run>.json
     Ils contiennent l'intégralité du run. C'est important : Kaggle TRONQUE le début
     des gros logs (sur v52/v53 les lignes d'itération ne commencent qu'à ~8000), donc
     le texte du log n'est PAS une source fiable pour ces runs.

  2. Les logs stdout à plat (ère Modal/Lightning, v19b→v22)
     experiments/raw_logs/<run>.log
     Conservés pour les runs antérieurs, qui n'ont pas de summary JSON.

Sorties : experiments/data/<run>_metrics.csv  (1 ligne / LOG_INTERVAL=50 iter)
          experiments/data/<run>_evals.csv    (1 ligne / EVAL, détail par achievement)

/!\\ Les logs Kaggle sont un tableau JSON ({"stream_name":..,"data":".."}), pas du texte
plat : le parseur de logs ci-dessous ne les lit PAS. Utiliser --summaries pour eux.

Usage :
    python experiments/parse_logs.py --summaries          # tous les runs Kaggle (JSON)
    python experiments/parse_logs.py                      # tous les raw_logs/ (texte plat)
    python experiments/parse_logs.py raw_logs/v21*.log    # sélection
"""

import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
KAGGLE = HERE.parent / "kaggle_outputs"

# iter  3000/8000 [37.5%] | WM wm=24.24 rec=19.14 kl=4.95 rew=0.135 con=0.012 |
# AC act=-0.000 crit=1.719 pg=-0.000 H=0.61 | img ret=0.51 val=0.50 scale=2.73
# p5=-0.42 p95=2.30 | r/step=0.0111 | 2.8 ips ETA 29m57
ITER_RE = re.compile(
    r"iter\s+(?P<iter>\d+)/(?P<total>\d+).*?"
    r"wm=(?P<loss_wm>-?[\d.]+) rec=(?P<loss_recon>-?[\d.]+) kl=(?P<loss_kl>-?[\d.]+) "
    r"rew=(?P<loss_reward>-?[\d.]+) con=(?P<loss_continue>-?[\d.]+).*?"
    r"act=(?P<loss_actor>-?[\d.]+) crit=(?P<loss_critic>-?[\d.]+) "
    r"pg=(?P<loss_pg>-?[\d.]+) H=(?P<entropy>-?[\d.]+).*?"
    r"ret=(?P<returns_mean>-?[\d.]+) val=(?P<values_mean>-?[\d.]+) "
    r"scale=(?P<return_scale>-?[\d.]+) p5=(?P<p5>-?[\d.]+) p95=(?P<p95>-?[\d.]+).*?"
    r"r/step=(?P<reward_per_step>-?[\d.]+).*?(?P<ips>[\d.]+) ips"
)

# >>> EVAL @ iter 8000 : score=1.70  length=186  achievements=2.60 =  sample=3.20  (best=2.60 @8000)
EVAL_RE = re.compile(
    r">>> EVAL @ iter (?P<iter>\d+) : score=(?P<score>-?[\d.]+)\s+length=(?P<length>[\d.]+)\s+"
    r"achievements=(?P<achievements>[\d.]+)(?:\s+[↑↓=])?(?:\s+sample=(?P<sample>[\d.]+))?"
)

# unlocked (5/22): collect_sapling=100%  wake_up=90%  ...
UNLOCKED_RE = re.compile(r"unlocked \((?P<n>\d+)/22\):\s*(?P<detail>.*)")
ACH_RE = re.compile(r"(\w+)=(\d+)%")

ENV_STEPS_PER_ITER = 32  # 2 boucles × 16 envs (constant sur tous les runs v17+)


def parse_log(path: Path, out_dir: Path):
    run = path.stem
    metrics_rows, eval_rows = [], []
    pending_eval = None

    for line in path.read_text(errors="replace").splitlines():
        m = ITER_RE.search(line)
        if m:
            d = {k: float(v) for k, v in m.groupdict().items()}
            d["iter"] = int(d["iter"])
            d["env_steps"] = d["iter"] * ENV_STEPS_PER_ITER
            metrics_rows.append(d)
            continue
        m = EVAL_RE.search(line)
        if m:
            pending_eval = {
                "iter": int(m["iter"]),
                "env_steps": int(m["iter"]) * ENV_STEPS_PER_ITER,
                "score": float(m["score"]),
                "length": float(m["length"]),
                "achievements": float(m["achievements"]),
                "sample": float(m["sample"]) if m["sample"] else "",
                "n_unlocked": "",
                "unlocked_detail": "",
            }
            eval_rows.append(pending_eval)
            continue
        m = UNLOCKED_RE.search(line)
        if m and pending_eval is not None:
            pending_eval["n_unlocked"] = int(m["n"])
            pending_eval["unlocked_detail"] = ";".join(
                f"{name}:{pct}" for name, pct in ACH_RE.findall(m["detail"])
            )
            pending_eval = None

    out_dir.mkdir(parents=True, exist_ok=True)
    if metrics_rows:
        with open(out_dir / f"{run}_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
            w.writeheader()
            w.writerows(metrics_rows)
    if eval_rows:
        with open(out_dir / f"{run}_evals.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(eval_rows[0].keys()))
            w.writeheader()
            w.writerows(eval_rows)
    print(f"{run}: {len(metrics_rows)} metrics, {len(eval_rows)} evals")


# Séries de history → colonnes attendues par plot_runs.fig_internals().
# `p5`/`p95` y sont nommées sans le préfixe `return_`, d'où le renommage.
SUMMARY_METRIC_KEYS = {
    "iter": "iter", "loss_wm": "loss_wm", "loss_recon": "loss_recon",
    "loss_kl": "loss_kl", "loss_reward": "loss_reward", "loss_continue": "loss_continue",
    "loss_actor": "loss_actor", "loss_critic": "loss_critic", "loss_pg": "loss_pg",
    "entropy": "entropy", "returns_mean": "returns_mean", "values_mean": "values_mean",
    "return_scale": "return_scale", "return_p5": "p5", "return_p95": "p95",
    "ips": "ips", "env_reward_per_step": "env_reward_per_step",
    # Diagnostics ajoutés après v53 : absents des runs plus anciens, d'où le .get()
    "rew_pred_ach": "rew_pred_ach", "rew_pred_zero": "rew_pred_zero",
    "img_rew_max": "img_rew_max", "img_rew_frac_hi": "img_rew_frac_hi",
}


def parse_summary(path: Path, out_dir: Path):
    """Convertit un summary JSON en <run>_metrics.csv + <run>_evals.csv."""
    d = json.loads(path.read_text())
    run = d.get("run_name") or path.stem.replace("dreamer_crafter_jax_summary_", "")
    h = d.get("history", {})
    steps_per_iter = int(d.get("env_steps_per_iter", 32))   # ne PAS hardcoder : varie avec n_envs

    n = len(h.get("iter", []))
    metrics_rows = []
    for i in range(n):
        row = {"env_steps": h["iter"][i] * steps_per_iter}
        for src, dst in SUMMARY_METRIC_KEYS.items():
            seq = h.get(src)
            row[dst] = seq[i] if seq is not None and i < len(seq) else ""
        metrics_rows.append(row)

    eval_rows = []
    for i, it in enumerate(h.get("eval_iter", [])):
        detail = h.get("eval_detail", [{}])[i] if i < len(h.get("eval_detail", [])) else {}
        # plot_runs.fig_spectrum attend "nom:pourcentage;..." et des POURCENTAGES,
        # alors que le JSON stocke des fractions [0,1].
        detail_str = ";".join(f"{k}:{100.0 * v:.0f}" for k, v in sorted(detail.items()))
        get = lambda key, default="": (h[key][i] if key in h and i < len(h[key]) else default)
        eval_rows.append({
            "iter": it,
            "env_steps": it * steps_per_iter,
            "score": get("eval_score"),
            "length": get("eval_length"),
            "achievements": get("eval_achievements"),
            "sample": get("eval_sample"),
            "crafter_score": get("eval_crafter_score"),
            "n_unlocked": len(detail),
            "unlocked_detail": detail_str,
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    for rows, suffix in ((metrics_rows, "metrics"), (eval_rows, "evals")):
        if not rows:
            continue
        with open(out_dir / f"{run}_{suffix}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"{run}: {len(metrics_rows)} metrics, {len(eval_rows)} evals  (summary JSON)")


def main():
    out_dir = HERE / "data"
    args = sys.argv[1:]

    if "--summaries" in args:
        found = sorted(KAGGLE.glob("*/runs/dreamer_crafter_jax_summary_*.json"))
        if not found:
            print(f"aucun summary JSON sous {KAGGLE}/*/runs/")
        for path in found:
            parse_summary(path, out_dir)
        return

    targets = [Path(p) for p in args] or sorted((HERE / "raw_logs").glob("*.log"))
    for path in targets:
        if path.suffix == ".json":
            parse_summary(path, out_dir)
        else:
            parse_log(path, out_dir)


if __name__ == "__main__":
    main()

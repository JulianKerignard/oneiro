"""
Parse les logs stdout des runs Oneiro en CSV exploitables (figures du paper).

Entrée  : experiments/raw_logs/<run>.log (stdout brut d'un job)
Sorties : experiments/data/<run>_metrics.csv  (1 ligne / LOG_INTERVAL=50 iter)
          experiments/data/<run>_evals.csv    (1 ligne / EVAL, avec le détail par achievement)

Usage :
    python experiments/parse_logs.py                     # parse tout raw_logs/
    python experiments/parse_logs.py raw_logs/v21*.log   # sélection
"""

import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent

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


def main():
    targets = [Path(p) for p in sys.argv[1:]] or sorted((HERE / "raw_logs").glob("*.log"))
    out_dir = HERE / "data"
    for path in targets:
        parse_log(path, out_dir)


if __name__ == "__main__":
    main()

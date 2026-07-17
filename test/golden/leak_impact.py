"""Leak-impact A/B study for KI-1.

Measures how much the MVP/VO cross-run recovery-state leak (resopairs /
_intr_init_vel surviving between runs that share a process) moves the aggregate
IPR, on small exp3/exp4/exp5-style batches sized for a 4-core box.

The fix state is controlled *externally* by the orchestrator (fix present in the
source = "fixed"; the same lines git-stashed = "leaky"); this script only tags its
output by the ``tag`` arg and does not know which mode it is in.

Caveat: the leak's magnitude scales with reps-per-worker (n_runs / n_jobs). The
published runs used ~10 reps/worker (n_runs=1000, n_jobs=100 -> ~90% of reps
contaminated); a small n_runs/n_jobs=4 batch has fewer, so the measured delta is
a rough, likely-conservative estimate of the published condition.

    python test/golden/leak_impact.py fixed     # run, tag output "fixed"
    python test/golden/leak_impact.py leaky      # run, tag output "leaky"
    python test/golden/leak_impact.py --compare  # print fixed-vs-leaky delta table
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np
from joblib import Parallel, delayed

from experiments.config import (
    CONFIG_PATH, ASAS_MARH, LOOKAHEAD, RECEPTION_PROB,
    KTS_TO_MS, SPEED_MIN_KTS, SPEED_MAX_KTS, SPEED_HOMOGEN_KTS,
    VEL_CI95, DEFAULT_GAMMA, BASE_SEED,
)
from sim.pairwise_stochastic.get_ipr_stochastic_env import get_ipr_stochastic_env
from sim.pairwise_stochastic.run_multiple_jobs import NB_PAIR

N_RUNS = int(os.environ.get("LEAK_N_RUNS", "12"))
N_JOBS = int(os.environ.get("LEAK_N_JOBS", "4"))

# Pre-drawn scenario parameters (shared across both modes and all conditions so
# the fixed-vs-leaky comparison is apples-to-apples on identical seeds).
_rng = np.random.default_rng(BASE_SEED)
DPSI_PP = _rng.uniform(0.0, 360.0, size=(N_RUNS, NB_PAIR))          # per-pair (exp3/exp5)
SPD_OWN = _rng.uniform(SPEED_MIN_KTS, SPEED_MAX_KTS, size=(N_RUNS, NB_PAIR))
SPD_INT = _rng.uniform(SPEED_MIN_KTS, SPEED_MAX_KTS, size=(N_RUNS, NB_PAIR))
DPSI_RUN = _rng.uniform(0.0, 360.0, size=N_RUNS)                    # per-run scalar (exp4)
SPD_HOM = SPEED_HOMOGEN_KTS * KTS_TO_MS

# (name, kind, recovery, actual_ci, assumed_ci)
CONDS = [
    ("exp3_normal_prob",  "hetero", "Probabilistic FTR", 10, None),
    ("exp3_normal_ftr",   "hetero", "FTR",               10, None),
    ("exp4_normal_ftr",   "homo",   "FTR",               10, None),
    ("exp5_mismatch_prob", "hetero", "Probabilistic FTR", 15, 10),
]


def _run(rep, kind, recovery, ci, assumed):
    if kind == "hetero":       # exp3 / exp5: per-pair angle + heterogeneous speeds
        extra = dict(dpsi=DPSI_PP[rep],
                     init_speed_ownship=SPD_OWN[rep] * KTS_TO_MS,
                     init_speed_intruder=SPD_INT[rep] * KTS_TO_MS)
    else:                      # exp4: per-run angle + homogeneous speed
        extra = dict(dpsi=float(DPSI_RUN[rep]),
                     init_speed_ownship=SPD_HOM, init_speed_intruder=SPD_HOM)
    _da, ipr, _t, _n = get_ipr_stochastic_env(
        asas_marh=ASAS_MARH, confidence_interval=ci, confidence_interval_velo=VEL_CI95,
        reception_prob=RECEPTION_PROB, lookahead_time=LOOKAHEAD, seed=BASE_SEED + rep,
        config_path=CONFIG_PATH, threshold_probability=DEFAULT_GAMMA,
        recovery_model=recovery, pos_dist=None, latency_s=0.0,
        assumed_confidence_interval=assumed, **extra,
    )
    return float(ipr)


def _overall(iprs):
    n_los = np.sum((1.0 - np.array(iprs)) * NB_PAIR)
    return float(1.0 - n_los / (len(iprs) * NB_PAIR))


def run(tag):
    out = {"n_runs": N_RUNS, "n_jobs": N_JOBS, "conditions": {}}
    print(f"[{tag}]  n_runs={N_RUNS}  n_jobs={N_JOBS}", flush=True)
    for name, kind, rec, ci, assumed in CONDS:
        iprs = Parallel(n_jobs=N_JOBS)(
            delayed(_run)(r, kind, rec, ci, assumed) for r in range(N_RUNS)
        )
        out["conditions"][name] = {"overall_ipr": _overall(iprs), "per_run": iprs}
        print(f"  {name:22s} overall_ipr={out['conditions'][name]['overall_ipr']:.5f}", flush=True)
    path = os.path.join(ROOT, "test", "golden", f"leak_impact_{tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}", flush=True)


def compare():
    d = {}
    for tag in ("fixed", "leaky"):
        with open(os.path.join(ROOT, "test", "golden", f"leak_impact_{tag}.json")) as f:
            d[tag] = json.load(f)
    print(f"\n{'condition':22s} {'fixed':>10} {'leaky':>10} {'delta':>11}")
    print("-" * 56)
    for name in d["fixed"]["conditions"]:
        f_ = d["fixed"]["conditions"][name]["overall_ipr"]
        l_ = d["leaky"]["conditions"][name]["overall_ipr"]
        print(f"{name:22s} {f_:>10.5f} {l_:>10.5f} {l_ - f_:>+11.5f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", nargs="?", default="fixed")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    compare() if a.compare else run(a.tag)

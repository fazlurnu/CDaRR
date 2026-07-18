"""L3 aggregate-pipeline baseline: capture + verify, around the Phase 4 switchover.

Captures run_multiple_jobs (refactor_fp.md's test_equiv_run_multiple_jobs.py
spec) and per-noise-model exp3-style cells (test_equiv_experiment_cells.py
spec) from the CURRENT sim.pairwise_stochastic.get_ipr_stochastic_env module
-- before the switchover, this is the legacy sim_models-class pipeline; after,
it's the functional loop.py pipeline aliased in by the same import path. So
"capture" then "verify" around the switchover edit is the direct evidence
that the switchover changed zero observable output, one level above the
per-call golden/L2 checks.

Usage (from the worktree root, under the cdarr env)::

    python test/golden/l3_baseline.py capture   # -> test/golden/l3_baseline.json
    python test/golden/l3_baseline.py verify     # re-run, compare, exit!=0 on drift
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from sim.pairwise_stochastic.run_multiple_jobs import run_multiple_jobs  # noqa: E402
from sim.pairwise_stochastic.get_ipr_stochastic_env import get_ipr_stochastic_env  # noqa: E402
from sim_models.noise_distributions import (  # noqa: E402
    make_mixture_gaussian, make_anisotropic_gaussian,
)

OUT_PATH = os.path.join(ROOT, "test", "golden", "l3_baseline.json")

CFG = "sim_configs/test_config_2x2.json"

# --- test_equiv_run_multiple_jobs.py spec: one baseline + one probabilistic config ---
RMJ_CASES = {
    "rmj_ftr": dict(
        n_runs=6, n_jobs=3, base_seed=42, asas_marh=1.05,
        confidence_interval=10, confidence_interval_velo=1, reception_prob=1.0,
        lookahead_time=120, dpsi=90, config_path=CFG, recovery_model="FTR",
    ),
    "rmj_probabilistic": dict(
        n_runs=6, n_jobs=3, base_seed=42, asas_marh=1.05,
        confidence_interval=10, confidence_interval_velo=1, reception_prob=1.0,
        lookahead_time=120, dpsi=2, config_path=CFG, recovery_model="Probabilistic FTR",
        threshold_probability=0.999,
    ),
}

# --- test_equiv_experiment_cells.py spec: one _one() call per exp3 noise model + exp5 mismatch ---
NOISE_MODELS = {
    "normal": None,
    "mixture": make_mixture_gaussian(3.0, 0.10),
    "anisotropic": make_anisotropic_gaussian(9.0),
}
BASE_CELL = dict(
    asas_marh=1.05, confidence_interval=10, confidence_interval_velo=1,
    reception_prob=1.0, lookahead_time=120, dpsi=30.0, seed=42,
    config_path=CFG, threshold_probability=0.999, recovery_model="Probabilistic FTR",
)


def _cell(pos_dist=None, **overrides):
    kw = dict(BASE_CELL)
    kw["pos_dist"] = pos_dist
    kw.update(overrides)
    da, ipr, t, n = get_ipr_stochastic_env(**kw)
    return {"ipr": ipr, "min_cpa": np.min(da, axis=0).tolist(), "sim_timer": t, "n_active": n}


def _run_all():
    result = {"rmj": {}, "cells": {}}

    for name, kw in RMJ_CASES.items():
        r = run_multiple_jobs(**kw)
        result["rmj"][name] = {
            "overall_ipr": r["overall_ipr"],
            "ipr": np.asarray(r["ipr"]).tolist(),
            "worst_cpa": np.asarray(r["worst_cpa"]).tolist(),
            "sim_timer": np.asarray(r["sim_timer"]).tolist(),
            "n_active_conflict": int(r["n_active_conflict"]),
            "worst_cpa_min": r["worst_cpa_min"],
        }
        print(f"  rmj:{name:20s} overall_ipr={r['overall_ipr']:.4f}")

    for label, pos_dist in NOISE_MODELS.items():
        result["cells"][f"exp3_{label}"] = _cell(pos_dist=pos_dist)
        print(f"  cell:exp3_{label:12s} ipr={result['cells'][f'exp3_{label}']['ipr']:.4f}")

    result["cells"]["exp5_mismatch"] = _cell(
        pos_dist=None, confidence_interval=15, assumed_confidence_interval=10)
    print(f"  cell:exp5_mismatch     ipr={result['cells']['exp5_mismatch']['ipr']:.4f}")

    return result


def capture():
    result = _run_all()
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"captured -> {OUT_PATH}")


def verify():
    with open(OUT_PATH) as f:
        ref = json.load(f)
    result = _run_all()

    mism = []

    def _cmp(path, a, b):
        if isinstance(a, dict):
            for k in a:
                _cmp(f"{path}.{k}", a[k], b[k])
        elif isinstance(a, list):
            if not np.array_equal(np.asarray(a), np.asarray(b)):
                mism.append(path)
        else:
            if a != b:
                mism.append(path)

    _cmp("root", ref, result)
    print(f"\n{'ALL MATCH' if not mism else 'MISMATCH: ' + ', '.join(mism)}")
    sys.exit(0 if not mism else 1)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "capture"
    {"capture": capture, "verify": verify}[mode]()

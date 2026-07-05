'''One-off search that identified PROBFAIL_SEED for regenerate_data.py.

At dpsi=2 the near-parallel prob_fails case needs a pair where Prob-FTR loses
separation AND all three strategies' CPA fall inside the 310 s plot window. Such
pairs are rare (Past-CPA chatters along the RPZ so its global CPA is usually
late), so this scans seeds 44..143 in parallel and reports every qualifying
(seed, pair). regenerate_data.py then hard-codes the chosen PROBFAIL_SEED so the
per-run regeneration stays fast; re-run this only if the scenario changes.

Requires the ``cdarr`` env and a CDaRR_FP checkout beside CDaRR_git.

Run::

    python scan_probfail.py            # writes scan_probfail.json
'''
import json
import os
import sys

import numpy as np
from joblib import Parallel, delayed

_HERE    = os.path.dirname(os.path.abspath(__file__))
_FP_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "CDaRR_FP"))
sys.path.insert(0, _FP_ROOT); os.chdir(_FP_ROOT)
from runners.stochastic_pairwise_hor_conflict import run_single  # noqa: E402

S = ("double_criteria", "probabilistic", "cpa")
RPZ, TMAX = 50.0, 310.0
SEEDS = list(range(44, 144))
OUT = os.path.join(_HERE, "scan_probfail.json")


def _kw(dpsi):
    return dict(pair_width=10, pair_height=10, rpz=RPZ, hpz=50.0, dtlookahead=120.0,
        init_speed_ownship=10.2889, init_speed_intruder=10.2889, aircraft_type="M600",
        dpsi=dpsi, pos_ci95=10.0, vel_ci95=1.0, reception_prob=1.0, start_lat=52.0,
        start_lon=4.0, delta_lat_lon=0.1, tmax=480.0, done_timeout=30.0, resofach=1.05,
        recovery_resofach=1.05, simdt_factor=4, record_history=True,
        prob_threshold=0.999, spawn_margin=1.5)


def _tcpa(res, p):
    return float(res.t_arr[int(np.argmin(res.dist_arr[:, p]))])


def _scan_seed(seed):
    runs = {l: run_single(crr=l, seed=seed, **_kw(2)) for l in S}
    hits = []
    for p in range(runs["cpa"].env.nb_pair):
        if runs["probabilistic"].min_dist[p] < RPZ:
            tt = {s: _tcpa(runs[s], p) for s in S}
            if all(tt[s] < TMAX for s in S):
                hits.append({"seed": seed, "pair": p,
                             "min": {s: round(float(runs[s].min_dist[p]), 1) for s in S},
                             "t_cpa": {s: round(tt[s]) for s in S},
                             "cpa_spread": round(max(tt.values()) - min(tt.values()))})
    return hits


if __name__ == "__main__":
    results = Parallel(n_jobs=8, verbose=5)(delayed(_scan_seed)(s) for s in SEEDS)
    flat = sorted((h for hs in results for h in hs), key=lambda h: (h["cpa_spread"], h["seed"]))
    with open(OUT, "w") as f:
        json.dump({"n_seeds": len(SEEDS), "hits": flat}, f, indent=2)
    print(f"DONE: {len(flat)} hits across {len(SEEDS)} seeds → {OUT}")

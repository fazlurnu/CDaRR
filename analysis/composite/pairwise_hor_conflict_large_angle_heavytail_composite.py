'''Large-angle (DPSI = 90 deg) heavy-tail composite CD&R figure — CDaRR_git.

Heavy-tail counterpart of the ``large_angle`` composite: the self-measurement
*position* noise is the heavy-tailed two-component Gaussian mixture
(``make_mixture_gaussian``, TAIL_RATIO=3.0, TAIL_WEIGHT=0.10 — exp3's
``heavy_tail`` model) rather than the plain Gaussian. It emits the ``ftr_wins``
showcase: a wide tail draw briefly collapses the probabilistic projected-CPA
estimate, so Prob-FTR *disengages the resolution manoeuvre early* and loses
separation, while the deterministic FTR (double-criteria) method holds its
manoeuvre and stays clear.

Selection targets the most convincing failure — among pairs where Prob-FTR loses
separation, the deterministic FTR stays safe, and all three strategies' CPA fall
inside the plot window — pick the deepest probabilistic loss (tie-break tighter
CPA cluster for legibility).

Self-contained (like the almost-parallel heavy-tail script): runs the CDaRR_FP
stochastic pairwise sim under heavy-tail noise, scans seeds, caches the chosen
run, and plots it. Re-runs replot straight from the cache unless ``--regen``.

Run::

    conda activate cdarr
    python pairwise_hor_conflict_large_angle_heavytail_composite.py [--latex] [--regen]
'''
import json
import math
import os
import pickle
import sys

import numpy as np
from joblib import Parallel, delayed

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from regenerate_data import (  # noqa: E402
    _simulate, _pick, _cpas, _t_cpas, _entry, _all_cpa_before, _cpa_spread,
    RPZ, DETAIL_T_MAX, STRATEGIES, DATA_DIR,
)
from composite_plotutils import (  # noqa: E402
    set_latex_style, plot_pair_cdr_composite, _R_EARTH, _FILE_PREFIX,
)
from sim_models.cns.distributions import make_mixture_gaussian  # noqa: E402

set_latex_style("--latex" in sys.argv)

FIGURE_DIR = os.path.join(_HERE, "figures")
YLIM_STEP  = 50.0
DPSI       = 90.0

# Heavy-tail position-noise model — matches exp3's ``heavy_tail`` condition.
TAIL_RATIO  = 3.0
TAIL_WEIGHT = 0.10
POS_DIST    = make_mixture_gaussian(TAIL_RATIO, TAIL_WEIGHT)

# Seeds scanned for a legible ftr_wins pair; the winner is cached (below) so the
# choice is reproducible and re-runs skip the scan.
SEEDS = list(range(44, 94))

RUNS_PATH = os.path.join(DATA_DIR, "large_angle_heavytail_ftrwins_runs.pkl")
PICK_PATH = os.path.join(DATA_DIR, "large_angle_heavytail_pick.json")
STEM      = f"{_FILE_PREFIX}_large_angle_heavytail_pair{{pair:03d}}_cdr_composite"


def _nice_ceil(value, step=YLIM_STEP):
    return math.ceil(value / step) * step


def _panel_max(res_by_label, pair, t_max, *attrs):
    peak = 0.0
    for res in res_by_label.values():
        mask = res.t_arr <= t_max
        for attr in attrs:
            peak = max(peak, np.nanmax(getattr(res, attr)[mask, pair]))
    return peak


def _pair_trajectory_bbox(res_by_label, pair, margin=100.0):
    xs, ys = [], []
    for res in res_by_label.values():
        env = res.env
        lat0 = float(res.lat_arr[0, env.ownship_idx[pair]])
        lon0 = float(res.lon_arr[0, env.ownship_idx[pair]])
        lat0r = np.deg2rad(lat0)
        for i in (env.ownship_idx[pair], env.intruder_idx[pair]):
            xs.append(np.deg2rad(res.lon_arr[:, i] - lon0) * _R_EARTH * np.cos(lat0r))
            ys.append(np.deg2rad(res.lat_arr[:, i] - lat0) * _R_EARTH)
    xs = np.concatenate(xs); ys = np.concatenate(ys)
    xlim = (float(np.nanmin(xs)) - margin, float(np.nanmax(xs)) + margin)
    ylim = (float(np.nanmin(ys)) - margin, float(np.nanmax(ys)) + margin)
    return xlim, ylim


def _ftr_wins(runs, p):
    '''Prob-FTR loses separation, deterministic FTR stays safe, and every
    strategy's CPA falls inside the plot window (legible single encounter).'''
    return (runs["probabilistic"].min_dist[p] < RPZ
            and runs["double_criteria"].min_dist[p] >= RPZ
            and _all_cpa_before(runs, p, DETAIL_T_MAX))


def _scan_seed(seed):
    '''Return (seed, pair, prob_min, spread) for the deepest ftr_wins probabilistic
    loss at ``seed`` under heavy-tail noise, or ``None`` if none qualifies.'''
    runs = _simulate(dpsi=DPSI, seed=seed, pos_dist=POS_DIST)
    pair = _pick(runs, _ftr_wins,
                 lambda r, p: (r["probabilistic"].min_dist[p], _cpa_spread(r, p)))
    if pair is None:
        return None
    return (seed, pair, round(float(runs["probabilistic"].min_dist[pair]), 1),
            round(_cpa_spread(runs, pair), 1))


def _find_ftr_wins():
    '''Scan SEEDS in parallel; return the global-best (seed, pair) by deepest
    probabilistic loss then tighter CPA cluster, or ``None``.'''
    print(f"scanning seeds {SEEDS[0]}..{SEEDS[-1]} for a heavy-tail ftr_wins pair "
          f"(dpsi={DPSI:g}) …")
    hits = [h for h in Parallel(n_jobs=8, verbose=5)(
        delayed(_scan_seed)(s) for s in SEEDS) if h is not None]
    if not hits:
        return None
    seed, pair, prob_min, spread = min(hits, key=lambda h: (h[2], h[3]))
    print(f"  {len(hits)} qualifying seed(s); best → seed {seed}, pair {pair} "
          f"(Prob CPA {prob_min} m, CPA spread {spread}s)")
    return seed, pair


def _load_or_build(regen):
    '''Return (runs, pair). Uses the cache unless ``regen`` or it is missing.'''
    if not regen and os.path.exists(RUNS_PATH) and os.path.exists(PICK_PATH):
        with open(PICK_PATH) as f:
            pick = json.load(f)
        with open(RUNS_PATH, "rb") as f:
            runs = pickle.load(f)
        print(f"cache → seed {pick['seed']}, pair {pick['pair']}  ({RUNS_PATH})")
        return runs, pick["pair"]

    found = _find_ftr_wins()
    if found is None:
        sys.exit("no heavy-tail ftr_wins pair found in the scanned seed range.")
    seed, pair = found
    runs = _simulate(dpsi=DPSI, seed=seed, pos_dist=POS_DIST)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(RUNS_PATH, "wb") as f:
        pickle.dump(runs, f)
    entry = _entry(runs, pair)
    entry["seed"] = seed
    with open(PICK_PATH, "w") as f:
        json.dump({"dpsi": DPSI, "tail_ratio": TAIL_RATIO, "tail_weight": TAIL_WEIGHT,
                   **entry}, f, indent=2)
    print(f"  cpa={_cpas(runs, pair)}  t_cpa={_t_cpas(runs, pair)}")
    return runs, pair


def main():
    runs, pair = _load_or_build("--regen" in sys.argv)

    dist_max = _nice_ceil(_panel_max(runs, pair, DETAIL_T_MAX, "dist_arr"))
    dcpa_max = 450.0
    traj_xlim, traj_ylim = _pair_trajectory_bbox(runs, pair)

    path = plot_pair_cdr_composite(runs, FIGURE_DIR, pair, t_max=DETAIL_T_MAX,
                                   dist_max=dist_max, dcpa_max=dcpa_max,
                                   traj_xlim=traj_xlim, traj_ylim=traj_ylim,
                                   stem=STEM.format(pair=pair))
    print(f"→ {path}")


# Guarded because the seed scan uses joblib; on macOS (spawn) worker processes
# re-import this module, and an unguarded scan would recursively spawn.
if __name__ == "__main__":
    main()

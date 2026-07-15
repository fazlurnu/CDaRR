'''Near-parallel (DPSI = 2 deg) heavy-tail composite CD&R figure — CDaRR_git.

Companion to ``pairwise_hor_conflict_almost_parallel_composite.py``, but the
self-measurement *position* noise is drawn from the heavy-tailed two-component
Gaussian mixture (``make_mixture_gaussian``, TAIL_RATIO=3.0, TAIL_WEIGHT=0.10 —
the ``heavy_tail`` / ``mixture`` model swept in ``experiments/exp3``) instead of
the plain Gaussian. It emits the ``prob_fails`` composite: a rare wide tail draw
inflates the perceived-CPA uncertainty just enough that the probabilistic method
disengages too early (or never engages hard enough) and loses separation on a
near-parallel pair — the distinctive failure heavy tails produce.

Unlike the Gaussian composite (pure-plotting from cached pickles written by
``regenerate_data.py``), this script is self-contained: it runs the CDaRR_FP
stochastic pairwise sim under heavy-tail noise, scans seeds for a legible
``prob_fails`` pair (Prob-FTR LoS AND all three strategies' CPA inside the plot
window), caches the chosen run, and plots it. Re-runs replot straight from the
cache unless ``--regen`` is given.

The sim harness (``_simulate`` and the selection helpers) is reused from
``regenerate_data.py`` so the geometry / RPZ / gamma stay identical to the
Gaussian figure; only ``pos_dist`` differs.

Run::

    conda activate cdarr
    python pairwise_hor_conflict_almost_parallel_heavytail_composite.py [--latex] [--regen]
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

# regenerate_data pulls in CDaRR_FP (sys.path + chdir) and exposes the sim
# harness + selection helpers; importing it no longer triggers a full run.
from regenerate_data import (  # noqa: E402
    _simulate, _pick, _cpas, _t_cpas, _entry, _all_cpa_before, _cpa_spread,
    _cpa_time, RPZ, DETAIL_T_MAX, STRATEGIES, DATA_DIR,
)
from composite_plotutils import (  # noqa: E402
    set_latex_style, plot_pair_cdr_composite, _R_EARTH, _FILE_PREFIX,
)
from sim_models.cns.distributions import make_mixture_gaussian  # noqa: E402

set_latex_style("--latex" in sys.argv)

FIGURE_DIR = os.path.join(_HERE, "figures")
YLIM_STEP  = 50.0

# Heavy-tail position-noise model — matches exp3's ``heavy_tail`` condition.
TAIL_RATIO  = 3.0
TAIL_WEIGHT = 0.10
POS_DIST    = make_mixture_gaussian(TAIL_RATIO, TAIL_WEIGHT)

# Seeds scanned for a legible prob_fails pair. The winner is cached (below) so
# the choice is reproducible and re-runs skip the scan; widen this only if the
# scenario changes and no pair qualifies.
SEEDS = list(range(44, 84))

RUNS_PATH = os.path.join(DATA_DIR, "almost_parallel_heavytail_probfail_runs.pkl")
PICK_PATH = os.path.join(DATA_DIR, "almost_parallel_heavytail_pick.json")
STEM      = f"{_FILE_PREFIX}_heavytail_pair{{pair:03d}}_cdr_composite"


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
    ys = []
    for res in res_by_label.values():
        env = res.env
        lat0 = float(res.lat_arr[0, env.ownship_idx[pair]])
        for i in (env.ownship_idx[pair], env.intruder_idx[pair]):
            ys.append(np.deg2rad(res.lat_arr[:, i] - lat0) * _R_EARTH)
    ys = np.concatenate(ys)
    return (float(np.nanmin(ys)) - margin, float(np.nanmax(ys)) + margin)


def _prob_fails(runs, p):
    '''Prob-FTR loses separation and every strategy's CPA is inside the window.'''
    return runs["probabilistic"].min_dist[p] < RPZ and _all_cpa_before(runs, p, DETAIL_T_MAX)


def _scan_seed(seed):
    '''Return (seed, best_pair, spread, prob_min) for the most legible prob_fails
    pair at ``seed`` under heavy-tail noise, or ``None`` if none qualifies.'''
    runs = _simulate(dpsi=2, seed=seed, pos_dist=POS_DIST)
    pair = _pick(runs, _prob_fails,
                 lambda r, p: (_cpa_spread(r, p), r["probabilistic"].min_dist[p]))
    if pair is None:
        return None
    return (seed, pair, round(_cpa_spread(runs, pair), 1),
            round(float(runs["probabilistic"].min_dist[pair]), 1))


def _find_prob_fails():
    '''Scan SEEDS in parallel; return the global-best (seed, pair) by tightest CPA
    cluster then deepest probabilistic loss, or ``None``.'''
    print(f"scanning seeds {SEEDS[0]}..{SEEDS[-1]} for a heavy-tail prob_fails pair …")
    hits = [h for h in Parallel(n_jobs=8, verbose=5)(
        delayed(_scan_seed)(s) for s in SEEDS) if h is not None]
    if not hits:
        return None
    seed, pair, spread, prob_min = min(hits, key=lambda h: (h[2], h[3]))
    print(f"  {len(hits)} qualifying seed(s); best → seed {seed}, pair {pair} "
          f"(CPA spread {spread}s, Prob CPA {prob_min} m)")
    return seed, pair


def _load_or_build(regen):
    '''Return (runs, pair). Uses the cache unless ``regen`` or it is missing.'''
    if not regen and os.path.exists(RUNS_PATH) and os.path.exists(PICK_PATH):
        with open(PICK_PATH) as f:
            pick = json.load(f)
        with open(RUNS_PATH, "rb") as f:
            runs = pickle.load(f)
        print(f"cache → seed {pick['seed']}, pair {pick['pair']}  "
              f"({RUNS_PATH})")
        return runs, pick["pair"]

    found = _find_prob_fails()
    if found is None:
        sys.exit("no heavy-tail prob_fails pair found in the scanned seed range.")
    seed, pair = found
    runs = _simulate(dpsi=2, seed=seed, pos_dist=POS_DIST)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(RUNS_PATH, "wb") as f:
        pickle.dump(runs, f)
    entry = _entry(runs, pair)
    entry["seed"] = seed
    with open(PICK_PATH, "w") as f:
        json.dump({"tail_ratio": TAIL_RATIO, "tail_weight": TAIL_WEIGHT, **entry}, f, indent=2)
    print(f"  cpa={_cpas(runs, pair)}  t_cpa={_t_cpas(runs, pair)}")
    return runs, pair


def main():
    runs, pair = _load_or_build("--regen" in sys.argv)

    dist_max  = _nice_ceil(_panel_max(runs, pair, DETAIL_T_MAX, "dist_arr"))
    dcpa_max  = 450.0   # fixed projected-CPA y-limit, matching the Gaussian figure
    traj_ylim = _pair_trajectory_bbox(runs, pair)

    path = plot_pair_cdr_composite(runs, FIGURE_DIR, pair, t_max=DETAIL_T_MAX,
                                   dist_max=dist_max, dcpa_max=dcpa_max,
                                   traj_xlim=(-1000.0, 1000.0), traj_ylim=traj_ylim,
                                   stem=STEM.format(pair=pair))
    print(f"→ {path}")


# Execution is guarded because the seed scan uses joblib; on macOS (spawn) the
# worker processes re-import this module, and an unguarded scan would recursively
# spawn. Keep all runtime work inside main().
if __name__ == "__main__":
    main()

'''Near-parallel (DPSI = 2 deg) composite CD&R figures — CDaRR_git.

Pure-plotting: reads the sanitised runs in ``data/almost_parallel_runs.pkl`` and
the pairs chosen by ``regenerate_data.py`` in ``data/selected_pairs.json`` (run
that first). Emits one composite figure per near-parallel case:
  • prob_wins  — Past-CPA & FTR LoS, Prob-FTR safe
  • prob_fails — Prob-FTR loses separation
strategy = column, rows = trajectory / actual-distance / projected-CPA.
Figures → figures/.

Run::

    python pairwise_hor_conflict_almost_parallel_composite.py [--latex]
'''
import json
import math
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from composite_plotutils import set_latex_style, plot_pair_cdr_composite, _R_EARTH

set_latex_style("--latex" in sys.argv)

_HERE        = os.path.dirname(os.path.abspath(__file__))
PAIRS_PATH   = os.path.join(_HERE, "data", "selected_pairs.json")
FIGURE_DIR   = os.path.join(_HERE, "figures")
DETAIL_T_MAX = 310.0
YLIM_STEP    = 50.0
# Each case names the pickle it is drawn from (prob_fails comes from a different
# seed than prob_wins — see regenerate_data.py).
CASE_DATA    = {"prob_wins":  os.path.join(_HERE, "data", "almost_parallel_runs.pkl"),
                "prob_fails": os.path.join(_HERE, "data", "almost_parallel_probfail_runs.pkl")}


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


with open(PAIRS_PATH) as f:
    selected = json.load(f)["cases"]["almost_parallel"]

for case, data_path in CASE_DATA.items():
    entry = selected.get(case)
    if entry is None:
        print(f"  {case}: no pair satisfied the criteria — skipped")
        continue
    with open(data_path, "rb") as f:
        res_single = pickle.load(f)
    pair = entry["pair"]
    print(f"  {case:<11} → pair {pair:03d} (seed {entry.get('seed', 'AP')})  {entry['cpa']}")

    dist_max = _nice_ceil(_panel_max(res_single, pair, DETAIL_T_MAX, "dist_arr"))
    dcpa_max = 450.0   # fixed projected-CPA y-limit across all columns
    traj_ylim = _pair_trajectory_bbox(res_single, pair)

    path = plot_pair_cdr_composite(res_single, FIGURE_DIR, pair, t_max=DETAIL_T_MAX,
                                   dist_max=dist_max, dcpa_max=dcpa_max,
                                   traj_xlim=(-1000.0, 1000.0), traj_ylim=traj_ylim)
    print(f"    → {path}")

'''Large-angle (DPSI = 90 deg) composite CD&R figure — all-strategies-safe case.

Pure-plotting: reads the sanitised runs in ``data/large_angle_runs.pkl`` and the
pair chosen by ``regenerate_data.py``/manual selection in
``data/selected_pairs.json`` (case ``all_safe_prob_tight``). Unlike ``ftr_wins``,
here every strategy keeps separation, but Prob-FTR's CPA distance sits much
closer to the RPZ boundary than Past-CPA's — illustrating that Prob-FTR trades
margin for tighter, still-safe resolutions. strategy = column, rows =
trajectory / actual-distance / projected-CPA. Figures → figures/.

Run::

    python pairwise_hor_conflict_large_angle_all_safe_composite.py [--latex]
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
DATA_PATH    = os.path.join(_HERE, "data", "large_angle_runs.pkl")
PAIRS_PATH   = os.path.join(_HERE, "data", "selected_pairs.json")
FIGURE_DIR   = os.path.join(_HERE, "figures")
DETAIL_T_MAX = 310.0
YLIM_STEP    = 50.0


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
        i_own, i_int = env.ownship_idx[pair], env.intruder_idx[pair]
        lat0, lon0 = float(res.lat_arr[0, i_own]), float(res.lon_arr[0, i_own])
        lat0r = np.deg2rad(lat0)
        for i in (i_own, i_int):
            xs.append(np.deg2rad(res.lon_arr[:, i] - lon0) * _R_EARTH * np.cos(lat0r))
            ys.append(np.deg2rad(res.lat_arr[:, i] - lat0) * _R_EARTH)
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    return ((float(np.nanmin(xs)) - margin, float(np.nanmax(xs)) + margin),
            (float(np.nanmin(ys)) - margin, float(np.nanmax(ys)) + margin))


with open(DATA_PATH, "rb") as f:
    res_single = pickle.load(f)
with open(PAIRS_PATH) as f:
    entry = json.load(f)["cases"]["large_angle"]["all_safe_prob_tight"]

if entry is None:
    sys.exit("  all_safe_prob_tight: no pair satisfied the criteria "
              "(Past-CPA, Prob-FTR, FTR all safe; Prob-FTR < Past-CPA).")

pair = entry["pair"]
print(f"  all_safe_prob_tight → pair {pair:03d}  {entry['cpa']}")

dist_max = _nice_ceil(_panel_max(res_single, pair, DETAIL_T_MAX, "dist_arr"))
dcpa_max = 450.0   # fixed projected-CPA y-limit across all columns
traj_xlim, traj_ylim = _pair_trajectory_bbox(res_single, pair)

path = plot_pair_cdr_composite(res_single, FIGURE_DIR, pair, t_max=DETAIL_T_MAX,
                               dist_max=dist_max, dcpa_max=dcpa_max,
                               traj_xlim=traj_xlim, traj_ylim=traj_ylim)
print(f"    → {path}")

'''Aggregate resolution-flag comparison — CDaRR_git port.

Pure-plotting port: for each scenario it pools the avoidance (resolution) flag of
all 100 pairs per strategy and overlays the mean resolution fraction over time —
Past-CPA vs FTR vs Probabilistic FTR on one axis. Reads the sanitised run
namespaces in ``data/*_runs.pkl`` (no BlueSky / FP stack needed).

Note: the aggregate pools the same single 100-pair run used for the composites
(the sanitised detail cache), so it is a 100-pair mean per strategy. Figures →
figures/.

Run::

    python avoidance_aggregate_compare.py            # default
    python avoidance_aggregate_compare.py --latex    # LaTeX-friendly
'''
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from composite_plotutils import set_latex_style, plot_avoidance_aggregate

set_latex_style("--latex" in sys.argv)

_HERE      = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(_HERE, "figures")
DETAIL_T_MAX = 310.0

SCENARIOS = {
    "almost_parallel": "almost_parallel_runs.pkl",
    "large_angle":     "large_angle_runs.pkl",
}

for scenario, fname in SCENARIOS.items():
    with open(os.path.join(_HERE, "data", fname), "rb") as f:
        res_single = pickle.load(f)
    # plot_avoidance_aggregate expects {label: [run, ...]}.
    results_by_label = {label: [res] for label, res in res_single.items()}
    name = f"stochastic_pairwise_hor_conflict_avoidance_aggregate_{scenario}.png"
    path = plot_avoidance_aggregate(results_by_label, FIGURE_DIR, name=name,
                                    select="all", t_max=DETAIL_T_MAX)
    print(f"  {scenario:<16} aggregate resolution flag  → {path}")

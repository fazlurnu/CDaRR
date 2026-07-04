'''Experiment 2 (CDARR_Claude) — Effect of confidence threshold gamma on IPR and
median dCPA.

Port of CDaRR_FP experiments/exp2-gamma.py onto this project's run_multiple_jobs
API. Sweeps crossing angle for all four uncertainty levels and five gamma values
{0.999, 0.99, 0.9, 0.75, 0.5} using probabilistic recovery, with deterministic
FTR as a benchmark.

Design
------
* Crossing angle : 2, 4, ..., 180 deg
* Uncertainty    : 4 levels {pos 3/10 m} x {vel 1/3 m/s}
* gamma          : {0.999, 0.99, 0.9, 0.75, 0.5}  (Probabilistic FTR)
* Benchmark      : FTR (double-criteria), gamma-independent
* Monte Carlo    : 100 runs x 100 pairs = 10 000 pairs per configuration

Results saved to experiments/results/exp2.npz.  Run directly::

    python experiments/exp2-gamma.py
'''
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from experiments.config import (
    CONFIG_PATH, ASAS_MARH, LOOKAHEAD, RECEPTION_PROB,
    CROSSING_ANGLES, UNCERTAINTY_LEVELS, GAMMA_VALUES,
    SWEEP_N_RUNS, N_JOBS, BASE_SEED, RESULTS_DIR,
)
from sim.pairwise_stochastic.run_multiple_jobs import run_multiple_jobs


def _sweep(recovery_model, ci, civ, gamma, tag=""):
    """Angle sweep for one (recovery, uncertainty, gamma) config.

    Returns (ipr[n_angles], median_dcpa[n_angles]).
    """
    n_angles = len(CROSSING_ANGLES)
    ipr = np.full(n_angles, np.nan)
    med = np.full(n_angles, np.nan)
    for ai, angle in enumerate(CROSSING_ANGLES):
        res = run_multiple_jobs(
            n_runs=SWEEP_N_RUNS, n_jobs=N_JOBS, base_seed=BASE_SEED,
            asas_marh=ASAS_MARH,
            lookahead_time=LOOKAHEAD,
            confidence_interval=ci,
            confidence_interval_velo=civ,
            reception_prob=RECEPTION_PROB,
            dpsi=float(angle),
            config_path=CONFIG_PATH,
            threshold_probability=gamma,
            recovery_model=recovery_model,
        )
        ipr[ai] = res["overall_ipr"]
        med[ai] = float(np.median(res["worst_cpa"]))
        print(f'  [{ai + 1:>2}/{n_angles}] {tag} dpsi={angle:>3}  '
              f'IPR={ipr[ai]:.4f}  medCPA={med[ai]:6.1f} m', flush=True)
    return ipr, med


# ── Storage ───────────────────────────────────────────────────────────────────
n_unc    = len(UNCERTAINTY_LEVELS)
n_gamma  = len(GAMMA_VALUES)
n_angles = len(CROSSING_ANGLES)

ipr_prob         = np.full((n_unc, n_gamma, n_angles), np.nan)
median_dcpa_prob = np.full((n_unc, n_gamma, n_angles), np.nan)
ipr_ftr          = np.full((n_unc, n_angles), np.nan)
median_dcpa_ftr  = np.full((n_unc, n_angles), np.nan)

# ── Main sweep ────────────────────────────────────────────────────────────────
for ui, unc in enumerate(UNCERTAINTY_LEVELS):

    # FTR benchmark for this uncertainty level (gamma-independent)
    print(f'Running FTR benchmark: {unc["label"]} ...', flush=True)
    ipr_ftr[ui], median_dcpa_ftr[ui] = _sweep(
        "FTR", unc["ci"], unc["civ"], None, tag="FTR   "
    )

    # Probabilistic sweep over gamma
    for gi, gamma in enumerate(GAMMA_VALUES):
        print(f'Running: {unc["label"]} / gamma={gamma} ...', flush=True)
        ipr_prob[ui, gi], median_dcpa_prob[ui, gi] = _sweep(
            "Probabilistic FTR", unc["ci"], unc["civ"], gamma,
            tag=f"g={gamma:<5}"
        )
        print(f'  done — mean IPR = {ipr_prob[ui, gi, :].mean():.4f}', flush=True)

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
out_path = os.path.join(RESULTS_DIR, 'exp2.npz')
np.savez(
    out_path,
    crossing_angles=np.array(CROSSING_ANGLES),
    uncertainty_labels=np.array([u["label"] for u in UNCERTAINTY_LEVELS]),
    uncertainty_titles=np.array([u["title"] for u in UNCERTAINTY_LEVELS]),
    gamma_values=np.array(GAMMA_VALUES),
    ipr_prob=ipr_prob,
    median_dcpa_prob=median_dcpa_prob,
    ipr_ftr=ipr_ftr,
    median_dcpa_ftr=median_dcpa_ftr,
    n_runs=SWEEP_N_RUNS,
)
print(f'\nSaved -> {out_path}')

# ── Summary ───────────────────────────────────────────────────────────────────
print(f'\n{"Uncertainty":<14} {"gamma":>6} {"Mean IPR":>9} {"Min IPR":>8}')
print('-' * 42)
for ui, unc in enumerate(UNCERTAINTY_LEVELS):
    for gi, gamma in enumerate(GAMMA_VALUES):
        row = ipr_prob[ui, gi, :]
        print(f'{unc["label"]:<14} {gamma:>6.3f} {row.mean():>9.4f} {row.min():>8.4f}')

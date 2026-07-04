'''Experiment 1 (CDARR_Claude) — IPR and median dCPA vs crossing angle.

Port of CDaRR_FP experiments/exp1-crossing-angle.py onto this project's
run_multiple_jobs API (the same sweep that compare_crr.py performs, but written
as a single tidy script with a self-contained .npz output).

Design
------
* Crossing angle : 2, 4, ..., 180 deg  (90 values)
* Uncertainty    : 4 levels {pos 3/10 m} x {vel 1/3 m/s}
* Recovery       : Past-CPA / FTR / Probabilistic FTR (gamma=0.999)
* Monte Carlo    : 100 runs x 100 pairs = 10 000 pairs per configuration

Dependent variables per configuration:
  - IPR         : intrusion prevention rate
  - median dCPA : median closest-point-of-approach distance across pairs [m]

Results saved to experiments/results/exp1.npz.  Run directly::

    python experiments/exp1-crossing-angle.py
'''
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from experiments.config import (
    CONFIG_PATH, ASAS_MARH, LOOKAHEAD, RECEPTION_PROB,
    CROSSING_ANGLES, UNCERTAINTY_LEVELS, SWEEP_RECOVERY_METHODS,
    SWEEP_N_RUNS, N_JOBS, BASE_SEED, DEFAULT_GAMMA, RESULTS_DIR,
)
from sim.pairwise_stochastic.run_multiple_jobs import run_multiple_jobs

# ── Storage ───────────────────────────────────────────────────────────────────
n_unc     = len(UNCERTAINTY_LEVELS)
n_methods = len(SWEEP_RECOVERY_METHODS)
n_angles  = len(CROSSING_ANGLES)

ipr_arr         = np.full((n_unc, n_methods, n_angles), np.nan)
median_dcpa_arr = np.full((n_unc, n_methods, n_angles), np.nan)

# ── Main sweep ────────────────────────────────────────────────────────────────
for ui, unc in enumerate(UNCERTAINTY_LEVELS):
    for mi, (short, disp, model) in enumerate(SWEEP_RECOVERY_METHODS):
        # Probabilistic recovery uses gamma; deterministic methods ignore it.
        gamma = DEFAULT_GAMMA if model == "Probabilistic FTR" else None
        print(f'Running: {unc["label"]} / {disp} ...', flush=True)

        for ai, angle in enumerate(CROSSING_ANGLES):
            res = run_multiple_jobs(
                n_runs=SWEEP_N_RUNS, n_jobs=N_JOBS, base_seed=BASE_SEED,
                asas_marh=ASAS_MARH,
                lookahead_time=LOOKAHEAD,
                confidence_interval=unc["ci"],
                confidence_interval_velo=unc["civ"],
                reception_prob=RECEPTION_PROB,
                dpsi=float(angle),
                config_path=CONFIG_PATH,
                threshold_probability=gamma,
                recovery_model=model,
            )
            med = float(np.median(res["worst_cpa"]))
            ipr_arr[ui, mi, ai]         = res["overall_ipr"]
            median_dcpa_arr[ui, mi, ai] = med
            print(f'  [{ai + 1:>2}/{n_angles}] dpsi={angle:>3}  '
                  f'IPR={res["overall_ipr"]:.4f}  medCPA={med:6.1f} m', flush=True)

        print(f'  done — mean IPR = {ipr_arr[ui, mi, :].mean():.4f}', flush=True)

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
out_path = os.path.join(RESULTS_DIR, 'exp1.npz')
np.savez(
    out_path,
    crossing_angles=np.array(CROSSING_ANGLES),
    uncertainty_labels=np.array([u["label"] for u in UNCERTAINTY_LEVELS]),
    uncertainty_titles=np.array([u["title"] for u in UNCERTAINTY_LEVELS]),
    methods=np.array([m[0] for m in SWEEP_RECOVERY_METHODS]),
    method_labels=np.array([m[1] for m in SWEEP_RECOVERY_METHODS]),
    recovery_models=np.array([m[2] for m in SWEEP_RECOVERY_METHODS]),
    gamma=DEFAULT_GAMMA,
    ipr=ipr_arr,
    median_dcpa=median_dcpa_arr,
    n_runs=SWEEP_N_RUNS,
)
print(f'\nSaved -> {out_path}')

# ── Summary ───────────────────────────────────────────────────────────────────
print(f'\n{"Uncertainty":<14} {"Method":<16} {"Mean IPR":>9} {"Min IPR":>8}')
print('-' * 50)
for ui, unc in enumerate(UNCERTAINTY_LEVELS):
    for mi, (short, disp, model) in enumerate(SWEEP_RECOVERY_METHODS):
        row = ipr_arr[ui, mi, :]
        print(f'{unc["label"]:<14} {disp:<16} {row.mean():>9.4f} {row.min():>8.4f}')

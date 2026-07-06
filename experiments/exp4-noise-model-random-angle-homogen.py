'''Experiment 4 (CDARR_Claude) — Noise model comparison, random crossing angle,
homogeneous speed.

Same design as ``exp3-noise-model-random-angle.py`` but every aircraft flies at
the same fixed speed (20 kts -> 10.2889 m/s) instead of an independent
Uniform(10, 30) kts draw -- isolating the noise-model / recovery-method effect
from any speed-heterogeneity confound.

Design
------
* Uncertainty    : pos_ci95 in {10 m, 3 m}, vel_ci95=1 m/s  (2 levels)
* Noise model    : Normal Gaussian / Latency bias / Heavy-tail Gaussian /
                   Anisotropic Gaussian (along-track stdev = 3x cross-track
                   stdev, i.e. variance ratio 9, radial CI95 still 10 m) /
                   Latency + Anisotropic / Heavy-tail + Anisotropic
                   (6 conditions)
* Recovery       : Probabilistic FTR / FTR  (2 conditions)
* Crossing angle : drawn i.i.d. from Uniform(0, 360 deg) per run (seeded, shared)
* Speed          : fixed at 20 kts for every aircraft (converted to m/s)
* Pairs per run  : 10 x 10 = 100
* Runs per model : 1 000    ->   100 x 1 000 = 100 000 pairs per condition

See ``exp3-noise-model-random-angle.py`` for the reviewer-response rationale
behind these six noise conditions (incl. the ADS-B MLAT-accuracy paper
citation) and the LATENCY_S=100 ms stress-test choice (shared via
experiments/config.py). Note: the latency bias (`latency`/`latency_aniso`) is
intruder-only -- applied only to the intruder's position as perceived by
ownship, never to ownship's own state -- via get_ipr_stochastic_env's
ownship_adsl/intruder_adsl split; see my-observation.md #13.

Results saved to experiments/results/exp4.npz.  Run directly::

    python experiments/exp4-noise-model-random-angle-homogen.py
'''
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from joblib import Parallel, delayed

from experiments.config import (
    CONFIG_PATH, ASAS_MARH, LOOKAHEAD, RECEPTION_PROB,
    KTS_TO_MS, SPEED_HOMOGEN_KTS,
    POS_CI95_LEVELS, VEL_CI95,
    LATENCY_S, TAIL_RATIO, TAIL_WEIGHT, ANISO_VAR_RATIO,
    N_RUNS, N_JOBS, BASE_SEED, DEFAULT_GAMMA,
    RECOVERY_METHODS, RECOVERY_LABELS, RESULTS_DIR,
)
from sim.pairwise_stochastic.get_ipr_stochastic_env import get_ipr_stochastic_env
from sim.pairwise_stochastic.run_multiple_jobs import NB_PAIR
from sim_models.noise_distributions import (
    make_mixture_gaussian, make_anisotropic_gaussian, make_anisotropic_mixture_gaussian,
)

# Fixed homogeneous speed (m/s)
SPEED_MS = SPEED_HOMOGEN_KTS * KTS_TO_MS

# ── Noise model definitions: (label, pos_dist, latency_s) ─────────────────────
NOISE_MODELS = [
    ('normal',           None,                                                                   0.0),
    ('latency',          None,                                                                   LATENCY_S),
    ('heavy_tail',       make_mixture_gaussian(TAIL_RATIO, TAIL_WEIGHT),                        0.0),
    ('anisotropic',      make_anisotropic_gaussian(ANISO_VAR_RATIO),                             0.0),
    ('latency_aniso',    make_anisotropic_gaussian(ANISO_VAR_RATIO),                             LATENCY_S),
    ('heavy_tail_aniso', make_anisotropic_mixture_gaussian(ANISO_VAR_RATIO, TAIL_RATIO, TAIL_WEIGHT), 0.0),
]
NOISE_LABELS = [m[0] for m in NOISE_MODELS]

# ── Pre-generate per-run angles (seeded, shared across conditions) ────────────
rng = np.random.default_rng(BASE_SEED)
dpsi_values = rng.uniform(0.0, 360.0, size=N_RUNS)

# ── Storage ───────────────────────────────────────────────────────────────────
n_ci       = len(POS_CI95_LEVELS)
n_recovery = len(RECOVERY_METHODS)
n_models   = len(NOISE_MODELS)
ipr_arr    = np.full((n_ci, n_recovery, n_models, N_RUNS),          np.nan)
mincpa_arr = np.full((n_ci, n_recovery, n_models, N_RUNS, NB_PAIR), np.nan)


def _one(rep, pos_ci95, pos_dist, latency_s, recovery_model):
    dist_arr, ipr, _t, _n = get_ipr_stochastic_env(
        asas_marh=ASAS_MARH,
        confidence_interval=pos_ci95,
        confidence_interval_velo=VEL_CI95,
        reception_prob=RECEPTION_PROB,
        lookahead_time=LOOKAHEAD,
        dpsi=float(dpsi_values[rep]),
        seed=BASE_SEED + rep,
        config_path=CONFIG_PATH,
        threshold_probability=DEFAULT_GAMMA,
        recovery_model=recovery_model,
        pos_dist=pos_dist,
        latency_s=latency_s,
        init_speed_ownship=SPEED_MS,
        init_speed_intruder=SPEED_MS,
    )
    return ipr, np.min(dist_arr, axis=0)  # min over time -> shape (nb_pair,)


for ci_i, pos_ci95 in enumerate(POS_CI95_LEVELS):
    for ri, (recovery_label, recovery_model) in enumerate(RECOVERY_METHODS):
        for mi, (model_label, pos_dist, latency_s) in enumerate(NOISE_MODELS):
            print(f'\npos_ci95={pos_ci95} m  |  Recovery: {recovery_label}  |  Noise model: {model_label}', flush=True)
            print(f'  {N_RUNS} runs x {NB_PAIR} pairs = {N_RUNS * NB_PAIR:,} pairs', flush=True)

            results = Parallel(n_jobs=N_JOBS)(
                delayed(_one)(r, pos_ci95, pos_dist, latency_s, recovery_model) for r in range(N_RUNS)
            )
            for r, (ipr, min_cpa) in enumerate(results):
                ipr_arr[ci_i, ri, mi, r]    = ipr
                mincpa_arr[ci_i, ri, mi, r] = min_cpa

            n_los     = np.sum((1.0 - ipr_arr[ci_i, ri, mi]) * NB_PAIR)
            total_ipr = 1.0 - n_los / float(N_RUNS * NB_PAIR)
            print(f'  overall IPR = {total_ipr:.4f}', flush=True)

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
out_path = os.path.join(RESULTS_DIR, 'exp4.npz')
np.savez(
    out_path,
    noise_labels=np.array(NOISE_LABELS),
    recovery_labels=np.array(RECOVERY_LABELS),
    pos_ci95_levels=np.array(POS_CI95_LEVELS),
    ipr=ipr_arr,
    min_cpa=mincpa_arr,
    dpsi_values=dpsi_values,
    vel_ci95=VEL_CI95,
    speed_kts=SPEED_HOMOGEN_KTS,
    speed_ms=SPEED_MS,
    n_runs=N_RUNS,
    nb_pair=NB_PAIR,
)
print(f'\nSaved -> {out_path}')

# ── Summary ───────────────────────────────────────────────────────────────────
print(f'\n{"pos_ci95":>10} {"Recovery":<16} {"Noise":<18} {"Overall IPR":>12} {"Min run IPR":>12}')
print('-' * 65)
for ci_i, pos_ci95 in enumerate(POS_CI95_LEVELS):
    for ri, recovery_label in enumerate(RECOVERY_LABELS):
        for mi, model_label in enumerate(NOISE_LABELS):
            n_los     = np.sum((1.0 - ipr_arr[ci_i, ri, mi]) * NB_PAIR)
            total_ipr = 1.0 - n_los / float(N_RUNS * NB_PAIR)
            print(f'{pos_ci95:>10} {recovery_label:<16} {model_label:<18} {total_ipr:>12.4f} {ipr_arr[ci_i, ri, mi].min():>12.4f}')

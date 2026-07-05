'''Experiment 3 (CDARR_Claude) — Noise model comparison, random crossing angle,
heterogeneous speed.

Port of CDaRR_FP experiments/exp3-noise-model-random-angle.py onto this project's
get_ipr_stochastic_env API.

Design
------
* Uncertainty    : pos_ci95 in {10 m, 3 m}, vel_ci95=1 m/s  (2 levels)
* Noise model    : Normal Gaussian / Latency bias / Heavy-tail Gaussian /
                   Anisotropic Gaussian (along-track stdev = 3x cross-track
                   stdev, i.e. variance ratio 9, radial CI95 still 10 m) /
                   Latency + Anisotropic / Heavy-tail + Anisotropic
                   (6 conditions)
* Recovery       : Probabilistic FTR / FTR  (2 conditions)
* Crossing angle : drawn i.i.d. from Uniform(0, 360 deg) PER PAIR (seeded, shared
                   across all conditions for comparability)
* Speed          : ownship and intruder each drawn i.i.d. from
                   Uniform(10, 30) kts PER PAIR (converted to m/s at the boundary)
* Pairs per run  : 10 x 10 = 100  (fixed by the env / config)
* Runs per model : 1 000    ->   100 x 1 000 = 100 000 pairs per condition

NB — both crossing angle and speed heterogeneity are now per-pair, matching
CDaRR_FP: each of the 100 pairs in a run gets its own crossing angle and its own
ownship/intruder speed. This is enabled by PairwiseHorConflict accepting
length-NB_PAIR dpsi/speed arrays (it still spawns via creconfs / lookahead time).

Rationale (reviewer response)
------------------------------
Reviewers flagged that isotropic Gaussian position noise doesn't capture
real-world effects: non-Gaussian / heavy-tailed errors, biased errors, and
anisotropic errors. The six noise conditions above are our response:
* Latency bias   -> biased error (deterministic along-track lag).
* Heavy-tail Gaussian -> heavy-tailed / non-Gaussian error (e.g. multipath),
  implemented as a two-component Gaussian mixture (make_mixture_gaussian).
* Anisotropic Gaussian -> along-track/cross-track anisotropy (along-track
  stdev = 3x cross-track stdev; radial CI95 held at whichever pos_ci95 level
  is being run, 10 m or 3 m).
* Latency + Anisotropic -> latency bias layered on top of the anisotropic
  Gaussian (same pos_dist as 'anisotropic', with latency_s=LATENCY_S).
* Heavy-tail + Anisotropic -> heavy-tailed anisotropic Gaussian: both mixture
  components share the along-/cross-track ratio, and the tail component's
  axes are both scaled by TAIL_RATIO (make_anisotropic_mixture_gaussian).
The pos_ci95 in {10 m, 3 m} sweep (POS_CI95_LEVELS in experiments/config.py)
tests whether these effects hold up as the nominal accuracy tightens.
Schaefer & Jonas (2025, "ADS-B Positional Accuracy and Anomalies: A
Comprehensive Analysis Using High-Resolution MLAT Data") back this up
empirically: cross-track deviation is a zero-mean Gaussian (std ~3.2-3.4 m),
while along-track deviation is non-Gaussian, latency-driven, and roughly 3x
larger in stdev (mean offset -14.93 m) -- i.e. real ADS-B position noise is
both anisotropic and non-Gaussian, dominated by along-track latency bias. Per
that paper, ADS-B v2 latency averages only ~66 ms in practice; LATENCY_S is
deliberately set to 100 ms (see experiments/config.py) to stress-test beyond
the reported figure. We do not model correlated position/velocity errors --
out of scope for this sweep.

Results saved to experiments/results/exp3.npz.  Run directly::

    python experiments/exp3-noise-model-random-angle.py
'''
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from joblib import Parallel, delayed

from experiments.config import (
    CONFIG_PATH, ASAS_MARH, LOOKAHEAD, RECEPTION_PROB,
    KTS_TO_MS, SPEED_MIN_KTS, SPEED_MAX_KTS,
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

# ── Pre-generate per-pair angles and speeds (seeded, shared across conditions) ─
# Both crossing angle and speed are heterogeneous PER PAIR: each of the NB_PAIR
# pairs in a run gets its own crossing angle ~ Uniform(0, 360) deg and its own
# ownship/intruder speed ~ Uniform(SPEED_MIN, SPEED_MAX) kts.
rng = np.random.default_rng(BASE_SEED)
dpsi_values   = rng.uniform(0.0, 360.0, size=(N_RUNS, NB_PAIR))
speed_own_kts = rng.uniform(SPEED_MIN_KTS, SPEED_MAX_KTS, size=(N_RUNS, NB_PAIR))
speed_int_kts = rng.uniform(SPEED_MIN_KTS, SPEED_MAX_KTS, size=(N_RUNS, NB_PAIR))

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
        dpsi=dpsi_values[rep],   # per-pair, shape (NB_PAIR,)
        seed=BASE_SEED + rep,
        config_path=CONFIG_PATH,
        threshold_probability=DEFAULT_GAMMA,
        recovery_model=recovery_model,
        pos_dist=pos_dist,
        latency_s=latency_s,
        init_speed_ownship=speed_own_kts[rep] * KTS_TO_MS,   # per-pair, shape (NB_PAIR,)
        init_speed_intruder=speed_int_kts[rep] * KTS_TO_MS,  # per-pair, shape (NB_PAIR,)
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
out_path = os.path.join(RESULTS_DIR, 'exp3.npz')
np.savez(
    out_path,
    noise_labels=np.array(NOISE_LABELS),
    recovery_labels=np.array(RECOVERY_LABELS),
    pos_ci95_levels=np.array(POS_CI95_LEVELS),
    ipr=ipr_arr,
    min_cpa=mincpa_arr,
    dpsi_values=dpsi_values,
    speed_own_kts=speed_own_kts,
    speed_int_kts=speed_int_kts,
    vel_ci95=VEL_CI95,
    speed_min_kts=SPEED_MIN_KTS,
    speed_max_kts=SPEED_MAX_KTS,
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

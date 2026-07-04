"""Shared experiment parameters for CDARR_Claude — mirrors CDaRR_FP's
experiments/config.py, adapted to this project's get_ipr_stochastic_env API.

Speed convention (see the CDaRR_FP speed-units note): the 10/20/30 figures are
KNOTS. BlueSky / this env consumes speed in m/s, so we convert at the boundary
with ``KTS_TO_MS``. This is the "done correctly" fix relative to CDaRR_FP, which
labelled speeds kts but fed the raw number to BlueSky as m/s (~2x too fast).
"""
import os
import sys
import multiprocessing as _mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Simulation environment ────────────────────────────────────────────────────
CONFIG_PATH = os.path.join(PROJECT_ROOT, "sim_configs", "sim_config.json")
ASAS_MARH     = 1.05
LOOKAHEAD     = 120.0     # s tactical look-ahead horizon
RECEPTION_PROB = 1.0

# Speed (knots -> m/s at the boundary).
KTS_TO_MS       = 0.514444
SPEED_MIN_KTS   = 10.0    # exp3 heterogeneous range
SPEED_MAX_KTS   = 30.0
SPEED_HOMOGEN_KTS = 20.0  # exp4 fixed speed

# ── Uncertainty (single level for exp3/exp4) ──────────────────────────────────
POS_CI95 = 10.0           # confidence_interval (m)
VEL_CI95 =  1.0           # confidence_interval_velo (m/s)

# ── Noise model parameters ────────────────────────────────────────────────────
LATENCY_S   = 0.0661      # s — ADS-B v2 mean latency
TAIL_RATIO  = 3.0
TAIL_WEIGHT = 0.10

# ── Monte Carlo ────────────────────────────────────────────────────────────────
N_RUNS   = 1_000          # 1000 runs x 100 pairs = 100 000 pairs per condition
_ncpu    = _mp.cpu_count()
N_JOBS   = 100 if _ncpu > 100 else (4 if _ncpu > 4 else 1)
BASE_SEED = 42

# ── Probabilistic recovery ─────────────────────────────────────────────────────
DEFAULT_GAMMA = 0.999

# ── Recovery methods (label -> CDARR_Claude recovery_model string) ─────────────
# FP 'probabilistic' == "Probabilistic FTR"; FP 'double_criteria'/'ftr' == "FTR".
RECOVERY_METHODS = [
    ("probabilistic", "Probabilistic FTR"),
    ("ftr",           "FTR"),
]
RECOVERY_LABELS = [r[0] for r in RECOVERY_METHODS]

# ── exp1 / exp2 angle-sweep parameters ────────────────────────────────────────
# Crossing angle grid, matching CDaRR_FP and compare_crr (2, 4, ..., 180 deg).
CROSSING_ANGLES = list(range(2, 181, 2))

# Monte Carlo per angle: 100 runs x 100 pairs = 10 000 pairs (compare_crr convention).
SWEEP_N_RUNS = 100

# Four uncertainty levels (position CI in m, velocity CI in m/s).
UNCERTAINTY_LEVELS = [
    dict(ci=3,  civ=1, label="pos3_vel1",  title="pos=3 m, vel=1 m/s"),
    dict(ci=3,  civ=3, label="pos3_vel3",  title="pos=3 m, vel=3 m/s"),
    dict(ci=10, civ=1, label="pos10_vel1", title="pos=10 m, vel=1 m/s"),
    dict(ci=10, civ=3, label="pos10_vel3", title="pos=10 m, vel=3 m/s"),
]

# Recovery methods for exp1 (short label, display label, model string).
SWEEP_RECOVERY_METHODS = [
    ("cpa",           "Past-CPA",      "CPA"),
    ("ftr",           "FTR",           "FTR"),
    ("probabilistic", "Probabilistic", "Probabilistic FTR"),
]

# Confidence thresholds for exp2 (probabilistic recovery only).
GAMMA_VALUES = [0.999, 0.99, 0.9, 0.75, 0.5]

# ── Output paths ──────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

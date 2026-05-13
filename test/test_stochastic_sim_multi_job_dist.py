from sim.pairwise_stochastic.run_multiple_jobs import run_multiple_jobs
from sim.utils import get_configs

import numpy as np

# Runs the dist-based scenario:
#   - per-pair randomized dpsi (0-360 deg), speed (15-35 kts), dcpa (0-RPZ)
#   - intruders created at the specified initial distance

DIST_M           = 600.0                # initial separation distance (m)
ASAS_MARH        = 1.0                  # ASAS margin
TMAX             = 600                  # simulation time limit (s)
RESOLUTION_MODEL = "MVP"                # "MVP" or "VO"
RECOVERY_MODEL   = "Probabilistic FTR"  # "CPA", "FTR", or "Probabilistic FTR"
THRESHOLD_PROB   = 0.750                # only used when RECOVERY_MODEL == "Probabilistic FTR"

results = run_multiple_jobs(
    n_runs=12,
    n_jobs=4,
    confidence_interval=30,
    confidence_interval_velo=1,
    reception_prob=0.99,
    dist_m=DIST_M,
    asas_marh=ASAS_MARH,
    tmax=TMAX,
    lookahead_time=300.0,
    resolution_model=RESOLUTION_MODEL,
    recovery_model=RECOVERY_MODEL,
    threshold_probability=THRESHOLD_PROB,
    dist_mode=True,
)

cfg = get_configs()

print("Overall IPR:", results["overall_ipr"])
print("Sim timer array:", results["sim_timer"])
print("N active conflict:", results["n_active_conflict"])

all_cpa = results["worst_cpa"].flatten()  # shape: (1200,)

percentiles = np.percentile(all_cpa, [1, 5, 50, 95, 99])
print(f"\nCPA statistics (m) over {len(all_cpa)} pairs:")
print(f"  1%:    {percentiles[0]:.3f}")
print(f"  5%:    {percentiles[1]:.3f}")
print(f"  Mean:  {np.mean(all_cpa):.3f}")
print(f"  Median:{percentiles[2]:.3f}")
print(f"  95%:   {percentiles[3]:.3f}")
print(f"  99%:   {percentiles[4]:.3f}")
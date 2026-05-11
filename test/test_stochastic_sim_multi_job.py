from sim.pairwise_stochastic.run_multiple_jobs import run_multiple_jobs
from sim.utils import get_configs

import numpy as np

# this is is to test that worst_cpa is unique
# because it shows that each run is unique
# and also the value make sense (it shouldn't be above 2*horizontal_sep)

results = run_multiple_jobs(
    n_runs=12,
    n_jobs=4,
    asas_marh=1.2,
    confidence_interval=15,
    confidence_interval_velo=0.5,
    reception_prob=0.95,
    lookahead_time=240,
    dpsi=2,
)

cfg = get_configs()

# Optional debug prints
print("Overall IPR:", results["overall_ipr"])
print("Worst CPA array:", results["worst_cpa"])
print("Sim timer array:", results["sim_timer"])
print("N active conflict:", results["n_active_conflict"])

worst_cpa = np.asarray(results["worst_cpa"])
# One worst-CPA value per run (min across that run's pairs)
worst_per_run = worst_cpa.min(axis=1) if worst_cpa.ndim > 1 else worst_cpa
unique_worst_per_run = np.unique(worst_per_run)
assert len(worst_per_run) == len(unique_worst_per_run), (
    f"Duplicate per-run worst CPA values found: {worst_per_run}"
)

max_allowed_cpa = 2.0 * cfg.horizontal_sep
assert np.all(worst_cpa <= max_allowed_cpa), (
    f"Some worst_cpa values exceed the expected upper bound. "
    f"Maximum allowed: {max_allowed_cpa}, "
    f"Found: {results['worst_cpa']}"
)

print("Test passed: worst_cpa uniqueness and separation constraints satisfied.")

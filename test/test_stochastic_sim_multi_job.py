from sim.pairwise_stochastic.run_multiple_jobs import run_multiple_jobs
from sim.utils import get_configs

import numpy as np

# this is is to test that worst_cpa is unique
# because it shows that each run is unique
# and also the value make sense (it shouldn't be above 2*horizontal_sep)

results = run_multiple_jobs(
    n_runs=12,
    n_jobs=4,
    asas_marh=1.05,
    confidence_interval=42.3,
    confidence_interval_velo=5.0,
    reception_prob=0.95,
    lookahead_time=15,
    dpsi=45,
)

cfg = get_configs()

# Optional debug prints
print("Overall IPR:", results["overall_ipr"])
print("Worst CPA array:", results["worst_cpa"])
print("Sim timer array:", results["sim_timer"])


unique_worst_cpa = np.unique(results["worst_cpa"])
assert len(results["worst_cpa"]) == len(unique_worst_cpa), (
    f"Duplicate values found in worst_cpa: {unique_worst_cpa}"
)

min_allowed_cpa = 2.0 * cfg.horizontal_sep
assert np.all(results["worst_cpa"] <= min_allowed_cpa), (
    f"Some worst_cpa values violate separation constraint. "
    f"Minimum allowed: {min_allowed_cpa}, "
    f"Found: {results['worst_cpa']}"
)

print("Test passed: worst_cpa uniqueness and separation constraints satisfied.")

"""
Run multiple Monte-Carlo jobs of get_ipr_stochastic_env with different seeds.
"""

from joblib import Parallel, delayed
import numpy as np
from typing import Dict, Any, List, Tuple

from sim.pairwise_stochastic.get_ipr_stochastic_env import get_ipr_stochastic_env
from sim.pairwise_stochastic.get_ipr_stochastic_env import get_ipr_stochastic_env_randomized

# Number of independent conflict pairs per simulation run
NB_PAIR = 100


def _run_one(
    rep: int, base_seed: int, kwargs: Dict[str, Any]
) -> Tuple[float, float, float, int]:
    """
    One Monte-Carlo repetition.

    Returns:
        ipr (float)
        worst_cpa (float)
        sim_timer (float)
        n_active_conflict (int)
    """
    seed = base_seed + rep

    # take the kwargs here, randomized_speed_heading is only used in get_ipr_stochastic_env_randomized, so it will be ignored if not present
    if(kwargs.get("randomized_speed_heading", False)):
        distance_array, ipr, sim_timer, n_active_conflict = get_ipr_stochastic_env_randomized(seed=seed, **kwargs)
    else:
        distance_array, ipr, sim_timer, n_active_conflict = get_ipr_stochastic_env(seed=seed, **kwargs)


    # Worst CPA over all pairs
    min_dist_per_pair = np.min(distance_array, axis=0)
    worst_cpa = np.round(min_dist_per_pair, 3)

    return float(ipr), worst_cpa, float(sim_timer), int(n_active_conflict)


def run_multiple_jobs(
    *,
    n_runs: int,
    n_jobs: int,
    base_seed: int = 42,
    **env_kwargs,
):
    """
    Run get_ipr_stochastic_env multiple times with different seeds.

    Args:
        n_runs: number of Monte-Carlo repetitions
        n_jobs: number of parallel workers
        base_seed: seed offset
        **env_kwargs: forwarded to get_ipr_stochastic_env

    Returns:
        results dict with arrays
    """

    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_one)(rep, base_seed, env_kwargs)
        for rep in range(n_runs)
    )

    # Recompute overall IPR across all runs (mean of per-run IPR is not correct
    # because each run has NB_PAIR pairs and LOS counts must be aggregated)
    ipr_arr, worst_cpa_arr, sim_timer_arr, n_active_conflict_arr = map(
        np.array, zip(*results)
    )

    n_los = np.sum((1.0 - ipr_arr) * NB_PAIR)
    overall_ipr = 1.0 - (n_los / float(n_runs * NB_PAIR))

    return {
        "overall_ipr": overall_ipr,
        "ipr": ipr_arr,
        "worst_cpa": np.round(worst_cpa_arr, 3),
        "sim_timer": sim_timer_arr,
        "n_active_conflict": np.sum(n_active_conflict_arr),
        "worst_cpa_min": float(worst_cpa_arr.min()),
    }

if __name__ == "__main__":
    # Example usage
    print("Run!")

    results = run_multiple_jobs(
        n_runs=100,
        n_jobs=100,
        asas_marh=1.05,
        confidence_interval=10,
        confidence_interval_velo=1,
        reception_prob=1.0,
        lookahead_time=300,
        dpsi=0.18,
        config_path="sim_configs/sim_config.json",
        recovery_model="Probabilistic FTR",
        threshold_probability=0.5
    )

    print("Overall IPR", results["overall_ipr"])

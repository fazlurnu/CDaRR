"""
Run multiple Monte-Carlo jobs of get_ipr_stochastic_env with different seeds.
"""

from joblib import Parallel, delayed
import numpy as np
from typing import Dict, Any, List, Tuple

from sim.get_ipr_stochastic_env import get_ipr_stochastic_env


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

    distance_array, ipr, sim_timer, n_active_conflict = get_ipr_stochastic_env(seed=seed, **kwargs)

    # Worst CPA over all pairs
    min_dist_per_pair = np.min(distance_array, axis=0)
    worst_cpa = float(np.min(min_dist_per_pair))

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

    # here we assume the nb of pair is always 100 per run
    # we use that number to recompute the ipr for the whole run
    # mean of IPR is not the IPR of the whole run

    ipr_arr, worst_cpa_arr, sim_timer_arr, n_active_conflict_arr = map(
        np.array, zip(*results)
    )

    n_los = np.sum((1.0 - ipr_arr) * 100.0)
    overall_ipr = 1.0 - (n_los / float(n_runs * 100.0))

    return {
        "overall_ipr:": overall_ipr,
        "ipr": ipr_arr,
        "worst_cpa": worst_cpa_arr,
        "sim_timer": sim_timer_arr,
        "n_active_conflict": n_active_conflict_arr,
        "ipr_mean": float(ipr_arr.mean()),
        "ipr_std": float(ipr_arr.std()),
        "worst_cpa_min": float(worst_cpa_arr.min()),
        "sim_timer_mean": float(sim_timer_arr.mean()),
        "sim_timer_max": float(sim_timer_arr.max()),
    }

if __name__ == "__main__":
    # Example usage
    results = run_multiple_jobs(
        n_runs=12,
        n_jobs=4,
        asas_marh=1.2,
        confidence_interval=42.3,
        confidence_interval_velo=5.0,
        reception_prob=0.95,
        lookahead_time=15,
        dpsi=45,
    )

    print("Overall IPR:", results["overall_ipr:"])
    print("Worst CPA array:", results["worst_cpa"])
    print("Sim timer array:", results["sim_timer"])

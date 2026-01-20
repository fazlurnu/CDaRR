from sim.get_ipr_stochastic_env import get_ipr_stochastic_env
import numpy as np

distance_array, ipr, sim_timer, n_active_conflict = get_ipr_stochastic_env(
    asas_marh=1.05,
    confidence_interval=15,
    confidence_interval_velo=1.5,
    reception_prob=1.0,
    lookahead_time=15,
    dpsi=180,
)

print(f"IPR: {ipr:.2f}")

min_dist_per_pair = np.min(distance_array, axis=0)
worst_cpa = np.min(min_dist_per_pair)
print(f"Worst CPA (m): {worst_cpa:.2f}")
print(f"Number of active conflicts at end: {n_active_conflict}, at time {sim_timer:.2f} s")
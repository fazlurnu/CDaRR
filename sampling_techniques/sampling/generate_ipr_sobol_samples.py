import matplotlib.pyplot as plt
import numpy as np

from sampling_techniques.utils.sobol_pool import generate_sobol_points
from sampling_techniques.utils.scaling import uniform_denorm
from sim.pairwise_stochastic.run_multiple_jobs import run_multiple_jobs
from types import SimpleNamespace

# let's define the bounds here and we generate the samples accordingly

# x1 is resofach, x2 is lookahead time
bounds = SimpleNamespace(
    x1_min=0.9,
    x1_max=1.2,
    x2_min=1.0,
    x2_max=120.0
)

def generate_ipr_sobol_samples(bounds, parameters, n_samples, seed):
    samples = generate_sobol_points(bounds, n_samples=n_samples, seed=seed)
    X_raw = uniform_denorm(bounds, samples)

    # Run initial simulations for the first tanh fitting
    results = []

    counter = 1

    for resofach, lookahead_time in X_raw:    
        print(f"{counter}/{len(X_raw)}. Running resofach: {resofach:.3f}, lookahead_time: {lookahead_time:.3f}")
        sim_results = run_multiple_jobs(
            n_runs=4,
            n_jobs=4,
            asas_marh=resofach,
            lookahead_time=lookahead_time,
            confidence_interval=parameters.confidence_interval,
            confidence_interval_velo=parameters.confidence_interval_velo,
            reception_prob=parameters.reception_prob,
            dpsi=parameters.dpsi,
        )

        results.append({
            "x1_resofach": resofach,
            "x2_lookahead_time": lookahead_time,
            "sim_results": sim_results,
        })

        counter += 1

    return results
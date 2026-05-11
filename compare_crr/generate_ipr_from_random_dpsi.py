from sim.pairwise_stochastic.run_multiple_jobs import run_multiple_jobs


def generate_ipr_from_random_dpsi(parameters):
    """Run simulations across all crossing angles for a given parameter set.

    Parameters
    ----------
    parameters : SimpleNamespace
        Must have: confidence_interval, confidence_interval_velo, reception_prob,
        config_path, threshold_probability, recovery_model.

    Returns
    -------
    dict : {dpsi: sim_results_dict}
    """
    results = {}

    sim_results = run_multiple_jobs(
        n_runs=10000,
        n_jobs=100,
        asas_marh=1.05,
        lookahead_time=120,
        confidence_interval=parameters.confidence_interval,
        confidence_interval_velo=parameters.confidence_interval_velo,
        reception_prob=parameters.reception_prob,
        dpsi=None,
        config_path=parameters.config_path,
        threshold_probability=parameters.threshold_probability,
        recovery_model=parameters.recovery_model,
        randomized_speed_heading = True,
    )

    # check if dpsi_list exists
    print(
        f"recovery: {parameters.recovery_model}, "
        f"overall_ipr: {sim_results['overall_ipr']}"
    )

    results['randomized'] = sim_results

    return results

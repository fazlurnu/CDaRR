from sim.pairwise_stochastic.run_multiple_jobs import run_multiple_jobs


def generate_ipr_from_dpsi_samples(parameters):
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

<<<<<<< HEAD
    dpsi_list = list(range(2, 43, 2)) + list(range(45, 181, 5))
=======
    dpsi_list = list(range(2, 181, 2))
>>>>>>> c842459 (modify variance)

    for i, dpsi in enumerate(dpsi_list, 1):
        sim_results = run_multiple_jobs(
            n_runs=100,
            n_jobs=100,
            asas_marh=1.05,
            lookahead_time=120,
            confidence_interval=parameters.confidence_interval,
            confidence_interval_velo=parameters.confidence_interval_velo,
            reception_prob=parameters.reception_prob,
            dpsi=dpsi,
            config_path=parameters.config_path,
            threshold_probability=parameters.threshold_probability,
            recovery_model=parameters.recovery_model,
        )

        print(
            f"{i}/{len(dpsi_list)} Done. dpsi: {dpsi}, "
            f"recovery: {parameters.recovery_model}, "
            f"overall_ipr: {sim_results['overall_ipr']}"
        )

        results[dpsi] = sim_results

    return results

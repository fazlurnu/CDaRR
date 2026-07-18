"""
Stochastic pairwise horizontal conflict simulation.

Inputs:
    - asas_marh
    - confidence_interval
    - confidence_interval_velo
    - reception_prob
    - lookahead_time
    - dpsi

Returns:
    - distance_array : np.ndarray, shape (T, nb_pair)
    - ipr            : float

Phase 4 switchover (refactor_fp.md): the three public functions below are now
thin wrappers over sim.pairwise_stochastic.loop's functional-core
reimplementation (cd/cr/crr/cns/cdarr), not the legacy sim_models Entity
classes. Same signatures, same defaults, same return contract -- every
caller (experiments/*, compare_crr/*, sim/pairwise_stochastic/run_multiple_jobs.py,
quick_test_ipr.py) needs zero edits. Verified bit-identical against the
frozen legacy implementation before this switchover: 16/16 golden baselines
(test/golden/baseline/, test_equiv_loop_golden.py), get_ipr_stochastic_env_randomized
against the legacy function directly (test_equiv_loop_randomized.py), and the
L3 aggregate pipeline (run_multiple_jobs + exp3/exp5-style cells,
test/golden/l3_baseline.py) -- captured from the legacy code immediately
before this edit, then re-verified against this wrapped version immediately
after.

The legacy sim_models-based implementation is preserved unchanged and still
directly testable (sim_models/cd_statebased.py, cr_mvp.py, cr_vo.py,
crr_resumenav_*.py, adsl_module.py, ...) -- Phase 5 will move it to legacy/
once the deprecation window closes; this module no longer references it.
"""

import bluesky as bs

from sim.pairwise_stochastic.loop import (
    get_ipr_stochastic_env as _get_ipr_stochastic_env,
    get_ipr_stochastic_env_randomized as _get_ipr_stochastic_env_randomized,
    get_ipr_stochastic_env_dist as _get_ipr_stochastic_env_dist,
)
from sim.utils import suppress_output

# ---------------------------------------------------------
# Safe BlueSky initialization (only once)
# ---------------------------------------------------------
# Kept here (in addition to loop.py's own identical guard) because joblib
# workers import this module directly and rely on import-time init.
if not getattr(bs, "_joblib_inited", False):
    with suppress_output():
        bs.init(mode="sim", detached=True)
        bs._joblib_inited = True


# ---------------------------------------------------------
# Main API
# ---------------------------------------------------------
def get_ipr_stochastic_env(
    asas_marh: float,
    confidence_interval: float,
    confidence_interval_velo: float,
    reception_prob: float,
    lookahead_time: float,
    dpsi: float,
    seed: int = 44,
    config_path: str = None,
    threshold_probability: float = None,
    recovery_model: str = None,
    pos_dist=None,
    latency_s: float = 0.0,
    init_speed_ownship: float = None,
    init_speed_intruder: float = None,
    assumed_confidence_interval: float = None,
    assumed_confidence_interval_velo: float = None,
):
    """
    Run a stochastic pairwise conflict simulation and compute IPR.

    Parameters
    ----------
    recovery_model : str, optional
        If provided, overrides the recovery model from the config file.
        Valid values: "CPA", "FTR", "Probabilistic FTR".
    assumed_confidence_interval, assumed_confidence_interval_velo : float, optional
        The position / velocity CI95 (m, m/s) the *probabilistic* recovery
        assumes for its worldview uncertainty (``conf.sigma_r`` / ``sigma_v``),
        decoupled from the ``confidence_interval[_velo]`` that generates the
        actual measurement noise. ``None`` (default) keeps them matched — the
        worldview equals the true noise, as in exp3/exp4. Set them different to
        model a *calibration mismatch* (exp5): e.g. the system believes CI95=10 m
        while the true noise is 15 m. Only the probabilistic recovery reads
        these; FTR / CPA recovery are unaffected.
    pos_dist : callable, optional
        Position noise distribution ``(n, ci95, rng) -> (n, 2)`` in metres.
        ``None`` (default) uses the 2D-Gaussian model. See
        ``sim_models.noise_distributions`` (e.g. ``make_mixture_gaussian``).
    latency_s : float, optional
        ADS-B reporting latency in seconds; adds an along-track position bias
        of ``-latency_s * gs``. ``0.0`` (default) disables it.
    init_speed_ownship, init_speed_intruder : float, optional
        Override the initial aircraft speeds (m/s). ``None`` uses the config
        values. Used by exp3/exp4 to inject per-run speeds.

    Returns
    -------
    distance_array : np.ndarray
        Distance over time for each pair, shape (T, nb_pair)
    ipr : float
        Intrusion Prevention Rate
    """
    return _get_ipr_stochastic_env(
        asas_marh=asas_marh,
        confidence_interval=confidence_interval,
        confidence_interval_velo=confidence_interval_velo,
        reception_prob=reception_prob,
        lookahead_time=lookahead_time,
        dpsi=dpsi,
        seed=seed,
        config_path=config_path,
        threshold_probability=threshold_probability,
        recovery_model=recovery_model,
        pos_dist=pos_dist,
        latency_s=latency_s,
        init_speed_ownship=init_speed_ownship,
        init_speed_intruder=init_speed_intruder,
        assumed_confidence_interval=assumed_confidence_interval,
        assumed_confidence_interval_velo=assumed_confidence_interval_velo,
    )


def get_ipr_stochastic_env_randomized(
    asas_marh: float,
    confidence_interval: float,
    confidence_interval_velo: float,
    reception_prob: float,
    lookahead_time: float,
    dpsi: float,
    seed: int = 44,
    config_path: str = None,
    threshold_probability: float = None,
    recovery_model: str = None,
    randomized_speed_heading: bool = True,
):
    """
    Run a stochastic pairwise conflict simulation and compute IPR.

    Parameters
    ----------
    recovery_model : str, optional
        If provided, overrides the recovery model from the config file.
        Valid values: "CPA", "FTR", "Probabilistic FTR".

    Returns
    -------
    distance_array : np.ndarray
        Distance over time for each pair, shape (T, nb_pair)
    ipr : float
        Intrusion Prevention Rate
    """
    return _get_ipr_stochastic_env_randomized(
        asas_marh=asas_marh,
        confidence_interval=confidence_interval,
        confidence_interval_velo=confidence_interval_velo,
        reception_prob=reception_prob,
        lookahead_time=lookahead_time,
        dpsi=dpsi,
        seed=seed,
        config_path=config_path,
        threshold_probability=threshold_probability,
        recovery_model=recovery_model,
        randomized_speed_heading=randomized_speed_heading,
    )


def get_ipr_stochastic_env_dist(
    confidence_interval: float,
    confidence_interval_velo: float,
    reception_prob: float,
    dist_m: float,
    asas_marh: float = 1.0,
    lookahead_time: float = 300.0,
    tmax: float = 1200.0,
    seed: int = 44,
    config_path: str = None,
    threshold_probability: float = None,
    resolution_model: str = None,
    recovery_model: str = None,
):
    """
    Stochastic pairwise conflict simulation with ASAS margin fixed at 1.0
    and intruders placed at a given initial distance.

    Per-pair randomization (seeded):
      - dpsi:            uniform 0-360 deg
      - ownship speed:   uniform 15-35 kts (7.72-18.01 m/s)
      - intruder speed:  uniform 15-35 kts (7.72-18.01 m/s)
      - dcpa:            uniform 0-RPZ (50 m)

    Parameters
    ----------
    confidence_interval      : position uncertainty (m)
    confidence_interval_velo : velocity uncertainty (m/s)
    reception_prob           : ADSL reception probability
    dist_m                   : initial separation distance at conflict creation (m)
    lookahead_time           : CDR lookahead horizon (s), default 300
    resolution_model         : override config resolution model ("MVP" or "VO")
    recovery_model            : override config recovery model ("CPA", "FTR", "Probabilistic FTR")
    """
    return _get_ipr_stochastic_env_dist(
        confidence_interval=confidence_interval,
        confidence_interval_velo=confidence_interval_velo,
        reception_prob=reception_prob,
        dist_m=dist_m,
        asas_marh=asas_marh,
        lookahead_time=lookahead_time,
        tmax=tmax,
        seed=seed,
        config_path=config_path,
        threshold_probability=threshold_probability,
        resolution_model=resolution_model,
        recovery_model=recovery_model,
    )

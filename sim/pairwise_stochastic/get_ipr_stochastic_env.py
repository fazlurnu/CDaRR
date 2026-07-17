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
"""

import numpy as np
import bluesky as bs

from sim_models.cd_statebased import StateBased
from envs.pairwise_conflict import PairwiseHorConflict, PairwiseHorConflictDist
from sim_models.cr_mvp import MVP
from sim_models.cr_vo import VO
from sim_models.adsl_module import ADSL
from sim_models.reception_model import ReceptionModel
from sim_models.crr_resumenav_cpa import resumenav as resumenav_cpa
from sim_models.crr_resumenav_ftr import resumenav_double_criteria
from sim_models.crr_resumenav_probabilistic_ftr import resumenav_probabilistic_ftr

from sim.utils import (
    _check_tcpa_tinhor_per_pair,
    done_with_timeout,
    get_configs,
    suppress_output
)

# ---------------------------------------------------------
# Safe BlueSky initialization (only once)
# ---------------------------------------------------------
if not getattr(bs, "_joblib_inited", False):
    with suppress_output():
        bs.init(mode="sim", detached=True)
        bs._joblib_inited = True


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
_RECOVERY_MODELS = {
    "CPA": resumenav_cpa,
    "FTR": resumenav_double_criteria,
    "Probabilistic FTR": resumenav_probabilistic_ftr,
}

_RESOLUTION_MODELS = {
    "MVP": MVP,
    "VO": VO,
}


def _create_cdr_models(cfg):
    """Create conflict detection, resolution, and recovery model instances."""
    detection = StateBased()
    detection_gt = StateBased()

    if cfg.resolution_model not in _RESOLUTION_MODELS:
        raise ValueError(f"Unsupported resolution model: {cfg.resolution_model}")
    resolution = _RESOLUTION_MODELS[cfg.resolution_model]()

    # BlueSky's EntityMeta makes MVP/VO process-wide singletons: the constructor
    # above returns a persistent instance whose __init__ ran only on first use, so
    # the per-run recovery bookkeeping (resopairs and _intr_init_vel) survives from
    # one call to the next. Aircraft IDs are reused every run (DRO###/DRI###), so a
    # stale resopairs makes a new run's first resolve treat already-in-conflict
    # pairs as already-resolved -- skipping the "new pair" branch that records the
    # initial intruder velocity -- and recovery then reuses the *previous* run's
    # _intr_init_vel. That order-dependent leak is the KI-1 in-process
    # nondeterminism (test/golden/KNOWN_ISSUES.md). Reset both so every run starts
    # from a clean resolution state, independent of what ran before it.
    resolution.resopairs = set()
    resolution._intr_init_vel = {}

    if cfg.recovery_model not in _RECOVERY_MODELS:
        raise ValueError(f"Unsupported recovery model: {cfg.recovery_model}")
    recovery = _RECOVERY_MODELS[cfg.recovery_model]

    return detection, detection_gt, resolution, recovery


def _create_adsl_stack(confidence_interval, confidence_interval_velo, reception_prob, seed,
                       pos_dist=None, latency_s=0.0):
    """Create the four ADSL nodes (bus, ownship, intruder, prev_intruder).

    ``pos_dist`` selects the general position-noise model (exp3/exp4
    noise-model sweep) and is forwarded to every node; only ``ownship_adsl``
    actually injects it via ``update_from_truth`` in the simulation loop.

    ``latency_s`` (ADS-B reporting latency) is deliberately *not* passed to
    ``ownship_adsl``: latency is a broadcast-transmission delay, so it should
    only affect the position of an aircraft as perceived by *someone else*
    receiving its (delayed) ADS-B message -- not an aircraft's own directly-
    known state. ``ownship_adsl`` is used as the "own" state in conflict
    detection/resolution/recovery, so it keeps the general position noise but
    always has ``latency_s=0.0``. ``intruder_adsl`` keeps the real
    ``latency_s``; the simulation loop applies the bias explicitly (via
    ``intruder_adsl.noise.add_latency_bias``) only to freshly-received
    aircraft, on top of the (already latency-free) general noise relayed in
    from ``ownship_adsl``. See the simulation loop below.
    """
    adsl_bus = ADSL(
        confidence_interval,
        confidence_interval_velo,
        reception_prob=1.0,
        seed=seed + 1,
        pos_dist=pos_dist,
        latency_s=latency_s,
    )

    ownship_adsl = ADSL(
        confidence_interval,
        confidence_interval_velo,
        reception_prob=1.0,
        seed=seed + 2,
        pos_dist=pos_dist,
        latency_s=0.0,
    )

    intruder_adsl = ADSL(
        confidence_interval,
        confidence_interval_velo,
        reception_prob=reception_prob,
        seed=seed + 3,
        pos_dist=pos_dist,
        latency_s=latency_s,
    )

    prev_intruder_adsl = ADSL(
        confidence_interval,
        confidence_interval_velo,
        reception_prob=1.0,
        seed=seed + 4,
        pos_dist=pos_dist,
        latency_s=latency_s,
    )

    rx_rng = np.random.default_rng(seed + 999)
    intruder_adsl.reception = ReceptionModel(
        reception_prob=reception_prob,
        rng=rx_rng,
    )

    return adsl_bus, ownship_adsl, intruder_adsl, prev_intruder_adsl


def _compute_ipr(distance_array, nb_pair, horizontal_sep):
    """Compute IPR from the distance history.

    Returns
    -------
    distance_array : np.ndarray, shape (T, nb_pair)
    ipr : float
    """
    distance_array = np.asarray(distance_array)[:, :nb_pair]
    min_dist = np.min(distance_array, axis=0)

    n_los = int(np.sum(min_dist < horizontal_sep))
    n_conflict = int(nb_pair)

    ipr = 1.0 - (n_los / float(n_conflict))
    return distance_array, ipr


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

    # ----------------------------
    # Load config
    # ----------------------------
    cfg = get_configs(config_path) if config_path else get_configs()

    if recovery_model is not None:
        cfg.recovery_model = recovery_model

    bs.settings.asas_marh = asas_marh

    # ----------------------------
    # CDR models
    # ----------------------------
    conf_detection, conf_detection_gt, conf_resolution, conf_recovery = _create_cdr_models(cfg)

    # ----------------------------
    # Environment
    # ----------------------------
    pairwise = PairwiseHorConflict(
        pair_width=cfg.width,
        pair_height=cfg.height,
        asas_pzr_m=cfg.horizontal_sep,
        dtlookahead=lookahead_time * 1.5,
        init_speed_ownship=cfg.init_speed_ownship if init_speed_ownship is None else init_speed_ownship,
        init_speed_intruder=cfg.init_speed_intruder if init_speed_intruder is None else init_speed_intruder,
        init_dpsi=dpsi,
        aircraft_type_ownship=cfg.aircraft_type,
        simdt_factor=cfg.SIMDT_FACTOR,
    )

    simdt = bs.settings.simdt * cfg.SIMDT_FACTOR
    tmax = 600

    # ----------------------------
    # ADSL setup
    # ----------------------------
    adsl_bus, ownship_adsl, intruder_adsl, prev_intruder_adsl = _create_adsl_stack(
        confidence_interval, confidence_interval_velo, reception_prob, seed,
        pos_dist=pos_dist, latency_s=latency_s,
    )

    # Probabilistic-recovery worldview uncertainty. By default it equals the true
    # measurement noise (combined over both aircraft). When an assumed CI95 is
    # supplied it is used instead — a deliberate mismatch between what the
    # recovery believes and the noise actually injected (exp5). CI95->1-sigma
    # uses the same 2.448 factor as the ADSL nodes.
    _CI95_TO_STD_2D = 2.448
    if assumed_confidence_interval is None:
        sigma_r_worldview = np.sqrt(ownship_adsl.pos_std**2 + intruder_adsl.pos_std**2)
    else:
        sigma_r_worldview = np.sqrt(2.0) * (assumed_confidence_interval / _CI95_TO_STD_2D)
    if assumed_confidence_interval_velo is None:
        sigma_v_worldview = np.sqrt(ownship_adsl.vel_std**2 + intruder_adsl.vel_std**2)
    else:
        sigma_v_worldview = np.sqrt(2.0) * (assumed_confidence_interval_velo / _CI95_TO_STD_2D)

    # ----------------------------
    # Simulation loop
    # ----------------------------
    sim_timer = 0.0
    next_event_t = 0.0
    event_dt = float(bs.settings.asas_dt)
    eps = np.finfo(float).eps * 100

    distance_array = []
    done_start_time = None
    initialized = False

    while sim_timer < tmax:

        states = pairwise._get_states()
        n = int(states.ntraf)

        if not initialized:
            ownship_adsl.update_from_truth(states)
            adsl_bus.send_data(intruder_adsl, ownship_adsl, indices=None)
            # Latency bias applies only to the perceived/received copy (see
            # _create_adsl_stack docstring), not to ownship_adsl own state.
            intruder_adsl.noise.add_latency_bias(intruder_adsl.msg, states, np.arange(n))
            adsl_bus.send_data(prev_intruder_adsl, intruder_adsl, indices=None)
            initialized = True

        if sim_timer + eps >= next_event_t:

            # --- ADSL update ---
            ownship_adsl.update_from_truth(states)

            idx_rx = intruder_adsl.reception.sample_indices(n)
            rx_mask = np.zeros(n, dtype=bool)
            rx_mask[idx_rx] = True
            idx_miss = np.where(~rx_mask)[0]

            if idx_rx.size > 0:
                adsl_bus.send_data(intruder_adsl, ownship_adsl, indices=idx_rx)
                # Add the along-track latency bias only for freshly-received
                # aircraft (a stale/held-over contact, relayed below via
                # idx_miss, already carries the bias from when it was fresh).
                intruder_adsl.noise.add_latency_bias(intruder_adsl.msg, states, idx_rx)
            if idx_miss.size > 0:
                adsl_bus.send_data(intruder_adsl, prev_intruder_adsl, indices=idx_miss)

            adsl_bus.send_data(prev_intruder_adsl, intruder_adsl, indices=None)

            # --- Detect / resolve ---
            conf_detection.detect(
                ownship_adsl,
                intruder_adsl,
                cfg.horizontal_sep,
                100.0,
                lookahead_time,
            )

            conf_detection_gt.detect(
                states,
                states,
                cfg.horizontal_sep,
                100.0,
                lookahead_time,
            )

            reso = conf_resolution.resolve(
                conf_detection,
                ownship_adsl,
                intruder_adsl,
                asas_marh
            )

            conf_detection.sigma_r = sigma_r_worldview
            conf_detection.sigma_v = sigma_v_worldview
            conf_detection.dcpa_prob_threshold = threshold_probability

            delpairs_noise = conf_recovery(conf_resolution, conf_detection, ownship_adsl, intruder_adsl)

            missed = (
                int(np.floor((sim_timer - next_event_t) / event_dt)) + 1
                if sim_timer > next_event_t
                else 1
            )
            next_event_t += missed * event_dt

        # --- Step environment ---
        dist = pairwise.step(reso)
        distance_array.append(dist)

        # --- Done logic ---
        done_now, is_active = _check_tcpa_tinhor_per_pair(
            bs.traf.id,
            conf_detection_gt.tcpa_all,
            conf_detection_gt.tinhor_all,
        )

        dist_hist = np.asarray(distance_array)
        min_dist_so_far = np.min(dist_hist, axis=0)

        n_active = int(np.sum(is_active & (min_dist_so_far > 50.0)))

        done_start_time, should_stop = done_with_timeout(
            done_now,
            done_start_time,
            sim_timer,
            cfg.DONE_TIMEOUT,
            verbose=False,
        )

        if should_stop:
            break

        sim_timer += simdt

    # ----------------------------
    # Cleanup & compute IPR
    # ----------------------------
    pairwise.reset()
    distance_array, ipr = _compute_ipr(distance_array, pairwise.nb_pair, cfg.horizontal_sep)

    return distance_array, ipr, sim_timer, n_active

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

    # ----------------------------
    # Load config
    # ----------------------------
    cfg = get_configs(config_path) if config_path else get_configs()

    if recovery_model is not None:
        cfg.recovery_model = recovery_model

    bs.settings.asas_marh = asas_marh

    # ----------------------------
    # CDR models
    # ----------------------------
    conf_detection, conf_detection_gt, conf_resolution, conf_recovery = _create_cdr_models(cfg)

    # ----------------------------
    # Environment
    # ----------------------------
    scenario_rng = np.random.default_rng(seed + 7919)
    pairwise = PairwiseHorConflict(
        pair_width=cfg.width,
        pair_height=cfg.height,
        asas_pzr_m=cfg.horizontal_sep,
        dtlookahead=lookahead_time * 1.5,
        init_speed_ownship=float(scenario_rng.uniform(10, 30)),
        init_speed_intruder=float(scenario_rng.uniform(10, 30)),
        init_dpsi=float(scenario_rng.uniform(0, 360)),
        aircraft_type_ownship=cfg.aircraft_type,
        simdt_factor=cfg.SIMDT_FACTOR,
    )

    simdt = bs.settings.simdt * cfg.SIMDT_FACTOR
    tmax = 600

    # ----------------------------
    # ADSL setup
    # ----------------------------
    adsl_bus, ownship_adsl, intruder_adsl, prev_intruder_adsl = _create_adsl_stack(
        confidence_interval, confidence_interval_velo, reception_prob, seed
    )

    # ----------------------------
    # Simulation loop
    # ----------------------------
    sim_timer = 0.0
    next_event_t = 0.0
    event_dt = float(bs.settings.asas_dt)
    eps = np.finfo(float).eps * 100

    distance_array = []
    done_start_time = None
    initialized = False

    while sim_timer < tmax:

        states = pairwise._get_states()
        n = int(states.ntraf)

        if not initialized:
            ownship_adsl.update_from_truth(states)
            adsl_bus.send_data(intruder_adsl, ownship_adsl, indices=None)
            adsl_bus.send_data(prev_intruder_adsl, intruder_adsl, indices=None)
            initialized = True

        if sim_timer + eps >= next_event_t:

            # --- ADSL update ---
            ownship_adsl.update_from_truth(states)

            idx_rx = intruder_adsl.reception.sample_indices(n)
            rx_mask = np.zeros(n, dtype=bool)
            rx_mask[idx_rx] = True
            idx_miss = np.where(~rx_mask)[0]

            if idx_rx.size > 0:
                adsl_bus.send_data(intruder_adsl, ownship_adsl, indices=idx_rx)
            if idx_miss.size > 0:
                adsl_bus.send_data(intruder_adsl, prev_intruder_adsl, indices=idx_miss)

            adsl_bus.send_data(prev_intruder_adsl, intruder_adsl, indices=None)

            # --- Detect / resolve ---
            conf_detection.detect(
                ownship_adsl,
                intruder_adsl,
                cfg.horizontal_sep,
                100.0,
                lookahead_time,
            )

            conf_detection_gt.detect(
                states,
                states,
                cfg.horizontal_sep,
                100.0,
                lookahead_time,
            )

            reso = conf_resolution.resolve(
                conf_detection,
                ownship_adsl,
                intruder_adsl,
                asas_marh
            )

            conf_detection.sigma_r = np.sqrt(ownship_adsl.pos_std**2 + intruder_adsl.pos_std**2)
            conf_detection.sigma_v = np.sqrt(ownship_adsl.vel_std**2 + intruder_adsl.vel_std**2)
            conf_detection.dcpa_prob_threshold = threshold_probability

            delpairs_noise = conf_recovery(conf_resolution, conf_detection, ownship_adsl, intruder_adsl)

            missed = (
                int(np.floor((sim_timer - next_event_t) / event_dt)) + 1
                if sim_timer > next_event_t
                else 1
            )
            next_event_t += missed * event_dt

        # --- Step environment ---
        dist = pairwise.step(reso)
        distance_array.append(dist)

        # --- Done logic ---
        done_now, is_active = _check_tcpa_tinhor_per_pair(
            bs.traf.id,
            conf_detection_gt.tcpa_all,
            conf_detection_gt.tinhor_all,
        )

        dist_hist = np.asarray(distance_array)
        min_dist_so_far = np.min(dist_hist, axis=0)

        n_active = int(np.sum(is_active & (min_dist_so_far > 50.0)))

        done_start_time, should_stop = done_with_timeout(
            done_now,
            done_start_time,
            sim_timer,
            cfg.DONE_TIMEOUT,
            verbose=False,
        )

        if should_stop:
            break

        sim_timer += simdt

    # ----------------------------
    # Cleanup & compute IPR
    # ----------------------------
    pairwise.reset()
    distance_array, ipr = _compute_ipr(distance_array, pairwise.nb_pair, cfg.horizontal_sep)

    return distance_array, ipr, sim_timer, n_active


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
    recovery_model           : override config recovery model ("CPA", "FTR", "Probabilistic FTR")
    """
    cfg = get_configs(config_path) if config_path else get_configs()
    if resolution_model is not None:
        cfg.resolution_model = resolution_model
    if recovery_model is not None:
        cfg.recovery_model = recovery_model

    bs.settings.asas_marh = asas_marh

    conf_detection, conf_detection_gt, conf_resolution, conf_recovery = _create_cdr_models(cfg)

    scenario_rng = np.random.default_rng(seed + 7919)
    dist_nm = dist_m / 1852.0
    pairwise = PairwiseHorConflictDist(
        pair_width=cfg.width,
        pair_height=cfg.height,
        asas_pzr_m=cfg.horizontal_sep,
        dtlookahead=lookahead_time * 1.5,
        dist_nm=dist_nm,
        aircraft_type_ownship=cfg.aircraft_type,
        speed_range=(7.7167, 18.0056),  # 15–35 kts in m/s
        dcpa_range_m=(0.0, cfg.horizontal_sep),
        simdt_factor=cfg.SIMDT_FACTOR,
        rng=scenario_rng,
    )

    simdt = bs.settings.simdt * cfg.SIMDT_FACTOR

    adsl_bus, ownship_adsl, intruder_adsl, prev_intruder_adsl = _create_adsl_stack(
        confidence_interval, confidence_interval_velo, reception_prob, seed
    )

    sim_timer = 0.0
    next_event_t = 0.0
    event_dt = float(bs.settings.asas_dt)
    eps = np.finfo(float).eps * 100

    distance_array = []
    done_start_time = None
    initialized = False

    while sim_timer < tmax:
        states = pairwise._get_states()
        n = int(states.ntraf)

        if not initialized:
            ownship_adsl.update_from_truth(states)
            adsl_bus.send_data(intruder_adsl, ownship_adsl, indices=None)
            adsl_bus.send_data(prev_intruder_adsl, intruder_adsl, indices=None)
            initialized = True

        if sim_timer + eps >= next_event_t:
            ownship_adsl.update_from_truth(states)

            idx_rx = intruder_adsl.reception.sample_indices(n)
            rx_mask = np.zeros(n, dtype=bool)
            rx_mask[idx_rx] = True
            idx_miss = np.where(~rx_mask)[0]

            if idx_rx.size > 0:
                adsl_bus.send_data(intruder_adsl, ownship_adsl, indices=idx_rx)
            if idx_miss.size > 0:
                adsl_bus.send_data(intruder_adsl, prev_intruder_adsl, indices=idx_miss)
            adsl_bus.send_data(prev_intruder_adsl, intruder_adsl, indices=None)

            conf_detection.detect(ownship_adsl, intruder_adsl, cfg.horizontal_sep, 100.0, lookahead_time)
            conf_detection_gt.detect(states, states, cfg.horizontal_sep, 100.0, lookahead_time)

            reso = conf_resolution.resolve(conf_detection, ownship_adsl, intruder_adsl, asas_marh)

            conf_detection.sigma_r = np.sqrt(ownship_adsl.pos_std**2 + intruder_adsl.pos_std**2)
            conf_detection.sigma_v = np.sqrt(ownship_adsl.vel_std**2 + intruder_adsl.vel_std**2)
            if threshold_probability is not None:
                conf_detection.dcpa_prob_threshold = threshold_probability

            conf_recovery(conf_resolution, conf_detection, ownship_adsl, intruder_adsl)

            missed = (
                int(np.floor((sim_timer - next_event_t) / event_dt)) + 1
                if sim_timer > next_event_t
                else 1
            )
            next_event_t += missed * event_dt

        dist = pairwise.step(reso)
        distance_array.append(dist)

        done_now, is_active = _check_tcpa_tinhor_per_pair(
            bs.traf.id,
            conf_detection_gt.tcpa_all,
            conf_detection_gt.tinhor_all,
        )

        dist_hist = np.asarray(distance_array)
        min_dist_so_far = np.min(dist_hist, axis=0)
        n_active = int(np.sum(is_active & (min_dist_so_far > cfg.horizontal_sep)))

        done_start_time, should_stop = done_with_timeout(
            done_now, done_start_time, sim_timer, cfg.DONE_TIMEOUT, verbose=False
        )
        if should_stop:
            break

        sim_timer += simdt

    pairwise.reset()
    distance_array, ipr = _compute_ipr(distance_array, pairwise.nb_pair, cfg.horizontal_sep)
    return distance_array, ipr, sim_timer, n_active

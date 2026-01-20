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
from envs.pairwise_conflict import PairwiseHorConflict
from sim_models.cr_mvp import MVP
from sim_models.cr_vo import VO
from sim_models.adsl_module import ADSL
from sim_models.reception_model import ReceptionModel

from sim.utils import (
    _check_tcpa_tinhor_per_pair,
    done_with_timeout,
    get_configs,
)


# ---------------------------------------------------------
# Safe BlueSky initialization (only once)
# ---------------------------------------------------------
if not getattr(bs, "_joblib_inited", False):
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
):
    """
    Run a stochastic pairwise conflict simulation and compute IPR.

    Returns
    -------
    distance_array : np.ndarray
        Distance over time for each pair, shape (T, nb_pair)
    ipr : float
        Intrusion Prevention Rate
    """

    # ----------------------------
    # Load config defaults
    # ----------------------------
    cfg = get_configs()

    width = cfg.width
    height = cfg.height
    horizontal_sep = cfg.horizontal_sep
    init_speed_ownship = cfg.init_speed_ownship
    init_speed_intruder = cfg.init_speed_intruder
    aircraft_type = cfg.aircraft_type
    SIMDT_FACTOR = cfg.SIMDT_FACTOR
    DONE_TIMEOUT = cfg.DONE_TIMEOUT

    # Override ASAS margin
    bs.settings.asas_marh = asas_marh

    # ----------------------------
    # Conflict detection / resolution
    # ----------------------------
    conf_detection = StateBased()
    conf_detection_gt = StateBased()

    if cfg.resolution_model == "MVP":
        conf_resolution = MVP()
    elif cfg.resolution_model == "VO":
        conf_resolution = VO()
    else:
        raise ValueError(f"Unsupported resolution model: {cfg.resolution_model}")

    # ----------------------------
    # Environment
    # ----------------------------
    pairwise = PairwiseHorConflict(
        pair_width=width,
        pair_height=height,
        asas_pzr_m=horizontal_sep,
        dtlookahead=lookahead_time + 1,
        init_speed_ownship=init_speed_ownship,
        init_speed_intruder=init_speed_intruder,
        init_dpsi=dpsi,
        aircraft_type_ownship=aircraft_type,
        simdt_factor=SIMDT_FACTOR,
    )

    simdt = bs.settings.simdt * SIMDT_FACTOR
    tmax = lookahead_time * cfg.tmax_factor

    # ----------------------------
    # ADSL setup
    # ----------------------------
    adsl_bus = ADSL(
        confidence_interval,
        confidence_interval_velo,
        reception_prob=1.0,
        seed=seed + 1,
    )

    ownship_adsl = ADSL(
        confidence_interval,
        confidence_interval_velo,
        reception_prob=1.0,
        seed=seed + 2,
    )

    intruder_adsl = ADSL(
        confidence_interval,
        confidence_interval_velo,
        reception_prob=reception_prob,
        seed=seed + 3,
    )

    prev_intruder_adsl = ADSL(
        confidence_interval,
        confidence_interval_velo,
        reception_prob=1.0,
        seed=seed + 4,
    )

    rx_rng = np.random.default_rng(seed + 999)
    intruder_adsl.reception = ReceptionModel(
        reception_prob=reception_prob,
        rng=rx_rng,
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
                horizontal_sep,
                100.0,
                lookahead_time,
            )

            conf_detection_gt.detect(
                states,
                states,
                horizontal_sep,
                100.0,
                lookahead_time,
            )

            reso = conf_resolution.resolve(
                conf_detection,
                ownship_adsl,
                intruder_adsl,
            )

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
        done_now, n_active = _check_tcpa_tinhor_per_pair(
            bs.traf.id,
            conf_detection_gt.tcpa_all,
            conf_detection_gt.tinhor_all,
        )

        done_start_time, should_stop = done_with_timeout(
            done_now,
            done_start_time,
            sim_timer,
            DONE_TIMEOUT,
            verbose=False,
        )

        if should_stop:
            break

        sim_timer += simdt

    # ----------------------------
    # Cleanup
    # ----------------------------
    pairwise.reset()

    # ----------------------------
    # Compute IPR
    # ----------------------------
    distance_array = np.asarray(distance_array)[:, :pairwise.nb_pair]
    min_dist = np.min(distance_array, axis=0)

    n_los = int(np.sum(min_dist < horizontal_sep))
    n_conflict = int(pairwise.nb_pair)

    ipr = 1.0 - (n_los / float(n_conflict))

    return distance_array, ipr, sim_timer, n_active

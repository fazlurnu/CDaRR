''' Functional simulation shell -- Phase 3 (refactor_fp.md).

New implementation of ``get_ipr_stochastic_env`` built on the pure core
(``cd``/``cr``/``crr``/``cns``/``cdarr``), instead of the legacy
Entity-based ``sim_models`` classes. A deliberately SEPARATE module from
``sim/pairwise_stochastic/get_ipr_stochastic_env.py`` -- the switchover
(aliasing callers to this implementation) is Phase 4's job, not this one;
until then the legacy module stays the untouched, golden-verified reference.

RNG fidelity (refactor_fp.md section 5.2's ledger): only two streams are ever
actually drawn from in the legacy code -- ``seed+2`` (ownship: reception
draw, THEN position noise, THEN velocity noise, every ASAS tick after the
first) and ``seed+999`` (intruder reception). ``seed+1``/``+3``/``+4`` are
constructed in the legacy ADSL stack (one np.random.default_rng each) but
never drawn from -- intruder/bus/prev_intruder data arrives entirely via
relay (pure copy, no RNG), confirmed by reading the live loop line by line:
``intruder_adsl.update_from_truth`` is simply never called. This module does
NOT construct those three dead generators at all (there is nothing for them
to seed here) -- a deliberate omission, not an oversight, and the 16/16
golden-baseline match (test_equiv_loop_golden.py) is the proof that omitting
them has zero observable effect. Only ``ownship_rng`` (seed+2) and ``rx_rng``
(seed+999) exist, each a single persistent ``np.random.Generator`` object
reused (not recreated) for the life of a run, in the identical per-tick draw
order the legacy code used.
'''
from dataclasses import replace

import numpy as np
import bluesky as bs

import cd.statebased
import cns.link
import cns.noise
import cns.reception
from cdarr.core import cdarr, CdarrParams, make_dict_id2idx
from cr.common import ResolutionParams
from crr.common import RecoveryState

from envs.pairwise_conflict import PairwiseHorConflict, PairwiseHorConflictDist
from sim.utils import (
    _check_tcpa_tinhor_per_pair,
    done_with_timeout,
    get_configs,
    suppress_output,
)

if not getattr(bs, "_joblib_inited", False):
    with suppress_output():
        bs.init(mode="sim", detached=True)
        bs._joblib_inited = True

_CI95_TO_STD_2D = 2.448


def _update_node_from_truth(msg, states, first_call, node_reception_prob, rng,
                             pos_cov, vel_cov, pos_dist, pos_ci95, latency_s):
    ''' Mirrors ADSL.update_from_truth exactly -- including its own internal
    reception draw, which every node (ownship included, at reception_prob=1.0)
    performs on every call AFTER the first. ADSL.reception and ADSL.noise
    share one rng object, so this draw consumes the stream even though the
    mask ends up all-True at p=1.0 -- confirmed against
    sim_models/adsl_module.py's ADSL.__init__ (self.reception = ReceptionModel(
    reception_prob=reception_prob, rng=self.rng)) and matches this module's
    own RNG-ledger docstring (F9: "reception rng.random(n) # even at p=1.0").

    Draw order once idx is known: position (incl. latency bias, which draws
    nothing) THEN velocity, both from the same rng -- matching
    add_position_noise then add_velocity_noise's order in update_from_truth.
    '''
    n = int(states.ntraf)
    if first_call:
        idx = np.arange(n, dtype=int)
    else:
        idx = cns.reception.sample_received(n, node_reception_prob, rng)

    msg = cns.link.with_truth(msg, states, idx)

    new_lat, new_lon = cns.noise.position_noise(
        states.lat, states.lon, states.trk, states.gs, idx,
        pos_cov, rng, pos_dist=pos_dist, pos_ci95=pos_ci95, latency_s=latency_s)
    new_gsnorth, new_gseast = cns.noise.velocity_noise(states.trk, states.gs, idx, vel_cov, rng)

    lat = msg.lat.copy(); lat[idx] = new_lat
    lon = msg.lon.copy(); lon[idx] = new_lon
    gsnorth = msg.gsnorth.copy(); gsnorth[idx] = new_gsnorth
    gseast = msg.gseast.copy(); gseast[idx] = new_gseast

    return replace(msg, lat=lat, lon=lon, gsnorth=gsnorth, gseast=gseast)


def _apply_latency_bias(msg, states, idx, latency_s):
    ''' Mirrors NoiseModel.add_latency_bias applied standalone to an
    already-relayed message (the intruder-only, freshly-received-only use
    case -- see cns/noise.py's module docstring). No-op (no rng draw either
    way) if latency_s == 0 or idx is empty. '''
    if idx.size == 0 or not latency_s:
        return msg
    lat_bias, lon_bias = cns.noise.latency_bias(states.lat, states.trk, states.gs, idx, latency_s)
    lat = msg.lat.copy(); lat[idx] = msg.lat[idx] + lat_bias
    lon = msg.lon.copy(); lon[idx] = msg.lon[idx] + lon_bias
    return replace(msg, lat=lat, lon=lon)


def _worldview_sigma(assumed_ci, pos_or_vel_std):
    ''' Combined (over two aircraft) worldview uncertainty for the
    probabilistic recovery, matching get_ipr_stochastic_env.py's
    sigma_r_worldview/sigma_v_worldview construction exactly. '''
    if assumed_ci is None:
        return np.sqrt(pos_or_vel_std**2 + pos_or_vel_std**2)
    return np.sqrt(2.0) * (assumed_ci / _CI95_TO_STD_2D)


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
    ''' Functional-core reimplementation of
    sim.pairwise_stochastic.get_ipr_stochastic_env.get_ipr_stochastic_env.
    Same signature, same return contract (distance_array, ipr, sim_timer,
    n_active); see that module's docstring for parameter semantics. '''
    cfg = get_configs(config_path) if config_path else get_configs()
    if recovery_model is not None:
        cfg.recovery_model = recovery_model

    bs.settings.asas_marh = asas_marh

    resolution_params = ResolutionParams(resofach=asas_marh, resofacv=float(bs.settings.asas_marv))

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
    # CNS stack setup -- mirrors _create_adsl_stack's exact seed layout.
    # Only ownship_rng (seed+2) and rx_rng (seed+999) are ever drawn from;
    # the other three seeds (+1/+3/+4) have no analogue here since bus/
    # prev_intruder never draw noise in the legacy code either -- see this
    # module's docstring.
    # ----------------------------
    pos_std = float(confidence_interval) / _CI95_TO_STD_2D
    vel_std = float(confidence_interval_velo) / _CI95_TO_STD_2D
    pos_cov = cns.noise.make_covariance(pos_std)
    vel_cov = cns.noise.make_covariance(vel_std)
    pos_ci95 = pos_std * _CI95_TO_STD_2D

    ownship_rng = np.random.default_rng(seed + 2)
    rx_rng = np.random.default_rng(seed + 999)

    n0 = int(pairwise._get_states().ntraf)
    ownship_msg = cns.link.empty_message(n0)
    intruder_msg = cns.link.empty_message(n0)
    prev_intruder_msg = cns.link.empty_message(n0)

    cdarr_state = RecoveryState()

    sigma_r_worldview = _worldview_sigma(assumed_confidence_interval, pos_std)
    sigma_v_worldview = _worldview_sigma(assumed_confidence_interval_velo, vel_std)

    cdarr_params = CdarrParams(
        rpz=cfg.horizontal_sep, hpz=100.0, dtlookahead=lookahead_time,
        resolution=cfg.resolution_model, resolution_params=resolution_params,
        recovery=cfg.recovery_model, resofach=asas_marh,
        sigma_r=sigma_r_worldview, sigma_v=sigma_v_worldview,
        prob_threshold=threshold_probability, Ktheta=256,
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
    ownship_first_call = True
    action = None

    while sim_timer < tmax:

        states = pairwise._get_states()
        n = int(states.ntraf)

        if not initialized:
            idx_all = np.arange(n)
            ownship_msg = _update_node_from_truth(
                ownship_msg, states, ownship_first_call, 1.0, ownship_rng,
                pos_cov, vel_cov, pos_dist, pos_ci95, latency_s=0.0)
            ownship_first_call = False
            intruder_msg = cns.link.relay(intruder_msg, ownship_msg, idx=None)
            intruder_msg = _apply_latency_bias(intruder_msg, states, idx_all, latency_s)
            prev_intruder_msg = cns.link.relay(prev_intruder_msg, intruder_msg, idx=None)
            initialized = True

        if sim_timer + eps >= next_event_t:

            # --- CNS update ---
            ownship_msg = _update_node_from_truth(
                ownship_msg, states, ownship_first_call, 1.0, ownship_rng,
                pos_cov, vel_cov, pos_dist, pos_ci95, latency_s=0.0)
            ownship_first_call = False

            idx_rx = cns.reception.sample_received(n, reception_prob, rx_rng)
            rx_mask = np.zeros(n, dtype=bool)
            rx_mask[idx_rx] = True
            idx_miss = np.where(~rx_mask)[0]

            if idx_rx.size > 0:
                intruder_msg = cns.link.relay(intruder_msg, ownship_msg, idx=idx_rx)
                intruder_msg = _apply_latency_bias(intruder_msg, states, idx_rx, latency_s)
            if idx_miss.size > 0:
                intruder_msg = cns.link.relay(intruder_msg, prev_intruder_msg, idx=idx_miss)

            prev_intruder_msg = cns.link.relay(prev_intruder_msg, intruder_msg, idx=None)

            # --- Ground-truth detect (for the done-check) ---
            conf_gt = cd.statebased.detect(states, states, cfg.horizontal_sep, 100.0, lookahead_time)

            # --- cdarr: detect -> resolve -> recover, composed ---
            id2idx = make_dict_id2idx(ownship_msg.id)
            result = cdarr(ownship_msg, intruder_msg, cdarr_state, cdarr_params, id2idx=id2idx)
            cdarr_state = result.state

            # Action tuple shape matches envs.pairwise_conflict._do_action's
            # expectation exactly: (trk, gs, vs, alt, resopairs). _do_action
            # does its own membership test over resopairs, which is now the
            # post-recovery set -- identical to result.avoiding by
            # construction (both read the same new_state.resopairs).
            action = (result.command.trk, result.command.gs_capped,
                      result.command.vs_capped, result.command.alt,
                      set(cdarr_state.resopairs))

            missed = (
                int(np.floor((sim_timer - next_event_t) / event_dt)) + 1
                if sim_timer > next_event_t
                else 1
            )
            next_event_t += missed * event_dt

        # --- Step environment ---
        dist = pairwise.step(action)
        distance_array.append(dist)

        # --- Done logic ---
        done_now, is_active = _check_tcpa_tinhor_per_pair(
            bs.traf.id,
            conf_gt.tcpa_all,
            conf_gt.tinhor_all,
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
    distance_array = np.asarray(distance_array)[:, :pairwise.nb_pair]
    min_dist = np.min(distance_array, axis=0)
    n_los = int(np.sum(min_dist < cfg.horizontal_sep))
    ipr = 1.0 - (n_los / float(pairwise.nb_pair))

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
    ''' Functional-core reimplementation of
    sim.pairwise_stochastic.get_ipr_stochastic_env.get_ipr_stochastic_env_randomized.
    Same signature/contract (including ``randomized_speed_heading``, which the
    legacy function also accepts but never reads inside its body -- the
    dispatch on that flag happens one level up, in run_multiple_jobs; see
    that module's docstring). Differs from get_ipr_stochastic_env only in:
    randomized init speed/dpsi via a seeded scenario_rng, no pos_dist/latency_s
    (always the default Gaussian, zero latency), and worldview uncertainty
    always matched to the true noise (no assumed_confidence_interval support).
    '''
    cfg = get_configs(config_path) if config_path else get_configs()
    if recovery_model is not None:
        cfg.recovery_model = recovery_model

    bs.settings.asas_marh = asas_marh

    resolution_params = ResolutionParams(resofach=asas_marh, resofacv=float(bs.settings.asas_marv))

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

    pos_std = float(confidence_interval) / _CI95_TO_STD_2D
    vel_std = float(confidence_interval_velo) / _CI95_TO_STD_2D
    pos_cov = cns.noise.make_covariance(pos_std)
    vel_cov = cns.noise.make_covariance(vel_std)
    pos_ci95 = pos_std * _CI95_TO_STD_2D

    ownship_rng = np.random.default_rng(seed + 2)
    rx_rng = np.random.default_rng(seed + 999)

    n0 = int(pairwise._get_states().ntraf)
    ownship_msg = cns.link.empty_message(n0)
    intruder_msg = cns.link.empty_message(n0)
    prev_intruder_msg = cns.link.empty_message(n0)

    cdarr_state = RecoveryState()

    sigma_r_worldview = _worldview_sigma(None, pos_std)
    sigma_v_worldview = _worldview_sigma(None, vel_std)

    cdarr_params = CdarrParams(
        rpz=cfg.horizontal_sep, hpz=100.0, dtlookahead=lookahead_time,
        resolution=cfg.resolution_model, resolution_params=resolution_params,
        recovery=cfg.recovery_model, resofach=asas_marh,
        sigma_r=sigma_r_worldview, sigma_v=sigma_v_worldview,
        prob_threshold=threshold_probability, Ktheta=256,
    )

    sim_timer = 0.0
    next_event_t = 0.0
    event_dt = float(bs.settings.asas_dt)
    eps = np.finfo(float).eps * 100

    distance_array = []
    done_start_time = None
    initialized = False
    ownship_first_call = True
    action = None

    while sim_timer < tmax:

        states = pairwise._get_states()
        n = int(states.ntraf)

        if not initialized:
            ownship_msg = _update_node_from_truth(
                ownship_msg, states, ownship_first_call, 1.0, ownship_rng,
                pos_cov, vel_cov, None, pos_ci95, latency_s=0.0)
            ownship_first_call = False
            intruder_msg = cns.link.relay(intruder_msg, ownship_msg, idx=None)
            prev_intruder_msg = cns.link.relay(prev_intruder_msg, intruder_msg, idx=None)
            initialized = True

        if sim_timer + eps >= next_event_t:

            ownship_msg = _update_node_from_truth(
                ownship_msg, states, ownship_first_call, 1.0, ownship_rng,
                pos_cov, vel_cov, None, pos_ci95, latency_s=0.0)
            ownship_first_call = False

            idx_rx = cns.reception.sample_received(n, reception_prob, rx_rng)
            rx_mask = np.zeros(n, dtype=bool)
            rx_mask[idx_rx] = True
            idx_miss = np.where(~rx_mask)[0]

            if idx_rx.size > 0:
                intruder_msg = cns.link.relay(intruder_msg, ownship_msg, idx=idx_rx)
            if idx_miss.size > 0:
                intruder_msg = cns.link.relay(intruder_msg, prev_intruder_msg, idx=idx_miss)

            prev_intruder_msg = cns.link.relay(prev_intruder_msg, intruder_msg, idx=None)

            conf_gt = cd.statebased.detect(states, states, cfg.horizontal_sep, 100.0, lookahead_time)

            id2idx = make_dict_id2idx(ownship_msg.id)
            result = cdarr(ownship_msg, intruder_msg, cdarr_state, cdarr_params, id2idx=id2idx)
            cdarr_state = result.state

            action = (result.command.trk, result.command.gs_capped,
                      result.command.vs_capped, result.command.alt,
                      set(cdarr_state.resopairs))

            missed = (
                int(np.floor((sim_timer - next_event_t) / event_dt)) + 1
                if sim_timer > next_event_t
                else 1
            )
            next_event_t += missed * event_dt

        dist = pairwise.step(action)
        distance_array.append(dist)

        done_now, is_active = _check_tcpa_tinhor_per_pair(
            bs.traf.id,
            conf_gt.tcpa_all,
            conf_gt.tinhor_all,
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

    pairwise.reset()
    distance_array = np.asarray(distance_array)[:, :pairwise.nb_pair]
    min_dist = np.min(distance_array, axis=0)
    n_los = int(np.sum(min_dist < cfg.horizontal_sep))
    ipr = 1.0 - (n_los / float(pairwise.nb_pair))

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
    ''' Functional-core reimplementation of
    sim.pairwise_stochastic.get_ipr_stochastic_env.get_ipr_stochastic_env_dist.
    Same signature/contract; see that module's docstring for parameter
    semantics. Differs from get_ipr_stochastic_env only in using
    PairwiseHorConflictDist (fixed initial separation, per-pair randomized
    dpsi/speed/dcpa via a seeded scenario_rng) and using cfg.horizontal_sep
    (not the hardcoded 50.0) as the n_active distance floor -- both matching
    the legacy function exactly.
    '''
    cfg = get_configs(config_path) if config_path else get_configs()
    if resolution_model is not None:
        cfg.resolution_model = resolution_model
    if recovery_model is not None:
        cfg.recovery_model = recovery_model

    bs.settings.asas_marh = asas_marh

    resolution_params = ResolutionParams(resofach=asas_marh, resofacv=float(bs.settings.asas_marv))

    scenario_rng = np.random.default_rng(seed + 7919)
    dist_nm = dist_m / 1852.0
    pairwise = PairwiseHorConflictDist(
        pair_width=cfg.width,
        pair_height=cfg.height,
        asas_pzr_m=cfg.horizontal_sep,
        dtlookahead=lookahead_time * 1.5,
        dist_nm=dist_nm,
        aircraft_type_ownship=cfg.aircraft_type,
        speed_range=(7.7167, 18.0056),
        dcpa_range_m=(0.0, cfg.horizontal_sep),
        simdt_factor=cfg.SIMDT_FACTOR,
        rng=scenario_rng,
    )

    simdt = bs.settings.simdt * cfg.SIMDT_FACTOR

    pos_std = float(confidence_interval) / _CI95_TO_STD_2D
    vel_std = float(confidence_interval_velo) / _CI95_TO_STD_2D
    pos_cov = cns.noise.make_covariance(pos_std)
    vel_cov = cns.noise.make_covariance(vel_std)
    pos_ci95 = pos_std * _CI95_TO_STD_2D

    ownship_rng = np.random.default_rng(seed + 2)
    rx_rng = np.random.default_rng(seed + 999)

    n0 = int(pairwise._get_states().ntraf)
    ownship_msg = cns.link.empty_message(n0)
    intruder_msg = cns.link.empty_message(n0)
    prev_intruder_msg = cns.link.empty_message(n0)

    cdarr_state = RecoveryState()

    sigma_r_worldview = _worldview_sigma(None, pos_std)
    sigma_v_worldview = _worldview_sigma(None, vel_std)

    cdarr_params = CdarrParams(
        rpz=cfg.horizontal_sep, hpz=100.0, dtlookahead=lookahead_time,
        resolution=cfg.resolution_model, resolution_params=resolution_params,
        recovery=cfg.recovery_model, resofach=asas_marh,
        sigma_r=sigma_r_worldview, sigma_v=sigma_v_worldview,
        prob_threshold=threshold_probability, Ktheta=256,
    )

    sim_timer = 0.0
    next_event_t = 0.0
    event_dt = float(bs.settings.asas_dt)
    eps = np.finfo(float).eps * 100

    distance_array = []
    done_start_time = None
    initialized = False
    ownship_first_call = True
    action = None

    while sim_timer < tmax:
        states = pairwise._get_states()
        n = int(states.ntraf)

        if not initialized:
            ownship_msg = _update_node_from_truth(
                ownship_msg, states, ownship_first_call, 1.0, ownship_rng,
                pos_cov, vel_cov, None, pos_ci95, latency_s=0.0)
            ownship_first_call = False
            intruder_msg = cns.link.relay(intruder_msg, ownship_msg, idx=None)
            prev_intruder_msg = cns.link.relay(prev_intruder_msg, intruder_msg, idx=None)
            initialized = True

        if sim_timer + eps >= next_event_t:
            ownship_msg = _update_node_from_truth(
                ownship_msg, states, ownship_first_call, 1.0, ownship_rng,
                pos_cov, vel_cov, None, pos_ci95, latency_s=0.0)
            ownship_first_call = False

            idx_rx = cns.reception.sample_received(n, reception_prob, rx_rng)
            rx_mask = np.zeros(n, dtype=bool)
            rx_mask[idx_rx] = True
            idx_miss = np.where(~rx_mask)[0]

            if idx_rx.size > 0:
                intruder_msg = cns.link.relay(intruder_msg, ownship_msg, idx=idx_rx)
            if idx_miss.size > 0:
                intruder_msg = cns.link.relay(intruder_msg, prev_intruder_msg, idx=idx_miss)
            prev_intruder_msg = cns.link.relay(prev_intruder_msg, intruder_msg, idx=None)

            conf_gt = cd.statebased.detect(states, states, cfg.horizontal_sep, 100.0, lookahead_time)

            id2idx = make_dict_id2idx(ownship_msg.id)
            result = cdarr(ownship_msg, intruder_msg, cdarr_state, cdarr_params, id2idx=id2idx)
            cdarr_state = result.state

            action = (result.command.trk, result.command.gs_capped,
                      result.command.vs_capped, result.command.alt,
                      set(cdarr_state.resopairs))

            missed = (
                int(np.floor((sim_timer - next_event_t) / event_dt)) + 1
                if sim_timer > next_event_t
                else 1
            )
            next_event_t += missed * event_dt

        dist = pairwise.step(action)
        distance_array.append(dist)

        done_now, is_active = _check_tcpa_tinhor_per_pair(
            bs.traf.id,
            conf_gt.tcpa_all,
            conf_gt.tinhor_all,
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
    distance_array = np.asarray(distance_array)[:, :pairwise.nb_pair]
    min_dist = np.min(distance_array, axis=0)
    n_los = int(np.sum(min_dist < cfg.horizontal_sep))
    ipr = 1.0 - (n_los / float(pairwise.nb_pair))

    return distance_array, ipr, sim_timer, n_active

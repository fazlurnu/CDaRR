"""L1 equivalence: crr.probabilistic_ftr.step vs. the legacy
sim_models.crr_resumenav_probabilistic_ftr.resumenav_probabilistic_ftr.

The legacy function reads sigma_r/sigma_v/dcpa_prob_threshold/dcpa_prob_Ktheta by
smuggled attribute lookup on `conf` (conf.sigma_r = ..., set by the sim loop before
calling recovery) -- impossible with the new frozen ConflictData, and exactly the
"no hidden parameter channels" change crr/probabilistic_ftr.py's docstring explains.
So here: the OLD side uses a mutable `sim_models.cd_statebased.StateBased` instance
with those attributes set directly; the NEW side passes the *same* numeric values as
explicit keyword arguments. Both must produce identical decisions.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import bluesky as bs

from cd.statebased import detect as new_detect
from crr.common import RecoveryState, apply_recovery
from crr.probabilistic_ftr import step as new_step

if not getattr(bs, "_joblib_inited", False):
    bs.init(mode="sim", detached=True)
    bs._joblib_inited = True

# Representative worldview uncertainty, matching exp3/4/5's usage: CI95=10m pos,
# CI95=1 m/s vel, combined over two aircraft (sqrt(2) * ci/2.448), gamma=0.999,
# Ktheta=256 (experiments/config.py's DEFAULT_GAMMA and the module default).
_CI95_TO_STD = 2.448
SIGMA_R = np.sqrt(2.0) * (10.0 / _CI95_TO_STD)
SIGMA_V = np.sqrt(2.0) * (1.0 / _CI95_TO_STD)
PROB_THRESHOLD = 0.999
KTHETA = 256


def _make_env(dpsi, width=2, height=2):
    from envs.pairwise_conflict import PairwiseHorConflict
    return PairwiseHorConflict(
        pair_width=width, pair_height=height,
        asas_pzr_m=50, dtlookahead=120,
        init_speed_ownship=10.2889, init_speed_intruder=10.2889,
        init_dpsi=dpsi, aircraft_type_ownship="M600",
        simdt_factor=4.0,
    )


def _old_detect_with_worldview(states, rpz, hpz, dtlookahead):
    from sim_models.cd_statebased import StateBased
    old_conf = StateBased()
    old_conf.detect(states, states, rpz, hpz, dtlookahead)
    # Mirrors get_ipr_stochastic_env.py's attribute-smuggling exactly.
    old_conf.sigma_r = SIGMA_R
    old_conf.sigma_v = SIGMA_V
    old_conf.dcpa_prob_threshold = PROB_THRESHOLD
    old_conf.dcpa_prob_Ktheta = KTHETA
    return old_conf


def _old_tick(old_conf, ownship, intruder, reso):
    from sim_models.crr_resumenav_probabilistic_ftr import resumenav_probabilistic_ftr as old_step
    old_step(reso, old_conf, ownship, intruder)
    return reso


def _new_tick(conf, ownship, intruder, ntraf, state):
    new_state, changeactive, _delpairs = new_step(
        conf, ownship, intruder, state,
        sigma_r=SIGMA_R, sigma_v=SIGMA_V, prob_threshold=PROB_THRESHOLD, Ktheta=KTHETA,
    )
    active = np.zeros(ntraf, dtype=bool)
    apply_recovery(changeactive, active)
    return new_state, active


@pytest.mark.fast
@pytest.mark.parametrize("dpsi", [2, 90, 180])
def test_probabilistic_ftr_matches_legacy_two_ticks(dpsi):
    env = _make_env(dpsi)
    try:
        for _ in range(5):
            env.step(None)
        states = env._get_states()
        ntraf = states.ntraf

        old_conf1 = _old_detect_with_worldview(states, 50, 100.0, 15)
        new_conf1 = new_detect(states, states, 50, 100.0, 15)

        reso = SimpleNamespace(resopairs=set(), active=np.zeros(ntraf, dtype=bool))
        state = RecoveryState()

        _old_tick(old_conf1, states, states, reso)
        state, _active1 = _new_tick(new_conf1, states, states, ntraf, state)

        assert reso.resopairs == set(state.resopairs), "tick 1 resopairs mismatch"

        for _ in range(10):
            env.step(None)
        states = env._get_states()

        old_conf2 = _old_detect_with_worldview(states, 50, 100.0, 15)
        new_conf2 = new_detect(states, states, 50, 100.0, 15)

        _old_tick(old_conf2, states, states, reso)
        new_state2, active2 = _new_tick(new_conf2, states, states, ntraf, state)

        assert reso.resopairs == set(new_state2.resopairs), "tick 2 resopairs mismatch"
        assert np.array_equal(reso.active, active2), "tick 2 active mismatch"
        assert dict(reso._intr_init_vel) == dict(new_state2.init_vel), "init_vel mismatch"
    finally:
        env.reset()


@pytest.mark.fast
def test_probabilistic_ftr_matches_legacy_empty():
    env = _make_env(dpsi=90)
    try:
        states = env._get_states()
        ntraf = states.ntraf

        old_conf = _old_detect_with_worldview(states, 50, 100.0, 1e-9)
        new_conf = new_detect(states, states, 50, 100.0, 1e-9)
        assert old_conf.confpairs == [] and new_conf.confpairs == []

        reso = SimpleNamespace(resopairs=set(), active=np.zeros(ntraf, dtype=bool))
        state = RecoveryState()

        _old_tick(old_conf, states, states, reso)
        new_state, active = _new_tick(new_conf, states, states, ntraf, state)

        assert reso.resopairs == set(new_state.resopairs) == set()
        assert np.array_equal(reso.active, active)
    finally:
        env.reset()

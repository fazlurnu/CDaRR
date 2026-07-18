"""L1 equivalence: crr.ftr.step (+ crr.common.apply_recovery) vs. the legacy
sim_models.crr_resumenav_ftr.resumenav_double_criteria.

Calls both paths twice in sequence (mirroring how the live loop persists recovery
state tick-to-tick) so the "intruder reverts to velocity at conflict initiation
(Vi,i)" criterion actually differs from "intruder holds current velocity (Vi,c)" --
a single isolated call would have every pair look brand new, with Vi,i defaulting
to Vi,c, never exercising the memory path.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import bluesky as bs

from cd.statebased import detect as new_detect
from crr.common import RecoveryState, apply_recovery
from crr.ftr import step as new_step

if not getattr(bs, "_joblib_inited", False):
    bs.init(mode="sim", detached=True)
    bs._joblib_inited = True


def _make_env(dpsi, width=2, height=2):
    from envs.pairwise_conflict import PairwiseHorConflict
    return PairwiseHorConflict(
        pair_width=width, pair_height=height,
        asas_pzr_m=50, dtlookahead=120,
        init_speed_ownship=10.2889, init_speed_intruder=10.2889,
        init_dpsi=dpsi, aircraft_type_ownship="M600",
        simdt_factor=4.0,
    )


def _old_tick(conf, ownship, intruder, ntraf, reso):
    from sim_models.crr_resumenav_ftr import resumenav_double_criteria as old_step
    old_step(reso, conf, ownship, intruder)
    return reso


def _new_tick(conf, ownship, intruder, ntraf, state):
    new_state, changeactive, _delpairs = new_step(conf, ownship, intruder, state)
    active = np.zeros(ntraf, dtype=bool)
    apply_recovery(changeactive, active)
    return new_state, active


@pytest.mark.fast
@pytest.mark.parametrize("dpsi", [2, 90, 180])
def test_ftr_recovery_matches_legacy_two_ticks(dpsi):
    env = _make_env(dpsi)
    try:
        for _ in range(5):
            env.step(None)
        states = env._get_states()
        ntraf = states.ntraf
        conf1 = new_detect(states, states, 50, 100.0, 15)

        reso = SimpleNamespace(resopairs=set(), active=np.zeros(ntraf, dtype=bool))
        state = RecoveryState()

        _old_tick(conf1, states, states, ntraf, reso)
        state, _active1 = _new_tick(conf1, states, states, ntraf, state)

        assert reso.resopairs == set(state.resopairs), "tick 1 resopairs mismatch"

        # Advance the sim so intruder velocity moves (MVP resolution kicks in),
        # so tick 2's Vi,c differs from tick 1's recorded Vi,i for surviving pairs.
        for _ in range(10):
            env.step(None)
        states = env._get_states()
        conf2 = new_detect(states, states, 50, 100.0, 15)

        _old_tick(conf2, states, states, ntraf, reso)
        new_state2, active2 = _new_tick(conf2, states, states, ntraf, state)

        assert reso.resopairs == set(new_state2.resopairs), "tick 2 resopairs mismatch"
        assert np.array_equal(reso.active, active2), "tick 2 active mismatch"
        # init_vel bookkeeping must also match exactly (it's read by tick 3+).
        assert dict(reso._intr_init_vel) == dict(new_state2.init_vel), "init_vel mismatch"
    finally:
        env.reset()


@pytest.mark.fast
def test_ftr_recovery_matches_legacy_empty():
    env = _make_env(dpsi=90)
    try:
        states = env._get_states()
        ntraf = states.ntraf
        conf = new_detect(states, states, 50, 100.0, 1e-9)
        assert conf.confpairs == []

        reso = SimpleNamespace(resopairs=set(), active=np.zeros(ntraf, dtype=bool))
        state = RecoveryState()

        _old_tick(conf, states, states, ntraf, reso)
        new_state, active = _new_tick(conf, states, states, ntraf, state)

        assert reso.resopairs == set(new_state.resopairs) == set()
        assert np.array_equal(reso.active, active)
    finally:
        env.reset()

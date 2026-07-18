"""L1 equivalence: crr.cpa.step (+ crr.common.apply_recovery) vs. the legacy
sim_models.crr_resumenav_cpa.resumenav.

The legacy function expects a `reso`-like object with .resopairs (mutable set),
.resofach, .active (bool array) -- a lightweight SimpleNamespace stands in. Both
paths run against the real bluesky env (findact() always returns -1 here, since no
route is ever set -- see refactor_fp.md's F4 note -- so the waypoint-recovery side
effect is a no-op for both, safe to exercise identically).
"""
from types import SimpleNamespace

import numpy as np
import pytest
import bluesky as bs

from cd.statebased import detect as new_detect
from crr.common import RecoveryState, apply_recovery
from crr.cpa import step as new_step

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


def _old_resumenav(conf, ownship, intruder, resofach, ntraf, prior_resopairs):
    from sim_models.crr_resumenav_cpa import resumenav as old_resumenav
    reso = SimpleNamespace(
        resopairs=set(prior_resopairs),
        resofach=resofach,
        active=np.zeros(ntraf, dtype=bool),
    )
    old_resumenav(reso, conf, ownship, intruder)
    return reso.resopairs, reso.active


def _new_resumenav(conf, ownship, intruder, resofach, ntraf, prior_resopairs):
    state = RecoveryState(resopairs=frozenset(prior_resopairs), init_vel={})
    new_state, changeactive, _delpairs = new_step(conf, ownship, intruder, state, resofach)
    active = np.zeros(ntraf, dtype=bool)
    apply_recovery(changeactive, active)
    return set(new_state.resopairs), active


@pytest.mark.fast
@pytest.mark.parametrize("dpsi,steps", [
    (2, 5),      # near-parallel, likely still in conflict / hor_los
    (90, 5),     # crossing, mid-resolution
    (90, 40),    # crossing, well past CPA -- exercises the release branch
    (180, 5),    # head-on
])
def test_cpa_recovery_matches_legacy(dpsi, steps):
    env = _make_env(dpsi)
    try:
        for _ in range(steps):
            env.step(None)
        states = env._get_states()
        ntraf = states.ntraf
        conf = new_detect(states, states, 50, 100.0, 15)
        resofach = float(bs.settings.asas_marh)

        # Prior resopairs = whatever's currently in conflict, so both the "new
        # pair this tick" and "already tracked" branches get exercised.
        prior = set(conf.confpairs)

        old_resopairs, old_active = _old_resumenav(conf, states, states, resofach, ntraf, prior)
        new_resopairs, new_active = _new_resumenav(conf, states, states, resofach, ntraf, prior)

        assert old_resopairs == new_resopairs, "resopairs mismatch"
        assert np.array_equal(old_active, new_active), "active mismatch"
    finally:
        env.reset()


@pytest.mark.fast
def test_cpa_recovery_matches_legacy_empty():
    """No conflicts at all -> resopairs stays empty, no active changes."""
    env = _make_env(dpsi=90)
    try:
        states = env._get_states()
        ntraf = states.ntraf
        conf = new_detect(states, states, 50, 100.0, 1e-9)
        assert conf.confpairs == []

        resofach = float(bs.settings.asas_marh)
        old_resopairs, old_active = _old_resumenav(conf, states, states, resofach, ntraf, set())
        new_resopairs, new_active = _new_resumenav(conf, states, states, resofach, ntraf, set())

        assert old_resopairs == new_resopairs == set()
        assert np.array_equal(old_active, new_active)
    finally:
        env.reset()

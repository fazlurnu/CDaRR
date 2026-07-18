"""L1 equivalence: cr.vo.resolve vs. the legacy sim_models.cr_vo.VO.

VO's resolve() has the F6 quirk (refactor_fp.md): it calls a built-in
past-CPA-style release step internally, on top of whatever recovery model the
sim loop calls afterward -- so this test checks not just the (trk, gs_capped,
vs_capped, alt) command but also the resulting resopairs, across two ticks (so
the release branch, not just the "just detected" branch, gets exercised).
"""
from types import SimpleNamespace

import numpy as np
import pytest
import bluesky as bs

from cd.statebased import detect as new_detect
from cr.common import ResolutionParams
from cr.vo import resolve as new_resolve
from crr.common import RecoveryState

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


def _old_resolve(conf, ownship, intruder, resofach, old_vo):
    return old_vo.resolve(conf, ownship, intruder, resofach)


@pytest.mark.fast
@pytest.mark.parametrize("dpsi", [2, 90, 180])
def test_vo_resolve_matches_legacy_two_ticks(dpsi):
    from sim_models.cr_vo import VO
    env = _make_env(dpsi)
    try:
        for _ in range(5):
            env.step(None)
        states = env._get_states()
        conf1 = new_detect(states, states, 50, 100.0, 15)

        resofach = float(bs.settings.asas_marh)
        resofacv = float(bs.settings.asas_marv)

        old_vo = VO()
        old_vo.resopairs = set()  # KI-1: singleton, start clean

        old_trk1, old_gs1, old_vs1, old_alt1, old_resopairs1 = _old_resolve(
            conf1, states, states, resofach, old_vo)

        params = ResolutionParams(resofach=resofach, resofacv=resofacv)
        new_state = RecoveryState()
        new_cmd1, new_state, _changeactive1 = new_resolve(conf1, states, states, params, new_state)

        assert np.array_equal(old_trk1, new_cmd1.trk), "tick1 trk mismatch"
        assert np.array_equal(old_gs1, new_cmd1.gs_capped), "tick1 gs mismatch"
        assert np.array_equal(old_vs1, new_cmd1.vs_capped), "tick1 vs mismatch"
        assert np.array_equal(old_alt1, new_cmd1.alt), "tick1 alt mismatch"
        assert set(old_resopairs1) == set(new_state.resopairs), "tick1 resopairs mismatch"

        # Advance further so some pairs may pass CPA and release.
        for _ in range(15):
            env.step(None)
        states = env._get_states()
        conf2 = new_detect(states, states, 50, 100.0, 15)

        old_trk2, old_gs2, old_vs2, old_alt2, old_resopairs2 = _old_resolve(
            conf2, states, states, resofach, old_vo)
        new_cmd2, new_state2, _changeactive2 = new_resolve(conf2, states, states, params, new_state)

        assert np.array_equal(old_trk2, new_cmd2.trk), "tick2 trk mismatch"
        assert np.array_equal(old_gs2, new_cmd2.gs_capped), "tick2 gs mismatch"
        assert np.array_equal(old_vs2, new_cmd2.vs_capped), "tick2 vs mismatch"
        assert np.array_equal(old_alt2, new_cmd2.alt), "tick2 alt mismatch"
        assert set(old_resopairs2) == set(new_state2.resopairs), "tick2 resopairs mismatch"
    finally:
        env.reset()


@pytest.mark.fast
def test_vo_resolve_matches_legacy_no_conflict():
    from sim_models.cr_vo import VO
    env = _make_env(dpsi=90)
    try:
        states = env._get_states()
        conf = new_detect(states, states, 50, 100.0, 1e-9)
        assert conf.confpairs == []

        resofach = float(bs.settings.asas_marh)
        resofacv = float(bs.settings.asas_marv)

        old_vo = VO()
        old_vo.resopairs = set()
        old_trk, old_gs, old_vs, old_alt, old_resopairs = _old_resolve(
            conf, states, states, resofach, old_vo)

        params = ResolutionParams(resofach=resofach, resofacv=resofacv)
        new_cmd, new_state, _changeactive = new_resolve(
            conf, states, states, params, RecoveryState())

        assert np.array_equal(old_trk, new_cmd.trk)
        assert np.array_equal(old_gs, new_cmd.gs_capped)
        assert np.array_equal(old_vs, new_cmd.vs_capped)
        assert np.array_equal(old_alt, new_cmd.alt)
        assert set(old_resopairs) == set(new_state.resopairs) == set()
    finally:
        env.reset()

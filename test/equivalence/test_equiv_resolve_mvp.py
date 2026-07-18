"""L1 equivalence: cr.mvp.resolve vs. the legacy sim_models.cr_mvp.MVP.

Feeds the *same* ConflictData (produced by the already-verified cd.statebased.detect,
see test_equiv_detect.py) into both the legacy Entity-based MVP class and the new pure
resolve(), and asserts (trk, gs_capped, vs_capped, alt) are bitwise equal. ConflictData
is field-name-compatible with the legacy StateBased instance (see cd/common.py), so
the same conf object can be handed to both -- neither resolver ever writes to it.
"""
import numpy as np
import pytest
import bluesky as bs

from cd.statebased import detect as new_detect
from cr.common import ResolutionParams
from cr.mvp import resolve as new_resolve

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


def _old_resolve(conf, ownship, intruder, resofach):
    from sim_models.cr_mvp import MVP
    old = MVP()
    # KI-1: MVP is a bluesky Entity singleton -- reset the recovery bookkeeping
    # resolve() reads/writes noresoac/resooffac (sized by settrafarrays, not
    # resopairs/_intr_init_vel) so this reset isn't required for resolve() itself,
    # but keeps the instance clean in case a later test in the same process reads it.
    old.resopairs = set()
    return old.resolve(conf, ownship, intruder, resofach)


@pytest.mark.fast
@pytest.mark.parametrize("dpsi,label", [
    (2, "near_parallel"),
    (90, "crossing_90"),
    (180, "head_on"),
])
def test_resolve_matches_legacy(dpsi, label):
    env = _make_env(dpsi)
    try:
        for _ in range(5):
            env.step(None)
        states = env._get_states()
        conf = new_detect(states, states, 50, 100.0, 15)

        resofach = float(bs.settings.asas_marh)
        resofacv = float(bs.settings.asas_marv)

        old_trk, old_gs, old_vs, old_alt, _old_resopairs = _old_resolve(conf, states, states, resofach)

        params = ResolutionParams(resofach=resofach, resofacv=resofacv)
        new_cmd = new_resolve(conf, states, states, params)

        assert np.array_equal(old_trk, new_cmd.trk), "trk mismatch"
        assert np.array_equal(old_gs, new_cmd.gs_capped), "gs_capped mismatch"
        assert np.array_equal(old_vs, new_cmd.vs_capped), "vs_capped mismatch"
        assert np.array_equal(old_alt, new_cmd.alt), "alt mismatch"
    finally:
        env.reset()


@pytest.mark.fast
def test_resolve_matches_legacy_no_conflict():
    """Empty confpairs -> dv stays all-zero; both paths should be pure pass-through."""
    env = _make_env(dpsi=90)
    try:
        states = env._get_states()
        conf = new_detect(states, states, 50, 100.0, 1e-9)
        assert conf.confpairs == []

        resofach = float(bs.settings.asas_marh)
        resofacv = float(bs.settings.asas_marv)

        old_trk, old_gs, old_vs, old_alt, _ = _old_resolve(conf, states, states, resofach)
        params = ResolutionParams(resofach=resofach, resofacv=resofacv)
        new_cmd = new_resolve(conf, states, states, params)

        assert np.array_equal(old_trk, new_cmd.trk)
        assert np.array_equal(old_gs, new_cmd.gs_capped)
        assert np.array_equal(old_vs, new_cmd.vs_capped)
        assert np.array_equal(old_alt, new_cmd.alt)
    finally:
        env.reset()

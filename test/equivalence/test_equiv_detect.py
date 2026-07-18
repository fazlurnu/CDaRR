"""L1 equivalence: cd.statebased.detect vs. the legacy sim_models.cd_statebased.StateBased.

Replays the same (ownship, intruder, rpz, hpz, dtlookahead) through both the old,
attribute-mutating class and the new, pure function, and asserts every output field is
bitwise equal. This is what actually proves the Phase 2 port didn't drift from the
Phase 1 extraction it was copied from (refactor_fp.md L1).
"""
import numpy as np
import pytest
import bluesky as bs

from cd.statebased import detect as new_detect
from cd.common import ConflictData

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


def _old_detect(ownship, intruder, rpz, hpz, dtlookahead):
    from sim_models.cd_statebased import StateBased
    old = StateBased()
    old.detect(ownship, intruder, rpz, hpz, dtlookahead)
    return old


def _assert_field_equal(name, old_val, new_val):
    if isinstance(old_val, np.ndarray):
        assert np.array_equal(old_val, new_val), f"{name} mismatch"
    elif isinstance(old_val, (list, tuple)):
        assert list(old_val) == list(new_val), f"{name} mismatch"
    elif isinstance(old_val, (set, frozenset)):
        assert set(old_val) == set(new_val), f"{name} mismatch"
    else:
        assert old_val == new_val, f"{name} mismatch"


FIELDS = ["rpz", "hpz", "dtlookahead", "confpairs", "confpairs_unique", "lospairs",
          "qdr", "dist", "dcpa", "tcpa", "tLOS", "inconf", "tcpamax",
          "tcpa_all", "tinhor_all"]


def _compare(old, new: ConflictData):
    for f in FIELDS:
        _assert_field_equal(f, getattr(old, f), getattr(new, f))


@pytest.mark.fast
@pytest.mark.parametrize("dpsi,label", [
    (2, "near_parallel"),
    (90, "crossing_90"),
    (180, "head_on"),
])
def test_detect_matches_legacy_stepped(dpsi, label):
    """2x2 grid, several sim steps in (non-trivial geometry), observed-vs-observed."""
    env = _make_env(dpsi)
    try:
        for _ in range(5):
            env.step(None)
        states = env._get_states()
        old = _old_detect(states, states, 50, 100.0, 15)
        new = new_detect(states, states, 50, 100.0, 15)
        _compare(old, new)
    finally:
        env.reset()


@pytest.mark.fast
def test_detect_matches_legacy_gt_mode():
    """own == intr (the ground-truth detector's calling convention)."""
    env = _make_env(dpsi=90)
    try:
        for _ in range(3):
            env.step(None)
        states = env._get_states()
        old = _old_detect(states, states, 50, 100.0, 120)
        new = new_detect(states, states, 50, 100.0, 120)
        _compare(old, new)
    finally:
        env.reset()


@pytest.mark.fast
def test_detect_matches_legacy_empty_conflict():
    """dtlookahead effectively 0 -> no pair should ever qualify as a conflict."""
    env = _make_env(dpsi=90)
    try:
        states = env._get_states()
        old = _old_detect(states, states, 50, 100.0, 1e-9)
        new = new_detect(states, states, 50, 100.0, 1e-9)
        assert old.confpairs == [] and new.confpairs == []
        _compare(old, new)
    finally:
        env.reset()

"""L1 equivalence: cns.link vs. the legacy sim_models.adsl_message.ADSLMessage.

ADSLMessage is pure numpy (no bluesky dependency), so this uses lightweight
synthetic truth/message objects. Compares Message/with_truth/relay against
ADSLMessage's ensure_size + copy_from_states/copy_from_message, in the
fixed-size usage pattern the real pipeline actually follows (aircraft count
never changes mid-run -- see cns/link.py's module docstring).
"""
from types import SimpleNamespace

import numpy as np
import pytest

from cns.link import Message, empty_message, with_truth, relay
from sim_models.adsl_message import ADSLMessage

N = 6


def _truth(seed):
    rng = np.random.default_rng(seed)
    return SimpleNamespace(
        ntraf=N,
        lat=rng.uniform(40, 41, N), lon=rng.uniform(-70, -69, N),
        alt=rng.uniform(50, 150, N), hdg=rng.uniform(0, 360, N),
        trk=rng.uniform(0, 360, N), gs=rng.uniform(5, 15, N),
        tas=rng.uniform(5, 15, N), vs=rng.uniform(-2, 2, N),
        gseast=rng.uniform(-10, 10, N), gsnorth=rng.uniform(-10, 10, N),
        id=[f"AC{i:03d}" for i in range(N)],
        perf=object(), ap=object(), selalt=rng.uniform(50, 150, N),
    )


def _old_msg_state(msg: ADSLMessage):
    return dict(lat=msg.lat.copy(), lon=msg.lon.copy(), alt=msg.alt.copy(),
                hdg=msg.hdg.copy(), trk=msg.trk.copy(), gs=msg.gs.copy(),
                tas=msg.tas.copy(), vs=msg.vs.copy(),
                gseast=msg.gseast.copy(), gsnorth=msg.gsnorth.copy(),
                id=list(msg.id))


def _new_msg_state(msg: Message):
    return dict(lat=msg.lat, lon=msg.lon, alt=msg.alt, hdg=msg.hdg, trk=msg.trk,
                gs=msg.gs, tas=msg.tas, vs=msg.vs, gseast=msg.gseast, gsnorth=msg.gsnorth,
                id=list(msg.id))


def _assert_states_equal(old_state, new_state):
    for k in old_state:
        if k == "id":
            assert old_state[k] == new_state[k], f"{k} mismatch"
        else:
            assert np.array_equal(old_state[k], new_state[k]), f"{k} mismatch"


@pytest.mark.fast
def test_with_truth_matches_legacy_full_and_partial():
    states = _truth(1)

    old = ADSLMessage()
    old.ensure_size(N)
    old.copy_from_states(states, np.arange(N))

    new = empty_message(N)
    new = with_truth(new, states, np.arange(N))

    _assert_states_equal(_old_msg_state(old), _new_msg_state(new))
    assert old.perf is states.perf and new.perf is states.perf
    assert old.ap is states.ap and new.ap is states.ap

    # Partial update: only some indices.
    states2 = _truth(2)
    idx = np.array([0, 2, 4])
    old.copy_from_states(states2, idx)
    new = with_truth(new, states2, idx)
    _assert_states_equal(_old_msg_state(old), _new_msg_state(new))


@pytest.mark.fast
def test_with_truth_wrong_size_raises():
    states = _truth(3)
    msg = empty_message(N - 1)
    with pytest.raises(ValueError):
        with_truth(msg, states, np.arange(N))


@pytest.mark.fast
def test_relay_full_copy_matches_legacy():
    states = _truth(4)
    old_src = ADSLMessage(); old_src.ensure_size(N); old_src.copy_from_states(states, np.arange(N))
    new_src = with_truth(empty_message(N), states, np.arange(N))

    old_dst = ADSLMessage(); old_dst.ensure_size(N)
    old_dst.copy_from_message(old_src, idx=None)
    new_dst = relay(empty_message(N), new_src, idx=None)

    _assert_states_equal(_old_msg_state(old_dst), _new_msg_state(new_dst))
    assert old_dst.perf is old_src.perf and new_dst.perf is new_src.perf


@pytest.mark.fast
def test_relay_partial_copy_matches_legacy_preserves_perf():
    """Partial (idx-provided) relay must NOT touch perf/ap/selalt on the dst --
    this is the trickiest part of copy_from_message to replicate faithfully."""
    states_dst = _truth(5)
    states_src = _truth(6)

    old_dst = ADSLMessage(); old_dst.ensure_size(N); old_dst.copy_from_states(states_dst, np.arange(N))
    old_src = ADSLMessage(); old_src.ensure_size(N); old_src.copy_from_states(states_src, np.arange(N))

    new_dst = with_truth(empty_message(N), states_dst, np.arange(N))
    new_src = with_truth(empty_message(N), states_src, np.arange(N))

    idx = np.array([1, 3, 5])
    old_dst.copy_from_message(old_src, idx=idx)
    new_dst2 = relay(new_dst, new_src, idx=idx)

    _assert_states_equal(_old_msg_state(old_dst), _new_msg_state(new_dst2))
    # perf/ap/selalt must be dst's original (states_dst's), NOT src's -- the
    # legacy idx-provided branch never touches them.
    assert old_dst.perf is states_dst.perf
    assert new_dst2.perf is new_dst.perf is states_dst.perf


@pytest.mark.fast
def test_relay_mismatched_size_falls_back_to_full_copy():
    states = _truth(7)
    src = with_truth(empty_message(N), states, np.arange(N))
    dst = empty_message(N - 1)  # deliberately wrong size

    old_dst = ADSLMessage(); old_dst.ensure_size(N - 1)
    old_src = ADSLMessage(); old_src.ensure_size(N); old_src.copy_from_states(states, np.arange(N))
    old_dst.copy_from_message(old_src, idx=np.array([0]))  # size mismatch -> full copy

    new_dst = relay(dst, src, idx=np.array([0]))

    assert new_dst.ntraf == N  # grew to match src, matching the legacy fallback
    _assert_states_equal(_old_msg_state(old_dst), _new_msg_state(new_dst))

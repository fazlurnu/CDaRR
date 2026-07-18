'''Shared, pure building blocks for conflict-recovery (resume-navigation).

The recovery models decide which resolved conflicts may be released and which
aircraft should resume their route. This module holds the pieces they share:

* :class:`RecoveryState` -- the immutable, explicitly-threaded book-keeping
  (``resopairs`` still before CPA, and each intruder's velocity at conflict
  initiation). Replaces the mutable ``reso.resopairs`` / ``reso._intr_init_vel``
  pair the legacy Entity classes carried.
* small pure maths/geometry helpers, fresh copies of
  ``sim_models/crr_recovery_base.py``'s already-pure functions.

The impure parts -- looking up aircraft indices and commanding waypoint
recovery -- are injected as callables (:func:`default_id2idx`,
:func:`default_recover`) so the decision logic in cpa.py/ftr.py/
probabilistic_ftr.py stays testable in isolation, with no bluesky dependency.
'''
from dataclasses import dataclass, field, replace
from typing import Callable, Mapping, Tuple

import numpy as np


@dataclass(frozen=True)
class RecoveryState:
    '''Immutable state threaded through the recovery models.

    ``resopairs`` are conflicts that have been resolved but whose CPA is still
    ahead; ``init_vel`` records each intruder's velocity at conflict initiation
    (used by the "intruder reverts" criterion in ftr.py / probabilistic_ftr.py).
    Every "mutation" the legacy code performed on these (``.update()``,
    ``.pop()``, item assignment) becomes a fresh replacement object here.
    '''
    resopairs: frozenset = frozenset()
    init_vel: Mapping = field(default_factory=dict)


def empty_recovery_state() -> RecoveryState:
    '''A fresh, empty RecoveryState -- matches a brand-new MVP()/VO() instance's
    resopairs=set(), no _intr_init_vel attribute yet.'''
    return RecoveryState(frozenset(), {})


def default_id2idx(pair: Tuple[str, str]):
    '''Real production id2idx: looks up both aircraft of `pair` in the live
    bluesky traffic set. Impure -- the injectable default for step().'''
    import bluesky as bs
    return bs.traf.id2idx(pair)


def default_recover(idx: int) -> None:
    '''Real production waypoint-recovery side effect for aircraft `idx`: send it
    to the next active waypoint if one exists. Impure -- the injectable default
    for apply_recovery(). Mirrors the legacy apply_recovery's route.direct()
    call exactly (findact() returning -1, e.g. because no route was ever set,
    is a no-op here, same as the legacy code -- see refactor_fp.md's F4 note).
    '''
    import bluesky as bs
    iwpid = bs.traf.ap.route[idx].findact(idx)
    if iwpid != -1:
        bs.traf.ap.route[idx].direct(idx, bs.traf.ap.route[idx].wpname[iwpid])


def apply_recovery(changeactive: Mapping[int, bool], active: np.ndarray,
                    recover: Callable[[int], None] = default_recover) -> None:
    '''Apply the pure step()'s ASAS active-flag decisions and trigger waypoint
    recovery for aircraft going inactive. Impure: mutates `active` in place
    (matches the legacy `reso.active[idx] = ...` assignment) and calls
    `recover` for every aircraft the step is releasing.
    '''
    for idx, is_active in changeactive.items():
        active[idx] = is_active
        if not is_active:
            recover(idx)


def _val(a, idx):
    '''Safely extract a float from an array-like at index idx.'''
    try:
        return float(a[idx])
    except Exception:
        return None


def get_desired_ownship_velocity(ownship, idx, cache):
    '''Get the desired (pre-resolution) ownship velocity in (east, north) m/s.

    Looks up seltrk/selspd first, then falls back to ap.trk/ap.tas,
    then to the current track/groundspeed. ``cache`` (caller-owned dict,
    idx -> (u, v)) memoises results within one timestep.
    '''
    if idx in cache:
        return cache[idx]

    trk = None
    if hasattr(ownship, "seltrk"):
        trk = _val(ownship.seltrk, idx)
    if trk is None and hasattr(ownship, "ap") and hasattr(ownship.ap, "trk"):
        trk = _val(ownship.ap.trk, idx)
    if trk is None:
        trk = _val(ownship.trk, idx)

    spd = None
    if hasattr(ownship, "selspd"):
        spd = _val(ownship.selspd, idx)
    if spd is None and hasattr(ownship, "ap") and hasattr(ownship.ap, "tas"):
        spd = _val(ownship.ap.tas, idx)
    if spd is None:
        spd = _val(getattr(ownship, "gs", None), idx)
    if spd is None:
        spd = float(np.hypot(ownship.gseast[idx], ownship.gsnorth[idx]))

    r = np.radians(trk)
    u = spd * np.sin(r)
    v = spd * np.cos(r)
    cache[idx] = (u, v)
    return u, v


def compute_pair_positions(conf):
    '''Build a dict mapping conflict pair -> (dx, dy) in meters from conf arrays.'''
    pair_dxdy = {}
    if len(conf.confpairs) > 0:
        q = np.radians(conf.qdr)
        dxs = conf.dist * np.sin(q)
        dys = conf.dist * np.cos(q)
        pair_dxdy = dict(zip(conf.confpairs, zip(dxs.tolist(), dys.tolist())))
    return pair_dxdy


def get_relative_position(ownship, intruder, idx1, idx2):
    '''Compute relative position (dx, dy) in meters using flat-earth approximation.'''
    re = 6371000.0
    dlon = float(intruder.lon[idx2] - ownship.lon[idx1])
    dlat = float(intruder.lat[idx2] - ownship.lat[idx1])
    latm = 0.5 * np.radians(float(intruder.lat[idx2] + ownship.lat[idx1]))
    dx = re * np.radians(dlon) * np.cos(latm)
    dy = re * np.radians(dlat)
    return dx, dy


def get_pair_dxdy(conflict, pair_dxdy, ownship, intruder, idx1, idx2):
    '''Get (dx, dy) for a conflict pair, using precomputed values or flat-earth fallback.'''
    if conflict in pair_dxdy:
        dx, dy = pair_dxdy[conflict]
        return float(dx), float(dy)
    return get_relative_position(ownship, intruder, idx1, idx2)


def record_initial_intruder_velocity(state: RecoveryState, conf, intruder,
                                      id2idx: Callable = default_id2idx
                                      ) -> RecoveryState:
    '''Merge newly-detected conflict pairs into `state.resopairs`, recording each
    new pair's intruder velocity at this moment (the "Vi,i" used by the
    double-criteria release decision). Pure given `id2idx` -- returns a new
    RecoveryState rather than mutating.
    '''
    curpairs = set(conf.confpairs)
    newpairs = curpairs - state.resopairs
    new_resopairs = state.resopairs | curpairs

    init_vel = dict(state.init_vel)
    for pair in newpairs:
        idx1, idx2 = id2idx(pair)
        if idx1 >= 0 and idx2 >= 0:
            init_vel[pair] = (float(intruder.gseast[idx2]), float(intruder.gsnorth[idx2]))

    return replace(state, resopairs=new_resopairs, init_vel=init_vel)

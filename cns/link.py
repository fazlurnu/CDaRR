''' Immutable measurement message + relay/hold composition -- functional core.

Redesign of sim_models/adsl_message.py's mutable ``ADSLMessage`` into an
immutable :class:`Message` plus pure update functions (refactor_fp.md's cns/
target: "Message (immutable snapshot) + relay/hold composition").

Scope note: ``ADSLMessage.ensure_size`` supports growing/shrinking a message
to a different aircraft count. In this pipeline's actual usage
(``ADSL.update_from_truth`` calling ``ensure_size(states.ntraf)`` every
tick), the aircraft count never changes after the first call -- no aircraft
is ever created or deleted mid-run in ``PairwiseHorConflict`` -- so a resize
to a *different* size never fires in practice; the general case is dead code
here (same "reproduce what's used" scoping as the Phase 1 decisions to skip
``resumenav_triple_criteria``/``applyprio``). ``Message`` is therefore
fixed-size: callers create one with :func:`empty_message` up front and every
subsequent :func:`with_truth`/:func:`relay` call must match that size.

The "4-node ADSL stack" wiring (bus/ownship/intruder/prev_intruder, which
node relays to which, when, with what reception draw) is simulation-loop
orchestration, not reusable CNS algebra -- it belongs in the Phase 3/4 shell
rewrite (``cdarr``/``sim`` package), not here, the same way
``envs/pairwise_conflict.py`` was correctly out of scope for cd/cr/crr.
'''
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True, eq=False)
class Message:
    ''' Immutable snapshot of one node's last-known (possibly noisy)
    measurements. Field names mirror ADSLMessage's attributes 1:1. Numpy
    fields -> eq=False (see cd.common.ConflictData's docstring for why).
    '''
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    hdg: np.ndarray
    trk: np.ndarray
    gs: np.ndarray
    tas: np.ndarray
    vs: np.ndarray
    gseast: np.ndarray
    gsnorth: np.ndarray
    id: tuple
    perf: Any = None
    ap: Any = None
    selalt: Any = None

    @property
    def ntraf(self) -> int:
        return int(self.lat.size)


def empty_message(n: int) -> Message:
    ''' A fresh, all-NaN Message of size n (ids all ""). Matches a brand-new
    ADSLMessage() after its first ensure_size(n) call. '''
    z = lambda: np.full(n, np.nan, dtype=float)
    return Message(lat=z(), lon=z(), alt=z(), hdg=z(), trk=z(), gs=z(), tas=z(), vs=z(),
                   gseast=z(), gsnorth=z(), id=tuple([""] * n))


def with_truth(msg: Message, states: Any, idx: np.ndarray) -> Message:
    ''' New Message with entries at idx copied from truth `states`; entries
    outside idx are carried over unchanged from `msg`. Mirrors
    ADSLMessage.copy_from_states, but pure -- returns a fresh Message rather
    than mutating. Requires msg.ntraf == states.ntraf (see module docstring).
    '''
    n = int(states.ntraf)
    if msg.ntraf != n:
        raise ValueError(
            f"with_truth requires msg.ntraf ({msg.ntraf}) == states.ntraf ({n}); "
            "general resize is out of scope, see module docstring.")

    lat = msg.lat.copy(); lat[idx] = states.lat[idx]
    lon = msg.lon.copy(); lon[idx] = states.lon[idx]
    alt = msg.alt.copy(); alt[idx] = states.alt[idx]
    hdg = msg.hdg.copy(); hdg[idx] = states.hdg[idx]
    trk = msg.trk.copy(); trk[idx] = states.trk[idx]
    gs  = msg.gs.copy();  gs[idx]  = states.gs[idx]
    tas = msg.tas.copy(); tas[idx] = states.tas[idx]
    vs  = msg.vs.copy();  vs[idx]  = states.vs[idx]

    gseast = msg.gseast.copy()
    gsnorth = msg.gsnorth.copy()
    if hasattr(states, "gseast") and hasattr(states, "gsnorth"):
        gseast[idx] = states.gseast[idx]
        gsnorth[idx] = states.gsnorth[idx]

    ids = list(msg.id)
    if hasattr(states, "id"):
        sid = states.id
        for i in idx:
            ids[int(i)] = sid[int(i)]

    return Message(
        lat=lat, lon=lon, alt=alt, hdg=hdg, trk=trk, gs=gs, tas=tas, vs=vs,
        gseast=gseast, gsnorth=gsnorth, id=tuple(ids),
        perf=getattr(states, "perf", None), ap=getattr(states, "ap", None),
        selalt=getattr(states, "selalt", None),
    )


def relay(dst: Message, src: Message, idx: Optional[np.ndarray] = None) -> Message:
    ''' New Message combining `dst` and `src`: entries at idx (or all, if
    idx is None) taken from `src`; other entries preserved from `dst`.
    Mirrors ADSLMessage.copy_from_message, but pure. If dst.ntraf !=
    src.ntraf, always takes a full copy of `src` regardless of idx --
    matching copy_from_message's mismatched-size fallback exactly.
    '''
    if idx is None or dst.ntraf != src.ntraf:
        return Message(
            lat=src.lat.copy(), lon=src.lon.copy(), alt=src.alt.copy(),
            hdg=src.hdg.copy(), trk=src.trk.copy(), gs=src.gs.copy(),
            tas=src.tas.copy(), vs=src.vs.copy(),
            gseast=src.gseast.copy(), gsnorth=src.gsnorth.copy(),
            id=tuple(src.id), perf=src.perf, ap=src.ap, selalt=src.selalt,
        )

    lat = dst.lat.copy(); lat[idx] = src.lat[idx]
    lon = dst.lon.copy(); lon[idx] = src.lon[idx]
    alt = dst.alt.copy(); alt[idx] = src.alt[idx]
    hdg = dst.hdg.copy(); hdg[idx] = src.hdg[idx]
    trk = dst.trk.copy(); trk[idx] = src.trk[idx]
    gs  = dst.gs.copy();  gs[idx]  = src.gs[idx]
    tas = dst.tas.copy(); tas[idx] = src.tas[idx]
    vs  = dst.vs.copy();  vs[idx]  = src.vs[idx]

    gseast = dst.gseast.copy(); gseast[idx] = src.gseast[idx]
    gsnorth = dst.gsnorth.copy(); gsnorth[idx] = src.gsnorth[idx]

    ids = list(dst.id)
    for i in idx:
        ids[int(i)] = src.id[int(i)]

    return Message(
        lat=lat, lon=lon, alt=alt, hdg=hdg, trk=trk, gs=gs, tas=tas, vs=vs,
        gseast=gseast, gsnorth=gsnorth, id=tuple(ids),
        perf=dst.perf, ap=dst.ap, selalt=dst.selalt,
    )

''' The composed cdarr() function: detect -> resolve -> recover, in the exact
per-tick order the legacy simulation loop used (refactor_fp.md section 4).

The legacy loop's critical, easy-to-miss property (F1 in refactor_fp.md):
``reso = conf_resolution.resolve(...)`` returns ``self.resopairs`` *by
reference* -- the same mutable set object, not a snapshot -- and the
subsequent ``conf_recovery(...)`` call mutates that same object in place. So
by the time ``_do_action`` reads the resolve()-returned tuple's resopairs
element, it is already seeing the *post-recovery* set, even though it was
captured before recovery ran. Every aircraft's avoidance status is therefore
determined by resopairs membership AFTER recovery, not after resolve --
``cdarr()`` below reproduces this explicitly, since the pure design has no
mutable-by-reference trick to lean on.
'''
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Optional

import numpy as np

import cd.statebased
import cr.mvp
import cr.vo
from cr.common import ResolutionCommand, ResolutionParams
import crr.cpa
import crr.ftr
import crr.probabilistic_ftr
from crr.common import RecoveryState, default_id2idx
from cd.common import ConflictData


@dataclass(frozen=True)
class CdarrParams:
    ''' Everything cdarr() needs beyond the raw traffic arrays: detection
    zone sizes, and which resolution/recovery model to run plus their
    model-specific knobs. Mirrors the legacy _RESOLUTION_MODELS/
    _RECOVERY_MODELS dispatch in sim/pairwise_stochastic/get_ipr_stochastic_env.py.
    '''
    rpz: float
    hpz: float
    dtlookahead: float
    resolution: str                       # "MVP" | "VO"
    resolution_params: ResolutionParams
    recovery: str                         # "CPA" | "FTR" | "Probabilistic FTR"
    # CPA recovery's bouncing-conflict check; FTR ignores this.
    resofach: float = 1.0
    # Probabilistic FTR only (see crr/probabilistic_ftr.py's "no hidden
    # parameter channels" docstring -- these were smuggled onto `conf` in
    # the legacy code).
    sigma_r: Any = None
    sigma_v: Any = None
    prob_threshold: float = 0.9
    Ktheta: int = 256


@dataclass(frozen=True, eq=False)
class CdarrResult:
    ''' Output of one cdarr() call. Numpy fields -> eq=False (see
    cd.common.ConflictData's docstring for why). '''
    gs_cmd: np.ndarray
    trk_cmd: np.ndarray
    avoiding: np.ndarray
    state: RecoveryState
    conflicts: ConflictData
    command: ResolutionCommand
    changeactive: Mapping[int, bool] = field(default_factory=dict)


_RESOLUTION_MODELS = ("MVP", "VO")
_RECOVERY_MODELS = ("CPA", "FTR", "Probabilistic FTR")


def _resolve(resolution, conf, own, intr, params: CdarrParams, state: RecoveryState,
             noresoac, resooffac, id2idx):
    ''' Dispatch to cr.mvp.resolve / cr.vo.resolve. Returns
    (command, state_after_resolve) -- VO's F6 quirk means it can advance
    `state`; MVP never touches recovery state, so it passes `state` through
    unchanged. '''
    if resolution == "MVP":
        command = cr.mvp.resolve(conf, own, intr, params.resolution_params,
                                 noresoac=noresoac, resooffac=resooffac)
        return command, state
    if resolution == "VO":
        command, new_state, _changeactive = cr.vo.resolve(
            conf, own, intr, params.resolution_params, state,
            noresoac=noresoac, resooffac=resooffac, id2idx=id2idx)
        return command, new_state
    raise ValueError(f"Unsupported resolution model: {resolution!r} (expected one of {_RESOLUTION_MODELS})")


def _recover(recovery, conf, own, intr, state: RecoveryState, params: CdarrParams, id2idx):
    ''' Dispatch to crr.cpa.step / crr.ftr.step / crr.probabilistic_ftr.step.
    Returns (new_state, changeactive, delpairs) -- same contract as each
    individual step() function. '''
    if recovery == "CPA":
        return crr.cpa.step(conf, own, intr, state, params.resofach, id2idx=id2idx)
    if recovery == "FTR":
        return crr.ftr.step(conf, own, intr, state, id2idx=id2idx)
    if recovery == "Probabilistic FTR":
        return crr.probabilistic_ftr.step(
            conf, own, intr, state,
            sigma_r=params.sigma_r, sigma_v=params.sigma_v,
            prob_threshold=params.prob_threshold, Ktheta=params.Ktheta,
            id2idx=id2idx)
    raise ValueError(f"Unsupported recovery model: {recovery!r} (expected one of {_RECOVERY_MODELS})")


def membership_flags(ids, resopairs) -> np.ndarray:
    ''' Per-aircraft bool array: True iff ids[i] appears in any pair of
    `resopairs` (either position -- both aircraft of a resolved pair are
    avoiding). Vectorized form of the legacy env's
    ``any(target_id in pair for pair in resopairs)`` membership test. '''
    involved = set()
    for a, b in resopairs:
        involved.add(a)
        involved.add(b)
    return np.array([i in involved for i in ids], dtype=bool)


def cdarr(own, intr, state: RecoveryState, params: CdarrParams,
          noresoac: Optional[np.ndarray] = None, resooffac: Optional[np.ndarray] = None,
          id2idx: Callable = default_id2idx) -> CdarrResult:
    ''' detect -> resolve -> recover, composed in the exact per-tick order
    the legacy simulation loop used (see module docstring for the
    by-reference mutation subtlety this reproduces explicitly).

    `own`/`intr` are duck-typed traffic views (same shape cd.statebased.detect
    / cr.mvp.resolve / cr.vo.resolve / crr.*.step already expect: .lat, .lon,
    .alt, .trk, .gs, .vs, .gseast, .gsnorth, .id, .ntraf, .perf, .ap,
    .selalt) -- e.g. a bluesky traf object, a cns.link.Message, or the view
    built by cdarr_step()'s make_view() below.

    For avoiding[i] == False, gs_cmd[i] == own.gs[i] and trk_cmd[i] ==
    own.trk[i] (current-value pass-through) -- this is cdarr()'s own
    contract, not the same as the legacy env's "restore to NOMINAL
    heading/speed" behaviour; a caller wanting nominal-restore (like
    envs/pairwise_conflict.py's _do_action) applies that on top of the
    `avoiding` flag itself, exactly as the legacy env does today (see
    refactor_fp.md's F3 note).
    '''
    conf = cd.statebased.detect(own, intr, params.rpz, params.hpz, params.dtlookahead)

    command, state_after_resolve = _resolve(
        params.resolution, conf, own, intr, params, state, noresoac, resooffac, id2idx)

    new_state, changeactive, _delpairs = _recover(
        params.recovery, conf, own, intr, state_after_resolve, params, id2idx)

    # F3: avoidance status is post-recovery resopairs membership (F1 ordering).
    avoiding = membership_flags(own.id, new_state.resopairs)

    gs_cmd = np.where(avoiding, command.gs_capped, own.gs)
    trk_cmd = np.where(avoiding, command.trk, own.trk)

    return CdarrResult(gs_cmd=gs_cmd, trk_cmd=trk_cmd, avoiding=avoiding,
                       state=new_state, conflicts=conf, command=command,
                       changeactive=changeactive)


def _default_perf(n):
    return SimpleNamespace(
        vmin=np.full(n, -np.inf), vmax=np.full(n, np.inf),
        vsmin=np.full(n, -np.inf), vsmax=np.full(n, np.inf),
    )


def make_view(lat, lon, alt, gs, trk, vs, ids, perf=None, ap=None, selalt=None):
    ''' Build a duck-typed traffic view from raw per-aircraft arrays, for
    callers (like cdarr_step below) that don't already have a bluesky-traf-
    shaped object. gseast/gsnorth are derived from gs/trk. perf/ap/selalt
    default to "no vertical resolution desired" sentinels (unbounded speed
    caps, current vs as ap.vs, current alt as selalt) so the vertical-
    resolution/altitude math in cr.mvp.resolve/cr.vo.resolve never NaNs for
    callers who only care about the horizontal (gs_cmd/trk_cmd) outputs.
    '''
    lat = np.asarray(lat, dtype=float)
    n = lat.size
    trk_arr = np.asarray(trk, dtype=float)
    gs_arr = np.asarray(gs, dtype=float)
    trk_rad = np.radians(trk_arr)
    alt_arr = np.asarray(alt, dtype=float)
    vs_arr = np.asarray(vs, dtype=float)

    return SimpleNamespace(
        lat=lat, lon=np.asarray(lon, dtype=float), alt=alt_arr,
        trk=trk_arr, gs=gs_arr, vs=vs_arr,
        gseast=gs_arr * np.sin(trk_rad), gsnorth=gs_arr * np.cos(trk_rad),
        id=list(ids), ntraf=n,
        perf=perf if perf is not None else _default_perf(n),
        ap=ap if ap is not None else SimpleNamespace(vs=vs_arr, trk=None, tas=None),
        selalt=selalt if selalt is not None else alt_arr,
    )


def make_dict_id2idx(ids) -> Callable:
    ''' Pure id2idx hook with no bluesky dependency: a dict lookup over
    `ids`, matching bs.traf.id2idx's contract of returning -1 for an id not
    found. Used by cdarr_step() below so the single-traffic-set entry point
    never touches bluesky. '''
    lookup = {aid: i for i, aid in enumerate(ids)}

    def _id2idx(pair):
        a, b = pair
        return lookup.get(a, -1), lookup.get(b, -1)

    return _id2idx


def cdarr_step(lat, lon, alt, gs, trk, vs, ids, state: RecoveryState, params: CdarrParams,
                perf=None, ap=None, selalt=None,
                noresoac: Optional[np.ndarray] = None, resooffac: Optional[np.ndarray] = None,
                ):
    ''' Single-traffic-set entry point: every aircraft observes every other
    (own == intr) -- the same convention the legacy ground-truth detector
    uses (``conf_detection_gt.detect(states, states, ...)``), and the n x n
    detection with a masked diagonal already supports it.

    Returns (gs_cmd, trk_cmd, avoiding, new_state) as plain arrays, one entry
    per aircraft in `ids`' order. For avoiding[i] == False, gs_cmd[i] ==
    gs[i] and trk_cmd[i] == trk[i] (pass-through) -- see cdarr()'s docstring.
    Threads `state` explicitly: pass `result_state` back in as `state` on
    the next call to persist recovery bookkeeping across ticks.
    '''
    view = make_view(lat, lon, alt, gs, trk, vs, ids, perf=perf, ap=ap, selalt=selalt)
    id2idx = make_dict_id2idx(list(ids))
    result = cdarr(view, view, state, params,
                   noresoac=noresoac, resooffac=resooffac, id2idx=id2idx)
    return result.gs_cmd, result.trk_cmd, result.avoiding, result.state

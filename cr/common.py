''' Shared, pure building blocks for conflict resolution.

Both the MVP and VO resolvers are expressed as plain functions operating on
explicit inputs. The pieces they have in common live here -- moved verbatim
from the local copies each carried after Phase 1 (sim_models/cr_mvp.py and
sim_models/cr_vo.py had byte-identical select_command/cap_velocities/
resolve_altitude bodies; refactor_fp.md's Phase-2 note explicitly allows
unifying them here, since the L1 fixture tests prove both files' old callers
still get the same numbers through this single copy).

Nothing here touches global BlueSky state.
'''
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class ResolutionParams:
    ''' Immutable bag of resolution settings (formerly mutable instance switches
    on the MVP/VO Entity classes: self.resofach, self.swresohoriz, etc).

    swprio/priocode are read only by VO's applyprio path (not yet ported --
    see refactor_fp.md's Phase 1 note on low-coverage code); kept here so both
    resolvers share one params type.
    '''
    resofach: float
    resofacv: float
    swresohoriz: bool = True
    swresospd: bool = False
    swresohdg: bool = False
    swresovert: bool = False
    swprio: bool = False
    priocode: str = ''

    def with_resofach(self, resofach: float) -> 'ResolutionParams':
        ''' Return a copy with an updated horizontal resolution factor -- mirrors
        the legacy `self.resofach = resofach` line at the top of `resolve()`.
        '''
        return replace(self, resofach=resofach)


@dataclass(frozen=True, eq=False)
class ResolutionCommand:
    ''' Per-aircraft ASAS output of one resolve() call. Numpy fields -> eq=False
    (see cd.common.ConflictData's docstring for why). '''
    trk: np.ndarray
    gs_capped: np.ndarray
    vs_capped: np.ndarray
    alt: np.ndarray


def select_command(newv, ownship, swresohoriz, swresospd, swresohdg, swresovert):
    """Select (newtrack, newgs, newvs) from the resolved cartesian velocity
    ``newv`` (shape (3, ntraf)), per the configured resolution-direction
    switches. Pure; preserves every branch of the former inline logic in
    MVP.resolve / VO.resolve exactly (SPD-only / HDG-only / SPD+HDG /
    vertical-only / horizontal+vertical).
    """
    if swresohoriz:  # horizontal resolutions
        if swresospd and not swresohdg:  # SPD only
            newtrack = ownship.trk
            newgs    = np.sqrt(newv[0, :]**2 + newv[1, :]**2)
            newvs    = ownship.vs
        elif swresohdg and not swresospd:  # HDG only
            newtrack = (np.arctan2(newv[0, :], newv[1, :]) * 180 / np.pi) % 360
            newgs    = ownship.gs
            newvs    = ownship.vs
        else:  # SPD + HDG
            newtrack = (np.arctan2(newv[0, :], newv[1, :]) * 180 / np.pi) % 360
            newgs    = np.sqrt(newv[0, :]**2 + newv[1, :]**2)
            newvs    = ownship.vs
    elif swresovert:  # vertical resolutions
        newtrack = ownship.trk
        newgs    = ownship.gs
        newvs    = newv[2, :]
    else:  # horizontal + vertical
        newtrack = (np.arctan2(newv[0, :], newv[1, :]) * 180 / np.pi) % 360
        newgs    = np.sqrt(newv[0, :]**2 + newv[1, :]**2)
        newvs    = newv[2, :]
    return newtrack, newgs, newvs


def cap_velocities(newgs, newvs, perf):
    """Clamp ground speed and vertical speed to the aircraft performance envelope."""
    newgscapped = np.maximum(perf.vmin, np.minimum(perf.vmax, newgs))
    vscapped = np.maximum(perf.vsmin, np.minimum(perf.vsmax, newvs))
    return newgscapped, vscapped


def resolve_altitude(ownship, vscapped, timesolveV, dtlookahead, dv_vert, swresohoriz):
    """ASAS altitude command.

    Follows the projected vertical-resolve altitude when its direction agrees
    with the direction toward the selected altitude, only while the aircraft is
    actually inside a vertical conflict (``timesolveV < dtlookahead`` and
    ``dv_vert`` nonzero); otherwise keeps the selected altitude.
    Horizontal-only resolutions (``swresohoriz`` truthy) always keep the
    selected altitude -- preserved via the original blend-weight form
    (``alt * (1 - swresohoriz) + selalt * swresohoriz``) rather than an
    if/else, since ``swresohoriz`` here is used as a 0/1 weight, not just a
    bool.
    """
    asasalttemp = vscapped * timesolveV + ownship.alt
    signdvs = np.sign(vscapped - ownship.ap.vs * np.sign(ownship.selalt - ownship.alt))
    signalt = np.sign(asasalttemp - ownship.selalt)
    alt = np.where(np.logical_or(signdvs == 0, signdvs == signalt), asasalttemp, ownship.selalt)

    altCondition = np.logical_and(timesolveV < dtlookahead, np.abs(dv_vert) > 0.0)
    alt[altCondition] = asasalttemp[altCondition]

    alt = alt * (1 - swresohoriz) + ownship.selalt * swresohoriz
    return alt


def default_noreso_arrays(ntraf):
    ''' All-False noreso/resooff arrays -- matches what the legacy MVP/VO Entity
    classes hold when nothing has ever flagged an aircraft as noreso/resooff
    (this pipeline never sets either), see refactor_fp.md's risk table. '''
    return np.zeros(ntraf, dtype=bool), np.zeros(ntraf, dtype=bool)

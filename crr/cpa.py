'''Past-CPA conflict recovery -- functional core.

Fresh copy of the Phase-1 extraction in sim_models/crr_resumenav_cpa.py (not
an import -- see cd/statebased.py's docstring for why). Decides which
resolved conflicts have passed their closest point of approach (or are no
longer in horizontal LOS / bouncing) and may release the resolution maneuver.
'''
from dataclasses import replace

import numpy as np

from .common import RecoveryState, default_id2idx


def _anglediff(a, b):
    ''' Smallest relative angle (deg) between heading a and b. '''
    d = a - b
    if d > 180:
        return _anglediff(a, b + 360)
    elif d < -180:
        return _anglediff(a + 360, b)
    else:
        return d


def cpa_keep_active(ownship, intruder, conf, idx1, idx2, resofach):
    ''' Past-CPA "keep resolving" decision for one conflict pair.

    Pure: reads only its arguments. Returns True if the pair should stay
    engaged -- not yet past CPA, still in horizontal LOS, or bouncing in and
    out of conflict -- False if it's safe to release.
    '''
    # Distance vector using flat earth approximation
    re = 6371000.
    dist = re * np.array([np.radians(intruder.lon[idx2] - ownship.lon[idx1]) *
                            np.cos(0.5 * np.radians(intruder.lat[idx2] +
                                                    ownship.lat[idx1])),
                            np.radians(intruder.lat[idx2] - ownship.lat[idx1])])

    # Relative velocity vector
    vrel = np.array([ownship.gseast[idx1] - intruder.gseast[idx2],
                     ownship.gsnorth[idx1] - intruder.gsnorth[idx2]])

    # Check if conflict is past CPA
    past_cpa = np.dot(dist, vrel) < 0.0

    rpz = np.max(conf.rpz[[idx1, idx2]])
    # Aircraft should continue to resolve until there is no horizontal LOS
    hdist = np.linalg.norm(dist)
    hor_los = hdist < rpz

    # Bouncing conflicts: stay active until bouncing stops
    is_bouncing = \
        abs(_anglediff(ownship.trk[idx1], intruder.trk[idx2])) < 30.0 and \
        hdist < rpz * resofach

    return (not past_cpa) or hor_los or is_bouncing


def step(conf, ownship, intruder, state: RecoveryState, resofach, id2idx=default_id2idx):
    ''' Pure Past-CPA recovery decision.

    Returns (new_state, changeactive, delpairs):
      new_state   : RecoveryState with resopairs = (state.resopairs | new confpairs) - delpairs
      changeactive: {aircraft_idx: bool} -- True keep resolving, False release
      delpairs    : set of conflict pairs released this call

    No side effects -- the caller applies changeactive via
    crr.common.apply_recovery (which touches bs.traf).
    '''
    resopairs = state.resopairs | set(conf.confpairs)

    delpairs = set()
    changeactive = {}

    for conflict in resopairs:
        idx1, idx2 = id2idx(conflict)
        if idx1 < 0:
            delpairs.add(conflict)
            continue

        if idx2 >= 0:
            keep_active = cpa_keep_active(ownship, intruder, conf, idx1, idx2, resofach)

        # Start recovery if intruder is deleted, or if past CPA
        # and not in horizontal LOS or a bouncing conflict
        if idx2 >= 0 and keep_active:
            changeactive[idx1] = True
        else:
            changeactive[idx1] = changeactive.get(idx1, False)
            delpairs.add(conflict)

    new_state = replace(state, resopairs=resopairs - delpairs)
    return new_state, changeactive, delpairs

import numpy as np
import bluesky as bs

from legacy.sim_models.crr_recovery_base import apply_recovery


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


def resumenav(reso, conf, ownship, intruder):
    reso.resopairs.update(conf.confpairs)

    delpairs = set()
    changeactive = dict()

    for conflict in reso.resopairs:
        idx1, idx2 = bs.traf.id2idx(conflict)
        if idx1 < 0:
            delpairs.add(conflict)
            continue

        if idx2 >= 0:
            keep_active = cpa_keep_active(ownship, intruder, conf, idx1, idx2, reso.resofach)

        # Start recovery if intruder is deleted, or if past CPA
        # and not in horizontal LOS or a bouncing conflict
        if idx2 >= 0 and keep_active:
            changeactive[idx1] = True
        else:
            changeactive[idx1] = changeactive.get(idx1, False)
            delpairs.add(conflict)

    apply_recovery(changeactive, reso, delpairs)
    return delpairs

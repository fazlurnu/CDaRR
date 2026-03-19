"""Shared helpers for conflict recovery models."""

import numpy as np
import bluesky as bs


def _val(a, idx):
    """Safely extract a float from an array-like at index idx."""
    try:
        return float(a[idx])
    except Exception:
        return None


def get_desired_ownship_velocity(ownship, idx, cache):
    """Get the desired (pre-resolution) ownship velocity in (east, north) m/s.

    Looks up seltrk/selspd first, then falls back to ap.trk/ap.tas,
    then to the current track/groundspeed.

    Parameters
    ----------
    ownship : state-like object with .seltrk, .selspd, .trk, .gseast, .gsnorth, etc.
    idx : int
        Aircraft index.
    cache : dict
        Mutable cache (idx -> (u, v)) to avoid recomputing within one timestep.

    Returns
    -------
    (u, v) : tuple of float
        East and north velocity components.
    """
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
    """Build a dict mapping conflict pair -> (dx, dy) in meters from conf arrays.

    Returns
    -------
    dict : {pair_tuple: (dx_m, dy_m)}
    """
    pair_dxdy = {}
    if len(conf.confpairs) > 0:
        q = np.radians(conf.qdr)
        dxs = conf.dist * np.sin(q)
        dys = conf.dist * np.cos(q)
        pair_dxdy = dict(zip(conf.confpairs, zip(dxs.tolist(), dys.tolist())))
    return pair_dxdy


def get_relative_position(ownship, intruder, idx1, idx2):
    """Compute relative position (dx, dy) in meters using flat-earth approximation.

    Returns
    -------
    (dx, dy) : tuple of float
        East and north displacement from ownship to intruder.
    """
    re = 6371000.0
    dlon = float(intruder.lon[idx2] - ownship.lon[idx1])
    dlat = float(intruder.lat[idx2] - ownship.lat[idx1])
    latm = 0.5 * np.radians(float(intruder.lat[idx2] + ownship.lat[idx1]))
    dx = re * np.radians(dlon) * np.cos(latm)
    dy = re * np.radians(dlat)
    return dx, dy


def apply_recovery(changeactive, reso, delpairs):
    """Apply ASAS active flags and trigger waypoint recovery for resolved conflicts.

    Parameters
    ----------
    changeactive : dict
        {aircraft_idx: bool} — True means keep resolving, False means recover.
    reso : resolution object
        Must have .active array and .resopairs set.
    delpairs : set
        Conflict pairs to remove from resopairs.
    """
    for idx, active in changeactive.items():
        reso.active[idx] = active
        if not active:
            iwpid = bs.traf.ap.route[idx].findact(idx)
            if iwpid != -1:
                bs.traf.ap.route[idx].direct(idx, bs.traf.ap.route[idx].wpname[iwpid])

    reso.resopairs -= delpairs


def record_initial_intruder_velocity(reso, conf, intruder):
    """Record initial intruder velocity for new conflict pairs.

    Returns
    -------
    set : newly detected conflict pairs
    """
    if not hasattr(reso, "_intr_init_vel"):
        reso._intr_init_vel = {}

    curpairs = set(conf.confpairs)
    newpairs = curpairs - reso.resopairs
    reso.resopairs.update(curpairs)

    for pair in newpairs:
        idx1, idx2 = bs.traf.id2idx(pair)
        if idx1 >= 0 and idx2 >= 0:
            reso._intr_init_vel[pair] = (float(intruder.gseast[idx2]),
                                         float(intruder.gsnorth[idx2]))

    return newpairs


def get_pair_dxdy(conflict, pair_dxdy, ownship, intruder, idx1, idx2):
    """Get (dx, dy) for a conflict pair, using precomputed values or flat-earth fallback."""
    if conflict in pair_dxdy:
        dx, dy = pair_dxdy[conflict]
        return float(dx), float(dy)
    return get_relative_position(ownship, intruder, idx1, idx2)

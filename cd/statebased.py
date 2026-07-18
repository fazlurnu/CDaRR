''' State-based conflict detection -- functional core.

Pure port of ``sim_models/cd_statebased.py``'s Phase-1 extraction (see
refactor_fp.md, Phase 2). ``detect`` is memoryless: given the same
``(ownship, intruder, rpz, hpz, dtlookahead)`` it always returns a
bit-identical :class:`~cd.common.ConflictData`, with no attribute-based side
channel (the legacy ``StateBased`` class stashed its result on ``self``).

This module is a deliberate *fresh copy* of the Phase-1 helpers, not an
import from ``sim_models.cd_statebased`` -- importing would make old-vs-new
equivalence trivially true and defeat the point of the check (see
refactor_fp.md's L1 layer: it replays captured old-detect inputs through this
module's ``detect`` and asserts every output field is bitwise equal).
'''
import numpy as np
from bluesky.tools import geo
from bluesky.tools.aero import nm

from .common import ConflictData

# Large finite value masking the diagonal (ownship-vs-itself) so an aircraft never
# detects a conflict with its own track. Distinct from the 1e8 sentinels below.
_BIG = 1e9
_EPS = 1e-6


def _velocity_components(trk, gs, n):
    ''' Decompose track (deg) / ground-speed into (east, north) m/s row vectors, shape (1, n). '''
    trkrad = np.radians(trk)
    u = gs * np.sin(trkrad).reshape((1, n))
    v = gs * np.cos(trkrad).reshape((1, n))
    return u, v


def relative_bearing_distance(ownship, intruder, eye):
    ''' Bearing (deg) and distance (m) from every ownship to every intruder.

    The diagonal is pushed to _BIG via `eye` so self-pairs are never flagged.
    '''
    qdr, dist = geo.kwikqdrdist_matrix(np.asmatrix(ownship.lat), np.asmatrix(ownship.lon),
                                np.asmatrix(intruder.lat), np.asmatrix(intruder.lon))

    # Convert back to array to allow element-wise array multiplications later on
    # Convert to meters and add large value to own/own pairs
    qdr = np.asarray(qdr)
    dist = np.asarray(dist) * nm + _BIG * eye
    return qdr, dist


def horizontal_conflict(ownship, intruder, qdr, dist, rpz, eye):
    ''' Horizontal CPA geometry and entry/exit times of the protected zone.

    Returns (swhorconf, tcpa, dcpa2, dtinhor, tinhor, touthor, vrel, rpz_mat).
    `dtinhor` is exposed separately (not folded into `tinhor`) because callers need
    the *unmasked* half-in-zone time even outside a formal conflict (see `tinhor_all`).
    '''
    # Calculate horizontal closest point of approach (CPA)
    qdrrad = np.radians(qdr)
    dx = dist * np.sin(qdrrad)  # is pos j rel to i
    dy = dist * np.cos(qdrrad)  # is pos j rel to i

    # Note: ownship/intruder track+speed are reshaped by their OWN ntraf, not a
    # shared one -- ownship.ntraf and intruder.ntraf may differ transiently.
    ownu, ownv = _velocity_components(ownship.trk, ownship.gs, ownship.ntraf)
    intu, intv = _velocity_components(intruder.trk, intruder.gs, intruder.ntraf)

    du = ownu - intu.T  # Speed du[i,j] is perceived eastern speed of i to j
    dv = ownv - intv.T  # Speed dv[i,j] is perceived northern speed of i to j

    dv2 = du * du + dv * dv
    dv2 = np.where(np.abs(dv2) < _EPS, _EPS, dv2)  # limit lower absolute value
    vrel = np.sqrt(dv2)

    tcpa = -(du * dx + dv * dy) / dv2 + _BIG * eye

    # Calculate distance^2 at CPA (minimum distance^2)
    dcpa2 = np.abs(dist * dist - tcpa * tcpa * dv2)

    # Check for horizontal conflict
    # RPZ can differ per aircraft, get the largest value per aircraft pair
    rpz_mat = np.asarray(np.maximum(np.asmatrix(rpz), np.asmatrix(rpz).transpose()))
    R2 = rpz_mat * rpz_mat
    swhorconf = dcpa2 < R2  # conflict or not

    # Calculate times of entering and leaving horizontal conflict
    dxinhor = np.sqrt(np.maximum(0., R2 - dcpa2))  # half the distance travelled inzide zone
    dtinhor = dxinhor / vrel

    tinhor = np.where(swhorconf, tcpa - dtinhor, 1e8)  # Set very large if no conf
    touthor = np.where(swhorconf, tcpa + dtinhor, -1e8)  # set very large if no conf

    return swhorconf, tcpa, dcpa2, dtinhor, tinhor, touthor, vrel, rpz_mat


def vertical_conflict(ownship, intruder, hpz, eye):
    ''' Vertical separation and entry/exit times of the protected zone.

    Returns (dalt, tinver, toutver, hpz_mat).
    '''
    # Vertical crossing of disk (-dh,+dh)
    dalt = ownship.alt.reshape((1, ownship.ntraf)) - \
        intruder.alt.reshape((1, intruder.ntraf)).T  + _BIG * eye

    dvs = ownship.vs.reshape(1, ownship.ntraf) - \
        intruder.vs.reshape(1, intruder.ntraf).T
    dvs = np.where(np.abs(dvs) < _EPS, _EPS, dvs)  # prevent division by zero

    # Check for passing through each others zone
    # hPZ can differ per aircraft, get the largest value per aircraft pair
    hpz_mat = np.asarray(np.maximum(np.asmatrix(hpz), np.asmatrix(hpz).transpose()))
    tcrosshi = (dalt + hpz_mat) / -dvs
    tcrosslo = (dalt - hpz_mat) / -dvs
    tinver = np.minimum(tcrosshi, tcrosslo)
    toutver = np.maximum(tcrosshi, tcrosslo)

    return dalt, tinver, toutver, hpz_mat


def detect(ownship, intruder, rpz, hpz, dtlookahead) -> ConflictData:
    ''' Conflict detection between ownship (traf) and intruder (traf/adsb).

    Pure function: no attribute is written on `ownship`/`intruder`, and the
    result depends only on the arguments. Returns a fresh ConflictData.
    '''
    rpz_arr = np.array([rpz] * (ownship.ntraf))
    hpz_arr = np.array([hpz] * (ownship.ntraf))
    # NOTE: the per-aircraft broadcast list below is the *output* field
    # (ConflictData.dtlookahead, read downstream as conf.dtlookahead[idx]).
    # The `swconfl` comparison a few lines down instead uses the raw
    # `dtlookahead` scalar parameter directly (via np.asmatrix(dtlookahead)),
    # matching the legacy StateBased.detect exactly -- these are NOT the same
    # value used twice, despite sharing a name.
    dtlookahead_arr = [dtlookahead] * (ownship.ntraf)

    # Identity matrix of order ntraf: avoid ownship-ownship detected conflicts
    eye = np.eye(ownship.ntraf)

    # Horizontal conflict ------------------------------------------------------
    qdr, dist = relative_bearing_distance(ownship, intruder, eye)

    swhorconf, tcpa, dcpa2, dtinhor, tinhor, touthor, vrel, rpz_mat = horizontal_conflict(
        ownship, intruder, qdr, dist, rpz, eye)

    # this is to check the end condition, tcpa and tin should be < 0
    tcpa_all = tcpa
    tinhor_all = tcpa - dtinhor

    # Vertical conflict --------------------------------------------------------
    dalt, tinver, toutver, hpz_mat = vertical_conflict(ownship, intruder, hpz, eye)

    # Combine vertical and horizontal conflict----------------------------------
    tinconf = np.maximum(tinver, tinhor)
    toutconf = np.minimum(toutver, touthor)

    swconfl = np.array(swhorconf * (tinconf <= toutconf) * (toutconf > 0.0) *
                       np.asarray(tinconf < np.asmatrix(dtlookahead).T) * (1.0 - eye), dtype=bool)

    # --------------------------------------------------------------------------
    # Update conflict lists
    # --------------------------------------------------------------------------
    # Ownship conflict flag and max tCPA
    inconf = np.any(swconfl, 1)
    tcpamax = np.max(tcpa * swconfl, 1)

    # Select conflicting pairs: each a/c gets their own record
    confpairs = [(ownship.id[i], ownship.id[j]) for i, j in zip(*np.where(swconfl))]
    confpairs_unique = {frozenset(pair) for pair in confpairs}
    swlos = (dist < rpz_mat) * (np.abs(dalt) < hpz_mat)
    lospairs = [(ownship.id[i], ownship.id[j]) for i, j in zip(*np.where(swlos))]

    return ConflictData(
        rpz=rpz_arr,
        hpz=hpz_arr,
        dtlookahead=dtlookahead_arr,
        confpairs=confpairs,
        confpairs_unique=confpairs_unique,
        lospairs=lospairs,
        qdr=qdr[swconfl],
        dist=dist[swconfl],
        dcpa=np.sqrt(dcpa2[swconfl]),
        tcpa=tcpa[swconfl],
        tLOS=tinconf[swconfl],
        inconf=inconf,
        tcpamax=tcpamax,
        tcpa_all=tcpa_all,
        tinhor_all=tinhor_all,
    )

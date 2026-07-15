'''Propagate the CNS position-noise distributions through the CPA geometry and
histogram the resulting projected-CPA distance (dcpa).

Fixed crossing scenario (all quantities are the ones a CR system would actually
see through ADS-B self-measurements):

  * 90 deg crossing, both aircraft at 20 kts
  * current separation (range) = 100 m
  * true dcpa = 40 m  (a genuine conflict — inside the RPZ)
  * RPZ = 50 m
  * position CI95 = 10 m, velocity CI95 = 1 m/s  (per-axis sigma = CI95 / 2.448)

Both aircraft self-measure position *and* velocity with independent noise; the
observed relative position/velocity are formed from the two noisy self-reports
and the projected-CPA distance is recomputed exactly as ``cd.statebased`` does:

  dcpa = | r_obs x w_hat_obs |            (perpendicular distance of the relative
                                           position to the relative-velocity line)

The **position** noise distribution is swept over the four stochastic CNS models
(normal / heavy-tail / anisotropic / heavy-tail+anisotropic — the same factories
exp3/exp4 use); the velocity noise is always the isotropic Gaussian the sim
applies. For the two anisotropic models the along-track (wider) axis is oriented
per-aircraft by its own track, so ownship and intruder stretch in different
directions.

The velocity error propagates to the CPA over a lever arm set by the time-to-CPA,
which here follows from the current separation (``RANGE_M``) rather than being a
free constant: at 100 m / dcpa 40 m the CPA is only ~6 s away, so the estimate
sits in the near-CPA, position-noise-dominated regime and the four distributions
are distinguishable. At larger ranges the (always-Gaussian) velocity noise,
propagated over a longer lever arm, dominates and washes the distributions
together — increase ``RANGE_M`` to see that.

Output: one overlaid histogram of |dcpa| per distribution, with reference lines
at the true dcpa (30 m) and the RPZ (50 m); the legend reports the
missed-detection rate P(observed dcpa > RPZ) each distribution induces. A
summary table is printed to stdout.

Run directly::

    python analysis/noise_dist_sensitivity/dcpa_noise_propagation.py
'''
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.config import TAIL_RATIO, TAIL_WEIGHT, ANISO_VAR_RATIO, KTS_TO_MS
from sim_models.noise_distributions import (
    gaussian, make_mixture_gaussian, make_anisotropic_gaussian,
    make_anisotropic_mixture_gaussian,
)

# ── Scenario ──────────────────────────────────────────────────────────────────
SPEED_KTS  = 20.0
SPEED_MS   = SPEED_KTS * KTS_TO_MS       # ground speed of both aircraft (m/s)
RANGE_M    = 100.0                       # m — current inter-aircraft separation
TRUE_DCPA  = 40.0                        # m — genuine conflict, inside the RPZ
RPZ_M      = 50.0                        # m — protected-zone radius
POS_CI95   = 10.0                        # m
VEL_CI95   = 1.0                         # m/s
CI95_TO_STD_2D = 2.448                   # matches adsl_module / noise_distributions

# The estimate is taken at the current separation RANGE_M (not a free-floating
# time): the along-track closing distance is sqrt(RANGE_M**2 - TRUE_DCPA**2) and
# the time-to-CPA that the velocity error propagates over follows from it (see
# _geometry / the printed tcpa). RANGE_M >= TRUE_DCPA is required.

OWN_TRK_DEG = 0.0                        # ownship heading (north-up)
INT_TRK_DEG = 90.0                       # intruder heading (east) → 90 deg crossing

N_SAMPLES  = 400_000
SEED       = 0

# ── Noise-distribution models (position); velocity is always Gaussian ─────────
MODELS = [
    ('normal',           'Normal',                    'tab:blue',  gaussian),
    ('heavy_tail',       'Heavy-tail',                'tab:green',
        make_mixture_gaussian(TAIL_RATIO, TAIL_WEIGHT)),
    ('anisotropic',      'Anisotropic',               'tab:red',
        make_anisotropic_gaussian(ANISO_VAR_RATIO)),
    ('heavy_tail_aniso', 'Heavy-tail + anisotropic',  'tab:brown',
        make_anisotropic_mixture_gaussian(ANISO_VAR_RATIO, TAIL_RATIO, TAIL_WEIGHT)),
]

_HERE       = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR  = os.path.join(_HERE, 'figures')
OUT_PATH    = os.path.join(FIGURE_DIR, 'dcpa_noise_propagation_90deg.png')


def _velocity(trk_deg, speed):
    '''(east, north) velocity components for a track angle (deg from north).'''
    trk = np.deg2rad(trk_deg)
    return np.array([speed * np.sin(trk), speed * np.cos(trk)])


def _geometry():
    '''Ownship/intruder true states for the fixed 90 deg scenario at RANGE_M.

    Ownship sits at the origin; the intruder is placed at the current separation
    RANGE_M with a noise-free relative geometry giving exactly TRUE_DCPA. Returns
    ``(p_own, p_int, v_own, v_int)`` as (east, north) vectors in metres / m/s.
    '''
    if RANGE_M < TRUE_DCPA:
        raise ValueError(f'RANGE_M ({RANGE_M}) must be >= TRUE_DCPA ({TRUE_DCPA})')
    v_own = _velocity(OWN_TRK_DEG, SPEED_MS)
    v_int = _velocity(INT_TRK_DEG, SPEED_MS)
    w = v_own - v_int                                   # relative velocity (du, dv)
    wn = np.hypot(*w)
    along = w / wn                                      # unit vector along relative motion
    perp = np.array([-along[1], along[0]])             # unit perpendicular
    # Place the intruder at range RANGE_M: back off the along-track closing
    # distance sqrt(RANGE**2 - dcpa**2), offset TRUE_DCPA sideways. Sign of the
    # along-track term only sets past/future CPA; dcpa (a magnitude) is unaffected.
    closing = np.sqrt(RANGE_M ** 2 - TRUE_DCPA ** 2)
    p_int = -closing * along + TRUE_DCPA * perp
    p_own = np.zeros(2)
    return p_own, p_int, v_own, v_int


def _tcpa_s():
    '''Time-to-CPA implied by RANGE_M / TRUE_DCPA (the velocity-error lever arm).'''
    w = _velocity(OWN_TRK_DEG, SPEED_MS) - _velocity(INT_TRK_DEG, SPEED_MS)
    return np.sqrt(RANGE_M ** 2 - TRUE_DCPA ** 2) / np.hypot(*w)


def _dcpa(r, w):
    '''Projected-CPA distance for relative position ``r`` and relative velocity
    ``w`` (both shape ``(n, 2)``): |r x w| / |w|, the perpendicular distance of
    r to the line of relative motion — identical to cd.statebased's dcpa.'''
    cross = r[:, 0] * w[:, 1] - r[:, 1] * w[:, 0]
    return np.abs(cross) / np.hypot(w[:, 0], w[:, 1])


def _sample_dcpa(pos_dist, rng):
    '''Draw N observed dcpa values: both aircraft self-measure position (via
    ``pos_dist``) and velocity (isotropic Gaussian) independently.'''
    p_own, p_int, v_own, v_int = _geometry()
    n = N_SAMPLES
    own_trk = np.full(n, np.deg2rad(OWN_TRK_DEG))
    int_trk = np.full(n, np.deg2rad(INT_TRK_DEG))

    # Position self-measurement error (east, north) per aircraft, in metres.
    e_p_own = pos_dist(n, POS_CI95, rng, own_trk)
    e_p_int = pos_dist(n, POS_CI95, rng, int_trk)

    # Velocity self-measurement error — isotropic Gaussian, per-axis sigma.
    v_std = VEL_CI95 / CI95_TO_STD_2D
    e_v_own = rng.normal(0.0, v_std, size=(n, 2))
    e_v_int = rng.normal(0.0, v_std, size=(n, 2))

    r_obs = (p_int + e_p_int) - (p_own + e_p_own)      # observed relative position
    w_obs = (v_own + e_v_own) - (v_int + e_v_int)      # observed relative velocity
    return _dcpa(r_obs, w_obs)


def _draw_refs(ax):
    '''True-dcpa and RPZ reference verticals, labelled at the bottom of the axes.'''
    ax.axvline(TRUE_DCPA, color='black', linestyle='--', linewidth=1.4)
    ax.axvline(RPZ_M, color='crimson', linestyle=':', linewidth=1.6)
    ax.text(TRUE_DCPA, 0.02, f' true dcpa {TRUE_DCPA:.0f} m', rotation=90,
            va='bottom', ha='right', fontsize=8, transform=ax.get_xaxis_transform())
    ax.text(RPZ_M, 0.02, f' RPZ {RPZ_M:.0f} m', rotation=90, color='crimson',
            va='bottom', ha='right', fontsize=8, transform=ax.get_xaxis_transform())


def main():
    rng = np.random.default_rng(SEED)
    samples = {key: _sample_dcpa(pos_dist, rng) for key, _, _, pos_dist in MODELS}

    # Fine shared bins out to the fattest 99.9th-percentile tail (heavy-tail),
    # so the far tail — the heavy-tail's defining feature — is not clipped.
    hi = 10.0 * np.ceil(max(np.percentile(s, 99.9) for s in samples.values()) / 10.0)
    hi_bulk = 10.0 * np.ceil(max(np.percentile(s, 99.0) for s in samples.values()) / 10.0)
    bins = np.linspace(0.0, hi, 160)

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(15, 6))
    for key, pretty, color, _ in MODELS:
        s = samples[key]
        p_miss = float(np.mean(s > RPZ_M))             # observed "safe" while truly a conflict
        label = f'{pretty}  —  P(dcpa > RPZ) = {p_miss * 100:.2f}%'
        for ax in (ax_lin, ax_log):
            ax.hist(s, bins=bins, density=True, histtype='step', linewidth=2.0,
                    color=color, alpha=0.85, label=label)

    # Left: linear, bulk of the distribution. Right: log-y, same data, showing
    # where the heavy-tail's far tail crosses above the Normal (beyond ~RPZ).
    _draw_refs(ax_lin)
    ax_lin.set_xlim(0.0, hi_bulk)
    ax_lin.set_ylabel('Probability density [1/m]')
    ax_lin.set_title('Bulk (linear)', fontsize=10)

    _draw_refs(ax_log)
    ax_log.set_yscale('log')
    ax_log.set_xlim(0.0, hi)
    ax_log.set_title('Tail (log density)', fontsize=10)
    ax_log.legend(loc='upper right', fontsize=8)

    for ax in (ax_lin, ax_log):
        ax.set_xlabel('Observed projected-CPA distance, |dcpa| [m]')
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        f'dcpa under CNS position-noise distributions — '
        f'90 deg crossing, {SPEED_KTS:.0f} kts, range = {RANGE_M:.0f} m, '
        f'true dcpa = {TRUE_DCPA:.0f} m, RPZ = {RPZ_M:.0f} m, '
        f'pos CI95 = {POS_CI95:.0f} m, vel CI95 = {VEL_CI95:.0f} m/s '
        f'(tcpa = {_tcpa_s():.1f} s)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
    print(f'Saved -> {OUT_PATH}')

    # Numeric summary alongside the plot. P(dcpa>RPZ) is the missed-detection
    # rate; P(dcpa>2·RPZ) isolates the far-tail excess (where heavy tails bite).
    print(f'\nScenario: 90 deg, {SPEED_KTS:.0f} kts, range = {RANGE_M:.0f} m, '
          f'true dcpa = {TRUE_DCPA:.0f} m, RPZ = {RPZ_M:.0f} m, '
          f'pos CI95 = {POS_CI95:.0f} m, vel CI95 = {VEL_CI95:.1f} m/s, '
          f'tcpa = {_tcpa_s():.1f} s, N = {N_SAMPLES:,}')
    print(f'\n{"model":<26}{"mean":>7}{"std":>7}{"p95":>7}{"p99.9":>8}'
          f'{"P(>RPZ)":>10}{"P(>2·RPZ)":>11}')
    for key, pretty, _, _ in MODELS:
        s = samples[key]
        print(f'{pretty:<26}{np.mean(s):>7.1f}{np.std(s):>7.1f}'
              f'{np.percentile(s, 95):>7.1f}{np.percentile(s, 99.9):>8.1f}'
              f'{np.mean(s > RPZ_M) * 100:>9.2f}%{np.mean(s > 2 * RPZ_M) * 100:>10.3f}%')


if __name__ == '__main__':
    main()

'''Diagnostic plot of the six exp3/exp4 position-noise models at pos_ci95=10 m.

For each of the six ``NOISE_MODELS`` conditions used in
``exp3-noise-model-random-angle.py`` / ``exp4-noise-model-random-angle-homogen.py``,
draws a large Monte Carlo sample through the *actual* ``NoiseModel`` code path
(same class used by the simulation), then plots:

* a 2D scatter of the sampled position errors (east/north, metres), with a
  dashed reference circle at r=ci95 for visual containment comparison, and
* the empirical radial CDF, P(||error|| <= r), with a marker at r=ci95 to
  confirm (or, for the latency-biased models, show the deviation from) the
  95% containment guarantee.

A fixed non-zero heading is used for all six draws so the anisotropic /
latency along-track direction is visibly not axis-aligned (it rotates with
aircraft heading, not with east/north).

Run directly::

    python experiments/plot_noise_models.py
'''
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.config import (
    LATENCY_S, TAIL_RATIO, TAIL_WEIGHT, ANISO_VAR_RATIO,
    SPEED_HOMOGEN_KTS, KTS_TO_MS, RESULTS_DIR,
)
from sim_models.noise_distributions import (
    make_mixture_gaussian, make_anisotropic_gaussian, make_anisotropic_mixture_gaussian,
)
from sim_models.noise_model import NoiseModel

CI95 = 10.0
N_SAMPLES = 300_000
N_SCATTER = 3_000
HEADING_DEG = 35.0          # fixed non-zero heading, shared across all draws
GS_MS = SPEED_HOMOGEN_KTS * KTS_TO_MS   # representative groundspeed for latency bias
SEED = 0

# Same (label, pos_dist, latency_s) tuples as exp3/exp4's NOISE_MODELS.
NOISE_MODELS = [
    ('normal',           None,                                                                        0.0),
    ('latency',          None,                                                                        LATENCY_S),
    ('heavy_tail',       make_mixture_gaussian(TAIL_RATIO, TAIL_WEIGHT),                             0.0),
    ('anisotropic',      make_anisotropic_gaussian(ANISO_VAR_RATIO),                                  0.0),
    ('latency_aniso',    make_anisotropic_gaussian(ANISO_VAR_RATIO),                                  LATENCY_S),
    ('heavy_tail_aniso', make_anisotropic_mixture_gaussian(ANISO_VAR_RATIO, TAIL_RATIO, TAIL_WEIGHT), 0.0),
]

COLORS = dict(zip(
    [m[0] for m in NOISE_MODELS],
    ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown'],
))

# Paper-grade panel titles (concise, descriptive), keyed by NOISE_MODELS label.
TITLES = {
    'normal':           'Normal (isotropic Gaussian)',
    'latency':          'Latency (positional bias)',
    'heavy_tail':       'Heavy-tail (Gaussian mixture)',
    'anisotropic':      'Anisotropic (along-/cross-track)',
    'latency_aniso':    'Latency + anisotropic',
    'heavy_tail_aniso': 'Heavy-tail + anisotropic',
}


class _DummyStates:
    '''Minimal stand-in for the ``states`` object NoiseModel.add_position_noise
    expects: fixed lat/lon/heading/groundspeed for every sample.'''
    def __init__(self, n, lat0, lon0, trk_deg, gs_ms):
        self.lat = np.full(n, lat0)
        self.lon = np.full(n, lon0)
        self.trk = np.full(n, trk_deg)
        self.gs  = np.full(n, gs_ms)


class _DummyMsg:
    def __init__(self, n, lat0, lon0):
        self.lat = np.full(n, lat0)
        self.lon = np.full(n, lon0)


def sample_east_north(n, pos_dist, latency_s, seed):
    '''Draw n position-error samples (east_m, north_m) through the real
    NoiseModel.add_position_noise code path.'''
    lat0, lon0 = 40.0, -70.0
    states = _DummyStates(n, lat0, lon0, HEADING_DEG, GS_MS)
    msg = _DummyMsg(n, lat0, lon0)
    nm = NoiseModel(
        pos_std_m=CI95 / 2.448, vel_std_ms=1.0,
        rng=np.random.default_rng(seed),
        pos_dist=pos_dist, latency_s=latency_s,
    )
    nm.add_position_noise(msg, states, np.arange(n))
    coslat = np.cos(np.deg2rad(lat0))
    north_m = (msg.lat - lat0) * 111_320.0
    east_m  = (msg.lon - lon0) * 111_320.0 * coslat
    return east_m, north_m


def radial_cdf_curve(radii, r_grid):
    radii_sorted = np.sort(radii)
    return np.searchsorted(radii_sorted, r_grid, side='right') / radii_sorted.size


def main():
    samples = {}
    for label, pos_dist, latency_s in NOISE_MODELS:
        east, north = sample_east_north(N_SAMPLES, pos_dist, latency_s, seed=SEED)
        samples[label] = (east, north)

    # Shared scatter axis limit: cover the 99.5th percentile radial extent
    # across all six models, rounded up, so panels are visually comparable.
    all_r99 = [
        np.percentile(np.hypot(e, n), 99.5) for (e, n) in samples.values()
    ]
    lim = 5.0 * np.ceil(max(all_r99) / 5.0)

    fig = plt.figure(figsize=(14, 9.5))
    gs_fig = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.15], hspace=0.4, wspace=0.3)

    theta = np.linspace(0, 2 * np.pi, 200)
    circle_x, circle_y = CI95 * np.cos(theta), CI95 * np.sin(theta)

    for i, (label, _, _) in enumerate(NOISE_MODELS):
        ax = fig.add_subplot(gs_fig[i // 3, i % 3])
        east, north = samples[label]
        idx = np.random.default_rng(1).choice(east.size, size=N_SCATTER, replace=False)
        ax.scatter(east[idx], north[idx], s=3, alpha=0.35, color=COLORS[label], linewidths=0)
        ax.plot(circle_x, circle_y, '--', color='black', linewidth=1.2, label=f'r = {CI95:.0f} [m]')
        ax.axhline(0, color='grey', linewidth=0.5)
        ax.axvline(0, color='grey', linewidth=0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_title(TITLES[label], fontsize=11)
        ax.set_xlabel('East [m]', fontsize=8)
        ax.set_ylabel('North [m]', fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(loc='upper right', fontsize=7)

    ax_cdf = fig.add_subplot(gs_fig[2, :])
    r_grid = np.linspace(0, lim, 400)
    for label, _, _ in NOISE_MODELS:
        east, north = samples[label]
        radii = np.hypot(east, north)
        cdf = radial_cdf_curve(radii, r_grid)
        ax_cdf.plot(r_grid, cdf, color=COLORS[label], label=TITLES[label], linewidth=2.2, alpha=0.65)

    ax_cdf.axhline(0.95, color='black', linestyle=':', linewidth=1.0)
    ax_cdf.axvline(CI95, color='black', linestyle='--', linewidth=1.0)
    ax_cdf.text(CI95 + 0.3, 0.05, f'ci95 = {CI95:.0f} [m]', fontsize=9)
    ax_cdf.set_xlabel('Radial error r [m]')
    ax_cdf.set_ylabel('P(||error|| <= r) [-]')
    ax_cdf.set_title('Empirical radial CDF')
    ax_cdf.set_xlim(0, lim)
    ax_cdf.set_ylim(0, 1.0)
    ax_cdf.legend(loc='lower right', fontsize=8, ncol=2)

    os.makedirs(os.path.join(RESULTS_DIR, 'figures'), exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'figures', 'noise_models_ci95_10m.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved -> {out_path}')

    # Print the empirical 95th percentile radial error per model for a quick
    # numeric sanity check alongside the plot (should read ~10 m except for
    # the two latency-biased conditions, where the deterministic bias pushes
    # it slightly past 10 m).
    print(f'\n{"model":<18} {"95th pct radial [m]":>20}')
    for label, _, _ in NOISE_MODELS:
        east, north = samples[label]
        r95 = np.percentile(np.hypot(east, north), 95)
        print(f'{label:<18} {r95:>20.3f}')


if __name__ == '__main__':
    main()

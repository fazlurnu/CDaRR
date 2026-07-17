'''Distribution-example figure for the appendix "Position-Noise Models".

Draws one panel per position-noise model (the six compared in Experiment 3),
each showing a scatter of sampled measurement errors (east, north) in metres
with the shared CI95 = 10 m radial containment circle overlaid. The point is to
show that although the six models differ in shape (isotropic, biased,
heavy-tailed, anisotropic, and their combinations), all are calibrated to the
same 95% radial confidence interval.

Sampling mirrors the sim exactly:
  * the isotropic / mixture / anisotropic draws come straight from
    ``sim_models.noise_distributions`` (the same callables exp3/exp4 use), and
  * the deterministic along-track latency bias (-latency_s * gs, along the
    reported track) matches ``NoiseModel.add_latency_bias``.

Output: ``paper/Journal___JRESS___Probabilistic_Recovery_REVISED/fig_noise_models_ci95.pgf``

Run (from repo root, in the ``cdarr`` env, with TeX on PATH):
    conda activate cdarr
    python analysis/noise_models_example/plot_noise_models_ci95.py
'''
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# repo root on sys.path so sim_models / experiments import when run as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sim_models.noise_distributions import (
    gaussian, make_mixture_gaussian,
    make_anisotropic_gaussian, make_anisotropic_mixture_gaussian,
)
from experiments.config import (
    POS_CI95, LATENCY_S, TAIL_RATIO, TAIL_WEIGHT, ANISO_VAR_RATIO,
    SPEED_MAX_KTS, KTS_TO_MS,
)

# match the repo's pgf/font conventions (font size 14 throughout)
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'pgf.texsystem': 'pdflatex',
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})

# ── Example parameters ────────────────────────────────────────────────────────
CI95   = float(POS_CI95)                 # 10 m radial confidence interval
N      = 4000                            # samples per panel
SEED   = 12345
HDG    = np.deg2rad(45.0)                # fixed heading (from north) for the example
GS     = SPEED_MAX_KTS * KTS_TO_MS       # representative groundspeed for the latency bias
LIM    = 18.0                            # axis half-range (m)


def latency_bias():
    '''Deterministic along-track lag, matching NoiseModel.add_latency_bias.'''
    bias_at = -LATENCY_S * GS
    return np.array([bias_at * np.sin(HDG), bias_at * np.cos(HDG)])  # (east, north)


def sample_models():
    '''Return {label: (title, samples (N,2))} for the six models.'''
    rng = np.random.default_rng(SEED)
    trk = np.full(N, HDG)                 # per-sample track for anisotropic draws
    bias = latency_bias()

    mixture   = make_mixture_gaussian(TAIL_RATIO, TAIL_WEIGHT)
    aniso     = make_anisotropic_gaussian(ANISO_VAR_RATIO)
    aniso_mix = make_anisotropic_mixture_gaussian(ANISO_VAR_RATIO, TAIL_RATIO, TAIL_WEIGHT)

    return [
        ('normal',           'Normal',                    gaussian(N, CI95, rng)),
        ('latency',          'Latency (bias)',            gaussian(N, CI95, rng) + bias),
        ('heavy_tail',       'Heavy-tail',                mixture(N, CI95, rng)),
        ('anisotropic',      'Anisotropic',               aniso(N, CI95, rng, trk)),
        ('latency_aniso',    'Latency + anisotropic',     aniso(N, CI95, rng, trk) + bias),
        ('heavy_tail_aniso', 'Heavy-tail + anisotropic',  aniso_mix(N, CI95, rng, trk)),
    ]


def main():
    panels = sample_models()

    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.4), sharex=True, sharey=True)
    for ax, (_label, title, xy) in zip(axes.ravel(), panels):
        ax.scatter(xy[:, 0], xy[:, 1], s=4, alpha=0.12, color='C0',
                   edgecolors='none', rasterized=True)
        ax.add_patch(Circle((0, 0), CI95, fill=False, ls='--', lw=1.3, color='C3'))

        # empirical fraction inside the CI95 circle (should be ~0.95 for all)
        frac = np.mean(np.hypot(xy[:, 0], xy[:, 1]) <= CI95)
        ax.set_title(title)
        ax.text(0.5, 0.02, rf'${100*frac:.1f}\%$ within CI95',
                transform=ax.transAxes, ha='center', va='bottom', fontsize=12)

        ax.set_xlim(-LIM, LIM)
        ax.set_ylim(-LIM, LIM)
        ax.set_aspect('equal')
        ax.axhline(0, color='0.85', lw=0.6, zorder=0)
        ax.axvline(0, color='0.85', lw=0.6, zorder=0)

    for ax in axes[-1, :]:
        ax.set_xlabel(r'East error (m)')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'North error (m)')

    fig.tight_layout()

    out_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..',
        'paper', 'Journal___JRESS___Probabilistic_Recovery_REVISED'))
    out_path = os.path.join(out_dir, 'fig_noise_models_ci95.pgf')
    fig.savefig(out_path, bbox_inches='tight')
    print('wrote', out_path)
    print('heading = 45 deg (from north), groundspeed = %.2f m/s (%.0f kts), '
          'latency = %.2f s' % (GS, SPEED_MAX_KTS, LATENCY_S))


if __name__ == '__main__':
    main()

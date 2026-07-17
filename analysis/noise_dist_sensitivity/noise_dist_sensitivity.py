'''Noise-distribution sensitivity of IPR and CPA distance, across recovery methods.

One composite figure per experiment sweep: a bar chart of IPR (top panel) and
a boxplot of the raw CPA-distance distribution (bottom panel), both vs.
noise-distribution model, sharing the x-axis, grouped by recovery method
(Past-CPA / FTR / Probabilistic FTR), at pos_ci95=10 m. Run for both
experiment sweeps:

* exp3 -- heterogeneous speed (Uniform(10, 30) kts per pair)  -> "heterogenous_*"
* exp4 -- homogeneous speed                                    -> "homogenous_*"

Sources per sweep (all under experiments/results/):
* ``exp{3,4}.npz``             -- Past-CPA, all 6 noise models
* ``exp{3,4}_probftr_ftr.npz`` -- FTR + Probabilistic FTR, all 6 noise models
* ``exp{3,4}_latency.npz``     -- Past-CPA + FTR + Probabilistic FTR,
  latency-only models (``latency``, ``latency_aniso``)

The two latency models were re-run after ``exp{3,4}.npz`` /
``exp{3,4}_probftr_ftr.npz`` with an updated latency noise model (see commits
"bump LATENCY_S to 0.1s stress-test value" and "update latency noise model"),
so ``exp{3,4}_latency.npz`` is the authoritative source for ``latency`` /
``latency_aniso`` for all three recovery methods -- the other two files are
only used for the remaining four (non-latency) noise models.

Figures are saved under analysis/noise_dist_sensitivity/figures/. Run directly::

    python analysis/noise_dist_sensitivity/noise_dist_sensitivity.py
'''
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'experiments', 'results')
POS_CI95 = 10.0  # only level present in exp3_latency.npz / exp4_latency.npz

SWEEPS = [
    {'exp_prefix': 'exp3', 'out_prefix': 'heterogenous'},
    {'exp_prefix': 'exp4', 'out_prefix': 'homogenous'},
]

RECOVERY_ORDER = ['cpa', 'ftr', 'probabilistic']
RECOVERY_COLORS = {
    'cpa': 'tab:blue',
    'ftr': 'tab:green',
    'probabilistic': 'tab:orange',
}
RECOVERY_PRETTY = {
    'cpa': 'Past-CPA',
    'ftr': 'FTR',
    'probabilistic': 'Probabilistic FTR',
}

NOISE_MODEL_ORDER = [
    'normal', 'heavy_tail', 'anisotropic', 'heavy_tail_aniso',
    'latency', 'latency_aniso',
]
NOISE_MODEL_PRETTY = {
    'normal':           'Normal',
    'heavy_tail':       'Heavy-tail',
    'anisotropic':      'Anisotropic',
    'heavy_tail_aniso': 'Heavy-tail\n+ anisotropic',
    'latency':          'Latency',
    'latency_aniso':    'Latency\n+ anisotropic',
}
LATENCY_MODELS = {'latency', 'latency_aniso'}

GROUP_WIDTH = 0.8   # total width (in x-units) spanned by one noise-model's bars/boxes
BAR_GAP = 0.15      # fraction of each bar's slot left empty, as breathing room between bars
RPZ_M = 50.0        # protected-zone radius (see get_ipr_stochastic_env.py: "dcpa: uniform 0-RPZ (50 m)")


def _ci_index(pos_ci95_levels, target):
    idx = np.flatnonzero(np.isclose(pos_ci95_levels, target))
    if idx.size == 0:
        raise ValueError(f'pos_ci95={target} not found in levels {pos_ci95_levels}')
    return int(idx[0])


def _by_model(npz, recovery_label, ci_i, field, reduce_fn):
    '''npz[field][ci, recovery, model, ...] -> {model_label: reduce_fn(values)}.'''
    recovery_labels = list(npz['recovery_labels'])
    noise_labels = list(npz['noise_labels'])
    ri = recovery_labels.index(recovery_label)
    values = npz[field][ci_i, ri, :]  # (n_models, ...)
    return {model: reduce_fn(values[mi]) for mi, model in enumerate(noise_labels)}


def _load_table(exp_prefix, field, reduce_fn):
    '''Returns {noise_model: {recovery_label: reduced_value}} for NOISE_MODEL_ORDER,
    reducing over exp{prefix}.npz / exp{prefix}_probftr_ftr.npz /
    exp{prefix}_latency.npz per the module-level source rules.'''
    cpa_npz = np.load(os.path.join(RESULTS_DIR, f'{exp_prefix}.npz'), allow_pickle=True)
    ftr_npz = np.load(os.path.join(RESULTS_DIR, f'{exp_prefix}_probftr_ftr.npz'), allow_pickle=True)
    lat_npz = np.load(os.path.join(RESULTS_DIR, f'{exp_prefix}_latency.npz'), allow_pickle=True)

    ci_cpa = _ci_index(cpa_npz['pos_ci95_levels'], POS_CI95)
    ci_ftr = _ci_index(ftr_npz['pos_ci95_levels'], POS_CI95)
    ci_lat = _ci_index(lat_npz['pos_ci95_levels'], POS_CI95)

    cpa_by_model = _by_model(cpa_npz, 'cpa', ci_cpa, field, reduce_fn)
    ftr_by_model = _by_model(ftr_npz, 'ftr', ci_ftr, field, reduce_fn)
    prob_by_model = _by_model(ftr_npz, 'probabilistic', ci_ftr, field, reduce_fn)

    lat_by_recovery = {
        r: _by_model(lat_npz, r, ci_lat, field, reduce_fn) for r in RECOVERY_ORDER
    }

    table = {}
    for model in NOISE_MODEL_ORDER:
        if model in LATENCY_MODELS:
            table[model] = {r: lat_by_recovery[r][model] for r in RECOVERY_ORDER}
        else:
            table[model] = {
                'cpa': cpa_by_model[model],
                'ftr': ftr_by_model[model],
                'probabilistic': prob_by_model[model],
            }
    return table


def load_ipr_table(exp_prefix):
    '''Returns {noise_model: {recovery_label: mean_ipr}}.'''
    return _load_table(exp_prefix, 'ipr', lambda v: float(np.mean(v)))


def load_mincpa_distributions(exp_prefix):
    '''Returns {noise_model: {recovery_label: flat_array_of_cpa_m}}, the raw
    (runs x pairs) minimum-separation samples -- for the CPA boxplot panel.'''
    return _load_table(exp_prefix, 'min_cpa', lambda v: np.asarray(v).ravel())


def _draw_grouped_bars(ax, table, *, value_fmt=None, ylim=None):
    '''Draws the IPR bars (one cluster per noise model, one bar per recovery
    method, with a small gap between bars in the same cluster) onto ax.'''
    n_models = len(NOISE_MODEL_ORDER)
    n_recovery = len(RECOVERY_ORDER)
    x = np.arange(n_models)
    slot = GROUP_WIDTH / n_recovery
    bar_width = slot * (1.0 - BAR_GAP)

    for i, recovery in enumerate(RECOVERY_ORDER):
        values = [table[model][recovery] for model in NOISE_MODEL_ORDER]
        offset = (i - (n_recovery - 1) / 2) * slot
        bars = ax.bar(x + offset, values, width=bar_width,
                       color=RECOVERY_COLORS[recovery], label=RECOVERY_PRETTY[recovery])
        if value_fmt is not None:
            ax.bar_label(bars, labels=[value_fmt(v) for v in values],
                         padding=2, fontsize=11)

    ax.set_xticks(x)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _draw_grouped_boxplot(ax, table, *, ylim=None):
    '''Draws the CPA-distance boxes (one cluster per noise model, one box per
    recovery method) onto ax. Default Tukey whiskers (1.5*IQR); outlier points
    are hidden (showfliers=False) since the raw min_cpa tail is very long (up to
    ~1000 m) and would otherwise swamp the plot. Whisker reach conveys spread,
    not the fraction below RPZ -- that fraction is the IPR panel's job (the
    caption states this so the boxplot is not misread).'''
    n_models = len(NOISE_MODEL_ORDER)
    n_recovery = len(RECOVERY_ORDER)
    x = np.arange(n_models)
    slot = GROUP_WIDTH / n_recovery

    for i, recovery in enumerate(RECOVERY_ORDER):
        data = [table[model][recovery] for model in NOISE_MODEL_ORDER]
        offset = (i - (n_recovery - 1) / 2) * slot
        positions = x + offset
        bp = ax.boxplot(
            data, positions=positions, widths=slot * 0.9,
            patch_artist=True, showfliers=False,
            medianprops=dict(color='black'),
        )
        for patch in bp['boxes']:
            patch.set_facecolor(RECOVERY_COLORS[recovery])
            patch.set_alpha(0.8)

    ax.axhline(RPZ_M, color='tab:red', linestyle='--', linewidth=1.5, zorder=0)

    ax.set_xticks(x)
    if ylim is not None:
        ax.set_ylim(*ylim)


def plot_composite(ipr_table, cpa_dist_table, out_path):
    '''One figure, two panels sharing the noise-model x-axis: IPR bars on top,
    CPA-distance boxplot below. A single legend (recovery method) sits above
    both panels.'''
    fig, (ax_ipr, ax_box) = plt.subplots(
        2, 1, figsize=(12, 9), sharex=True,
        gridspec_kw={'height_ratios': [1.0, 1.3], 'hspace': 0.08},
    )

    _draw_grouped_bars(ax_ipr, ipr_table, value_fmt=lambda v: f'{v:.4f}', ylim=(0.9, 1.01))
    ax_ipr.set_ylabel('IPR [-]')
    ax_ipr.tick_params(axis='x', labelbottom=False)

    _draw_grouped_boxplot(ax_box, cpa_dist_table)
    ax_box.set_ylabel('CPA distance [m]')
    ax_box.set_xlabel('Noise distribution model')
    ax_box.set_xticklabels([NOISE_MODEL_PRETTY[m] for m in NOISE_MODEL_ORDER])

    handles = [plt.Rectangle((0, 0), 1, 1, color=RECOVERY_COLORS[r], alpha=0.8)
               for r in RECOVERY_ORDER]
    labels = [RECOVERY_PRETTY[r] for r in RECOVERY_ORDER]
    handles.append(plt.Line2D([0], [0], color='tab:red', linestyle='--', linewidth=1.5))
    labels.append(f'RPZ = {RPZ_M:g} m')
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95),
               ncol=len(handles), frameon=True)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {out_path}')


def _print_table(table, header, fmt, reduce_fn=None):
    print(f'\n{header:<20} ' + ' '.join(f'{RECOVERY_PRETTY[r]:>18}' for r in RECOVERY_ORDER))
    for model in NOISE_MODEL_ORDER:
        row = ' '.join(
            f'{(reduce_fn(table[model][r]) if reduce_fn else table[model][r]):>18{fmt}}'
            for r in RECOVERY_ORDER
        )
        print(f'{model:<20} {row}')


def run_sweep(exp_prefix, out_prefix, figures_dir):
    print(f'\n=== {out_prefix} ({exp_prefix}) ===')

    ipr_table = load_ipr_table(exp_prefix)
    _print_table(ipr_table, 'model (IPR)', '.4f')

    cpa_dist_table = load_mincpa_distributions(exp_prefix)
    _print_table(cpa_dist_table, 'model (median CPA [m])', '.1f', reduce_fn=np.median)

    plot_composite(
        ipr_table, cpa_dist_table,
        os.path.join(figures_dir, f'fig_{out_prefix}_noise_dist_sensitivity.pgf'),
    )


def main():
    figures_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    for sweep in SWEEPS:
        run_sweep(sweep['exp_prefix'], sweep['out_prefix'], figures_dir)


if __name__ == '__main__':
    main()

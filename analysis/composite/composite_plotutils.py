'''Self-contained plotting helpers for the composite CD&R figures.

Extracted from the CDaRR_FP ``plot_utils.py`` and trimmed to exactly what the
composite + aggregate-resolution-flag scripts need, so this package runs inside
CDaRR_git without pulling in the FP simulation stack (``runners``, ``envs`` …).

The scripts operate purely on the *sanitised* run namespaces stored under
``data/*.pkl`` (produced from the FP caches); each run is a ``SimpleNamespace``
with the fields ``t_arr, dist_arr, dcpa_gt_arr, dcpa_obs_arr, lat_arr, lon_arr,
avoid_arr, min_dist`` plus scalars ``rpz, dpsi, pos_ci95, vel_ci95,
reception_prob, latency_s, ipr`` and a light ``env`` (``ownship_idx,
intruder_idx, ownship_ids, intruder_ids, nb_pair``). No BlueSky run is needed.
'''
import os

import matplotlib
import numpy as np

_SAVE_DPI = 300
_SAVE_FORMATS = ("png",)
_R_EARTH = 6371000.0
_FILE_PREFIX = "fig_stochastic_pairwise_hor_conflict"

# Fixed colours + display names for the strategy comparison (match FP).
_COMPARE_COLORS = {
    "cpa": "tab:blue",
    "double_criteria": "tab:green",
    "probabilistic": "tab:orange",
}
_STRATEGY_ORDER  = ("cpa", "double_criteria", "probabilistic")
_STRATEGY_PRETTY = {
    "cpa":             "Past-CPA",
    "double_criteria": "FTR",
    "probabilistic":   "Probabilistic FTR",
}


def set_latex_style(enable=True):
    '''Publication typography (serif CM fonts, inward ticks, 300 dpi). ``enable``
    toggles the LaTeX-friendly variant; kept dependency-free (no usetex probe).'''
    global _SAVE_DPI, _SAVE_FORMATS
    rc = {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.grid": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
    if enable:
        rc.update({
            "text.usetex": True,
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 14,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        })
    else:
        rc.update({"mathtext.fontset": "cm"})
    matplotlib.rcParams.update(rc)
    _SAVE_DPI = 300
    _SAVE_FORMATS = ("pgf", "png") if enable and matplotlib.rcParams.get("text.usetex") else ("png",)


def _write(fig, figure_dir, stem):
    '''Save *fig* as ``stem.<fmt>`` for every configured format; close, return the
    first path.'''
    import matplotlib.pyplot as plt
    os.makedirs(figure_dir, exist_ok=True)
    paths = []
    for fmt in _SAVE_FORMATS:
        path = os.path.join(figure_dir, f"{stem}.{fmt}")
        fig.savefig(path, dpi=_SAVE_DPI)
        paths.append(path)
    plt.close(fig)
    return paths[0]


def _save(fig, figure_dir, name):
    fig.tight_layout()
    return _write(fig, figure_dir, os.path.splitext(name)[0])


def _title_suffix(res):
    lat_s = getattr(res, "latency_s", 0.0)
    lat_str = f", latency={lat_s} s" if lat_s else ""
    return (f"pos_ci95={res.pos_ci95} m, vel_ci95={res.vel_ci95} m/s, "
            f"p_rx={res.reception_prob}{lat_str}")


def _cpa_time(res, pair):
    '''Sim time (s) of the actual closest point of approach for the pair.'''
    return float(res.t_arr[int(np.argmin(res.dist_arr[:, pair]))])


def _draw_pair_trajectory(ax, res, pair):
    '''One pair's ownship-centric trajectory (nominal tracks, avoiding segments
    emphasised, CPA marker). Returns the CPA tick index.'''
    env = res.env
    lat_arr, lon_arr, avoid_arr = res.lat_arr, res.lon_arr, res.avoid_arr
    i_own, i_int = env.ownship_idx[pair], env.intruder_idx[pair]

    lat0, lon0 = float(lat_arr[0, i_own]), float(lon_arr[0, i_own])
    lat0r = np.deg2rad(lat0)
    x_own = np.deg2rad(lon_arr[:, i_own] - lon0) * _R_EARTH * np.cos(lat0r)
    y_own = np.deg2rad(lat_arr[:, i_own] - lat0) * _R_EARTH
    x_int = np.deg2rad(lon_arr[:, i_int] - lon0) * _R_EARTH * np.cos(lat0r)
    y_int = np.deg2rad(lat_arr[:, i_int] - lat0) * _R_EARTH

    own_av = avoid_arr[:, i_own] == 1.0
    int_av = avoid_arr[:, i_int] == 1.0
    ax.plot(x_own, y_own, color="lightskyblue", alpha=0.4, label="Ownship (nominal)")
    ax.plot(x_int, y_int, color="lightsalmon",  alpha=0.4, label="Intruder (nominal)")
    ax.plot(np.where(own_av, x_own, np.nan), np.where(own_av, y_own, np.nan),
            color="tab:blue", label="Ownship (resolving)")
    ax.plot(np.where(int_av, x_int, np.nan), np.where(int_av, y_int, np.nan),
            color="tab:red", label="Intruder (resolving)")

    i_cpa = int(np.argmin(res.dist_arr[:, pair]))
    ax.scatter([x_own[i_cpa], x_int[i_cpa]], [y_own[i_cpa], y_int[i_cpa]],
               color="dimgray", marker="x", s=80, zorder=5, label="CPA")
    return i_cpa


def _shade_avoidance(ax, res, pair, color="lightgray", alpha=0.6):
    '''Shade every time span where the ownship's avoidance flag is active.'''
    active = res.avoid_arr[:, res.env.ownship_idx[pair]] == 1.0
    edges  = np.flatnonzero(np.diff(np.concatenate(([0], active.astype(int), [0]))))
    starts, ends = edges[0::2], edges[1::2] - 1
    for k, (s, e) in enumerate(zip(starts, ends)):
        ax.axvspan(res.t_arr[s], res.t_arr[min(e, len(res.t_arr) - 1)],
                   color=color, alpha=alpha, zorder=0, linewidth=0,
                   label="Resolution active" if k == 0 else None)


def _draw_distance_shaded(ax, res, pair, dist_max=None):
    '''Actual distance panel with the avoidance-active span shaded in.'''
    _shade_avoidance(ax, res, pair)
    ax.plot(res.t_arr, res.dist_arr[:, pair], color="tab:blue", label="Actual distance")
    ax.axhline(res.rpz, color="tab:red", linestyle="--", label=f"RPZ = {res.rpz} m")
    if dist_max is not None:
        ax.set_ylim(-2, dist_max)


def _draw_dcpa_compare_shaded(ax, res, pair, dcpa_max=None):
    '''Projected-CPA-distance panel (truth vs observed) with the same shading.'''
    _shade_avoidance(ax, res, pair)
    ax.plot(res.t_arr, res.dcpa_gt_arr[:, pair], color="tab:green",
            label="Projected CPA (truth)")
    ax.plot(res.t_arr, res.dcpa_obs_arr[:, pair], color="tab:orange",
            alpha=0.85, label="Projected CPA (observed)")
    ax.axhline(res.rpz, color="tab:red", linestyle="--", label=f"RPZ = {res.rpz} m")
    if dcpa_max is not None:
        ax.set_ylim(-2, dcpa_max)


def plot_pair_cdr_composite(results_by_label, figure_dir, pair, *, t_max=None,
                            dist_max=None, dcpa_max=None, traj_xlim=None,
                            traj_ylim=None, order=_STRATEGY_ORDER, stem=None,
                            time_width_frac=1.0, col_width=3.3, square_traj=True):
    '''One composite figure, 3 rows × N strategy columns. Row 0 is the geometric
    ownship-centric trajectory, row 1 the actual distance vs time, row 2 the
    projected CPA distance (truth vs observed) vs time — the avoidance-active
    span is shaded on both time rows. Columns share trajectory axes and, per row,
    the time-domain y-axis, so the three strategies are compared on one identical
    encounter. Returns the saved path.

    ``time_width_frac`` (0 < f <= 1) narrows only the two time-domain rows
    (actual distance, projected CPA): each is centred inside its column with
    (1-f)/2 padding on each side, so the wide square trajectory panels above are
    left untouched. f=1.0 is the original full-width layout.

    ``col_width`` sets the canvas width per column (inches). The trajectory
    panels are aspect-locked vertical strips, so lowering it mainly trims the
    dead horizontal whitespace and narrows the time rows without shrinking the
    trajectories. A narrower canvas makes the whole figure a more portrait
    aspect, so at a fixed ``\\linewidth`` LaTeX renders it taller/larger. The
    default 3.3 gives a near-square canvas; 4.6 reproduces the old landscape
    width.'''
    import matplotlib.pyplot as plt
    labels = [l for l in order if l in results_by_label]
    n = len(labels)
    # Row index 1 is a thin spacer that reserves room for the trajectory legend
    # (drawn between the trajectory row and the two time-domain rows).
    fig = plt.figure(figsize=(col_width * n, 9.3))
    gs = fig.add_gridspec(4, n, height_ratios=[1.7, 0.1, 1.0, 1.0],
                          hspace=0.3, wspace=0.18,
                          top=0.925, bottom=0.14, left=0.075, right=0.955)
    axes_traj = [fig.add_subplot(gs[0, c]) for c in range(n)]

    def _time_axis(row, c):
        # Full-width (f=1) keeps the original single-cell axis; otherwise nest a
        # 3-column [pad, plot, pad] sub-grid so the plot is centred and narrower.
        if time_width_frac >= 1.0:
            return fig.add_subplot(gs[row, c])
        pad = (1.0 - time_width_frac) / 2.0
        sub = gs[row, c].subgridspec(1, 3, width_ratios=[pad, time_width_frac, pad],
                                     wspace=0.0)
        return fig.add_subplot(sub[0, 1])

    axes_dist = [_time_axis(2, c) for c in range(n)]
    axes_dcpa = [_time_axis(3, c) for c in range(n)]

    # A square trajectory box under equal aspect needs equal x/y spans, so widen
    # the shorter window symmetrically about its centre (never clips the paths).
    tx, ty = traj_xlim, traj_ylim
    if square_traj and tx is not None and ty is not None:
        span = max(tx[1] - tx[0], ty[1] - ty[0])
        xc, yc = 0.5 * (tx[0] + tx[1]), 0.5 * (ty[0] + ty[1])
        tx = (xc - 0.5 * span, xc + 0.5 * span)
        ty = (yc - 0.5 * span, yc + 0.5 * span)

    for col, l in enumerate(labels):
        res = results_by_label[l]
        ax_traj, ax_dist, ax_dcpa = axes_traj[col], axes_dist[col], axes_dcpa[col]
        t_cpa = _cpa_time(res, pair)

        # Row 0 — geometric trajectory (detect → resolve → recover → return).
        _draw_pair_trajectory(ax_traj, res, pair)
        if tx is not None:
            ax_traj.set_xlim(*tx)
        if ty is not None:
            ax_traj.set_ylim(*ty)
        ax_traj.set_aspect("equal", adjustable="box")
        ax_traj.set_xlabel("East [m]")
        los = res.min_dist[pair] < res.rpz
        ax_traj.set_title(
            f"{_STRATEGY_PRETTY.get(l, l)}\n"
            f"CPA = {res.min_dist[pair]:.1f} m ({'LOS' if los else 'safe'})",
            fontsize=14, linespacing=1.6,
        )

        # Rows 1 & 2 — time domain, recovery timing via the shaded active span.
        _draw_distance_shaded(ax_dist, res, pair, dist_max=dist_max if col == 0 else None)
        for ln in ax_dist.get_lines():           # recolour actual-distance line
            if ln.get_label() == "Actual distance":
                ln.set_color("tab:olive")
        _draw_dcpa_compare_shaded(ax_dcpa, res, pair, dcpa_max=dcpa_max if col == 0 else None)
        for ax in (ax_dist, ax_dcpa):
            ax.axvline(t_cpa, color="dimgray", linestyle=":", linewidth=1.5)
            if t_max is not None:
                ax.set_xlim(0, t_max)
        ax_dcpa.set_xlabel("Time [s]")

    # Share the y-axis across each time row; keep tick labels on the first column.
    for row_axes in (axes_dist, axes_dcpa):
        ymin, ymax = row_axes[0].get_ylim()
        for col, ax in enumerate(row_axes):
            ax.set_ylim(ymin, ymax)
            if col > 0:
                ax.tick_params(labelleft=False)

    # Trajectory columns share y-limits (traj_ylim); drop the redundant tick
    # labels on all but the leftmost column.
    if traj_ylim is not None:
        for ax in axes_traj[1:]:
            ax.tick_params(labelleft=False)

    axes_traj[0].set_ylabel("North [m]")
    axes_dist[0].set_ylabel("Actual distance [m]")
    axes_dcpa[0].set_ylabel("Projected CPA [m]")
    # The distance row can reach thousands while the CPA row stays in hundreds,
    # so their tick labels differ in width and the two y-labels would otherwise
    # sit at different x. Snap the left-column y-labels to a common x-position.
    fig.align_ylabels([axes_traj[0], axes_dist[0], axes_dcpa[0]])

    # Two legends: trajectory keys (+ CPA marker) under the trajectory row, the
    # time-domain keys under the bottom row. The CPA event is shown once, in the
    # trajectory legend, so it is dropped from the time-domain legend.
    h_traj, l_traj = axes_traj[0].get_legend_handles_labels()
    h_time, l_time = [], []
    for ax in (axes_dist[0], axes_dcpa[0]):
        for h, lbl in zip(*ax.get_legend_handles_labels()):
            if lbl in l_time or lbl == "CPA":
                continue
            h_time.append(h); l_time.append(lbl)

    # Both legends pack all their keys onto one row, so their width tracks the
    # canvas: shrink the font in proportion when col_width is trimmed, otherwise
    # the entries overflow the narrower figure edges.
    leg_fs = 14.0 * min(1.0, col_width / 4.6)
    y_traj_leg = axes_dist[0].get_position().y1 + 0.03
    fig.legend(h_traj, l_traj, loc="center", ncol=len(l_traj), fontsize=leg_fs,
               bbox_to_anchor=(0.5, y_traj_leg))
    fig.legend(h_time, l_time, loc="lower center", ncol=len(l_time), fontsize=leg_fs,
               bbox_to_anchor=(0.5, 0.03))
    return _write(fig, figure_dir, stem or f"{_FILE_PREFIX}_pair{pair:03d}_cdr_composite")


def plot_avoidance_aggregate(results_by_label, figure_dir,
                             name=f"{_FILE_PREFIX}_avoidance_aggregate.png",
                             select="all", t_max=None, conflict_start_t=60.0):
    '''Aggregate the avoidance (resolution) flag across runs per strategy and
    overlay the mean resolution fraction over time. ``results_by_label`` maps a
    strategy label -> list of run namespaces. Returns the saved path.'''
    import matplotlib.pyplot as plt
    all_runs = [r for runs in results_by_label.values() for r in runs]
    n_max    = max(r.avoid_arr.shape[0] for r in all_runs)
    t_axis   = max((r.t_arr for r in all_runs), key=len)

    def _sel(env):
        if select == "ownship":
            return list(env.ownship_idx)
        if select == "intruder":
            return list(env.intruder_idx)
        return list(env.ownship_idx) + list(env.intruder_idx)

    def _pooled_mean(runs):
        cols = []
        for r in runs:
            a = r.avoid_arr[:, _sel(r.env)]
            if a.shape[0] < n_max:
                a = np.vstack([a, np.zeros((n_max - a.shape[0], a.shape[1]))])
            cols.append(a)
        return np.hstack(cols).mean(axis=1)

    any_res = all_runs[0]
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, runs in results_by_label.items():
        n_pairs = sum(r.env.nb_pair for r in runs)
        ax.plot(t_axis, _pooled_mean(runs),
                color=_COMPARE_COLORS.get(label),
                label=f"{_STRATEGY_PRETTY.get(label, label)}  (n={n_pairs} pairs, {len(runs)} runs)")
    if conflict_start_t is not None:
        ax.axvline(conflict_start_t, color="black", linestyle="--", linewidth=1.2,
                   label=f"conflict start ({conflict_start_t:g} s)")
    ax.set_ylim(-0.02, 1.02)
    if t_max is not None:
        ax.set_xlim(t_axis[0], t_max)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"Average resolution flag ({select} aircraft)")
    ax.set_title(
        f"Aggregated average resolution — recovery strategy comparison — "
        f"dpsi={any_res.dpsi} deg\n{_title_suffix(any_res)}"
    )
    ax.legend()
    return _save(fig, figure_dir, name)

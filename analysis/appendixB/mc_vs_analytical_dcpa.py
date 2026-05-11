"""
Monte Carlo validation of the analytical P(dCPA > rpz) formula.

Compares the analytical result from analytical_dcpa_prob_gt (projected-normal
integration) against brute-force Monte Carlo sampling for randomly generated
conflict geometries.

Tunable parameters are collected at the top of the script.
"""

import sys, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "pgf.texsystem": "pdflatex",
    "text.latex.preamble": "\n".join([
        r"\usepackage{amsmath}",
        r"\newcommand{\norm}[1]{\lVert #1 \rVert}",
        r"\newcommand{\dCPA}{\mathbf{d}_{\mathrm{CPA}}}",
    ]),
    "pgf.preamble": "\n".join([
        r"\usepackage{amsmath}",
        r"\newcommand{\norm}[1]{\lVert #1 \rVert}",
        r"\newcommand{\dCPA}{\mathbf{d}_{\mathrm{CPA}}}",
    ]),
    "axes.labelsize": 14,
    "font.size": 14,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

# ---------------------------------------------------------------------------
# Allow imports from the project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from sim_models.crr_resumenav_probabilistic_ftr import (
    analytical_dcpa_prob_gt,
    _regularize_spd,
    _to_cov,
)

# =====================  TUNABLE PARAMETERS  =====================
K_THETA       = 256         # angular grid points for the analytical formula
N_MC          = 10_000      # Monte Carlo samples per geometry
N_DCPA        = 100         # number of dcpa sample points
N_DPSI        = 100         # number of dpsi sample points (total points = N_DCPA * N_DPSI)
RPZ           = 50.0        # protected zone radius [m]
SPEED_KTS     = 20.0        # ownship & intruder speed [kts]
SIGMA_R       = 10.0        # position uncertainty std [m]  (isotropic)
SIGMA_V       = 1.0         # velocity uncertainty std [m/s] (isotropic)
D_ALONG       = 500.0       # along-track separation so CPA is in the future [m]
SEED          = 42
# ================================================================

KTS_TO_MS = 0.514444
V = SPEED_KTS * KTS_TO_MS   # speed in m/s

rng = np.random.default_rng(SEED)

Sigma_r = _regularize_spd(_to_cov(SIGMA_R), eps=1e-9)
Sigma_v = _regularize_spd(_to_cov(SIGMA_V), eps=1e-9)


def build_geometry(dcpa_m, dpsi_deg):
    """Return (mu_r, mu_v) for a conflict with given dcpa and heading difference.

    Ownship flies north (heading 0).  Intruder flies at heading dpsi.
    """
    dpsi = np.radians(dpsi_deg)

    # Relative velocity (ownship minus intruder), east-north frame
    mu_v = np.array([
        -V * np.sin(dpsi),
        V * (1.0 - np.cos(dpsi)),
    ])

    v_norm = np.linalg.norm(mu_v)
    if v_norm < 1e-12:
        # Co-linear same direction: place intruder offset to the side
        mu_r = np.array([dcpa_m, D_ALONG])
        return mu_r, mu_v

    # Unit vectors parallel and perpendicular to relative velocity
    u_par  = mu_v / v_norm
    u_perp = np.array([-u_par[1], u_par[0]])

    # Place intruder so that deterministic dcpa equals desired value
    # and along-track component puts CPA in the future
    mu_r = dcpa_m * u_perp + D_ALONG * u_par

    return mu_r, mu_v


def mc_prob_dcpa_gt(mu_r, Sigma_r, mu_v, Sigma_v, rpz, n_mc, rng):
    """Estimate P(dCPA > rpz) by Monte Carlo."""
    # Sample relative position and velocity
    r_samples = rng.multivariate_normal(mu_r, Sigma_r, size=n_mc)   # (N,2)
    v_samples = rng.multivariate_normal(mu_v, Sigma_v, size=n_mc)   # (N,2)

    # dCPA = ||r - v (r.v)/(v.v)||
    rv = np.sum(r_samples * v_samples, axis=1)    # (N,)
    vv = np.sum(v_samples * v_samples, axis=1)    # (N,)

    # Avoid division by zero for tiny relative velocities
    safe = vv > 1e-20
    t_cpa = np.where(safe, rv / vv, 0.0)

    dcpa_vec = r_samples - v_samples * t_cpa[:, None]
    dcpa_mag = np.linalg.norm(dcpa_vec, axis=1)

    return np.mean(dcpa_mag > rpz)


# ---------------------------------------------------------------------------
# Generate sample points
# ---------------------------------------------------------------------------
dcpa_values = np.linspace(0.0, 100.0, N_DCPA)
dpsi_values = np.linspace(0.01, 360.0, N_DPSI)

p_analytical = []
p_mc = []

total = N_DCPA * N_DPSI
count = 0

for dcpa_m in dcpa_values:
    for dpsi_deg in dpsi_values:
        mu_r, mu_v = build_geometry(dcpa_m, dpsi_deg)

        pa = analytical_dcpa_prob_gt(RPZ, mu_r, Sigma_r, mu_v, Sigma_v, Ktheta=K_THETA)
        pm = mc_prob_dcpa_gt(mu_r, Sigma_r, mu_v, Sigma_v, RPZ, N_MC, rng)

        p_analytical.append(pa)
        p_mc.append(pm)

        count += 1
        if count % 1000 == 0:
            print(f"  {count}/{total} geometries done …")

p_analytical = np.array(p_analytical)
p_mc = np.array(p_mc)

# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------
rmse = np.sqrt(np.mean((p_analytical - p_mc) ** 2))
print(f"\nRMSE between analytical and MC: {rmse:.6f}")
print(f"  K_theta = {K_THETA},  N_MC = {N_MC},  N_points = {total}")

# ---------------------------------------------------------------------------
# Scatter plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(p_mc, p_analytical, s=2, alpha=0.3, label="Sample points", rasterized=True)
ax.plot([0, 1], [0, 1], "r--", linewidth=1.2, label="$y = x$")
ax.set_xlabel(r"$P(d_\mathrm{CPA} > R_\mathrm{PZ})$ --- Monte Carlo")
ax.set_ylabel(r"$P(d_\mathrm{CPA} > R_\mathrm{PZ})$ --- Analytical")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect("equal")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)

fig.tight_layout()
outpath = os.path.join(os.path.dirname(__file__), f"fig_mc_vs_analytical_Ktheta{K_THETA}.pgf")
fig.savefig(outpath)
print(f"Saved plot to {outpath}")
plt.close(fig)

'''Regenerate the composite/aggregate input data from a fresh, seeded simulation.

This is the single source of truth for the pickles under ``data/`` and the
selected pairs in ``data/selected_pairs.json``. It runs the CDaRR_FP stochastic
pairwise simulation directly (the sim lives in CDaRR_FP; CDaRR_git has no
equivalent yet), sanitises each run to a dependency-free ``SimpleNamespace``, and
scans the 100 pooled pairs of each scenario for the three illustrative cases:

  almost_parallel (DPSI = 2 deg, seed 44)
    • prob_wins  — Past-CPA LoS  AND  FTR LoS  AND  Prob-FTR safe
    • prob_fails — Prob-FTR LoS  (FTR outcome unconstrained), with every
                   strategy's CPA occurring inside the DETAIL_T_MAX plot window
                   so the CPA marker is visible in all three panels
  large_angle (DPSI = 90 deg, seed 137)
    • ftr_wins   — Past-CPA LoS  AND  Prob-FTR LoS  AND  FTR safe

Deterministic selection rules (documented so the choice is reproducible):
  prob_wins  : among candidates, min(FTR CPA)      — most severe baseline failure
  prob_fails : among Prob-FTR LoS pairs whose 3 CPAs all fall before DETAIL_T_MAX,
               min(Prob CPA)                        — worst visible probabilistic loss
  ftr_wins   : among candidates, min(Prob CPA)      — most severe Prob-FTR failure

Requires the ``cdarr`` conda env and a CDaRR_FP checkout beside CDaRR_git.

Run::

    conda activate cdarr
    python regenerate_data.py
'''
import json
import os
import pickle
import sys

import numpy as np
from types import SimpleNamespace

# CDaRR_FP is a sibling of CDaRR_git; its runners package provides run_single.
_HERE   = os.path.dirname(os.path.abspath(__file__))
_FP_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "CDaRR_FP"))
if not os.path.isdir(os.path.join(_FP_ROOT, "runners")):
    sys.exit(f"CDaRR_FP not found at {_FP_ROOT} (needed for the simulation).")
sys.path.insert(0, _FP_ROOT)
os.chdir(_FP_ROOT)                       # some FP imports resolve cwd-relative
from runners.stochastic_pairwise_hor_conflict import run_single  # noqa: E402

DATA_DIR = os.path.join(_HERE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

RPZ          = 50.0
STRATEGIES   = ("double_criteria", "probabilistic", "cpa")
GAMMA        = 0.999
DETAIL_T_MAX = 310.0   # plot window; a pair's CPA must fall inside it to be usable

# Seeds. prob_wins + the aggregate come from AP_SEED; the near-parallel prob_fails
# case needs a pair whose three CPAs all fall inside the 310 s window — rare at
# dpsi=2 because Past-CPA chatters along the RPZ and its global CPA is usually
# late — so it is drawn from PROBFAIL_SEED, found by scanning seeds 44..143 for
# {Prob-FTR LoS, all 3 CPAs < 310}. See scan_probfail.py for the one-off search.
AP_SEED        = 44
LA_SEED        = 137
PROBFAIL_SEED  = 89


def _run_kwargs(dpsi):
    return dict(
        pair_width=10, pair_height=10,
        rpz=RPZ, hpz=50.0, dtlookahead=120.0,
        init_speed_ownship=10.2889, init_speed_intruder=10.2889,
        aircraft_type="M600", dpsi=dpsi,
        pos_ci95=10.0, vel_ci95=1.0, reception_prob=1.0,
        start_lat=52.0, start_lon=4.0, delta_lat_lon=0.1,
        tmax=120.0 * 4, done_timeout=30.0,
        resofach=1.05, recovery_resofach=1.05,
        simdt_factor=4,
        record_history=True,
        prob_threshold=GAMMA,
        spawn_margin=1.5,
    )


# Fields the composite / aggregate plots consume (everything else is dropped).
RES_FIELDS = ["t_arr", "dist_arr", "dcpa_gt_arr", "dcpa_obs_arr",
              "lat_arr", "lon_arr", "avoid_arr", "min_dist"]
SCALARS    = ["rpz", "dpsi", "pos_ci95", "vel_ci95", "reception_prob", "latency_s", "ipr"]
ENV_FIELDS = ["ownship_idx", "intruder_idx", "ownship_ids", "intruder_ids", "nb_pair"]


def _sanitise(res):
    env  = res.env
    penv = SimpleNamespace(**{
        a: (np.asarray(getattr(env, a)) if a.endswith("idx") else getattr(env, a))
        for a in ENV_FIELDS if hasattr(env, a)
    })
    kw = {f: np.asarray(getattr(res, f)) for f in RES_FIELDS if hasattr(res, f)}
    for s in SCALARS:
        kw[s] = getattr(res, s, 0.0)
    kw["env"] = penv
    return SimpleNamespace(**kw)


def _simulate(dpsi, seed):
    kw = _run_kwargs(dpsi)
    return {label: _sanitise(run_single(crr=label, seed=seed, **kw))
            for label in STRATEGIES}


def _cpas(runs, p):
    return {s: round(float(runs[s].min_dist[p]), 1) for s in STRATEGIES}


def _cpa_time(res, p):
    '''Sim time (s) of the actual closest point of approach for the pair.'''
    return float(res.t_arr[int(np.argmin(res.dist_arr[:, p]))])


def _t_cpas(runs, p):
    return {s: round(_cpa_time(runs[s], p), 1) for s in STRATEGIES}


def _all_cpa_before(runs, p, t_max):
    '''True iff every strategy's CPA for pair ``p`` occurs before ``t_max``.'''
    return all(_cpa_time(runs[s], p) < t_max for s in STRATEGIES)


def _cpa_spread(runs, p):
    '''Time span between the earliest and latest strategy CPA (legibility proxy:
    a tight cluster reads as one clearly-visible encounter).'''
    ts = [_cpa_time(runs[s], p) for s in STRATEGIES]
    return max(ts) - min(ts)


def _entry(runs, p):
    return None if p is None else {"pair": p, "cpa": _cpas(runs, p), "t_cpa": _t_cpas(runs, p)}


def _pick(runs, predicate, key):
    '''Index minimising ``key(p)`` over all pairs satisfying ``predicate(p)``.'''
    cand = [p for p in range(runs["cpa"].env.nb_pair) if predicate(runs, p)]
    return min(cand, key=lambda p: key(runs, p)) if cand else None


selected = {}

# ── almost-parallel prob_wins + aggregate run (AP_SEED) ─────────────────────────────
print(f"== almost_parallel prob_wins/aggregate (dpsi=2, seed={AP_SEED}) ==")
ap = _simulate(dpsi=2, seed=AP_SEED)
for lbl in STRATEGIES:
    print(f"  IPR {lbl:<16}: {ap[lbl].ipr:.4f}")
with open(os.path.join(DATA_DIR, "almost_parallel_runs.pkl"), "wb") as f:
    pickle.dump(ap, f)

prob_wins = _pick(
    ap,
    lambda r, p: (r["cpa"].min_dist[p] < RPZ and r["double_criteria"].min_dist[p] < RPZ
                  and r["probabilistic"].min_dist[p] >= RPZ),
    lambda r, p: r["double_criteria"].min_dist[p])
print(f"  prob_wins  → pair {prob_wins}  {_cpas(ap, prob_wins) if prob_wins is not None else '—'}")

# ── almost-parallel prob_fails run (PROBFAIL_SEED) ──────────────────────────────────
print(f"== almost_parallel prob_fails (dpsi=2, seed={PROBFAIL_SEED}) ==")
ap_pf = _simulate(dpsi=2, seed=PROBFAIL_SEED)
with open(os.path.join(DATA_DIR, "almost_parallel_probfail_runs.pkl"), "wb") as f:
    pickle.dump(ap_pf, f)

# Prob-FTR LoS, all 3 CPAs visible (< DETAIL_T_MAX); pick tightest CPA cluster
# (most legible single encounter), tie-break on the deepest probabilistic loss.
prob_fails = _pick(
    ap_pf,
    lambda r, p: r["probabilistic"].min_dist[p] < RPZ and _all_cpa_before(r, p, DETAIL_T_MAX),
    lambda r, p: (_cpa_spread(r, p), r["probabilistic"].min_dist[p]))

pf_entry = _entry(ap_pf, prob_fails)
if pf_entry is not None:
    pf_entry["seed"] = PROBFAIL_SEED
selected["almost_parallel"] = {"prob_wins": _entry(ap, prob_wins), "prob_fails": pf_entry}
print(f"  prob_fails → pair {prob_fails}  "
      f"{_cpas(ap_pf, prob_fails) if prob_fails is not None else '—'}  "
      f"t_cpa={_t_cpas(ap_pf, prob_fails) if prob_fails is not None else '—'}")

# ── large-angle (DPSI = 90) ─────────────────────────────────────────────────────────
print(f"== large_angle (dpsi=90, seed={LA_SEED}) ==")
la = _simulate(dpsi=90, seed=LA_SEED)
for lbl in STRATEGIES:
    print(f"  IPR {lbl:<16}: {la[lbl].ipr:.4f}")
with open(os.path.join(DATA_DIR, "large_angle_runs.pkl"), "wb") as f:
    pickle.dump(la, f)

ftr_wins = _pick(
    la,
    lambda r, p: (r["cpa"].min_dist[p] < RPZ and r["probabilistic"].min_dist[p] < RPZ
                  and r["double_criteria"].min_dist[p] >= RPZ),
    lambda r, p: r["probabilistic"].min_dist[p])
selected["large_angle"] = {
    "ftr_wins": _entry(la, ftr_wins),
}
print(f"  ftr_wins   → pair {ftr_wins}  {_cpas(la, ftr_wins) if ftr_wins is not None else '—'}")

with open(os.path.join(DATA_DIR, "selected_pairs.json"), "w") as f:
    json.dump({"rpz": RPZ, "gamma": GAMMA,
               "seeds": {"almost_parallel": AP_SEED, "almost_parallel_probfail": PROBFAIL_SEED,
                         "large_angle": LA_SEED},
               "cases": selected}, f, indent=2)
print(f"\nwrote {DATA_DIR}/selected_pairs.json")

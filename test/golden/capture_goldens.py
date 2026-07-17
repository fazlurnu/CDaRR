"""Capture golden baselines from the FROZEN reference pipeline (Phase 0).

Runs ``get_ipr_stochastic_env`` / ``_dist`` over the L2 equivalence matrix
(refactor_fp.md section 6) using the fast 2x2 configs, and writes one ``.npz`` per case
plus a ``manifest.json`` of per-case output hashes. The refactored code on ``fp-refactor``
must reproduce these bitwise, in the environment pinned by ``ENVIRONMENT.md``.

Usage (from the worktree root, under the ``cdarr`` env)::

    python test/golden/capture_goldens.py            # capture -> test/golden/baseline/
    python test/golden/capture_goldens.py --verify   # re-run, compare to manifest, exit!=0 on drift

The digest covers the full float64 ``distance_array`` bytes plus ``(ipr, sim_timer,
n_active)`` — i.e. every value the pipeline actually produces and the experiments consume.
"""
import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from sim.pairwise_stochastic.get_ipr_stochastic_env import (  # noqa: E402
    get_ipr_stochastic_env,
    get_ipr_stochastic_env_dist,
)
from sim_models.noise_distributions import (  # noqa: E402
    make_mixture_gaussian,
    make_anisotropic_gaussian,
    make_anisotropic_mixture_gaussian,
)

CFG_MVP = "sim_configs/test_config_2x2.json"
CFG_VO = "sim_configs/test_config_2x2_vo.json"

# Position-noise factories keyed by a readable string (experiments/config.py values:
# TAIL_RATIO=3.0, TAIL_WEIGHT=0.10, ANISO_VAR_RATIO=9.0).
POS_DIST = {
    "none": None,
    "mixture": make_mixture_gaussian(3.0, 0.10),
    "anisotropic": make_anisotropic_gaussian(9.0),
    "aniso_mixture": make_anisotropic_mixture_gaussian(9.0, 3.0, 0.10),
}

BASE = dict(
    asas_marh=1.05, confidence_interval=10, confidence_interval_velo=1,
    reception_prob=1.0, lookahead_time=120, seed=44,
)

# (name, kind, overrides); kind in {"env", "dist"}. Mirrors the L2 matrix.
CASES = [
    ("c01_mvp_cpa_d2",          "env",  dict(config_path=CFG_MVP, recovery_model="CPA", dpsi=2)),
    ("c02_mvp_ftr_d2",          "env",  dict(config_path=CFG_MVP, recovery_model="FTR", dpsi=2)),
    ("c03_mvp_prob999_d2",      "env",  dict(config_path=CFG_MVP, recovery_model="Probabilistic FTR", threshold_probability=0.999, dpsi=2)),
    ("c04_mvp_prob5_d90",       "env",  dict(config_path=CFG_MVP, recovery_model="Probabilistic FTR", threshold_probability=0.5, dpsi=90)),
    ("c05_mvp_cpa_d90",         "env",  dict(config_path=CFG_MVP, recovery_model="CPA", dpsi=90)),
    ("c06_mvp_ftr_d180",        "env",  dict(config_path=CFG_MVP, recovery_model="FTR", dpsi=180)),
    ("c07_vo_cpa_d90",          "env",  dict(config_path=CFG_VO,  recovery_model="CPA", dpsi=90)),
    ("c08_vo_ftr_d2",           "env",  dict(config_path=CFG_VO,  recovery_model="FTR", dpsi=2)),
    ("c09_mvp_prob999_d2_rx07", "env",  dict(config_path=CFG_MVP, recovery_model="Probabilistic FTR", threshold_probability=0.999, dpsi=2, reception_prob=0.7)),
    ("c10_mvp_prob999_perpair", "env",  dict(config_path=CFG_MVP, recovery_model="Probabilistic FTR", threshold_probability=0.999, dpsi=[2.0, 45.0, 90.0, 137.0])),
    ("c11_mvp_ftr_d30_mix",     "env",  dict(config_path=CFG_MVP, recovery_model="FTR", dpsi=30, pos_dist="mixture")),
    ("c12_mvp_prob999_d30_ani", "env",  dict(config_path=CFG_MVP, recovery_model="Probabilistic FTR", threshold_probability=0.999, dpsi=30, pos_dist="anisotropic")),
    ("c13_mvp_cpa_d30_lat",     "env",  dict(config_path=CFG_MVP, recovery_model="CPA", dpsi=30, latency_s=0.1)),
    ("c14_mvp_ftr_d30_all",     "env",  dict(config_path=CFG_MVP, recovery_model="FTR", dpsi=30, pos_dist="aniso_mixture", latency_s=0.1, reception_prob=0.9)),
    ("c15_mvp_prob999_mismatch","env",  dict(config_path=CFG_MVP, recovery_model="Probabilistic FTR", threshold_probability=0.999, dpsi=30, confidence_interval=15, assumed_confidence_interval=10)),
    ("c16_dist_prob999_rx099",  "dist", dict(config_path=CFG_MVP, recovery_model="Probabilistic FTR", threshold_probability=0.999, reception_prob=0.99, dist_m=400.0, lookahead_time=120, tmax=600.0)),
]


def _resolve(ov):
    ov = dict(ov)
    if "pos_dist" in ov:
        ov["pos_dist"] = POS_DIST[ov["pos_dist"]]
    return ov


def run_case(kind, overrides):
    ov = _resolve(overrides)
    if kind == "env":
        kw = dict(BASE)
        kw.update(ov)
        da, ipr, t, n = get_ipr_stochastic_env(**kw)
    elif kind == "dist":
        kw = dict(
            confidence_interval=BASE["confidence_interval"],
            confidence_interval_velo=BASE["confidence_interval_velo"],
            reception_prob=ov.get("reception_prob", BASE["reception_prob"]),
            dist_m=ov["dist_m"], asas_marh=BASE["asas_marh"],
            lookahead_time=ov.get("lookahead_time", 300.0),
            tmax=ov.get("tmax", 1200.0), seed=BASE["seed"],
            config_path=ov["config_path"],
            threshold_probability=ov.get("threshold_probability"),
            recovery_model=ov.get("recovery_model"),
        )
        da, ipr, t, n = get_ipr_stochastic_env_dist(**kw)
    else:
        raise ValueError(kind)
    return np.asarray(da, dtype=float), float(ipr), float(t), int(n)


def digest(da, ipr, t, n):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(da, dtype=np.float64).tobytes())
    h.update(repr((round(ipr, 12), round(t, 12), n)).encode())
    return h.hexdigest()


def capture(outdir):
    os.makedirs(outdir, exist_ok=True)
    manifest, failures = {}, []
    for name, kind, ov in CASES:
        try:
            da, ipr, t, n = run_case(kind, ov)
            sha = digest(da, ipr, t, n)
            np.savez(os.path.join(outdir, name + ".npz"),
                     distance_array=da, ipr=ipr, sim_timer=t, n_active=n)
            manifest[name] = dict(ipr=ipr, sim_timer=t, n_active=n,
                                  shape=list(da.shape), sha256=sha)
            print(f"  {name:28s} ipr={ipr:.4f} shape={tuple(da.shape)} sha={sha[:12]}")
        except Exception as e:  # noqa: BLE001 -- want every failure surfaced, not aborted
            failures.append((name, repr(e)))
            print(f"  {name:28s} FAILED: {e!r}")
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest, failures


def verify(outdir):
    with open(os.path.join(outdir, "manifest.json")) as f:
        ref = json.load(f)
    mism = []
    for name, kind, ov in CASES:
        da, ipr, t, n = run_case(kind, ov)
        sha = digest(da, ipr, t, n)
        exp = ref.get(name, {}).get("sha256")
        ok = sha == exp
        print(f"  {name:28s} {'MATCH' if ok else 'MISMATCH'} sha={sha[:12]} exp={str(exp)[:12]}")
        if not ok:
            mism.append(name)
    return mism


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "test", "golden", "baseline"))
    a = ap.parse_args()

    if a.verify:
        mism = verify(a.outdir)
        print(f"\n{'ALL MATCH' if not mism else 'DRIFT: ' + ','.join(mism)}")
        sys.exit(0 if not mism else 1)

    manifest, failures = capture(a.outdir)
    print(f"\ncaptured {len(manifest)} cases to {a.outdir}")
    if failures:
        print("FAILURES:")
        for n, e in failures:
            print(f"  {n}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

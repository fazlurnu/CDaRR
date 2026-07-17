"""Determinism guards.

The TRUE guarantee — a fixed-order fresh process reproduces the baselines — is enforced by
``test_golden_baseline.py`` (subprocess). ``test_n_jobs_invariance`` below pins n_jobs
invariance as a real assertion: it was an xfail documenting KI-1 (in-process
order-dependent nondeterminism from the MVP/VO recovery-singleton leak) until that leak
was fixed (KNOWN_ISSUES.md, KI-1) — verified XPASS once the fix landed, then promoted here.
"""
import numpy as np
import pytest


@pytest.mark.slow
def test_n_jobs_invariance():
    """n_jobs must not change results: n_jobs=1 runs all reps in one process (would
    accumulate cross-run recovery state pre-KI-1-fix); n_jobs=2 splits them across
    workers. Both must produce identical per-run values.
    """
    from sim.pairwise_stochastic.run_multiple_jobs import run_multiple_jobs

    kw = dict(
        n_runs=4, base_seed=42, asas_marh=1.05,
        confidence_interval=10, confidence_interval_velo=1, reception_prob=1.0,
        lookahead_time=120, dpsi=90,
        config_path="sim_configs/test_config_2x2.json", recovery_model="FTR",
    )
    r1 = run_multiple_jobs(n_jobs=1, **kw)
    r2 = run_multiple_jobs(n_jobs=2, **kw)
    assert np.array_equal(np.asarray(r1["worst_cpa"]), np.asarray(r2["worst_cpa"]))

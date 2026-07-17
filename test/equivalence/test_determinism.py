"""Determinism guards.

The TRUE guarantee — a fixed-order fresh process reproduces the baselines — is enforced by
``test_golden_baseline.py`` (subprocess). Here we pin the KNOWN in-process order-dependence
(KNOWN_ISSUES.md, KI-1) as an xfail, so if a future change makes runs order-independent it
surfaces as an XPASS rather than silently passing.
"""
import numpy as np
import pytest


@pytest.mark.slow
@pytest.mark.xfail(
    reason="KI-1: in-process order-dependent nondeterminism (test/golden/KNOWN_ISSUES.md); "
           "frozen behavior, not yet fixed",
    strict=False,
)
def test_n_jobs_invariance():
    """Would hold if runs were order-independent; currently xfail per KI-1.

    n_jobs=1 runs all reps in one process (history accumulates); n_jobs=2 splits them
    across workers (different per-worker history) -> per-run values can differ.
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

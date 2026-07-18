"""L2 equivalence: sim.pairwise_stochastic.loop (the new functional shell) vs.
the golden baselines captured from the frozen legacy reference.

Reuses test/golden/capture_goldens.py's exact CASES matrix and digest function
so this is a direct, apples-to-apples re-run of the same 16-case suite through
the NEW implementation -- the actual Phase 3 gate (refactor_fp.md): "new-loop
L2 green old-vs-new".

Runs in a subprocess (like test_golden_baseline.py) for the same reason: KI-1
taught us that in-process reruns can pick up unrelated bluesky global state
between cases (see KNOWN_ISSUES.md) -- a fixed-order fresh process is the only
configuration this whole test suite trusts.
"""
import json
import subprocess
import sys

import pytest

ROOT_MARKER = "test/golden/capture_goldens.py"

_RUNNER = r'''
import json
import numpy as np
from sim.pairwise_stochastic.loop import get_ipr_stochastic_env as new_env, get_ipr_stochastic_env_dist as new_env_dist
from test.golden.capture_goldens import CASES, BASE, _resolve, digest

with open("test/golden/baseline/manifest.json") as f:
    manifest = json.load(f)


def run_case(kind, ov):
    ov = _resolve(ov)
    if kind == "env":
        kw = dict(BASE)
        kw.update(ov)
        da, ipr, t, n = new_env(**kw)
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
        da, ipr, t, n = new_env_dist(**kw)
    else:
        raise ValueError(kind)
    return np.asarray(da, dtype=float), float(ipr), float(t), int(n)


mismatches = []
for name, kind, overrides in CASES:
    da, ipr, t, n = run_case(kind, overrides)
    sha = digest(da, ipr, t, n)
    exp = manifest[name]["sha256"]
    if sha != exp:
        mismatches.append(name)

print(json.dumps({"mismatches": mismatches, "n_cases": len(CASES)}))
'''


def _run_subprocess():
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER],
        capture_output=True, text=True, cwd=".", timeout=300,
    )
    assert proc.returncode == 0, f"subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    # capture_goldens.py's CASES emit some BlueSky stack echo noise on stderr;
    # the real result is the last stdout line (a JSON blob).
    last_line = proc.stdout.strip().splitlines()[-1]
    return json.loads(last_line)


@pytest.mark.slow
def test_new_loop_matches_all_golden_cases():
    result = _run_subprocess()
    assert result["n_cases"] == 16, f"expected 16 cases, ran {result['n_cases']}"
    assert not result["mismatches"], f"new loop diverged from golden baseline: {result['mismatches']}"

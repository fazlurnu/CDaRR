"""L2 equivalence: sim.pairwise_stochastic.loop.get_ipr_stochastic_env_randomized
vs. the legacy sim.pairwise_stochastic.get_ipr_stochastic_env.get_ipr_stochastic_env_randomized.

Not covered by test/golden/baseline (capture_goldens.py's CASES only exercise
"env"/"dist" kinds), so this is a direct old-vs-new comparison instead of a
manifest-hash check, run in a subprocess for the same KI-1-driven reason as
test_equiv_loop_golden.py.
"""
import json
import subprocess
import sys

import pytest

_RUNNER = r'''
import json
import numpy as np
from sim.pairwise_stochastic.loop import get_ipr_stochastic_env_randomized as new_fn
from sim.pairwise_stochastic.get_ipr_stochastic_env import get_ipr_stochastic_env_randomized as old_fn

mismatches = []
for rm in ["CPA", "FTR", "Probabilistic FTR"]:
    kw = dict(asas_marh=1.05, confidence_interval=10, confidence_interval_velo=1,
              reception_prob=0.9, lookahead_time=120, dpsi=2, seed=44,
              config_path="sim_configs/test_config_2x2.json", recovery_model=rm,
              threshold_probability=0.999)
    old_da, old_ipr, old_t, old_n = old_fn(**kw)
    new_da, new_ipr, new_t, new_n = new_fn(**kw)
    ok = (old_da.shape == new_da.shape and np.array_equal(old_da, new_da)
          and old_ipr == new_ipr and old_t == new_t and old_n == new_n)
    if not ok:
        mismatches.append(rm)

print(json.dumps({"mismatches": mismatches}))
'''


@pytest.mark.slow
def test_randomized_matches_legacy():
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER],
        capture_output=True, text=True, cwd=".", timeout=120,
    )
    assert proc.returncode == 0, f"subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert not result["mismatches"], f"diverged for: {result['mismatches']}"

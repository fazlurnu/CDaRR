"""Golden regression (process-isolated).

The frozen pipeline has in-process, order-dependent nondeterminism (see
`test/golden/KNOWN_ISSUES.md`, KI-1): only a FIXED-ORDER, FRESH-PROCESS run reproduces the
captured baselines (which were themselves captured in-sequence). So this test shells out to
``capture_goldens.py --verify`` in a subprocess rather than running the cases in-process.
Any drift in distance_array / ipr / sim_timer / n_active fails here — the primary
regression floor for the refactor (refactor_fp.md §§5-6).
"""
import os
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CG = os.path.join(_ROOT, "test", "golden", "capture_goldens.py")


@pytest.mark.golden
@pytest.mark.slow
def test_golden_baselines_reproduce():
    proc = subprocess.run(
        [sys.executable, _CG, "--verify"],
        cwd=_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0 and "ALL MATCH" in proc.stdout, (
        "golden drift — if order-related see KNOWN_ISSUES.md KI-1:\n"
        + proc.stdout[-3000:] + "\n--- stderr ---\n" + proc.stderr[-1000:]
    )

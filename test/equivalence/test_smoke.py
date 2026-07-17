"""Fast sanity checks — no simulation, no bluesky init.

Pins the pure-module import surface and the test configs. The end-to-end reproducibility
check lives in ``test_golden_baseline.py`` (subprocess). (An earlier version asserted that
a single call reproduces in-process; that property does NOT hold in general — see
KNOWN_ISSUES.md KI-1 — so it was removed.)
"""
import json
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.fast
def test_pure_modules_import():
    from sim_models import noise_distributions as nd
    from sim_models import crr_recovery_base as rb

    assert callable(nd.gaussian)
    assert callable(nd.make_mixture_gaussian(3.0, 0.1))
    assert callable(nd.make_anisotropic_gaussian(9.0))
    assert hasattr(rb, "get_desired_ownship_velocity")


@pytest.mark.fast
def test_test_configs_present():
    for name in ("test_config_2x2.json", "test_config_2x2_vo.json"):
        with open(os.path.join(_ROOT, "sim_configs", name)) as f:
            cfg = json.load(f)
        assert cfg["scenario"]["width"] == 2 and cfg["scenario"]["height"] == 2
    # the VO variant differs only in the resolution model
    with open(os.path.join(_ROOT, "sim_configs", "test_config_2x2_vo.json")) as f:
        assert json.load(f)["conflict_models"]["resolution"] == "VO"

"""L2 shadow-mode equivalence: cdarr.cdarr() vs. the legacy per-tick loop.

The strongest direct check of the composition: drives a REAL, running
simulation using the legacy resolve()+recovery() output as the actual control
action (so geometry evolves non-trivially tick to tick, exactly like the real
sim loop), and at every tick ALSO calls cdarr() on the identical ground-truth
observed states, then compares:

  - command.trk / command.gs_capped  vs. the legacy resolve()'s raw
    newtrack/newgscapped -- these should be bit-identical, since it's the
    same math (this catches a broken port of cr.mvp/cr.vo).
  - avoiding + new_state.resopairs   vs. the legacy's post-recovery
    membership test -- this is what cdarr()'s composition actually adds
    (F1 ordering: avoidance status computed from resopairs AFTER recovery,
    not after resolve -- see cdarr/core.py's module docstring for why this
    is subtle in the legacy code, which relies on resolve() returning
    self.resopairs *by reference*).

Uses ground-truth self-observation (own == intr == states, no ADSL noise
layer -- noise/reception is cns/link.py's separate concern, composed by the
shell on top of cdarr(), not part of cdarr() itself) across every
resolution/recovery combination, including VO's F6 quirk.
"""
import numpy as np
import pytest
import bluesky as bs

from cdarr.core import cdarr, CdarrParams, make_dict_id2idx
from cr.common import ResolutionParams
from crr.common import RecoveryState

if not getattr(bs, "_joblib_inited", False):
    bs.init(mode="sim", detached=True)
    bs._joblib_inited = True

RPZ, HPZ, DTLOOK = 50, 100.0, 15


def _make_env(dpsi, width=2, height=2):
    from envs.pairwise_conflict import PairwiseHorConflict
    return PairwiseHorConflict(
        pair_width=width, pair_height=height,
        asas_pzr_m=RPZ, dtlookahead=DTLOOK,
        init_speed_ownship=10.2889, init_speed_intruder=10.2889,
        init_dpsi=dpsi, aircraft_type_ownship="M600",
        simdt_factor=4.0,
    )


def _make_old_models(resolution, recovery):
    from sim_models.cd_statebased import StateBased
    if resolution == "MVP":
        from sim_models.cr_mvp import MVP as Resolver
    else:
        from sim_models.cr_vo import VO as Resolver
    resolver = Resolver()
    resolver.resopairs = set()  # KI-1: singleton, start clean

    if recovery == "CPA":
        from sim_models.crr_resumenav_cpa import resumenav as recover_fn
    elif recovery == "FTR":
        from sim_models.crr_resumenav_ftr import resumenav_double_criteria as recover_fn
    elif recovery == "Probabilistic FTR":
        from sim_models.crr_resumenav_probabilistic_ftr import resumenav_probabilistic_ftr as recover_fn
    else:
        raise ValueError(recovery)

    return StateBased(), resolver, recover_fn


# Same realistic exp3/4/5-style worldview as test_equiv_recovery_probabilistic_ftr.py.
_CI95_TO_STD = 2.448
_SIGMA_R = np.sqrt(2.0) * (10.0 / _CI95_TO_STD)
_SIGMA_V = np.sqrt(2.0) * (1.0 / _CI95_TO_STD)
_PROB_THRESHOLD = 0.999
_KTHETA = 256


def _old_avoiding(ids, resopairs):
    return np.array([any(i in pair for pair in resopairs) for i in ids], dtype=bool)


CASES = [
    ("MVP", "CPA", 90),
    ("MVP", "FTR", 2),
    ("MVP", "FTR", 180),
    ("VO", "CPA", 90),                    # exercises VO's F6 quirk end-to-end
    ("MVP", "Probabilistic FTR", 2),      # the paper's actual contribution
]


@pytest.mark.slow
@pytest.mark.parametrize("resolution,recovery,dpsi", CASES)
def test_shadow_mode_matches_legacy(resolution, recovery, dpsi):
    env = _make_env(dpsi)
    try:
        old_detect, old_resolver, old_recover_fn = _make_old_models(resolution, recovery)
        resofach = float(bs.settings.asas_marh)
        resofacv = float(bs.settings.asas_marv)

        new_params = CdarrParams(
            rpz=RPZ, hpz=HPZ, dtlookahead=DTLOOK,
            resolution=resolution,
            resolution_params=ResolutionParams(resofach=resofach, resofacv=resofacv),
            recovery=recovery, resofach=resofach,
            sigma_r=_SIGMA_R if recovery == "Probabilistic FTR" else None,
            sigma_v=_SIGMA_V if recovery == "Probabilistic FTR" else None,
            prob_threshold=_PROB_THRESHOLD, Ktheta=_KTHETA,
        )
        new_state = RecoveryState()

        n_ticks = 6
        for tick in range(n_ticks):
            states = env._get_states()
            ids = list(states.id)

            # --- legacy: detect -> resolve (F6 baked in for VO) -> recover ---
            old_detect.detect(states, states, RPZ, HPZ, DTLOOK)
            if recovery == "Probabilistic FTR":
                # Mirrors the legacy attribute-smuggling (get_ipr_stochastic_env.py
                # does `conf_detection.sigma_r = ...` before calling recovery).
                old_detect.sigma_r = _SIGMA_R
                old_detect.sigma_v = _SIGMA_V
                old_detect.dcpa_prob_threshold = _PROB_THRESHOLD
                old_detect.dcpa_prob_Ktheta = _KTHETA
            reso = old_resolver.resolve(old_detect, states, states, resofach)
            old_newtrack, old_newgscapped, old_vscapped, old_alt, _resopairs_ref = reso
            old_recover_fn(old_resolver, old_detect, states, states)
            # _resopairs_ref IS old_resolver.resopairs (by reference) -- now post-recovery.
            old_resopairs_after = set(old_resolver.resopairs)
            old_avoiding = _old_avoiding(ids, old_resopairs_after)

            # --- new: single cdarr() call ---
            id2idx = make_dict_id2idx(ids)
            result = cdarr(states, states, new_state, new_params, id2idx=id2idx)
            new_state = result.state

            assert np.array_equal(old_newtrack, result.command.trk), \
                f"tick {tick}: command.trk mismatch"
            assert np.array_equal(old_newgscapped, result.command.gs_capped), \
                f"tick {tick}: command.gs_capped mismatch"
            assert np.array_equal(old_avoiding, result.avoiding), \
                f"tick {tick}: avoiding mismatch"
            assert old_resopairs_after == set(result.state.resopairs), \
                f"tick {tick}: resopairs mismatch"

            # Drive the sim forward with the legacy action, so geometry evolves
            # non-trivially between ticks (matches how the real loop works).
            action = (old_newtrack, old_newgscapped, old_vscapped, old_alt, old_resopairs_after)
            for _ in range(3):
                env.step(action)
    finally:
        env.reset()

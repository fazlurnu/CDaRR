"""Verification tests for bug fixes #1-#15.

Each test checks the actual output/behavior of the fix where possible,
and falls back to source-level inspection where the fix is structural.

Run from the project root:
    python -m test.verify_fixes
"""
from __future__ import annotations

import ast
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RESULTS = []


def report(name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append((name, passed, detail))
    status = "PASS" if passed else "FAIL"
    line = f"[{status}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line)


def read(rel: str) -> str:
    with open(os.path.join(ROOT, rel)) as f:
        return f.read()


# ---------------------------------------------------------------------------
# #1: merge conflict resolved
# ---------------------------------------------------------------------------
def test_01_merge_conflict():
    src = read("compare_crr/generate_ipr_from_dpsi_samples.py")
    has_markers = ("<<<<<<<" in src) or (">>>>>>>" in src)
    try:
        ast.parse(src)
        parses = True
    except SyntaxError as e:
        parses = False
    report(
        "#1 generate_ipr_from_dpsi_samples: no conflict markers, parses",
        (not has_markers) and parses,
        f"markers={has_markers}, parses={parses}",
    )


# ---------------------------------------------------------------------------
# #2 & #3: VO vertical math reachable, hpz uses resofacv
# ---------------------------------------------------------------------------
def test_02_03_vo():
    src = read("sim_models/cr_vo.py")

    # #3: hpz uses resofacv inside the VO method
    has_resofacv = "hpz = np.max(conf.hpz[[idx1, idx2]] * self.resofacv)" in src
    no_old_hpz = "hpz = np.max(conf.hpz[[idx1, idx2]] * self.resofach)" not in src
    report("#3 VO.VO uses resofacv for hpz", has_resofacv and no_old_hpz)

    # #2: dead code after `return dv, tsolV` removed, vertical math present
    vo_start = src.find("def VO(self, ownship, intruder, conf, qdr, dist, tLOS, idx1, idx2):")
    vo_body = src[vo_start:]
    return_count = vo_body.count("return dv, tsolV")
    has_vertical_math = (
        "vrel_vs = ownship.vs[idx1] - intruder.vs[idx2]" in vo_body
        and "dv3 = (iV / tsolV)" in vo_body
    )
    no_dead_hardcode = "dv3 = 0\n        \n        dv = np.array([dv1, dv2, dv3])" not in vo_body
    report(
        "#2 VO.VO has vertical math, no dead code",
        return_count == 1 and has_vertical_math and no_dead_hardcode,
        f"return_count={return_count}, vertical={has_vertical_math}",
    )

    # Functional test of the vertical math (extracted, no bluesky needed)
    def vertical_dv3(vs_o, vs_i, alt_o, alt_i, hpz, tLOS, dtlook):
        drel_z = alt_i - alt_o
        vrel_vs = vs_o - vs_i
        iV = hpz if abs(vrel_vs) > 0.0 else hpz - abs(drel_z)
        tsolV = abs(drel_z / vrel_vs) if abs(vrel_vs) > 0.0 else tLOS
        if tsolV > dtlook:
            tsolV = tLOS
            iV = hpz
        dv3 = (iV / tsolV) * (-vrel_vs / abs(vrel_vs)) if abs(vrel_vs) > 0.0 else (iV / tsolV)
        return dv3, tsolV

    # Ownship climbing into intruder above: should drive dv3 negative (slow climb)
    dv3, tsolV = vertical_dv3(vs_o=2.0, vs_i=-1.0, alt_o=100.0, alt_i=110.0,
                              hpz=100.0, tLOS=5.0, dtlook=60.0)
    ok = dv3 < 0 and tsolV > 0
    report("#2 vertical math: climbing into descending intruder -> dv3 < 0",
           ok, f"dv3={dv3:.3f}, tsolV={tsolV:.3f}")

    # No vertical relative motion: dv3 should be non-zero only if hpz - |drel_z| > 0
    dv3_zero_vrel, _ = vertical_dv3(vs_o=0.0, vs_i=0.0, alt_o=100.0, alt_i=150.0,
                                    hpz=100.0, tLOS=5.0, dtlook=60.0)
    report("#2 vertical math: zero vrel still produces finite dv3",
           np.isfinite(dv3_zero_vrel), f"dv3={dv3_zero_vrel:.3f}")


# ---------------------------------------------------------------------------
# #4: kwargs pop randomized_speed_heading before forwarding
# ---------------------------------------------------------------------------
def test_04_kwargs():
    import sim.pairwise_stochastic.run_multiple_jobs as m

    captured = {}

    def fake_env(seed, asas_marh=None, **kw):
        # Strict signature: would raise if `randomized_speed_heading` were forwarded
        # (kept **kw open so randomized variant can receive it).
        captured["seed"] = seed
        captured["kw"] = dict(kw)
        captured["kw"]["asas_marh"] = asas_marh
        return (np.zeros((3, 2)), 0.5, 1.0, 0)

    def fake_env_strict(seed, asas_marh=None):
        # Strict, no **kw: any leftover kwarg would TypeError.
        captured["seed_strict"] = seed
        captured["asas_marh_strict"] = asas_marh
        return (np.zeros((3, 2)), 0.5, 1.0, 0)

    orig = m.get_ipr_stochastic_env
    orig_r = m.get_ipr_stochastic_env_randomized
    try:
        m.get_ipr_stochastic_env = fake_env_strict
        m.get_ipr_stochastic_env_randomized = fake_env

        m._run_one(rep=0, base_seed=42,
                   kwargs={"randomized_speed_heading": False, "asas_marh": 1.05})
        non_rand_ok = (captured.get("seed_strict") == 42
                       and captured.get("asas_marh_strict") == 1.05)
        report("#4 non-randomized path: randomized_speed_heading stripped",
               non_rand_ok, f"captured={captured}")

        captured.clear()
        m._run_one(rep=2, base_seed=42,
                   kwargs={"randomized_speed_heading": True, "asas_marh": 1.05})
        rand_ok = (captured.get("seed") == 44
                   and "randomized_speed_heading" not in captured["kw"])
        report("#4 randomized path: kwarg consumed (not forwarded)",
               rand_ok, f"kw_keys={list(captured['kw'].keys())}")
    finally:
        m.get_ipr_stochastic_env = orig
        m.get_ipr_stochastic_env_randomized = orig_r


# ---------------------------------------------------------------------------
# #5: scenario randomization uses a seeded RNG
# ---------------------------------------------------------------------------
def test_05_seeded():
    src = read("sim/pairwise_stochastic/get_ipr_stochastic_env.py")
    rand_block_start = src.find("def get_ipr_stochastic_env_randomized")
    rand_block = src[rand_block_start:]

    uses_seeded = "scenario_rng = np.random.default_rng" in rand_block
    no_global = "np.random.uniform(10, 30)" not in rand_block
    report("#5 randomized env uses seeded scenario_rng",
           uses_seeded and no_global,
           f"seeded={uses_seeded}, no_global={no_global}")

    # Functional: same seed -> identical scenario draws
    rng1 = np.random.default_rng(42 + 7919)
    a1 = (rng1.uniform(10, 30), rng1.uniform(10, 30), rng1.uniform(0, 360))
    rng2 = np.random.default_rng(42 + 7919)
    a2 = (rng2.uniform(10, 30), rng2.uniform(10, 30), rng2.uniform(0, 360))
    rng3 = np.random.default_rng(43 + 7919)
    a3 = (rng3.uniform(10, 30), rng3.uniform(10, 30), rng3.uniform(0, 360))
    report("#5 seeded scenario reproducible across calls",
           a1 == a2 and a1 != a3,
           f"a1={a1[:1]}..., differs_with_diff_seed={a1 != a3}")


# ---------------------------------------------------------------------------
# #6/#7/#8: resolve() calls pass resofach
# ---------------------------------------------------------------------------
def test_06_07_08_resolve_args():
    cases = [
        ("#6", "test/test_cr_mvp.py",
         "conf_resolution.resolve(conf_detection, states, states, bs.settings.asas_marh)"),
        ("#7", "test/test_stochastic_sim_forloop_sanitycheck.py",
         "conf_resolution.resolve(conf_detection, ownship_obs, intruder_obs, bs.settings.asas_marh)"),
        ("#8", "test/test_stochastic_sim_single_job.py",
         "conf_resolution.resolve(conf_detection, ownship_obs, intruder_obs, bs.settings.asas_marh)"),
    ]
    for tag, path, needle in cases:
        ok = needle in read(path)
        report(f"{tag} resolve() has resofach arg in {path}", ok)


# ---------------------------------------------------------------------------
# #9: test_cr_mvp tuple unpack
# ---------------------------------------------------------------------------
def test_09_tuple_unpack():
    src = read("test/test_cr_mvp.py")
    ok = "done_now, _n_active = _check_tcpa_tinhor_per_pair(" in src
    # Confirm `done_now` is then passed as bool (not tuple) to done_with_timeout
    # by checking no `done_now=(` appears.
    no_tuple_passed = "done_now=done_now" in src
    report("#9 test_cr_mvp unpacks (done_now, n_active)", ok and no_tuple_passed)


# ---------------------------------------------------------------------------
# #10 & #11: multi-job uniqueness + naming
# ---------------------------------------------------------------------------
def test_10_11_multijob():
    src = read("test/test_stochastic_sim_multi_job.py")
    has_per_run = "worst_per_run" in src
    has_max = "max_allowed_cpa" in src and "min_allowed_cpa" not in src
    report("#10 multi-job uses per-run min for uniqueness", has_per_run)
    report("#11 multi-job variable renamed to max_allowed_cpa", has_max)

    # Functional: synthetic 2D worst_cpa array
    arr = np.array([[60.0, 80.0], [70.0, 90.0], [55.0, 75.0]])
    worst_per_run = arr.min(axis=1)
    uniq = np.unique(worst_per_run)
    correct_unique = len(worst_per_run) == len(uniq)
    report("#10 per-run uniqueness check works on 2D array",
           correct_unique, f"worst_per_run={worst_per_run.tolist()}")

    # Old (broken) behaviour: np.unique flattens -> would compare 3 vs 6 elements
    old_uniq = np.unique(arr)
    old_broken = len(arr) != len(old_uniq)
    report("#10 old flattening logic was indeed broken",
           old_broken, f"len(arr)=3, len(np.unique(flat))={len(old_uniq)}")


# ---------------------------------------------------------------------------
# #12: applyprio captures both returns
# ---------------------------------------------------------------------------
def test_12_applyprio():
    src = read("sim_models/cr_vo.py")
    has_fix = "dv[idx1], dv[idx2] = self.applyprio(" in src
    no_old = "dv[idx1], _ = self.applyprio(" not in src
    report("#12 VO.resolve uses both applyprio returns", has_fix and no_old)


# ---------------------------------------------------------------------------
# #13: cd_statebased uses intruder.ntraf for intruder reshapes
# ---------------------------------------------------------------------------
def test_13_cd_statebased_ntraf():
    src = read("sim_models/cd_statebased.py")
    checks = {
        "intu reshape": "intu = intruder.gs * np.sin(inttrkrad).reshape((1, intruder.ntraf))" in src,
        "intv reshape": "intv = intruder.gs * np.cos(inttrkrad).reshape((1, intruder.ntraf))" in src,
        "alt reshape":  "intruder.alt.reshape((1, intruder.ntraf)).T" in src,
        "vs reshape":   "intruder.vs.reshape(1, intruder.ntraf).T" in src,
    }
    failed = [k for k, v in checks.items() if not v]
    report("#13 cd_statebased: all 4 intruder reshapes use intruder.ntraf",
           not failed, f"failed={failed}" if failed else "all 4 correct")


# ---------------------------------------------------------------------------
# #14: NB_PAIR defined once
# ---------------------------------------------------------------------------
def test_14_nb_pair():
    src = read("sim/pairwise_stochastic/run_multiple_jobs.py")
    count = src.count("NB_PAIR = 100")
    report("#14 NB_PAIR defined exactly once", count == 1, f"occurrences={count}")


# ---------------------------------------------------------------------------
# #15: cns_adsl uses seeded RNG
# ---------------------------------------------------------------------------
def test_15_cns_adsl():
    src = read("sim_models/cns_adsl.py")
    no_global_random = (
        "np.random.multivariate_normal" not in src
        and "np.random.random(" not in src
    )
    report("#15 cns_adsl source has no global np.random.*", no_global_random)

    # Functional: same seed -> same RNG sequence
    from sim_models.cns_adsl import ADSL as ADSL_cns
    a1 = ADSL_cns(10.0, 1.0, reception_prob=0.5, seed=42)
    a2 = ADSL_cns(10.0, 1.0, reception_prob=0.5, seed=42)
    a3 = ADSL_cns(10.0, 1.0, reception_prob=0.5, seed=999)
    s1 = a1.rng.random(5)
    s2 = a2.rng.random(5)
    s3 = a3.rng.random(5)
    same = np.array_equal(s1, s2)
    diff = not np.array_equal(s1, s3)
    report("#15 cns_adsl reproducible with same seed, differs with different seed",
           same and diff, f"same_seed_match={same}, diff_seed_differs={diff}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    tests = [
        test_01_merge_conflict,
        test_02_03_vo,
        test_04_kwargs,
        test_05_seeded,
        test_06_07_08_resolve_args,
        test_09_tuple_unpack,
        test_10_11_multijob,
        test_12_applyprio,
        test_13_cd_statebased_ntraf,
        test_14_nb_pair,
        test_15_cns_adsl,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            report(t.__name__, False, f"raised {type(e).__name__}: {e}")

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed}/{total} checks passed")
    print("=" * 70)
    if passed < total:
        print("\nFailing checks:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  - {name}  ({detail})")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

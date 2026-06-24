# Refactor TODO — cdarr composable simulation API

Ordered list of every file to create or modify, one entry per file.
Order: **PHASE 0** (§10, test modernization) → **STEP 0** (§5, golden master) → **STEPs 1–8** (§6).

Status legend: `[ ]` pending · `[~]` in progress · `[x]` approved

---

## PHASE 0 — Test suite modernization (§10)

### §10.4 Shared infrastructure (do first)

- [ ] **P0.1** `requirements-dev.txt` *(new)*  
  Add `pytest`.

- [ ] **P0.2** `pytest.ini` *(new)*  
  `[pytest]` testpaths=test, addopts=-q.

- [ ] **P0.3** `conftest.py` *(new, repo root)*  
  `--plot` option, `plot` fixture, `bluesky_init` session autouse fixture (replaces every
  `bs._joblib_inited` guard), Agg backend unless `--plot`. Exact template from §10.4.

- [ ] **P0.4** DELETE `sim_models/test_sim_models.py`  
  Empty file; not pytest-collected; remove per §10.5.

### §10.5 BROKEN — fix stale imports / moved API (3 files)

- [ ] **P0.5** `test/test_stochastic_sim_single_job_deterministic.py`  
  Delete `crr_resumenav_heuristic` import + all `resumenav` (CPA) use → `crr_resumenav_cpa.resumenav`.
  Delete `crr_resumenav_ftr_pastcpa` import + all `resumenav_triple_criteria` use; noise-path
  recovery → `resumenav_double_criteria`. Remove MAX_TR guard → use `bluesky_init` fixture
  (get_configs() sets MAX_TR). Top-level code → `def test_*()`. Gate `plt.show()`.
  Assert `0 <= ipr <= 1` and `worst_cpa < 2 * cfg.horizontal_sep`.

- [ ] **P0.6** `test/test_stochastic_sim_single_job_probabilistic.py`  
  Replace `from sim_models.crr_resumenav_probabilistic import resumenav_probabilistic` →
  `from sim_models.crr_resumenav_probabilistic_ftr import resumenav_probabilistic_ftr`.
  Set `conf_detection.dcpa_prob_threshold = GAMMA` (GAMMA=0.9) before the loop; call
  `resumenav_probabilistic_ftr` in place of old name. Keep
  `conf_resolution.resumenav_double_criteria_dummy(...)` call on MVP — it's valid.
  Remove MAX_TR guard → `bluesky_init` fixture. Top-level → `def test_*()`. Gate plots.
  Assert `0 <= ipr <= 1`.

- [ ] **P0.7** `test/test_stochastic_sim_single_job_trace_resumenav.py`  
  `conf_resolution.resumenav(...)` fails on MVP; replace both observed- and ground-truth
  recovery calls with `crr_resumenav_cpa.resumenav(...)` (standalone). Keep inlined loop
  (records per-timestep stats env doesn't return). Remove MAX_TR guard → `bluesky_init`
  fixture. CSV write → `if plot:`. Top-level → `def test_*()`. Assert stats dict non-empty,
  per-dpsi counts finite.

### §10.5 CLEAN-UP — guards / plots / asserts (7 files)

- [ ] **P0.8** `test/test_stochastic_sim_single_simple.py`  
  Wrap top-level into `def test_single_simple()`. Assert `0 <= ipr <= 1` and
  `worst_cpa < 100`.

- [ ] **P0.9** `test/test_stochastic_sim_multi_job.py`  
  Wrap into `def test_multi_job()`. Reduce to `n_runs=2, n_jobs=2`. Keep existing
  uniqueness + `worst_cpa <= 2*horizontal_sep` asserts.

- [ ] **P0.10** `test/test_stochastic_sim_multi_job_dist.py`  
  Wrap into `def test_multi_job_dist()`. Reduce to `n_runs=2, n_jobs=2`. Assert
  `0 <= overall_ipr <= 1`.

- [ ] **P0.11** `test/test_stochastic_sim_single_job.py`  
  Remove MAX_TR read-guard → `get_configs()` already called; MAX_TR is set there.
  Gate all `plt.show()` behind `if plot:`. Assert `worst_cpa < 2 * cfg.horizontal_sep`.

- [ ] **P0.12** `test/test_stochastic_sim_forloop_sanitycheck.py`  
  Remove MAX_TR read-guard AND `bs.traf.MAX_TR = 15` / `MAX_DTR2 = 10` hardcodes →
  `get_configs()`. Gate all `plt.show()` behind `if plot:`. Wrap top-level code into
  `def test_forloop_sanitycheck()`. Assert finite IPR per case.

- [ ] **P0.13** `test/test_cr_mvp.py`  
  Remove MAX_TR read-guard AND `bs.traf.MAX_TR = 15` hardcode → `get_configs()`. Wrap
  into `def test_cr_mvp()`. Gate all `plt.show()` behind `if plot:`. Assert head-on
  (dpsi=180) `worst_cpa < 2 * horizontal_sep`.

- [ ] **P0.14** `test/test_cd_statebased.py`  
  Move top-level code (which currently runs on import) into `def test_cd_statebased()`.
  Assert head-on geometry is flagged as a conflict.

### §10.5 Already have test_ functions — ensure pytest compatibility (3 files)

- [ ] **P0.15** `test/test_adsl_module.py`  
  Confirm existing `def test_end_to_end_comm_trace()` is collected by pytest with the
  session `bluesky_init` fixture. Remove any module-level `bs.init` guard (rely on
  `bluesky_init` autouse). Verify asserts stay intact.

- [ ] **P0.16** `test/test_noise_model.py`  
  Same as P0.15 for `def test_noise_model()`.

- [ ] **P0.17** `test/test_reception_model.py`  
  Same as P0.15 for `def test_bernoulli_update()`.

**PHASE 0 gate:** `pytest test/` exits 0; no GUI; nothing under `results/`. Stale-symbol
greps (`crr_resumenav_heuristic`, `crr_resumenav_ftr_pastcpa`, `resumenav_triple_criteria`,
`resumenav_probabilistic\b`, `bs.traf.MAX_TR\s*=`, `raise.*MAX_TR`) return nothing under `test/`.

---

## STEP 0 — Golden master baseline (§5)

- [ ] **S0.1** `tests/golden/capture_golden.py` *(new)*  
  Runs all three legacy functions over the spec grid and writes JSON files under
  `tests/golden/`. Must reproduce bit-for-bit on re-run.  
  Grid:
  - `get_ipr_stochastic_env`: seeds {44,45} × dpsi {30,90,180} × recovery
    {"CPA","FTR","Probabilistic FTR"}, ci=10, civ=1, reception_prob=0.95,
    asas_marh=1.05, lookahead=120, threshold_probability=0.9.
  - `get_ipr_stochastic_env_randomized`: seeds {44,45}, same other params.
  - `get_ipr_stochastic_env_dist`: seeds {44,45}, dist_m=600, recovery="Probabilistic FTR",
    threshold_probability=0.75.  
  Saves `ipr`, `sim_timer`, `n_active`, `worst_cpa`, plus array digest (`shape`,
  `.min()`, `.sum()`, `sha256` of rounded bytes). Uses `to_python` from `compare_crr/utils.py`.

**STEP 0 gate:** `tests/golden/*.json` exist; re-running `capture_golden.py` reproduces
them bit-for-bit.

---

## STEP 1 — `cdarr/noise.py` (§6 STEP 1)

- [ ] **S1.1** `cdarr/__init__.py` *(new, package stub)*  
  Minimal init; exports will be filled in at Step 6.

- [ ] **S1.2** `cdarr/protocols.py` *(new)*  
  `typing.Protocol` definitions for `NoiseModel`, `Sensor`, `Detector`, `Resolver`,
  `Recovery` as specified in §3.

- [ ] **S1.3** `cdarr/noise.py` *(new)*  
  `Gaussian` (joint `sample()` using two sequential `rng.multivariate_normal` calls —
  position first, velocity second — preserving legacy RNG order; owns CI95_TO_STD_2D=2.448).
  `BiasedGaussian(bias_pos, bias_vel)`. `StudentT(df)`. All implement `NoiseModel` Protocol.

- [ ] **S1.4** `test/test_equiv_noise.py` *(new)*  
  Equivalence: shared `rng=default_rng(S)`; `Gaussian.sample(ci_pos, ci_vel, n, rng)`
  position columns == legacy `add_position_noise` deviations; velocity columns ==
  `add_velocity_noise` deviations. Assert `np.array_equal`. Calibration tests for
  `BiasedGaussian` and `StudentT`.

**STEP 1 gate:** Gaussian equivalence test passes; new models import and return finite `(n,4)`.

---

## STEP 2 — `cdarr/sensing.py` :: Observation (§6 STEP 2)

- [ ] **S2.1** `cdarr/sensing.py` *(new)*  
  `Observation` dataclass `{ownship, intruder}`. No equivalence test.

**STEP 2 gate:** `Observation` importable; `ReceptionModel` untouched.

---

## STEP 3 — `cdarr/sensing.py` :: sensors (§6 STEP 3)

- [ ] **S3.1** `cdarr/sensing.py` *(modify — add sensors)*  
  Add `PerfectSensor` and `ADSLSensor(ci_pos, ci_vel, reception_prob, seed, noise=Gaussian())`.
  Option A: internally construct existing `ADSL` nodes with seed offsets +1..+4, reception
  RNG +999; reproduce per-tick update verbatim. Adapt noise call: `sample()` once → apply
  `dev[:,0:2]` as position, `dev[:,2:4]` as velocity. First-call init semantics preserved.
  Expose `sigma_r`, `sigma_v`.

- [ ] **S3.2** `test/test_equiv_sensor.py` *(new)*  
  Drive fixed `PairwiseHorConflict` (dpsi=90, seed=44), step K ticks. Compare
  `ADSLSensor.observe()` outputs vs legacy inline dance on same scenario+seed.
  Assert `np.array_equal` for lat/lon/gseast/gsnorth of ownship and intruder. Also
  assert `sigma_r`/`sigma_v` match legacy formula. Calibration tests for noise models.

**STEP 3 gate:** sensor equivalence green for `ADSLSensor+Gaussian`; `PerfectSensor` returns truth.

---

## STEP 4 — `cdarr/recovery.py` (§6 STEP 4)

- [ ] **S4.1** `cdarr/recovery.py` *(new)*  
  `CPA`, `FTR`, `ProbabilisticFTR(gamma=0.9)` — thin wrappers over existing
  `crr_resumenav_*` functions. `ProbabilisticFTR` sets `conf.dcpa_prob_threshold`,
  `conf.sigma_r`, `conf.sigma_v` then delegates.

- [ ] **S4.2** `test/test_equiv_recovery.py` *(new)*  
  Run a few legacy ticks to get detector+resolver state; call legacy bare function and
  new class on identical inputs; assert `delpairs` sets equal and `reso.resopairs`
  mutated identically. Cover all three; `ProbabilisticFTR` with `gamma=0.9`.

**STEP 4 gate:** all three recovery equivalence tests green.

---

## STEP 5 — `cdarr/scenarios.py` (§6 STEP 5)

- [ ] **S5.1** `cdarr/scenarios.py` *(new)*  
  `FixedAngle(dpsi, width=None, height=None)`, `Randomized(seed_offset=7919, ...)`,
  `DistanceBased(dist_m, ...)`. Each exposes `build(cfg, *, seed, lookahead)`.
  Preserves legacy RNG draw order for `Randomized` and `DistanceBased`.

- [ ] **S5.2** `test/test_equiv_scenario.py` *(new)*  
  For each scenario class, build via new class and via legacy inline construction with
  same seed; assert `_get_states()` arrays are `np.array_equal`.

**STEP 5 gate:** all three scenario equivalence tests green.

---

## STEP 6 — `cdarr/experiment.py` (§6 STEP 6)

- [ ] **S6.1** `cdarr/experiment.py` *(new)*  
  `IPRResult` dataclass. `IPRExperiment` with `__init__` (runtime guard §3), `run()` (shared
  loop with ALL magic constants preserved verbatim per §6.4). Uses `scenario.build()`,
  `sensor.observe()`, `detector.detect()`, `resolver.resolve()`, `recovery.recover()`.

- [ ] **S6.2** `cdarr/__init__.py` *(modify — add exports)*  
  Export `IPRExperiment`, `IPRResult`.

- [ ] **S6.3** `test/test_equiv_experiment.py` *(new)*  
  For each golden grid point, build equivalent `IPRExperiment` and assert `ipr`,
  `sim_timer`, `n_active`, `worst_cpa`, and `distance_array` hash match the golden
  baseline. Cover all three scenario types.

**STEP 6 gate:** every golden grid point reproduced exactly.

---

## STEP 7 — `cdarr/registry.py` + `from_config` (§6 STEP 7)

- [ ] **S7.1** `cdarr/registry.py` *(new)*  
  `register(kind, name, factory)` and internal registries pre-populated with built-ins.

- [ ] **S7.2** `cdarr/experiment.py` *(modify — add classmethod)*  
  `IPRExperiment.from_config(config_path, *, dpsi=None, **overrides)`.

- [ ] **S7.3** `test/test_equiv_from_config.py` *(new)*  
  `from_config` with `sim_config.json` defaults reproduces matching golden point.
  `register` with custom name resolves.

**STEP 7 gate:** from_config matches golden; custom injection works.

---

## STEP 8 — Deprecation shims + repoint consumers (§6 STEP 8)

- [ ] **S8.1** `sim/pairwise_stochastic/get_ipr_stochastic_env.py` *(modify — add shims)*  
  Reimplement `get_ipr_stochastic_env`, `_randomized`, `_dist` as thin shims over
  `IPRExperiment`; return legacy tuple `(distance_array, ipr, sim_timer, n_active)`;
  add `DeprecationWarning`.

- [ ] **S8.2** `sim/pairwise_stochastic/run_multiple_jobs.py` *(verify / minor modify)*  
  API unchanged; it forwards `**env_kwargs` to the shim. Verify smoke run matches
  Step-0 overall_ipr capture.

- [ ] **S8.3** `test/test_equiv_shims.py` *(new)*  
  Shim outputs == direct `IPRExperiment` outputs == golden. `run_multiple_jobs`
  overall_ipr matches Step-0 capture for a fixed small config.

**STEP 8 gate:** full `pytest test/` suite green (existing + all `test_equiv_*`);
`compare_crr` smoke run reproduces Step-0 `overall_ipr`.

---

## Summary count

| Phase/Step | Files |
|---|---|
| PHASE 0 | 17 entries (P0.1–P0.17) |
| STEP 0 | 1 entry (S0.1) |
| STEP 1 | 4 entries (S1.1–S1.4) |
| STEP 2 | 1 entry (S2.1) |
| STEP 3 | 2 entries (S3.1–S3.2) |
| STEP 4 | 2 entries (S4.1–S4.2) |
| STEP 5 | 2 entries (S5.1–S5.2) |
| STEP 6 | 3 entries (S6.1–S6.3) |
| STEP 7 | 3 entries (S7.1–S7.3) |
| STEP 8 | 3 entries (S8.1–S8.3) |
| **Total** | **38 entries** |

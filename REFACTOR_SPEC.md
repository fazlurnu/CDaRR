# Refactor Specification — `cdarr` composable simulation API

**Audience:** the engineer (Sonnet) who will implement this, file by file.
**Status:** specification. No code has been written yet.

---

## 0. Purpose

Refactor the stochastic pairwise-conflict simulator into a composable, dependency-injected
API whose end goal is:

```python
from cdarr import IPRExperiment
from cdarr.scenarios import FixedAngle
from cdarr.sensing import ADSLSensor, PerfectSensor
from cdarr.noise import Gaussian, BiasedGaussian, StudentT
from sim_models.cd_statebased import StateBased
from sim_models.cr_mvp import MVP
from cdarr.recovery import CPA, FTR, ProbabilisticFTR

result = IPRExperiment(
    scenario = FixedAngle(dpsi=90),
    sensor   = ADSLSensor(ci_pos=10, ci_vel=1, reception_prob=0.95, seed=44, noise=Gaussian()),
    detector = StateBased(),
    resolver = MVP(),
    recovery = ProbabilisticFTR(gamma=0.9),
    asas_marh=1.05, lookahead=120,
).run()      # -> IPRResult(ipr, worst_cpa, distance_array, sim_timer, n_active)
```

The user must be able to inject their own detector / resolver / recovery / noise model
(Tier-1 duck typing — see §3).

---

## 0.1 Working agreement (process — read first)

This refactor is executed **incrementally, one file at a time, with the human as approver.**
The coding agent does NOT implement the whole spec in one pass and does NOT commit to git.
Concretely:

1. **Start by producing a todo list.** Before writing any code, the coding agent writes a
   todo list (e.g. in a `REFACTOR_TODO.md` or the task tracker) enumerating, in order, each
   file that will be created or modified. The order is: **PHASE 0 (§10, test-suite
   modernization) FIRST**, then STEP 0 (§5, golden master), then STEPs 1–8 (§6). One entry
   per file. Present it to the human (the boss) for review.
2. **Implement one file, then stop and report.** For each todo item: implement exactly that
   one file plus its equivalence test, run the test, and report the result (pass/fail +
   evidence). Then **wait for the boss to say it's okay** before starting the next file.
3. **The boss approves each step.** The human reviews the file and the equivalence result and
   explicitly approves before the agent proceeds. If not okay, fix and re-report — do not move on.
4. **The boss owns git.** The coding agent does **not** run `git add`/`commit`/`push`. The
   human commits to the repository themselves, at whatever granularity they choose, after
   approving.
5. **Keep the todo list current.** Mark each item done only after the boss approves it, so the
   list always reflects what is finished, in-progress, and pending.

In short: agent proposes the file plan → implements one file + its equivalence test → reports →
boss approves → next file. No batch implementation, no autonomous commits.

## 1. NON-NEGOTIABLE PRINCIPLE: behavior must be identical

This is a **pure refactor**. For every existing code path there is an equivalent new
code path, and the two MUST produce **the same output for the same inputs and the same
seed**. We prove this, we do not assume it.

### 1.1 The equivalence rules

1. **Golden master first (Step 0).** Before changing anything, capture reference outputs
   from the *current* code over a small parameter grid. Every later step is validated
   against this frozen baseline.
2. **One file at a time.** Implement exactly one new module per step. Do not start the
   next step until the current step's equivalence test passes.
3. **Bottom-up.** Start at the leaf components (noise) and build up to `IPRExperiment`.
4. **Preserve RNG draw order exactly.** Numerical equivalence depends on the random
   number generator being seeded identically *and consumed in the same order*. This is
   the single biggest risk. Any reordering of `rng` calls breaks bitwise equivalence.
   See §1.3.
5. **Preserve every magic constant.** The legacy loop hardcodes values that MUST be
   carried over verbatim (see §6.4): `tmax = 600`, `dtlookahead = lookahead * 1.5`,
   detection `hpz = 100.0`, the `min_dist_so_far > 50.0` term in the `n_active` count,
   `DONE_TIMEOUT` from config, ADSL `CI95_TO_STD_2D = 2.448`, ADSL seed offsets
   `+1/+2/+3/+4` and reception RNG offset `+999`, scenario RNG offset `+7919`.
6. **The legacy code stays until its replacement is proven.** Do not delete
   `get_ipr_stochastic_env*` until Step 8; they are the equivalence oracle. At Step 8 they
   become thin shims over `IPRExperiment`.

### 1.2 What "same output" means per step

- **Numeric paths** (noise, sensor, full sim): bitwise-identical arrays when RNG order is
  preserved. Assert with `np.array_equal` first; if a legitimate float-reassociation makes
  that impossible, fall back to `np.allclose(rtol=0, atol=1e-9)` AND document why exact
  equality was not achievable.
- **Side-effecting paths** (detect/resolve/recover): identical resulting `delpairs` set and
  identical post-call attributes / `reso` tuple (these classes are reused unchanged, so this
  should be trivially true — see §4).

### 1.3 RNG equivalence detail (read before Step 1 and Step 3)

The legacy ADSL node (`sim_models/adsl_module.py`) builds **one** `rng =
np.random.default_rng(seed)` and shares it across reception + noise. Within
`update_from_truth` the order is: (a) `reception.sample_indices` (only after first tick),
then (b) `noise.add_position_noise` (`rng.multivariate_normal`), then (c)
`noise.add_velocity_noise` (`rng.multivariate_normal`). The four ADSL nodes use seeds
`seed+1..+4`; the loop's separate reception mask uses a node's own `reception`. Any new
`ADSLSensor`/`NoiseModel` MUST reproduce this exact construction and call order, or golden
equivalence will fail. When in doubt, the new `ADSLSensor` should *contain the existing
`ADSL` objects unchanged* rather than re-implement them (see §6.3, Option A).

---

## 2. Inputs (existing files to read)

The implementer should treat these as authoritative references:

| Concern | File |
|---|---|
| Legacy orchestrators (the oracle) | `sim/pairwise_stochastic/get_ipr_stochastic_env.py` (3 functions) |
| Parallel wrapper | `sim/pairwise_stochastic/run_multiple_jobs.py` |
| Config loader | `sim/utils.py` (`get_configs`, `done_with_timeout`, `_check_tcpa_tinhor_per_pair`) |
| Noise (to split) | `sim_models/noise_model.py` |
| Reception | `sim_models/reception_model.py` |
| ADSL node | `sim_models/adsl_module.py`, `sim_models/adsl_message.py` |
| Detection | `sim_models/cd_statebased.py` |
| Resolution | `sim_models/cr_mvp.py`, `sim_models/cr_vo.py` |
| Recovery (to wrap) | `sim_models/crr_resumenav_cpa.py`, `crr_resumenav_ftr.py`, `crr_resumenav_probabilistic_ftr.py`, `crr_recovery_base.py` |
| Scenarios | `envs/pairwise_conflict.py` (`PairwiseHorConflict`, `PairwiseHorConflictDist`) |
| Config values | `sim_configs/sim_config.json` |
| Downstream consumer | `compare_crr/compare_crr.py`, `compare_crr/generate_ipr_from_dpsi_samples.py` |

---

## 3. Extensibility contract (Tier-1 duck typing)

Users extend by passing any object that matches the documented shape. No subclassing
required. The four seams:

| Seam | Required method | Output contract |
|---|---|---|
| **NoiseModel** | `sample(ci_pos, ci_vel, n, rng) -> (n,4) float` | one **joint** draw returning columns `[east_m, north_m, vn, ve]`; owns its own CI→scale calibration. Joint (not two separate methods) so future models can correlate position↔velocity — see §3.1 |
| **Sensor** | `observe(truth) -> Observation`; attrs `sigma_r`, `sigma_v` | returns `(ownship_obs, intruder_obs)` state-like objects |
| **Detector** | `detect(own, intr, rpz, hpz, lookahead) -> None` | **side effect**: populates `confpairs`, `tcpa_all`, `tinhor_all`, `rpz` on self |
| **Resolver** | `resolve(conf, own, intr, resofach) -> reso` | returns the `reso` tuple consumed by `pairwise.step()` (positional: hdg, spd, alt, vs, resopairs) |
| **Recovery** | `recover(reso, conf, own, intr, *, sigma_r, sigma_v) -> set` | **side effect**: mutates resolution/`bs.traf` to resume nav; returns resumed `delpairs` (observability only) |

Define these as `typing.Protocol` (documentation + optional static checking). Additionally,
`IPRExperiment.__init__` performs a **lightweight runtime guard**: check each injected
component exposes the required callable(s) and raise a clear `TypeError` naming the missing
method. Do NOT rely on `@runtime_checkable`/`isinstance` alone (it checks presence, not
signatures).

Detector and Resolver are **kept as the existing classes** (`StateBased`, `MVP`, `VO`).
They already satisfy the contract; we do not rewrite them.

### 3.1 Why the noise draw is joint (read before Step 1)

The `NoiseModel` interface is a single **joint** draw `sample(ci_pos, ci_vel, n, rng) ->
(n,4)` returning `[east_m, north_m, vn, ve]`, NOT two separate `sample_position` /
`sample_velocity` methods. Reason: position↔velocity correlation lives entirely in the
*draw* (a joint 4D distribution with nonzero cross-covariance between the position block and
the velocity block); the *application* of the deviations (east→lon, north→lat, vn/ve→velocity
components) is per-column and identical regardless of distribution. Two independent draws
structurally cannot be correlated; one joint draw can. Keeping the draw joint now means a
future correlated model is just a new class — no interface change, no edits to the sensor or
other noise models.

**Crucial equivalence caveat:** a joint *interface* does NOT force a joint *draw*. The
legacy Gaussian draws position (2D) then velocity (2D) as two sequential
`rng.multivariate_normal` calls; a single 4D `multivariate_normal` consumes the RNG
differently (different standard-normal stream layout) and would break bitwise equivalence
even with a block-diagonal covariance. Therefore the **default `Gaussian` must implement
`sample()` by doing the two legacy draws internally (position first, then velocity) and
concatenating** — this preserves the legacy RNG order and keeps the golden baseline
bitwise-identical. Only genuinely new models (e.g. `CorrelatedGaussian`) do a true 4D draw,
and those have no legacy equivalent so they need not match any golden value.

The sensor always calls `sample()` once per updated set of aircraft and splits the columns:
`dev[:, 0:2]` → position application, `dev[:, 2:4]` → velocity application, through the
single shared (distribution-agnostic) geometry helper.

---

## 4. Target package layout

New package `cdarr/` (sibling to `sim/`, `sim_models/`, `envs/`):

```
cdarr/
  __init__.py            # exports IPRExperiment, IPRResult
  protocols.py           # NoiseModel, Sensor, Detector, Resolver, Recovery (Protocols)
  noise.py               # Gaussian (default), BiasedGaussian, StudentT
  sensing.py             # Observation, PerfectSensor, ADSLSensor
  recovery.py            # CPA, FTR, ProbabilisticFTR (thin wrappers)
  scenarios.py           # FixedAngle, Randomized, DistanceBased
  experiment.py          # IPRResult, IPRExperiment (the shared loop + .run + .from_config)
  registry.py            # register(), build_from_config() (Tier-2, optional)
tests/golden/            # frozen baseline outputs from Step 0
test/                    # existing pytest suite (keep; add equivalence tests here)
```

Existing `sim_models/*` stay where they are. `cdarr.recovery` wraps the existing
`crr_resumenav_*` functions; it does not move or rewrite them.

---

## 5. STEP 0 — Golden master baseline (first refactor step; PHASE 0 §10 precedes it)

> PHASE 0 (§10, test-suite modernization) is built before this. STEP 0 is the first
> step of the refactor proper — no production code changes yet, just capture the baseline.

**Goal:** freeze the current behavior so every later step has a reference.

**Create:** `tests/golden/capture_golden.py` and the data files it writes under
`tests/golden/`.

**What to capture:** run each existing function over a small deterministic grid and save
the full outputs to JSON (use the existing `to_python` helper in `compare_crr/utils.py`
for numpy→json):

- `get_ipr_stochastic_env`: grid = seeds `{44, 45}` × dpsi `{30, 90, 180}` ×
  recovery `{"CPA", "FTR", "Probabilistic FTR"}`, with `ci=10, civ=1, reception_prob=0.95,
  asas_marh=1.05, lookahead=120, threshold_probability=0.9`. Save `ipr`, `sim_timer`,
  `n_active`, and `worst_cpa = float(distance_array.min())`, plus a hash of
  `distance_array` (full array is large — store `shape`, `distance_array.min()`,
  `distance_array.sum()`, and `sha256` of the rounded bytes).
- `get_ipr_stochastic_env_randomized`: seeds `{44, 45}`, same other params.
- `get_ipr_stochastic_env_dist`: seeds `{44, 45}`, `dist_m=600`, `recovery="Probabilistic FTR"`,
  `threshold_probability=0.75`.

**Must run in the `cdarr` conda env** (see README — numpy/navdata constraint).

**Done when:** `tests/golden/*.json` exist and re-running `capture_golden.py` reproduces
them bit-for-bit (proves the baseline itself is deterministic).

---

## 6. Implementation steps (one module per step)

Each step uses the template: **Goal → Read → Create → Contract → Equivalence test → Done**.
Do not proceed until "Done" is green.

### STEP 1 — `cdarr/noise.py`

- **Goal:** split the *distribution* out of `sim_models/noise_model.py`. The meters→lat/lon
  conversion and velocity decomposition stay in the application layer (the sensor / a shared
  helper); only the random draw becomes pluggable.
- **Read:** `sim_models/noise_model.py`, `sim_models/adsl_module.py` (the
  `CI95_TO_STD_2D = 2.448` calibration at lines 28, 39–40).
- **Create:** `Gaussian` (default), `BiasedGaussian(bias_pos=(e,n), bias_vel=(vn,ve))`,
  `StudentT(df)`. Each implements the **joint** `sample(ci_pos, ci_vel, n, rng) -> (n,4)`
  returning columns `[east_m, north_m, vn, ve]`, and **owns its CI→scale calibration**
  (Gaussian uses the existing 2.448 factor so 95% of 2D radii fall within CI).
  - **`Gaussian` MUST preserve legacy RNG order** (§3.1): implement `sample()` as a position
    `multivariate_normal((0,0), pos_cov, size=n)` draw FOLLOWED BY a velocity
    `multivariate_normal((0,0), vel_cov, size=n)` draw, with covariances built from
    `ci/2.448`, then `np.hstack`. Do NOT use a single 4D draw for the default Gaussian.
- **Contract:** matches `NoiseModel` Protocol (§3, §3.1).
- **Equivalence test** (`test/test_equiv_noise.py`): `Gaussian.sample(...)` must reproduce the
  legacy `NoiseModel` draws exactly. With a shared `rng = default_rng(S)`, the position
  columns `dev[:, 0:2]` must equal the legacy `add_position_noise` deviations and the velocity
  columns `dev[:, 2:4]` the legacy `add_velocity_noise` deviations — same
  `rng.multivariate_normal((0,0), cov, size=n)` calls, **position before velocity**. Assert
  `np.array_equal`. `BiasedGaussian`/`StudentT` are NEW — assert their *calibration* property
  only (see Step 3 test note); they have no legacy equivalent.
- **Done:** Gaussian equivalence test passes; new models import and produce finite `(n,4)`.

### STEP 2 — `cdarr/sensing.py` :: reception reuse

- **Goal:** no new reception logic — reuse `sim_models/reception_model.py::ReceptionModel`
  unchanged. This step just confirms it and defines the `Observation` container.
- **Read:** `sim_models/reception_model.py`.
- **Create (in `sensing.py`):** `Observation` dataclass `{ownship, intruder}` (lightweight;
  may just hold the two ADSL message views).
- **Contract:** none beyond holding two state-like objects.
- **Equivalence test:** none (no behavior change).
- **Done:** `Observation` importable; `ReceptionModel` left untouched.

### STEP 3 — `cdarr/sensing.py` :: sensors

- **Goal:** encapsulate the per-tick ADSL "4-node bus + reception mask + hold-last buffer"
  dance (legacy loop lines 223–244 / 408–429) behind `Sensor.observe(truth)`.
- **Read:** `get_ipr_stochastic_env.py` lines 200–244 (`_create_adsl_stack` + init block +
  per-event ADSL update), `sim_models/adsl_module.py`.
- **Create:**
  - `PerfectSensor`: `observe(truth) -> Observation(truth, truth)`; `sigma_r = sigma_v = 0.0`.
    This unifies the deterministic/ground-truth path.
  - `ADSLSensor(ci_pos, ci_vel, reception_prob, seed, noise=Gaussian())`: **Option A
    (preferred for equivalence)** — internally construct the *existing* `ADSL` nodes
    (`adsl_bus, ownship, intruder, prev_intruder`) with the legacy seed offsets `+1..+4`
    and reception RNG `+999`, and reproduce `update_from_truth` + the rx/miss masking +
    `prev` buffer update verbatim. `observe(truth)` runs one tick of that dance and returns
    `Observation(ownship_node, intruder_node)`. Expose `sigma_r = sqrt(own.pos_std²+intr.pos_std²)`,
    `sigma_v = sqrt(own.vel_std²+intr.vel_std²)` (legacy lines 270–271/455–456).
    The `noise` model is injected into the ADSL nodes (replace the hardcoded
    `NoiseModel(...)` at `adsl_module.py:47`); with `noise=Gaussian()` behavior is unchanged.
    Because the new `NoiseModel` exposes the joint `sample(ci_pos, ci_vel, n, rng) -> (n,4)`
    (§3.1) instead of `add_position_noise`/`add_velocity_noise`, the ADSL node's noise
    application is adapted to: call `sample()` once for the updated indices, then apply
    `dev[:,0:2]` as position (the existing meters→lat/lon conversion) and `dev[:,2:4]` as
    velocity (the existing gs/trk decomposition). The conversion math is unchanged and lives
    in one shared helper; only the call into the noise model changes. With the default
    `Gaussian` (position draw then velocity draw internally) the RNG order and outputs are
    bitwise-identical to legacy.
  - First-call semantics: the first `observe` must do the full no-loss init (legacy
    `initialized` block) then the regular update, matching legacy ordering.
- **Contract:** matches `Sensor` Protocol (§3).
- **Equivalence test** (`test/test_equiv_sensor.py`): drive a fixed `PairwiseHorConflict`
  (FixedAngle dpsi=90, seed 44) and step it K times. At each tick compare
  `ADSLSensor.observe(states)` outputs (`ownship_obs.lat/lon/gseast/gsnorth`, `intruder_obs.*`)
  against the legacy inline dance run on the same scenario with the same seed. Assert
  `np.array_equal`. Also assert `sigma_r/sigma_v` equal the legacy formula.
  Separately, a *calibration* test for noise models: with a static truth and many `observe`
  calls, `Gaussian` gives ~95% of position radii ≤ `ci_pos` (reuse `test_noise_model` logic);
  `StudentT` asserts its own calibrated 95%; `BiasedGaussian` asserts the empirical mean
  offset ≈ its bias.
- **Done:** sensor equivalence test green for `ADSLSensor`+`Gaussian`; `PerfectSensor`
  returns truth unchanged.

### STEP 4 — `cdarr/recovery.py`

- **Goal:** wrap the three existing recovery functions as classes that hold their own params,
  removing the detector monkey-patch (legacy lines 270–272 / 455–457).
- **Read:** `crr_resumenav_cpa.py`, `crr_resumenav_ftr.py`,
  `crr_resumenav_probabilistic_ftr.py`, `crr_recovery_base.py`.
- **Create:**
  - `CPA`: `recover(reso, conf, own, intr, *, sigma_r=None, sigma_v=None)` → calls
    `crr_resumenav_cpa.resumenav(reso, conf, own, intr)`. Ignores sigmas.
  - `FTR`: → `crr_resumenav_ftr.resumenav_double_criteria(...)`. Ignores sigmas.
  - `ProbabilisticFTR(gamma=0.9)`: before delegating, set `conf.dcpa_prob_threshold = self.gamma`,
    `conf.sigma_r = sigma_r`, `conf.sigma_v = sigma_v` (the existing
    `resumenav_probabilistic_ftr` reads these off `conf`), then call it. This keeps the proven
    math untouched; the difference is that `gamma` now lives on the recovery object and the
    sigmas arrive as explicit args from the sensor instead of being assigned in the loop.
- **Contract:** matches `Recovery` Protocol (§3).
- **Equivalence test** (`test/test_equiv_recovery.py`): construct a detector+resolver state by
  running a few legacy ticks; call the legacy bare function and the new class on identical
  inputs (same `conf` with same `sigma_r/sigma_v/dcpa_prob_threshold`); assert the returned
  `delpairs` sets are equal and `reso.resopairs` mutated identically. For `ProbabilisticFTR`,
  set the legacy path's `conf.dcpa_prob_threshold = 0.9` to match `gamma=0.9`.
- **Done:** all three recovery equivalence tests green.

### STEP 5 — `cdarr/scenarios.py`

- **Goal:** extract the three scenario constructions; the loop body is shared (Step 6).
- **Read:** `get_ipr_stochastic_env.py` scenario blocks — fixed (lines 184–194), randomized
  (368–379), dist (`get_ipr_stochastic_env_dist`, ~530+); `envs/pairwise_conflict.py`.
- **Create:**
  - `FixedAngle(dpsi, width=None, height=None)`: builds `PairwiseHorConflict` with config
    speeds and `init_dpsi=dpsi`, `dtlookahead = lookahead*1.5`. (width/height default to cfg.)
  - `Randomized(seed_offset=7919, speed_range=(10,30), angle_range=(0,360))`: builds
    `PairwiseHorConflict` with `scenario_rng = default_rng(seed + 7919)` draws **in legacy
    order**: ownship speed, intruder speed, dpsi.
  - `DistanceBased(dist_m, ...)`: builds `PairwiseHorConflictDist` reproducing
    `get_ipr_stochastic_env_dist` per-pair randomization exactly.
  - Each exposes `build(cfg, *, seed, lookahead) -> PairwiseHorConflict`.
- **Contract:** matches `Scenario` Protocol (§3).
- **Equivalence test** (`test/test_equiv_scenario.py`): for each scenario, build via the new
  class and via the legacy inline construction with the same seed; assert the initial
  `_get_states()` arrays (lat/lon/gs/trk/id, nb_pair) are `np.array_equal`. Critical for
  `Randomized` (RNG draw order) and `DistanceBased` (per-pair draws).
- **Done:** all three scenario equivalence tests green.

### STEP 6 — `cdarr/experiment.py`

- **Goal:** the single shared loop + result object. This replaces the bodies of all three
  legacy functions.
- **Read:** `get_ipr_stochastic_env.py` full loop (lines 206–318), `sim/utils.py`
  (`done_with_timeout`, `_check_tcpa_tinhor_per_pair`), `run_multiple_jobs.py` (worst_cpa def).
- **Create:**
  - `IPRResult` dataclass: `ipr: float, worst_cpa: float, distance_array: np.ndarray,
    sim_timer: float, n_active: int`. `worst_cpa = float(distance_array.min())`.
  - `IPRExperiment(scenario, sensor, detector, resolver, recovery, *, asas_marh, lookahead,
    seed=44, config_path=None, cfg=None)`:
    - `__init__` runs the §3 runtime guard.
    - `run() -> IPRResult` contains the shared loop, structured as:
      1. resolve cfg (`get_configs(config_path)` or injected `cfg`); set `bs.settings.asas_marh`.
      2. `pairwise = scenario.build(cfg, seed=seed, lookahead=lookahead)`.
      3. detector_gt = a second `StateBased` for ground truth (as legacy does).
      4. loop while `sim_timer < tmax` (**tmax = 600 preserved**):
         - `truth = pairwise._get_states()`
         - on first/every ASAS event tick: `obs = sensor.observe(truth)`;
           `detector.detect(obs.ownship, obs.intruder, cfg.horizontal_sep, 100.0, lookahead)`;
           `detector_gt.detect(truth, truth, cfg.horizontal_sep, 100.0, lookahead)`;
           `reso = resolver.resolve(detector, obs.ownship, obs.intruder, asas_marh)`;
           `recovery.recover(reso, detector, obs.ownship, obs.intruder,
           sigma_r=sensor.sigma_r, sigma_v=sensor.sigma_v)`.
         - `dist = pairwise.step(reso)`; append; done-logic (`_check_tcpa_tinhor_per_pair`
           on detector_gt, `min_dist_so_far > 50.0`, `done_with_timeout`); `sim_timer += simdt`.
      5. `pairwise.reset()`; compute `ipr` via the legacy `_compute_ipr` formula; return `IPRResult`.
    - **Preserve all magic constants** (§6.4) and the **event-tick scheduling math** exactly
      (`next_event_t`, `missed`, `eps`).
- **Contract:** the wrapper; no new external contract.
- **Equivalence test** (`test/test_equiv_experiment.py`): for the Step-0 golden grid, build
  the equivalent `IPRExperiment` (`FixedAngle`+`ADSLSensor(Gaussian)`+`StateBased`+`MVP`+
  recovery class) and assert `ipr`, `sim_timer`, `n_active`, `worst_cpa`, and the
  `distance_array` hash **match the golden baseline**. Cover all three scenarios
  (FixedAngle vs `get_ipr_stochastic_env`; Randomized vs `_randomized`; DistanceBased vs `_dist`).
- **Done:** every golden grid point reproduced exactly.

#### 6.4 Magic constants to carry verbatim
`tmax = 600`; `dtlookahead = lookahead * 1.5`; detection `hpz = 100.0`; `n_active`
filter `min_dist_so_far > 50.0`; `DONE_TIMEOUT = cfg.DONE_TIMEOUT`; `simdt =
bs.settings.simdt * cfg.SIMDT_FACTOR`; `event_dt = bs.settings.asas_dt`; `eps =
finfo(float).eps*100`; ADSL `CI95_TO_STD_2D = 2.448`; seed offsets `+1/+2/+3/+4`,
reception `+999`, scenario `+7919`.

### STEP 7 — `cdarr/registry.py` + `IPRExperiment.from_config`

- **Goal:** Tier-2 named construction so config/CLI/`compare_crr` keep working. Objects remain
  the primitive; config just names them.
- **Read:** `sim/utils.py::get_configs`, legacy `_RESOLUTION_MODELS`/`_RECOVERY_MODELS` dicts,
  `sim_config.json`.
- **Create:**
  - `register(kind, name, factory)` and internal registries for `"noise"`, `"detector"`,
    `"resolver"`, `"recovery"`, `"scenario"`, pre-populated with the built-ins.
  - `IPRExperiment.from_config(config_path, *, dpsi=None, **overrides)`: read cfg, look up
    `resolution`/`recovery`/`noise` names, build the objects, construct an `IPRExperiment`.
    Must reproduce how `compare_crr`/`run_multiple_jobs` currently select models.
- **Equivalence test** (`test/test_equiv_from_config.py`): `IPRExperiment.from_config(...)`
  with `sim_config.json` defaults reproduces the matching golden point.
- **Done:** from_config path matches golden; `register` adds a custom name that resolves.

### STEP 8 — Deprecation shims + repoint consumers

- **Goal:** make the legacy entry points thin wrappers so nothing downstream breaks, then the
  refactor is complete.
- **Read:** `get_ipr_stochastic_env.py`, `run_multiple_jobs.py`, `generate_ipr_from_dpsi_samples.py`.
- **Do:**
  - Reimplement `get_ipr_stochastic_env`, `_randomized`, `_dist` as shims that build the
    equivalent `IPRExperiment` and return the legacy tuple `(distance_array, ipr, sim_timer,
    n_active)`. Add a `DeprecationWarning`.
  - Leave `run_multiple_jobs` API unchanged (it forwards `**env_kwargs`); it now calls the shim
    (or directly builds `IPRExperiment` per rep with seed offset). Its return dict is unchanged.
  - `compare_crr` requires **no change** (it goes through `run_multiple_jobs`); verify it still
    runs and, for one config, produces the same `overall_ipr` as before (compare to a value
    captured in Step 0 from `run_multiple_jobs`).
- **Equivalence test** (`test/test_equiv_shims.py`): the shim outputs equal direct
  `IPRExperiment` outputs (both equal golden). `run_multiple_jobs` overall_ipr matches a Step-0
  capture for a fixed small config.
- **Done:** full pytest suite green (existing + all `test_equiv_*`); `compare_crr` smoke run
  matches the captured `overall_ipr`.

---

## 7. Global acceptance criteria

- [ ] `tests/golden/` baseline exists and is reproducible.
- [ ] Each `test/test_equiv_*.py` passes (noise, sensor, recovery, scenario, experiment,
      from_config, shims).
- [ ] The pre-existing `test/` suite still passes unchanged (run in `cdarr` env).
- [ ] The end-goal snippet in §0 runs and returns an `IPRResult`.
- [ ] A user-supplied custom object (e.g. a `Laplace` noise model, or a trivial custom
      recovery) can be injected into `IPRExperiment` without touching `cdarr` internals.
- [ ] `compare_crr` runs unchanged and reproduces the Step-0 `overall_ipr` for a fixed config.
- [ ] No detector monkey-patching remains in the loop (`grep -n "dcpa_prob_threshold\s*=" sim/ cdarr/`
      shows assignments only inside `ProbabilisticFTR`).

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **RNG draw-order drift** breaks bitwise equivalence | §1.3; prefer Option A (reuse existing `ADSL` objects inside `ADSLSensor`); equivalence test at every numeric step |
| Removing the detector monkey-patch changes prob-FTR results | `ProbabilisticFTR` sets the exact same `conf` attributes the legacy loop set, just from the recovery object; Step-4 equivalence test pins it |
| `Randomized`/`DistanceBased` per-pair draws reordered | Step-5 compares initial `_get_states()` arrays bitwise |
| Float reassociation from refactoring sums | allowed only with documented `atol=1e-9` fallback (§1.2) |
| numpy/navdata env mismatch | run everything in the `cdarr` conda env (README) |
| Scope creep into rewriting CD/CR | explicitly out of scope — detector/resolver are reused unchanged (§3) |

---

## 9. Out of scope (future, not now)

- Making `detect` return a `ConflictResult` instead of mutating the detector (would require
  rewriting `MVP`/`VO`; deferred — see prior design discussion).
- New metrics beyond IPR.
- Replacing BlueSky as the propagator.

---

## 10. PHASE 0 — Test suite modernization (build FIRST, before STEP 0)

**Ordering:** PHASE 0 is the FIRST implementation work on the branch, BEFORE the golden
master (STEP 0) and the refactor (STEPs 1–8). It rebuilds `test/` as a headless pytest
suite that exercises the **legacy** code, giving a working safety net for everything after.
It touches only `test/*`, `conftest.py`, `pytest.ini`, `requirements-dev.txt` — never
`sim/`, `sim_models/`, or `envs/`.

**Starting point:** the branch is cut from clean `main`, so `test/` is in its ORIGINAL
pre-refactor state — plain scripts with top-level side effects, stale imports, and
`MAX_TR` read-guards/hardcodes. PHASE 0 fixes all of that from scratch.

**Dependency:** `sim_models/cr_mvp.py` must be in its committed `main` state — the
`resumenav_double_criteria_dummy` method must be present, because the probabilistic test
calls it. (If a working-tree edit removed it, revert before PHASE 0.)

**Run everything in the `cdarr` conda env** (correct numpy for the bluesky navdata cache;
the three `run_multiple_jobs`/`get_ipr_stochastic_env` tests fail to import otherwise).

### 10.1 Legacy architecture the tests target

- **Entry points** (`sim/pairwise_stochastic/`):
  - `get_ipr_stochastic_env(asas_marh, confidence_interval, confidence_interval_velo,
    reception_prob, lookahead_time, dpsi, seed=44, config_path=None,
    threshold_probability=None, recovery_model=None)` → `(distance_array[T,nb_pair], ipr,
    sim_timer, n_active)`. Variants: `_randomized`, `_dist`.
  - `run_multiple_jobs(*, n_runs, n_jobs, base_seed=42, **env_kwargs)` → dict:
    `overall_ipr`, `ipr[]`, `worst_cpa[]`, `sim_timer[]`, `n_active_conflict`, `worst_cpa_min`.
- **Recovery = standalone functions**, registry at `get_ipr_stochastic_env.py:50-52`:
  `"CPA"`→`crr_resumenav_cpa.resumenav`; `"FTR"`→`crr_resumenav_ftr.resumenav_double_criteria`;
  `"Probabilistic FTR"`→`crr_resumenav_probabilistic_ftr.resumenav_probabilistic_ftr`.
  Call convention: `delpairs = recovery_fn(conf_resolution, conf_detection, ownship, intruder)`.
- **`threshold_probability` (gamma)**: set `conf_detection.dcpa_prob_threshold = <gamma>`
  before the loop; `resumenav_probabilistic_ftr` reads `getattr(conf, "dcpa_prob_threshold", 0.9)`.
- **Resolution**: `cfg.resolution_model` (`"MVP"`/`"VO"`); `sim_config.json` = `"MVP"`. MVP has
  `resolve`, `resumenav_double_criteria_dummy`, but **no `resumenav`** (VO-only) — the trace bug.
- **`MAX_TR`/`MAX_DTR2`** (M600 turn-rate limits 15/10): NOT built-in bluesky attrs.
  `sim/utils.py::get_configs()` (lines ~161-163) assigns them onto `bs.traf` from
  `sim_config.json`. Calling `get_configs()` is the ONLY correct way to set them.

### 10.2 Stale-symbol find-and-replace map

| Old (in original tests) | Current |
|---|---|
| `from sim_models.crr_resumenav_heuristic import resumenav` | `from sim_models.crr_resumenav_cpa import resumenav` |
| `crr_resumenav_ftr.resumenav_double_criteria` | unchanged |
| `crr_resumenav_ftr_pastcpa.resumenav_triple_criteria` | **DELETE entirely** (triple-criteria is gone) |
| `crr_resumenav_probabilistic.resumenav_probabilistic` | `crr_resumenav_probabilistic_ftr.resumenav_probabilistic_ftr` |
| `conf_resolution.resumenav(...)` (MVP method) | standalone recovery fn (call convention §10.1) |
| `bs.traf.MAX_TR` read-and-raise guard | call `get_configs()`; delete the guard |
| `bs.traf.MAX_TR = 15` / `MAX_DTR2 = 10` hardcode | call `get_configs()`; delete the hardcode |

### 10.3 Locked decisions

1. Triple-criteria / past-CPA recovery is permanently gone — strip it; deterministic test → CPA + FTR only.
2. Adopt pytest — plain scripts → `test_*()` functions with `assert`s.
3. Headless by default, `--plot` to show figures — no `plt.show()`, window, or file/CSV write
   on a normal run. `pytest test/` = headless; `pytest test/ --plot` = figures.
4. Keep tests small/fast — stochastic tests use `n_runs<=2, n_jobs<=2`, a few dpsi values.

### 10.4 Shared infrastructure (do FIRST within PHASE 0)

**`requirements-dev.txt`** (new): `pytest`.

**`pytest.ini`** (new, repo root):
```ini
[pytest]
testpaths = test
addopts = -q
```

**`conftest.py`** (new, repo root) — `--plot` option, `plot` fixture, session BlueSky-init
fixture (replaces every `bs._joblib_inited` guard), Agg backend unless `--plot`:
```python
import pytest

def pytest_addoption(parser):
    parser.addoption("--plot", action="store_true", default=False,
                     help="render matplotlib figures during tests")

def pytest_configure(config):
    if not config.getoption("--plot", default=False):
        import matplotlib
        matplotlib.use("Agg")

@pytest.fixture
def plot(request):
    return request.config.getoption("--plot")

@pytest.fixture(scope="session", autouse=True)
def bluesky_init():
    import bluesky as bs
    if not getattr(bs, "_joblib_inited", False):
        bs.init(mode="sim", detached=True)
        bs._joblib_inited = True
    return bs
```

### 10.5 Per-file target state

Every test file: top-level body → `def test_*()`; delete `bs.init` guard (use `bluesky_init`
fixture); gate every `plt.show()` and file/CSV write behind `if plot:`; turn-rate limits via
`get_configs()` (never read-guard, never hardcode); ≥1 `assert`.

**BROKEN — fix stale imports / moved API:**
- `test_stochastic_sim_single_job_deterministic.py` — delete `crr_resumenav_heuristic` +
  `crr_resumenav_ftr_pastcpa` imports and all `resumenav_triple_criteria` use; noise-path
  recovery → `resumenav_double_criteria` (FTR). Assert `0<=ipr<=1` and `worst_cpa < 2*cfg.horizontal_sep`.
- `test_stochastic_sim_single_job_probabilistic.py` — import `resumenav_probabilistic_ftr`;
  set `conf_detection.dcpa_prob_threshold = GAMMA` (`GAMMA=0.9`) before the loop; call it in
  place of `resumenav_probabilistic`. `conf_resolution.resumenav_double_criteria_dummy(...)` is
  OK on MVP — keep. Assert `0<=ipr<=1`.
- `test_stochastic_sim_single_job_trace_resumenav.py` — `conf_resolution.resumenav(...)` fails on
  MVP; use `crr_resumenav_cpa.resumenav` for both observed and ground-truth calls. Keep inlined
  (records per-timestep stats the env doesn't return). CSV write → `if plot:`. Assert stats dict
  non-empty, per-dpsi counts finite.

**CLEAN-UP — guards/plots/asserts only (API already current):**
- `test_stochastic_sim_single_simple.py` — wrap; assert `0<=ipr<=1`, `worst_cpa < 100`.
- `test_stochastic_sim_multi_job.py` — wrap; keep uniqueness + `worst_cpa <= 2*horizontal_sep`; `n_runs=2,n_jobs=2`.
- `test_stochastic_sim_multi_job_dist.py` — wrap; assert `0 <= overall_ipr <= 1`; `n_runs=2,n_jobs=2`.
- `test_stochastic_sim_single_job.py` — keep inlined; MAX_TR→`get_configs()`; gate plots; assert `worst_cpa < 2*cfg.horizontal_sep`.
- `test_stochastic_sim_forloop_sanitycheck.py` — keep inlined sweep; remove MAX_TR guard AND
  `bs.traf.MAX_TR = 15` hardcode → `get_configs()`; gate plots; assert finite IPR per case.
- `test_cr_mvp.py` — remove MAX_TR guard + hardcode → `get_configs()`; wrap; gate plots; assert
  head-on (dpsi=180) `worst_cpa < 2*horizontal_sep`.
- `test_cd_statebased.py` — wrap; assert head-on geometry flagged as a conflict.
- `test_adsl_module.py`, `test_noise_model.py`, `test_reception_model.py` — already have good
  asserts; just ensure each lives in a `test_*()` function pytest collects. No MAX_TR here.

**OTHER:**
- `sim_models/test_sim_models.py` — empty; **delete**.
- `test/verify_fixes.py` — standalone `__main__` harness, not pytest-collected (filename isn't
  `test_*`). Run `python -m test.verify_fixes`, record pass/fail, leave standalone.

### 10.6 Execution order + acceptance

Order: (1) shared infra §10.4 + `pip install pytest`; (2) delete empty `test_sim_models.py`;
(3) fix the 3 BROKEN files; (4) `python -m py_compile test/*.py` clean; (5) clean-up files one
at a time, `pytest test/<file>.py` green before moving on; (6) `python -m test.verify_fixes`;
(7) full-suite gate.

Acceptance:
- [ ] `python -m py_compile test/*.py` exits 0.
- [ ] `pytest test/` (in `cdarr` env) exits 0; every file ≥1 passing test; no GUI window opens;
      nothing written under `results/`.
- [ ] `pytest test/ --plot` renders figures.
- [ ] These greps return nothing under `test/`:
  `crr_resumenav_heuristic\|crr_resumenav_ftr_pastcpa\|resumenav_triple_criteria`;
  `resumenav_probabilistic\b` (the `_ftr` form is fine);
  `bs.traf.MAX_TR\s*=`; `Required BlueSky.*MAX_TR\|raise.*MAX_TR`.

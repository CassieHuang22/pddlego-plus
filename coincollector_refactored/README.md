# coincollector

Refactored CoinCollector (TextWorldExpress) experiment codebase for testing three methods:

1. **Baseline**: LLM acts as a *direct planner* (outputs TextWorld actions)
2. **Translator (PDDLego+)**: LLM acts as a *translator* (builds DF/PF), then a classical planner produces actions
3. **Translator + initDF**: same as (2), but the LLM is additionally given an initial/seed domain file (initDF) to improve PDDL quality 

---

## Repository layout

```
coincollector/
├── test_all_modes.py
├── best_seed.log
└── src/
    ├── config.py
    ├── pddl_engine.py
    ├── initial_df_builder.py
    ├── prompts.py
    ├── solver.py
    ├── utils.py
    └── runner.py
```

* **Root**

  * `test_all_modes.py`: main experiment driver (build initDF + run experiments)
  * `best_seed.log`: saved output from seed scan (best seed selection)

* **src/**

  * `config.py`: CoinCollector environment parameters, step limits, and model prefixes
  * `pddl_engine.py`: baseline/translator/initDF runners + LLM interface
  * `initial_df_builder.py`: seed scan, BFS unit-test collection, DF training, and artifact saving
  * `prompts.py`: prompt templates for baseline, PDDL translation, and unit-test training
  * `solver.py`: planutils wrapper (dual-bfws-ffparser)
  * `utils.py`: action parsing, feedback mapping, sanitizers, etc.
  * `runner.py`: small wrapper around core runner functions (optional)

---

## Quickstart: run everything

The main driver is **`test_all_modes.py`**.

```bash
python test_all_modes.py
```

By default, it can:

* build an initDF (offline) using the action-diversity pipeline
* optionally run **initDF mode** with the generated DF
* optionally run **translator** (no initDF)
* optionally run **baseline** (direct planner)

---

## Running the three methods

All toggles are at the top of `test_all_modes.py`:

* `MODEL_NAME` : which LLM to use
* `START_TRIAL`, `END_TRIAL` : which random seeds to evaluate
* `DO_BUILD_ACTION` : build initDF using action diversity
* `RUN_AFTER_BUILD` : run initDF experiments right after DF is built
* `DO_RUN_TRANSLATOR` : run translator without initDF
* `DO_RUN_DIRECT_PLANNER` : run baseline

Example pattern:

* **initDF experiments**: enable `DO_BUILD_ACTION=True`, `RUN_AFTER_BUILD=True`
* **translator-only**: set `DO_BUILD_ACTION=False`, `DO_RUN_TRANSLATOR=True`
* **baseline-only**: set `DO_BUILD_ACTION=False`, `DO_RUN_DIRECT_PLANNER=True`

---

## initDF (initial domain file) — CoinCollector-specific workflow

CoinCollector is small enough to do deterministic BFS exploration within a seed environment.
So instead of choosing “trials” (ALFWorld), we choose a single best seed (or a few seeds) that maximize the diversity of reachable actions, and then collect unit-test samples by BFS.

### Step 1) Find the best seed (action diversity)

`src/initial_df_builder.py` can scan a seed range (e.g., 0–499) and choose the best seed based on action coverage.

A saved run output is provided as:

* `best_seed.log`

It reports the final choice like:

* `best seed = 217` with full action coverage and full env coverage (as recorded).

You can treat this as your “best seed” for initDF building.

### Step 2) Collect unit-test samples via BFS

The initDF builder focuses on two action schemas:

* `move <dir>`
* `open door to <dir>`

across the four directions: `north/south/east/west`.

It BFS-explores from the reset state and records **UnitTestSample** items:

* `seed`
* `prefix_actions` (how we got to a state)
* `observation`
* `action` (the expected next action)
* `schema` + `direction`

The goal is to cover each (schema, direction) combination up to a target count:

* `per_schema_target` (e.g., 1 or 2)
* `require_all_directions` (usually True for strict coverage)

### Step 3) Train an initial domain file using unit tests

Once samples are collected, the builder trains a DF iteratively:

For each sample:

1. replay the `prefix_actions` to reconstruct the state
2. ask the LLM to generate/update DF/PF with a unit-test constraint (“planner’s first action should equal expected_action”)
3. run the classical planner on DF/PF
4. check whether the planner’s first action matches the expected action

This repeats:

* per-sample tries (`per_sample_tries`)
* across global rounds (`max_global_rounds`)

until all unit tests pass or max rounds are reached.

### Step 4) Save initDF artifacts

The builder saves artifacts under:

```
output/<FOLDER_NAME>/action/
    <DATE>_<MODEL>_seed<SEED>_action_initial_df.pddl
    <DATE>_<MODEL>_seed<SEED>_action_unit_samples.json
    <DATE>_<MODEL>_seed<SEED>_action_report.json
    <DATE>_<MODEL>_seed<SEED>_action_meta.json
```

`test_all_modes.py` can automatically call this saver.

---

## Outputs

### Per-trial logs

Written under:

* `output/<FOLDER_NAME>/`

Typical filenames include model + mode + seed/trial index.

### Summary CSV

A run appends results to:

* `output/<RESULT_NAME>.csv`

### DF artifacts (initDF build)

Saved under:

* `output/<FOLDER_NAME>/action/` (see above)
# alfworld_refactored

Refactored ALFWorld (TextWorld) experiment codebase for testing three methods:

1. **Baseline**: LLM acts as a *direct planner* (outputs TextWorld actions)
2. **Translator (PDDLego+)**: LLM acts as a *translator* (builds DF/PF), then a classical planner produces actions
3. **Translator + initDF**: same as (2), but the LLM is additionally given an initial/seed domain file (initDF) to improve PDDL quality 

You can run these in a typed way (e.g., `basic`, `clean`, `heat`, `cool`, `use`, or `all`) via `test_all_typed_modes.py`. 

---

## Repository layout

```
alfworld_refactored/
├── test_all_typed_modes.py
├── build_seed_df_alfworld.py
├── analyze_problems.py
├── analyze_problem_distribution.py
├── problem_index.json
├── problem_distribution_analysis.json
└── src/
    ├── __init__.py
    ├── config.py
    ├── pddl_engine.py
    ├── problem_selector.py
    ├── prompts.py
    ├── runner.py
    ├── solver.py
    └── utils.py
```

* **Root**

  * `test_all_typed_modes.py`: main experiment driver (typed runs + comparison across the 3 methods). 
  * `build_seed_df_alfworld.py`: offline script to build a seed DF (a learned initial domain file) for initDF mode (see “Seed DF (initDF) workflow”).
  * `analyze_problems.py`: scans ALFWorld problems and produces a problem index used for typed selection. 
  * `analyze_problem_distribution.py`: summarizes per-split task-type counts and writes a JSON report (included as `problem_distribution_analysis.json` in this repo).
  * `problem_index.json`: precomputed index of tasks by split/type, plus a mapping to simplified categories. 
  * `problem_distribution_analysis.json`: aggregate counts by split and type/category (useful to pick evaluation split sizes).

* **src/**

  * `pddl_engine.py`: core implementation of the 3 methods + typed runners + environment setup. 
  * `problem_selector.py`: typed problem selection by loading `problem_index.json` (preferred) or scanning the dataset folders. 
  * `prompts.py`: prompt templates for (baseline / PDDL translation / initDF).
  * `solver.py`: wrapper around a local PDDL planner via planutils (`dual-bfws-ffparser` + optional `val`).
  * `utils.py`: JSON repair helpers, ALFWorld observation parsing, solver→TextWorld action mapping, loop/duplicate detection, etc.
  * `config.py`: shared constants such as step limits and allowed OpenAI model prefixes. 

---

## Quickstart: run experiments (typed)

All commands should be run from the repo root (because `test_all_typed_modes.py` adds the local path and imports `src.*`). 

**Baseline (LLM as direct planner):**

```bash
python3 test_all_typed_modes.py \
  --mode baseline \
  --type basic \
  --split valid_unseen \
  --count 10 \
  --model o3-mini \
  --output-folder 0113_o3-mini_baseline \
  --result-file 0113_o3-mini_baseline
```

**Translator (PDDLego+):**

```bash
python3 test_all_typed_modes.py \
  --mode translator \
  --type clean \
  --split valid_unseen \
  --count 10 \
  --model o3-mini \
  --output-folder 0113_o3-mini_translator \
  --result-file 0113_o3-mini_translator
```

**Translator + initDF:**

```bash
python3 test_all_typed_modes.py \
  --mode initdf \
  --type heat \
  --split valid_unseen \
  --count 10 \
  --model o3-mini \
  --init-df output/seeddf_all/seed_domain.pddl \
  --output-folder 0113_o3-mini_initdf \
  --result-file 0113_o3-mini_initdf
```

**All:**

```bash
python3 test_all_typed_modes.py \
  --mode all \
  --type basic \
  --split valid_unseen \
  --count 10 \
  --model o3-mini \
  --output-folder compare_basic_o3mini \
  --result-file compare_basic_o3mini
```

This triggers baseline → translator → initDF sequentially. 

---

## Outputs

All runs write:

* **Per-trial logs**: `output/<folder_name>/*.txt`
* **A single CSV summary**: `output/<result_name>.csv`
* **Errors**: `output/<folder_name>/errors.txt`

The unified runners in `src/pddl_engine.py` append a row to the CSV containing: date, model, mode, game_type, goal_type, trial index, success flag, and a loop trace summary. 

---

## Typed problem selection

`src/problem_selector.py` is responsible for mapping detailed ALFWorld directory prefixes (e.g., `pick_clean_then_place_in_recep-*`) into simplified categories (`clean`, `heat`, `cool`, `basic`, `use`). 

### `problem_index.json`

This file is a precomputed index used for fast, accurate typed selection. It contains:

* `by_split[split][type] -> [problem_names...]`
* `by_type[type] -> [{split, name, path}, ...]`
* `type_mapping` (detailed → category)
* `paths` (split/name → initial_state.pddl path)

Without the index file, `ProblemSelector` falls back to scanning the dataset, which may not respect `split` filtering in all environments. 

---

### `analyze_problems.py` → `problem_index.json`

Scans all splits, checks for overlaps, and exports a comprehensive JSON index for typed selection. 

### `analyze_problem_distribution.py` → `problem_distribution_analysis.json`

Runs `ProblemSelector` on each split and summarizes:

* category counts (`basic/clean/heat/cool/use`)
* detailed type counts
* totals per split

---

## Seed DF (initDF) workflow

### What initDF expects

In initDF mode, the main runner feeds an initial domain file into the LLM prompt during the first PDDL generation attempt, asking it to minimally extend/fix it while generating DF/PF for the current partial observation. For subsequent iterations within the same trial, the system uses the previously generated (evolved) domain file, not the original seed.

### How to build a seed DF

* The offline seed-DF construction loop (unit tests → iterative DF refinement) is provided in the script: `build_seed_df_alfworld.py`.

A typical workflow:

Goal: collect a deterministic set of (environment, goal, action/observation traces) samples and feed them to an LLM to produce a shared initial domain file used across held-out trials.

1. Select candidate trials from valid_unseen/ (small but diverse split).
2. Run a short deterministic probe rollout per trial to collect action schemas and traces.
3. Choose a set of trials maximizing action-type coverage (e.g., go/open/take/move/use/heat/cool/clean/slice).
4. Feed collected samples to an initDF-building LLM.
5. Keep the best initDF.
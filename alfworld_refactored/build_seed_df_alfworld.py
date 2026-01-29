from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.pddl_engine import prepare_problem_flexible, reset_env_with_prefix, run_llm
from src.utils import get_action_from_pddl, sanitize_valid_actions, summarize_obs
import src.problem_selector as problem_selector_mod
from src.problem_selector import ProblemSelector


class PDDLResponse(BaseModel):
    df: str
    pf: str

SYS_PROMPT_PDDL_UNIT = (
    'You are a PDDL expert. Output must be ONLY valid JSON with keys "df" and "pf". '
    'No explanation, no markdown.'
)

DEFAULT_SEED_DF = """(define (domain exploration)
  (:requirements :strips :typing :negative-preconditions)
  (:types
    receptacle object - thing
    microwaveReceptacle sinkbasinReceptacle fridgeReceptacle - receptacle
    sharpObject - object
  )
  (:predicates
    (at ?r - receptacle)
    (opened ?r - receptacle)
    (in ?o - object ?r - receptacle)
    (holding ?o - object)
    (handempty)
    (heated ?o - object)
    (cleaned ?o - object)
    (cooled ?o - object)
    (sliced ?o - object)
    (on ?o - object)
  )
  (:action GotoLocation
    :parameters (?from - receptacle ?to - receptacle)
    :precondition (at ?from)
    :effect (and (not (at ?from)) (at ?to))
  )
  (:action OpenObject
    :parameters (?r - receptacle)
    :precondition (and (at ?r) (not (opened ?r)))
    :effect (opened ?r)
  )
  (:action CloseObject
    :parameters (?r - receptacle)
    :precondition (and (at ?r) (opened ?r))
    :effect (not (opened ?r))
  )
  (:action PickupObject
    :parameters (?o - object ?r - receptacle)
    :precondition (and (at ?r) (in ?o ?r) (handempty))
    :effect (and (holding ?o) (not (in ?o ?r)) (not (handempty)))
  )
  (:action PutObject
    :parameters (?o - object ?r - receptacle)
    :precondition (and (at ?r) (holding ?o))
    :effect (and (in ?o ?r) (handempty) (not (holding ?o)))
  )
  (:action useObject
    :parameters (?o - object)
    :precondition (holding ?o)
    :effect (on ?o)
  )
  (:action HeatObject
    :parameters (?o - object ?r - microwaveReceptacle)
    :precondition (and (at ?r) (holding ?o))
    :effect (heated ?o)
  )
  (:action CleanObject
    :parameters (?o - object ?r - sinkbasinReceptacle)
    :precondition (and (at ?r) (holding ?o))
    :effect (cleaned ?o)
  )
  (:action CoolObject
    :parameters (?o - object ?r - fridgeReceptacle)
    :precondition (and (at ?r) (holding ?o))
    :effect (cooled ?o)
  )
  (:action SliceObject
    :parameters (?r - receptacle ?co - object ?sharp_o - sharpObject)
    :precondition (and (at ?r) (in ?co ?r) (holding ?sharp_o))
    :effect (sliced ?co)
  )
)
"""


@dataclass
class UnitTestSample:
    # Problem identity
    problem_path: str
    problem_name: str
    problem_type: str
    game_type: str

    prefix_actions: List[str]

    brief_obs: str
    overall_memory: str

    valid_actions: List[str]
    expected_action: str

    schema: str



_SCHEMA_ORDER = [
    "goto",
    "open",
    "pickup",
    "put",
    "use",
    "heat",
    "clean",
    "cool",
    "slice",
]


def norm_ws(s: str) -> str:
    return " ".join((s or "").strip().split())


def classify_action(action: str) -> Optional[str]:
    """Classify a TextWorld ALFWorld action string into a PDDLego+ schema."""
    a = norm_ws(action).lower()
    if a.startswith("go to "):
        return "goto"
    if a.startswith("open "):
        return "open"
    if a.startswith("take ") and " from " in a:
        return "pickup"
    if a.startswith("move ") and " to " in a:
        return "put"
    if a.startswith("use "):
        return "use"
    if a.startswith("heat ") and " with " in a:
        return "heat"
    if a.startswith("clean ") and " with " in a:
        return "clean"
    if a.startswith("cool ") and " with " in a:
        return "cool"
    if a.startswith("slice ") and " with " in a:
        return "slice"
    return None


def tw_name_to_pddl(name: str) -> str:
    """Convert a demangled TW name like 'armchair 1' -> 'armchair1'."""
    return re.sub(r"\s+", "", norm_ws(name).lower())


def goal_hint_from_expected_action(expected_action: str) -> Tuple[str, Dict[str, str]]:
    a = norm_ws(expected_action)
    al = a.lower()

    # goto
    if al.startswith("go to "):
        recep_txt = a[len("go to ") :]
        r = tw_name_to_pddl(recep_txt)
        return f"(at {r})", {"recep": r}

    # open
    m = re.match(r"^open (.+)$", al)
    if m:
        r = tw_name_to_pddl(m.group(1))
        return f"(opened {r})", {"recep": r}

    # pickup
    m = re.match(r"^take (.+) from (.+)$", al)
    if m:
        o = tw_name_to_pddl(m.group(1))
        r = tw_name_to_pddl(m.group(2))
        return f"(holding {o})", {"obj": o, "recep": r}

    # put
    m = re.match(r"^move (.+) to (.+)$", al)
    if m:
        o = tw_name_to_pddl(m.group(1))
        r = tw_name_to_pddl(m.group(2))
        return f"(in {o} {r})", {"obj": o, "recep": r}

    # use
    m = re.match(r"^use (.+)$", al)
    if m:
        o = tw_name_to_pddl(m.group(1))
        return f"(on {o})", {"obj": o}

    # heat / clean / cool
    m = re.match(r"^(heat|clean|cool) (.+) with (.+)$", al)
    if m:
        verb = m.group(1)
        o = tw_name_to_pddl(m.group(2))
        r = tw_name_to_pddl(m.group(3))
        pred = {"heat": "heated", "clean": "cleaned", "cool": "cooled"}[verb]
        return f"({pred} {o})", {"obj": o, "recep": r}

    # slice
    m = re.match(r"^slice (.+) with (.+)$", al)
    if m:
        co = tw_name_to_pddl(m.group(1))
        sharp = tw_name_to_pddl(m.group(2))
        return f"(sliced {co})", {"obj": co, "tool": sharp}

    # Fallback
    return "(handempty)", {}


def extract_expert_plan(infos: dict) -> Optional[List[str]]:
    if not infos:
        return None

    for key in ("expert_plan", "extra.expert_plan", "extra.expertPlan"):
        plan = infos.get(key)
        if plan:
            return _coerce_plan(plan)

    extras = infos.get("extras") or infos.get("extra")
    if isinstance(extras, dict):
        for key in ("expert_plan", "expertPlan"):
            if key in extras:
                return _coerce_plan(extras[key])

    return None


def _coerce_plan(plan_obj) -> Optional[List[str]]:
    if plan_obj is None:
        return None
    if isinstance(plan_obj, list):
        return [norm_ws(x) for x in plan_obj if norm_ws(x)]
    if isinstance(plan_obj, str):
        s = plan_obj.strip()
        try:
            j = json.loads(s)
            if isinstance(j, list):
                return [norm_ws(x) for x in j if norm_ws(x)]
        except Exception:
            pass
        lines = [norm_ws(x) for x in s.splitlines() if norm_ws(x)]
        return lines if lines else None
    return None


def verify_action_succeeds(
    gamefile: str,
    expert,
    prefix_actions: List[str],
    candidate_action: str,
) -> bool:

    env = None
    try:
        env, obs, infos = reset_env_with_prefix(gamefile, prefix_actions, expert)
        obs2, _, _, _ = env.step(candidate_action)
        return "Nothing happens." not in (obs2 or "")
    except Exception:
        return False
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# Collection
# -----------------------------------------------------------------------------


def collect_unit_tests(
    model_name: str,
    split: str,
    problem_type: str,
    num_problems: int,
    per_schema: int,
    max_steps_per_problem: int,
    verify: bool,
    problem_index_path: Optional[str],
    rng_seed: int,
    log_fh,
) -> List[UnitTestSample]:
    """Collect unit-test samples by walking expert trajectories and sampling admissible actions."""

    random.seed(rng_seed)

    if problem_index_path:
        problem_selector_mod.PROBLEM_INDEX_PATH = problem_index_path

    selector = ProblemSelector(split=split)
    problems = selector.get_problems_by_type(problem_type, max_count=num_problems, offset=0)

    need: Dict[str, int] = {k: per_schema for k in _SCHEMA_ORDER}
    samples: List[UnitTestSample] = []
    seen_actions: set[str] = set()

    def all_done() -> bool:
        return all(v <= 0 for v in need.values())

    log_fh.write(f"[collect] split={split} problem_type={problem_type} num_problems={len(problems)}\n")
    log_fh.write(f"[collect] targets per schema: {need}\n")

    for pi, (problem_path, ptype, gtype) in enumerate(problems):
        if all_done():
            break

        env = None
        try:
            env, expert, meta, init_scene, goal, valid_actions, infos = prepare_problem_flexible(
                problem_path=problem_path
            )

            problem_name = meta.get("problem_name", os.path.basename(os.path.dirname(problem_path)))
            gamefile = meta["gamefile"]

            expert_plan = extract_expert_plan(infos)

            brief_obs = "Action: look around\n" + summarize_obs(init_scene) + "\n"
            overall_memory = brief_obs
            prefix_actions: List[str] = []

            log_fh.write(f"\n[collect] Problem {pi+1}/{len(problems)}: {problem_name} ({ptype} -> {gtype})\n")
            log_fh.write(f"[collect] Path: {problem_path}\n")
            log_fh.write(f"[collect] Expert plan length: {len(expert_plan) if expert_plan else 'None'}\n")

            steps = 0
            done = False
            while steps < max_steps_per_problem and not done:
                current_valid = sanitize_valid_actions(infos)

                candidates = list(current_valid)
                random.shuffle(candidates)

                for cand in candidates:
                    schema = classify_action(cand)
                    if not schema:
                        continue
                    if need.get(schema, 0) <= 0:
                        continue
                    cand_norm = norm_ws(cand).lower()
                    if cand_norm in seen_actions:
                        continue

                    if verify:
                        ok = verify_action_succeeds(gamefile, expert, prefix_actions, cand)
                        if not ok:
                            continue

                    sample = UnitTestSample(
                        problem_path=problem_path,
                        problem_name=problem_name,
                        problem_type=ptype,
                        game_type=gtype,
                        prefix_actions=list(prefix_actions),
                        brief_obs=brief_obs,
                        overall_memory=overall_memory,
                        valid_actions=current_valid,
                        expected_action=norm_ws(cand),
                        schema=schema,
                    )
                    samples.append(sample)
                    seen_actions.add(cand_norm)
                    need[schema] -= 1

                    log_fh.write(
                        f"[collect] + sample schema={schema:5s} remaining={need[schema]:3d} action='{cand}'\n"
                    )

                    if all_done():
                        break

                if all_done():
                    break

                next_action: Optional[str] = None
                if expert_plan and steps < len(expert_plan):
                    next_action = expert_plan[steps]
                else:
                    non_meta = [a for a in current_valid if classify_action(a) in ("goto", "open", "pickup", "put")]
                    if non_meta:
                        next_action = random.choice(non_meta)

                if not next_action:
                    break

                obs, _, done_flag, infos = env.step(next_action)
                prefix_actions.append(next_action)

                if infos.get("won"):
                    done = True

                brief_obs = f"Action: {next_action}\n" + summarize_obs(obs) + "\n"
                overall_memory += brief_obs

                steps += 1

        except Exception as e:
            log_fh.write(f"[collect] !! error on problem {pi}: {e}\n")
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

    log_fh.write("\n[collect] Coverage remaining:\n")
    for k in _SCHEMA_ORDER:
        log_fh.write(f"  - {k:5s}: {need[k]}\n")

    return samples


# -----------------------------------------------------------------------------
# Build seed DF from unit tests
# -----------------------------------------------------------------------------


def format_unit_prompt(
    *,
    current_df: str,
    sample: UnitTestSample,
    attempt_id: int,
    prev_feedback: str = "",
    include_memory: bool = True,
) -> str:
    goal_hint, entities = goal_hint_from_expected_action(sample.expected_action)

    valid_actions = sample.valid_actions
    max_va = 120
    va_txt = json.dumps(valid_actions[:max_va], ensure_ascii=False)
    if len(valid_actions) > max_va:
        va_txt = va_txt[:-1] + f", ... (truncated {len(valid_actions) - max_va})]"

    memory_block = sample.overall_memory if include_memory else sample.brief_obs

    prompt = f"""Please provide the output in strict JSON format with keys df and pf only.

You are building a *general* PDDL domain file (DF) for ALFWorld-like text environments.

Hard constraints:
- DF must NOT contain any problem-specific objects (only types/predicates/actions).
- Keep the DF changes minimal relative to the provided current DF.
- PF must contain ONLY what is needed for this unit test.
- Use :typing. Include :negative-preconditions if you use (not ...).
- Do NOT invent objects/relations not supported by the observation/memory.

Unit-test requirement:
The external planner will solve PF using DF, and we will execute ONLY the FIRST planned step.
That FIRST step, after mapping back to TextWorld action strings, MUST equal:
EXPECTED_ACTION: "{sample.expected_action}"

Goal hint (you may use this exact goal literal):
SUGGESTED_GOAL_LITERAL: {goal_hint}

Action mapping used by the checker (for your reference):
- (GOTOLOCATION ?from ?to)  ->  go to <to>
- (OPENOBJECT ?r)           ->  open <r>
- (PICKUPOBJECT ?o ?r)      ->  take <o> from <r>
- (PUTOBJECT ?o ?r)         ->  move <o> to <r>
- (USEOBJECT ?o)            ->  use <o>
- (HEATOBJECT ?o ?mw)       ->  heat <o> with <mw>
- (CLEANOBJECT ?o ?sink)    ->  clean <o> with <sink>
- (COOLOBJECT ?o ?fridge)   ->  cool <o> with <fridge>
- (SLICEOBJECT ?r ?co ?k)   ->  slice <co> with <k>

Note on naming:
- In DF/PF, object names should be lowercase with spaces removed (e.g., "armchair 1" -> armchair1).

Current DF (seed to minimally edit):
|DF_START|
{current_df}
|DF_END|

Observation / memory (state before EXPECTED_ACTION):
|OBS_START|
{memory_block}
|OBS_END|

Admissible actions at this state (truncated if long):
{va_txt}

"""

    if prev_feedback:
        prompt += f"""\nPrevious attempt feedback (use this to fix DF/PF):
{prev_feedback}
\n"""

    prompt += """Now output JSON with df and pf."""

    return prompt


def build_seed_df_from_unit_tests(
    *,
    model_name: str,
    samples: List[UnitTestSample],
    initial_df_text: str,
    out_log_path: str,
    max_rounds: int,
    per_sample_tries: int,
    include_memory: bool,
    shuffle_each_round: bool,
    rng_seed: int,
) -> str:
    """Iteratively update DF until it passes the unit tests (best-effort)."""

    random.seed(rng_seed)

    df = initial_df_text

    with open(out_log_path, "a", encoding="utf-8") as log_fh:
        log_fh.write("\n\n====================\n")
        log_fh.write("[build] Starting DF build\n")
        log_fh.write(f"[build] #samples={len(samples)} max_rounds={max_rounds} per_sample_tries={per_sample_tries}\n")

        for round_id in range(max_rounds):
            log_fh.write(f"\n[build] ===== Round {round_id+1}/{max_rounds} =====\n")

            all_pass = True

            idxs = list(range(len(samples)))
            if shuffle_each_round:
                random.shuffle(idxs)

            for j, si in enumerate(idxs):
                s = samples[si]

                # Safety check: expected_action should be admissible in recorded state.
                exp_norm = norm_ws(s.expected_action).lower()
                admissible_set = {norm_ws(a).lower() for a in (s.valid_actions or [])}
                if exp_norm not in admissible_set:
                    log_fh.write(
                        f"[build] (skip) sample {si} expected_action not in valid_actions: {s.expected_action}\n"
                    )
                    continue

                passed = False
                feedback = ""

                for attempt in range(per_sample_tries):
                    prompt = format_unit_prompt(
                        current_df=df,
                        sample=s,
                        attempt_id=attempt,
                        prev_feedback=feedback,
                        include_memory=include_memory,
                    )

                    resp, raw = run_llm(
                        prompt,
                        model_name,
                        system_prompt=SYS_PROMPT_PDDL_UNIT,
                        response_model=PDDLResponse,
                    )

                    df_candidate = resp.get("df", "") or ""
                    pf_candidate = resp.get("pf", "") or ""

                    actions, solver_stderr, plan_text = get_action_from_pddl(df_candidate, pf_candidate)
                    got = actions[0] if actions else None

                    ok = norm_ws(got or "").lower() == exp_norm

                    log_fh.write(
                        f"\n[build] sample {j+1}/{len(idxs)} id={si} schema={s.schema} attempt={attempt}\n"
                    )
                    log_fh.write(f"[build] expected: {s.expected_action}\n")
                    log_fh.write(f"[build] got     : {got}\n")
                    log_fh.write(f"[build] ok={ok}\n")
                    log_fh.write("[build] --- PROMPT ---\n")
                    log_fh.write(prompt + "\n")
                    log_fh.write("[build] --- RAW LLM ---\n")
                    log_fh.write(raw + "\n")
                    log_fh.write("[build] --- SOLVER STDERR ---\n")
                    log_fh.write((solver_stderr or "") + "\n")
                    log_fh.write("[build] --- PLAN ---\n")
                    log_fh.write((plan_text or "") + "\n")

                    if ok:
                        df = df_candidate
                        passed = True
                        break

                    # Prepare feedback for next attempt
                    feedback = (
                        f"Expected first action: '{s.expected_action}'\n"
                        f"But planner produced: '{got}'\n\n"
                        f"Solver stderr:\n{solver_stderr}\n\n"
                        f"Plan text:\n{plan_text}\n"
                    )

                if not passed:
                    all_pass = False
                    log_fh.write(f"[build] !! sample {si} FAILED after {per_sample_tries} attempts\n")

            if all_pass:
                log_fh.write(f"\n[build] All samples passed in round {round_id+1}. Stopping.\n")
                break

        log_fh.write("\n[build] Done.\n")

    return df


def save_samples_jsonl(samples: List[UnitTestSample], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")


def load_samples_jsonl(path: str) -> List[UnitTestSample]:
    out: List[UnitTestSample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(UnitTestSample(**d))
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Build ALFWorld seed DF via action unit tests")

    p.add_argument("--model", default="o3-mini", help="Model name (deepseek, o3-mini, Qwen/Qwen3-32B, etc)")

    p.add_argument(
        "--split",
        default="valid_train",
        choices=["train", "valid_train", "valid_seen", "valid_unseen"],
        help="ALFWorld split",
    )

    p.add_argument(
        "--problem-type",
        default="all",
        choices=["basic", "clean", "heat", "cool", "use", "all"],
        help="Typed category to sample problems from",
    )

    p.add_argument("--num-problems", type=int, default=10, help="How many problems to sample")
    p.add_argument(
        "--per-schema",
        type=int,
        default=10,
        help="Target number of unit tests per action schema",
    )

    p.add_argument(
        "--max-steps-per-problem",
        type=int,
        default=25,
        help="How many env steps to advance per problem when collecting",
    )

    p.add_argument(
        "--no-verify",
        action="store_true",
        help="Do NOT verify collected actions (faster, but less safe)",
    )

    p.add_argument(
        "--problem-index",
        default=None,
        help="Optional path to problem_index.json (overrides src.problem_selector.PROBLEM_INDEX_PATH)",
    )

    p.add_argument(
        "--initial-df",
        default=None,
        help="Optional path to an initial DF template. If omitted, uses a minimal built-in DF.",
    )

    p.add_argument("--max-rounds", type=int, default=3, help="Max global rounds over all samples")
    p.add_argument("--per-sample-tries", type=int, default=3, help="LLM attempts per sample")

    p.add_argument(
        "--include-memory",
        action="store_true",
        help="Include full overall_memory in unit-test prompt (default off: only brief_obs)",
    )

    p.add_argument(
        "--shuffle-each-round",
        action="store_true",
        help="Shuffle sample order each round",
    )

    p.add_argument("--seed", type=int, default=0, help="RNG seed")

    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: output/seeddf_<type>_<timestamp>)",
    )

    p.add_argument(
        "--collect-only",
        action="store_true",
        help="Only collect unit tests; do not build DF",
    )

    p.add_argument(
        "--build-only",
        action="store_true",
        help="Only build DF; requires --samples-jsonl",
    )

    p.add_argument(
        "--samples-jsonl",
        default=None,
        help="Path to samples JSONL (for --build-only or to reuse collection)",
    )

    args = p.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or f"output/seeddf_{args.problem_type}_{ts}"
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "build_log.txt")
    samples_path = args.samples_jsonl or os.path.join(out_dir, "unit_tests.jsonl")
    df_out_path = os.path.join(out_dir, "seed_domain.pddl")

    # Load or init DF template
    if args.initial_df:
        with open(args.initial_df, "r", encoding="utf-8") as f:
            initial_df_text = f.read()
    else:
        initial_df_text = DEFAULT_SEED_DF

    # Collect samples
    if args.build_only:
        if not args.samples_jsonl:
            raise SystemExit("--build-only requires --samples-jsonl")
        samples = load_samples_jsonl(samples_path)
    else:
        with open(log_path, "w", encoding="utf-8") as log_fh:
            log_fh.write(f"Seed DF Builder started: {datetime.now().isoformat()}\n")
            log_fh.write(f"Args: {vars(args)}\n")

            samples = collect_unit_tests(
                model_name=args.model,
                split=args.split,
                problem_type=args.problem_type,
                num_problems=args.num_problems,
                per_schema=args.per_schema,
                max_steps_per_problem=args.max_steps_per_problem,
                verify=not args.no_verify,
                problem_index_path=args.problem_index,
                rng_seed=args.seed,
                log_fh=log_fh,
            )

        save_samples_jsonl(samples, samples_path)

    print(f"Collected {len(samples)} unit tests -> {samples_path}")

    if args.collect_only:
        print("collect-only: skipping DF build")
        return

    # Build DF
    seed_df = build_seed_df_from_unit_tests(
        model_name=args.model,
        samples=samples,
        initial_df_text=initial_df_text,
        out_log_path=log_path,
        max_rounds=args.max_rounds,
        per_sample_tries=args.per_sample_tries,
        include_memory=args.include_memory,
        shuffle_each_round=args.shuffle_each_round,
        rng_seed=args.seed,
    )

    with open(df_out_path, "w", encoding="utf-8") as f:
        f.write(seed_df)

    print(f"Seed DF saved -> {df_out_path}")
    print(f"Build log -> {log_path}")


if __name__ == "__main__":
    main()

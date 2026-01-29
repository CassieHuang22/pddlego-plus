import os
from datetime import date
import json
import re

from dotenv import load_dotenv
load_dotenv()

from textworld_express import TextWorldExpressEnv
from .config import ENV_PARAMS
from .utils import summarize_obs, sanitize_valid_actions, get_action_from_pddl, map_env_feedback_to_large_loop_error
from .pddl_engine import llm_to_pddl


from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Literal
import copy
from collections import deque


VALID_DIRS = ("north", "south", "east", "west")

VALID_ACTIONS_CANONICAL = [
    "move east", "move north", "move south", "move west",
    "open door to east", "open door to north", "open door to south", "open door to west"
]

_SHARED_ENV = None

def _get_shared_env():
    global _SHARED_ENV
    if _SHARED_ENV is None:
        _SHARED_ENV = TextWorldExpressEnv(envStepLimit=100)
        _SHARED_ENV.load(**ENV_PARAMS)
    return _SHARED_ENV

def _classify_action(action: str) -> Tuple[str, Optional[str]]:
    a = action.strip().lower()
    if a.startswith("open door to "):
        return "open-door", a.replace("open door to ", "").strip()
    if a.startswith("move "):
        return "move", a.replace("move ", "").strip()
    return "unknown", None

def _replay_to_state(seed: int, prefix_actions: List[str]) -> Tuple[str, dict, TextWorldExpressEnv, str]:
    env = _get_shared_env()
    obs, infos = env.reset(seed=seed, gameFold="train", generateGoldPath=True)

    overall_memory = "Action: look around\n" + summarize_obs(obs) + '\n'
    for act in prefix_actions:
        obs, reward, done, infos = env.step(act)
        overall_memory += f"Action: {act}\n" + summarize_obs(obs) + '\n'
    return obs, infos, env, overall_memory

@dataclass
class UnitTestSample:
    seed: int
    prefix_actions: List[str]  
    observation: str            
    action: str                 
    schema: str               
    direction: Optional[str]   


# ---------------------------------------------------------------------
# BFS
# ---------------------------------------------------------------------

def collect_unit_tests_bfs(
    seed: int,
    per_schema_target: int = 1,
    require_all_directions: bool = True,
    max_depth: int = 40,
    max_nodes: int = 5000,
    max_children_per_node: Optional[int] = None,
    verbose: bool = True,
) -> List[UnitTestSample]:
    
    samples: List[UnitTestSample] = []
    coverage_dirs: Dict[str, Dict[str, int]] = {
        "open-door": {"north": 0, "south": 0, "east": 0, "west": 0},
        "move":      {"north": 0, "south": 0, "east": 0, "west": 0},
    }
    directions = VALID_DIRS

    def coverage_met() -> bool:
        if not require_all_directions:
            return all(
                sum(coverage_dirs[s][d] for d in directions) >= per_schema_target
                for s in ("open-door", "move")
            )
        for s in ("open-door", "move"):
            for d in directions:
                if coverage_dirs[s][d] < per_schema_target:
                    return False
        return True

    def need_score(schema: str, direction: Optional[str]) -> int:
        if schema not in ("open-door", "move") or direction not in directions:
            return 0
        return max(0, per_schema_target - coverage_dirs[schema][direction])

    q = deque([([], 0)])  # (prefix, depth)
    visited_obs = set()
    nodes = 0

    while q and not coverage_met():
        prefix_actions, depth = q.popleft()
        if depth > max_depth:
            continue

        nodes += 1
        if nodes > max_nodes:
            if verbose:
                print(f"[BFS] node budget exceeded: {nodes}>{max_nodes}")
            break

        obs, infos, _, _ = _replay_to_state(seed, prefix_actions)
        obs_key = obs.strip()
        if obs_key in visited_obs:
            continue
        visited_obs.add(obs_key)

        brief_obs = "Action: look around\n" + summarize_obs(obs) + "\n"

        successes: List[Tuple[str, bool]] = []
        cand_order = list(sanitize_valid_actions(infos))  
        for cand in cand_order:
            _obs_cur, _infos_cur, env_tmp, _ = _replay_to_state(seed, prefix_actions)
            obs2, reward, done2, infos2 = env_tmp.step(cand)
            brief_after = "Action: " + cand + "\n" + summarize_obs(obs2) + "\n"
            msg, code = map_env_feedback_to_large_loop_error(brief_after, cand)
            ok = (code is None)
            if ok:
                successes.append((cand, bool(done2)))

        if not successes:
            if verbose:
                print(f"[BFS] no success at depth={depth}, prefix={prefix_actions}")
            continue

        for cand, _done in successes:
            schema, direction = _classify_action(cand)
            if schema in ("open-door", "move") and direction in directions:
                if coverage_dirs[schema][direction] < per_schema_target:
                    samples.append(UnitTestSample(
                        seed=seed,
                        prefix_actions=copy.deepcopy(prefix_actions),
                        observation=brief_obs,
                        action=cand,
                        schema=schema,
                        direction=direction,
                    ))
                    coverage_dirs[schema][direction] += 1
                    if verbose:
                        print(f"[BFS] +sample: depth={depth} {schema}/{direction} -> "
                              f"{coverage_dirs[schema][direction]}/{per_schema_target} via {cand}")
                    if coverage_met():
                        return samples

        expandable = [
            (cand, done) for cand, done in successes
            if not done
        ]
        if expandable:
            expandable.sort(key=lambda cd: (
                need_score(*_classify_action(cd[0])),
                cd[0].startswith("open door")
            ), reverse=True)

            if max_children_per_node is not None:
                expandable = expandable[:max_children_per_node]

            for cand, _done in expandable:
                q.append((prefix_actions + [cand], depth + 1))

    return samples


def _analyze_seed_diversity(
    seed: int,
    per_schema_target: int = 1,
    require_all_directions: bool = True,
    verbose_collect: bool = False,
) -> Dict:

    unit_samples = collect_unit_tests_bfs(
        seed=seed,
        per_schema_target=per_schema_target,
        require_all_directions=require_all_directions,
        verbose=verbose_collect,
    )
    actions_covered_set = {s.action for s in unit_samples}
    actions_covered = sorted(actions_covered_set)
    action_full = set(VALID_ACTIONS_CANONICAL).issubset(actions_covered_set)

    return {
        "seed": seed,
        "actions_covered": actions_covered,
        "num_actions": len(actions_covered),
        "action_full": action_full,
    }


def search_best_seed(
    seeds: List[int],
    per_schema_target: int = 1,
    require_all_directions: bool = True,
    verbose: bool = True,
):

    all_results: List[Dict] = []
    best_results: List[Dict] = []
    best_key: Optional[Tuple[int, int]] = None

    for s in seeds:
        info = _analyze_seed_diversity(
            seed=s,
            per_schema_target=per_schema_target,
            require_all_directions=require_all_directions,
            verbose_collect=False,  
        )
        all_results.append(info)

        key = (
            1 if info["action_full"] else 0,
            info["num_actions"],
        )

        if best_key is None or key > best_key:
            best_results = [info]
            best_key = key
        elif key == best_key:
            best_results.append(info)

        if verbose:
            print(
                f"[seed-scan] seed={s:3d} | "
                f"actions {info['num_actions']}/8 {info['actions_covered']} | "
                f"act_full={info['action_full']}"
            )

    if verbose and len(best_results) > 1:
        print(f"\n[seed-scan] Found {len(best_results)} equally best seeds:")
        for result in best_results:
            print(f"  - seed={result['seed']:3d} | "
                  f"actions {result['num_actions']}/8 {result['actions_covered']}")

    best = best_results[0] if best_results else None
    return all_results, best


# ---------------------------------------------------------------------
# DF trainer
# ---------------------------------------------------------------------

def build_initial_df_from_unit_tests(
    model_name: str,
    samples: List[UnitTestSample],
    max_global_rounds: int = 3,
    per_sample_tries: int = 3,
) -> Tuple[str, Dict]:
 
    df = ""
    pf = ""
    report = {"rounds": []}

    for round_id in range(max_global_rounds):
        all_pass = True
        round_log = {"round": round_id, "samples": []}

        for idx, s in enumerate(samples):
            obs, infos, _env_tmp, overall_memory = _replay_to_state(s.seed, s.prefix_actions)
            sample_pass = False

            for try_id in range(per_sample_tries):
                df, pf, _prompt_used, _raw_response = llm_to_pddl(
                    model_name=model_name,
                    brief_obs="Action: look around\n" + summarize_obs(obs) + '\n',
                    valid_actions=sanitize_valid_actions(infos),
                    prev_df=df,
                    prev_pf=pf,
                    overall_memory=overall_memory,
                    unit=True,
                    expected_action=s.action
                )

                actions_mapped, _stderr, plan_text = get_action_from_pddl(df, pf)
                first_plan_action = actions_mapped[0] if actions_mapped else None
                pass_now = (first_plan_action == s.action)

                round_log["samples"].append({
                    "sample_index": idx,
                    "try": try_id,
                    "expected": s.action,
                    "plan_text": plan_text,
                    "mapped": actions_mapped,
                    "pass": pass_now,
                    "_stderr": _stderr,
                })

                if pass_now:
                    sample_pass = True
                    break
                else:
                    all_pass = False

            print(f"[df-train/action] round={round_id} sample={idx}: {'PASS' if sample_pass else 'FAIL'}")

        report["rounds"].append(round_log)
        if all_pass:
            print(f"[df-train/action] All samples passed at round {round_id}.")
            break

    return df, report


# ---------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------

def make_initial_df_pipeline_action(
    model_name_for_pddl: str,
    seed: int,
    per_schema_target: int = 2,
    require_all_directions: bool = False,
) -> Tuple[str, List[UnitTestSample], Dict]:
    samples = collect_unit_tests_bfs(
        seed=seed,
        per_schema_target=per_schema_target,
        require_all_directions=require_all_directions,
    )
    print(f"[pipeline-action] collected {len(samples)} samples.")
    df, report = build_initial_df_from_unit_tests(
        model_name=model_name_for_pddl,
        samples=samples,
        max_global_rounds=2,
        per_sample_tries=3,
    )
    return df, samples, report


# =========================
# Artifact saver 
# =========================

def _sanitize_model_name_for_path(model_name: Optional[str]) -> str:
    if not model_name:
        return "model"
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_name)

def _summarize_unit_samples(samples: List[UnitTestSample]) -> Dict:
    coverage_schema_dir: Dict[str, int] = {}
    action_counts: Dict[str, int] = {}
    for s in samples:
        key = f"{s.schema}/{s.direction}"
        coverage_schema_dir[key] = coverage_schema_dir.get(key, 0) + 1
        action_counts[s.action] = action_counts.get(s.action, 0) + 1
    actions_covered = sorted(action_counts.keys())
    return {
        "type": "unit_samples",
        "num_samples": len(samples),
        "coverage_schema_dir": coverage_schema_dir,
        "actions_covered": actions_covered,
        "action_counts": action_counts,
    }

def save_artifacts(
    folder_name: str,
    seed: int,
    mode: str,                       # Now only "action" is supported
    df: str,
    items: List[UnitTestSample],
    report: Dict,
    model_name: Optional[str] = None,
    extra_meta: Optional[Dict] = None,
) -> Dict[str, str]:
    base_dir = os.path.join("output", folder_name, mode)
    os.makedirs(base_dir, exist_ok=True)

    today = str(date.today())
    model_tag = _sanitize_model_name_for_path(model_name)
    tag = f"{today}_{model_tag}_seed{seed}_{mode}"

    df_path     = os.path.join(base_dir, f"{tag}_initial_df.pddl")
    items_path  = os.path.join(base_dir, f"{tag}_unit_samples.json")
    report_path = os.path.join(base_dir, f"{tag}_report.json")
    meta_path   = os.path.join(base_dir, f"{tag}_meta.json")

    # 1) DF
    with open(df_path, "w", encoding="utf-8") as f:
        f.write(df or "")

    # 2) items (dataclass → dict)
    if items and hasattr(items[0], "__dataclass_fields__"):
        items_serialized = [asdict(x) for x in items]
    else:
        items_serialized = items  

    with open(items_path, "w", encoding="utf-8") as f:
        json.dump(items_serialized, f, ensure_ascii=False, indent=2)

    # 3) report
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 4) meta 
    summary = _summarize_unit_samples(items) if items else {"type": "unit_samples", "num_samples": 0}

    meta = {
        "seed": seed,
        "mode": mode,
        "model_name": model_name,
        "folder_name": folder_name,
        "paths": {
            "df": df_path,
            "items": items_path,
            "report": report_path,
        },
        "summary": summary,
    }
    if extra_meta:
        meta["extra_meta"] = extra_meta

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[save_artifacts] saved: \n  df={df_path}\n  items={items_path}\n  report={report_path}\n  meta={meta_path}")
    return {"df": df_path, "items": items_path, "report": report_path, "meta": meta_path}


def make_seed_df(
    mode: Literal["action"],  # Only action mode is supported now
    model_name_for_pddl: str,
    seed: int,
    **kwargs,
):

    if mode != "action":
        raise ValueError(f"Only 'action' mode is supported, got: {mode}")

    return make_initial_df_pipeline_action(model_name_for_pddl=model_name_for_pddl, seed=seed, **kwargs)


if __name__ == "__main__":
    model_id3 = "Qwen/Qwen3-32B"  
    CANDIDATE_SEEDS = range(0, 500)

    print("[MAIN] searching best seed among:", CANDIDATE_SEEDS)
    _all, best = search_best_seed(
        seeds=CANDIDATE_SEEDS,
        per_schema_target=1,
        require_all_directions=True,
        verbose=True,
    )
    print("[MAIN] best seed =", best["seed"], "actions:", best["actions_covered"])
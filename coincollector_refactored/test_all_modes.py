import os
from datetime import date
import argparse
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_THIS_DIR)
print("[INFO] CWD set to:", os.getcwd())

import traceback, time
from src.initial_df_builder import make_seed_df, save_artifacts, search_best_seed
from src.pddl_engine import (
    run_iterative_model_initDF,
    run_iterative_model,
    run_baseline_model,
)

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="which model to use")
parser.add_argument("--build_action", action='store_true')
parser.add_argument("--save_artifacts", action='store_true')
parser.add_argument("--run_after_build", action='store_true')
parser.add_argument("--run_translator", action='store_true')
parser.add_argument("--run_direct_planner", action='store_true')
args = parser.parse_args()
MODEL_NAME   = args.model # allenai/Olmo-3-32B-Think, Qwen/Qwen3-32B, gpt-4.1, o3-mini, deepseek-reasoner
start_date = date.today()
FOLDER_NAME  = f"CC_{start_date}_{MODEL_NAME}"
RESULT_NAME  = FOLDER_NAME


START_TRIAL  = 0
END_TRIAL    = 100


SEED         = 217

GOAL_TYPE    = "detailed"    # or "subgoal"

# only used when building action-mode DF
ACTION_PER_SCHEMA_TARGET      = 1
ACTION_REQUIRE_ALL_DIRECTIONS = True

DO_BUILD_ACTION = args.build_action  # Only action diversity approach is used

SAVE_ARTIFACTS  = args.save_artifacts
RUN_AFTER_BUILD = args.run_after_build   # if True, call run_iterative_model_initDF with the generated DF

# other modes without seed DF
DO_RUN_TRANSLATOR        = args.run_translator  # run_iterative_model
DO_RUN_DIRECT_PLANNER    = args.run_direct_planner  # run_baseline_model

def banner(title: str):
    print("\n" + "="*80)
    print(title)
    print("="*80)

def build_and_maybe_run(mode: str):
    banner(f"[BUILD] mode={mode}")
    build_kwargs = {}
    if mode == "action":
        build_kwargs["per_schema_target"] = ACTION_PER_SCHEMA_TARGET
        if ACTION_REQUIRE_ALL_DIRECTIONS:
            build_kwargs["require_all_directions"] = True

    t0 = time.time()
    df_text, items, report = make_seed_df(
        mode=mode,
        model_name_for_pddl=MODEL_NAME,
        seed=SEED,
        **build_kwargs,
    )
    print(f"[BUILD] completed in {time.time()-t0:.1f}s, DF length={len(df_text)} chars")

    saved_paths = None
    if SAVE_ARTIFACTS:
        banner(f"[SAVE] mode={mode}")
        saved_paths = save_artifacts(
            folder_name=FOLDER_NAME,
            seed=SEED,
            mode=mode,
            df=df_text,
            items=items,
            report=report,
            model_name=MODEL_NAME,
            extra_meta={"note": "created by test_all_modes.py"},
        )
        print("[SAVE] artifacts:", saved_paths)

    if RUN_AFTER_BUILD:
        banner(f"[RUN with seed DF] mode={mode}")
        t1 = time.time()
        run_iterative_model_initDF(
            model_name=MODEL_NAME,
            start_trial=START_TRIAL,
            end_trial=END_TRIAL,
            folder_name=FOLDER_NAME,
            result_name=RESULT_NAME,
            initial_df=df_text,
            goal_type=GOAL_TYPE,
            mode=mode,
        )
        print(f"[RUN with seed DF] completed in {time.time()-t1:.1f}s")

    return df_text, saved_paths


def main():
    # Build initial DF using action diversity approach
    if DO_BUILD_ACTION:
        try:
            build_and_maybe_run("action")
        except Exception as e:
            print(f"[ERROR] build/run failed for action mode: {e}")
            traceback.print_exc()

    # modes that don't use initial DF
    if DO_RUN_TRANSLATOR:
        banner("[RUN] translator (no seed DF)")
        run_iterative_model(MODEL_NAME, START_TRIAL, END_TRIAL, FOLDER_NAME, RESULT_NAME)

    if DO_RUN_DIRECT_PLANNER:
        banner("[RUN] direct planner baseline (LLM as planner)")
        run_baseline_model(MODEL_NAME, START_TRIAL, END_TRIAL, FOLDER_NAME, RESULT_NAME)

    banner("ALL DONE!")

if __name__ == "__main__":
    main()

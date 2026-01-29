from __future__ import annotations

import os
import time
from datetime import date
import csv
import json
import random
import re
import glob
from os.path import join as pjoin
from typing import List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from pydantic import BaseModel

import textworld
import textworld.gym
from alfworld.info import ALFWORLD_DATA
from alfworld.agents.utils.misc import add_task_to_grammar
from alfworld.agents.environment.alfred_tw_env import (
    AlfredExpert,
    AlfredDemangler,
    AlfredExpertType,
)

from .config import (
    MAX_STEPS,
    MAX_EPISODE_STEPS,
    DEFAULT_PROBLEM_ID_RANGE,
    OPENAI_MODELS_LIST,
)

from .utils import (
    detect_duplicates,
    get_action_from_pddl,
    build_large_loop_error_message as build_basic_error_message,
    extract_json,
    sanitize_valid_actions,
    summarize_obs,
    parse_alfworld_obs,
)

from .prompts import *

try:
    from .problem_selector import ProblemSelector
except ImportError:
    ProblemSelector = None


# =============================================================
# ALFWorld / TextWorld setup
# =============================================================

def _scan_problems() -> List[str]:
    """Scan for all available ALFWorld problems."""
    problems = glob.glob(pjoin(ALFWORLD_DATA, "**", "initial_state.pddl"), recursive=True)
    problems = [p for p in problems if "movable_recep" not in p]
    if not problems:
        raise ValueError(
            f"Can't find problem files in {ALFWORLD_DATA}. Did you run alfworld-data?"
        )
    return problems


# Scan once at module load
PROBLEMS = _scan_problems()

# Read base logic once and reuse
DOMAIN_PATH = pjoin(ALFWORLD_DATA, "logic", "alfred.pddl")
GRAMMAR_PATH = pjoin(ALFWORLD_DATA, "logic", "alfred.twl2")
with open(DOMAIN_PATH, "r", encoding="utf-8") as _f:
    BASE_DOMAIN = _f.read()
with open(GRAMMAR_PATH, "r", encoding="utf-8") as _f:
    BASE_GRAMMAR = _f.read()

REQUEST_INFOS = textworld.EnvInfos(
    won=True,
    admissible_commands=True,
    score=True,
    max_score=True,
    intermediate_reward=True,
    extras=["expert_plan"],
)


# =============================================================
# Problem preparation
# =============================================================

def prepare_problem_flexible(problem_path: str = None, problem_id: int = None):
    if problem_path:
        if "initial_state.pddl" in problem_path:
            pddl_file = problem_path
            trial_dir = os.path.dirname(pddl_file)
            problem_dir = os.path.dirname(trial_dir)
        else:
            problem_dir = problem_path
            trials = glob.glob(os.path.join(problem_dir, "trial_*"))
            if trials:
                trial_dir = trials[0]
                pddl_file = os.path.join(trial_dir, "initial_state.pddl")
            else:
                raise ValueError(f"No trial directory found in {problem_dir}")
        json_file = os.path.join(trial_dir, "traj_data.json")
    else:
        # ID-based selection (for random/legacy runs)
        if problem_id is None:
            problem_id = _pick_problem_id()
        pddl_file = PROBLEMS[problem_id]
        trial_dir = os.path.dirname(pddl_file)
        problem_dir = os.path.dirname(trial_dir)
        json_file = os.path.join(trial_dir, "traj_data.json")

    # Extract problem type from directory name
    problem_name = os.path.basename(problem_dir)
    if '-' in problem_name:
        problem_type = problem_name.split('-')[0]
    else:
        problem_type = 'unknown'

    game_type = 'unknown'
    if ProblemSelector:
        game_type = ProblemSelector.PROBLEM_TYPE_MAPPING.get(problem_type, 'unknown')

    with open(json_file, "r", encoding="utf-8") as f:
        traj_data = json.load(f)

    grammar = add_task_to_grammar(BASE_GRAMMAR, traj_data)
    gamefile = os.path.join(trial_dir, "game.tw-pddl")
    gamedata = {
        "pddl_domain": BASE_DOMAIN,
        "grammar": grammar,
        "pddl_problem": open(pddl_file, "r", encoding="utf-8").read(),
    }
    json.dump(gamedata, open(gamefile, "w", encoding="utf-8"))

    expert = AlfredExpert(expert_type=AlfredExpertType.HANDCODED)
    env_id = textworld.gym.register_game(
        gamefile,
        REQUEST_INFOS,
        max_episode_steps=MAX_EPISODE_STEPS,
        wrappers=[AlfredDemangler(), expert],
    )
    env = textworld.gym.make(env_id)

    # Get initial state
    obs, infos = env.reset()
    scene, goal = parse_alfworld_obs(obs)
    valid_actions = sanitize_valid_actions(infos)

    meta = {
        "problem_id": problem_id if problem_id is not None else -1,
        "problem_dir": problem_dir,
        "trial_dir": trial_dir,
        "problem_name": problem_name,
        "problem_type": problem_type,
        "game_type": game_type,
        "gamefile": gamefile,
    }
    return env, expert, meta, scene, goal, valid_actions, infos

def prepare_problem(problem_id: int):
    """Legacy wrapper for ID-based problem preparation."""
    env, expert, meta, scene, goal, valid_actions, infos = prepare_problem_flexible(problem_id=problem_id)
    return env, expert, meta, scene, goal, valid_actions, infos


def reset_env_with_prefix(gamefile: str, successful_actions: List[str], expert: AlfredExpert):
    env_id = textworld.gym.register_game(
        gamefile,
        REQUEST_INFOS,
        max_episode_steps=MAX_EPISODE_STEPS,
        wrappers=[AlfredDemangler(), expert],
    )
    env = textworld.gym.make(env_id)
    obs, infos = env.reset()
    for act in successful_actions:
        obs, _, _, infos = env.step(act)
    return env, obs, infos


def make_output_file(
    folder_name: str,
    today: date,
    model_name: str,
    tag: str,
    trial: int,
    goal_type: Optional[str] = None,
    game_type: Optional[str] = None,
):
    fixed_model_name = model_name.replace("/", "_")
    folder_path = f"output/{folder_name}"
    os.makedirs(folder_path, exist_ok=True)
    parts = [str(today), fixed_model_name, tag]
    if goal_type:
        parts.append(goal_type)
    if game_type:
        parts.append(game_type)
    parts.append(str(trial))
    return f"{folder_path}/{'_'.join(parts)}.txt"


def build_enhanced_error_message(step_trace: str, taken_action: str) -> Tuple[str, str]:
    """Enhanced error messages with detailed guidance and handling strategy.

    Returns (message, handling) where handling is 'retry' or 'ignore'
    """
    msg = (
        "In this step, you take the following actions and observations from those actions:\n"
        f"{step_trace}\n\n"
    )
    a = taken_action.lower()

    if "go to" in a:
        msg += (
            f"This is the action you take and got something wrong: {taken_action}. "
            "You are trying to go to a receptacle but nothing happens. "
            "You may already been at this receptacle, in other words, you have already "
            "went to this place and do not need to go to this receptacle again. "
            "Otherwise, there is no the receptacle you are aiming to."
        )
        return msg, "ignore"

    elif "open" in a:
        msg += (
            f"This is the action you take and got something wrong: {taken_action}. "
            "You are trying to open a receptacle but nothing happens. "
            "You should first go to this receptacle to open it. "
            "But if you have already go to this receptacle and still seeing this error message, "
            "it means that this receptacle cannot be opened and you can directly see objects "
            "after you go to it. Do not try to open it!!"
        )
        return msg, "retry"

    elif "take" in a:
        msg += (
            f"This is the action you take and got something wrong: {taken_action}. "
            "You are trying to take something from a receptacle. "
            "You should first go to this receptacle to take the object. "
            "But if you have already go to this receptacle and still seeing this error message, "
            "it means that this receptacle doesn't have this object. "
            "You should go to other receptacle to find your aim object. "
            "Remember do not assume you can take the object from the receptable but should "
            "always set the initial goal as finding that aim object."
        )
        return msg, "retry"

    elif "move" in a:
        msg += (
            f"This is the action you take and got something wrong: {taken_action}. "
            "You want to move some object to a receptacle but failed. "
            "You should first find that object somewhere by going to an unvisited receptacle "
            "and open if necessary. Then pick up the aiming object so that you can go to "
            "your aim receptacle and put it there."
        )
        return msg, "retry"

    elif "slice" in a:
        msg += (
            f"This is the action you take and got something wrong: {taken_action}. "
            "You are trying to slice an object with a sharp object. "
            "You should first pickup the sharp object (this should be the only object you pick up) "
            "then take the slice action directly without picking up the aim object! "
            "Don't forget to put the sharp object back to the receptacle after you finish slicing."
        )
        return msg, "retry"

    elif "cool" in a:
        msg += (
            f"This is the action you take and got something wrong: {taken_action}. "
            "You are trying to cool an object with a fridge. "
            "You need to find the object and pick it up from other receptacle. "
            "Then go to frige and cool the object directly. "
            "Notice: do not move the object to the fridge but cool directly!"
        )
        return msg, "retry"

    elif "heat" in a:
        msg += (
            f"This is the action you take and got something wrong: {taken_action}. "
            "You are trying to heat an object but nothing happens. "
            "You usually must pick up the object, then go to the microwave and heat it. "
            "Do not try to move object to microwave, use heat action directly."
        )
        return msg, "retry"

    elif "clean" in a:
        msg += (
            f"This is the action you take and got something wrong: {taken_action}. "
            "You are trying to clean an object but nothing happens. "
            "You usually must pick up the object, then go to the sinkbasin and clean it. "
            "Do not try to move object to sinkbasin, use clean action directly."
        )
        return msg, "retry"

    elif "use" in a:
        msg += (
            f"This is the action you take and got something wrong: {taken_action}. "
            "You are trying to use an object. "
            "You can only use a lamp to turn it on and look at or examine other objects. "
            "Note: to look at or examine other objects, you should first pick it up."
        )
        return msg, "retry"

    elif any(k in a for k in ("fridge", "sinkbasin", "microwave")) and (
        "move" in a or "take" in a
    ):
        msg += (
            f"This is the action you take and got something wrong: {taken_action}. "
            "You are trying to move or take an object to or from a fridge/sinkbasin/microwave. "
            "You don't need to take this action! "
            "You should go to the appliance receptacle, use the action (cool/clean/heat) directly, "
            "then go to another receptacle."
        )
        return msg, "ignore"

    else:
        msg += "This action failed. Review your PDDL model and adjust based on observations."
        return msg, "retry"


# =============================================================
# LLM response models (Pydantic)
# =============================================================

class PDDLResponse(BaseModel):
    df: str
    pf: str

class ActionResponse(BaseModel):
    actions: List[str]


# =============================================================
# LLM calls
# =============================================================

def run_llm(prompt: str, model_name: str, system_prompt, response_model = None) -> tuple:
    """Run LLM with the given prompt and return structured response."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    raw_response = None

    if "deepseek" in model_name.lower():
        client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages,
            response_format={'type': 'json_object'}
        )
        response_content = response.choices[0].message.content
        raw_response = response_content

        try:
            result = json.loads(response_content)
        except json.JSONDecodeError:
            try:
                repaired_content = extract_json(response_content)
                result = json.loads(repaired_content)
            except json.JSONDecodeError:
                raise ValueError(
                    f"Model response is not valid JSON after repair attempts:\n{response_content}"
                )
    else:
        params = {
            "model": model_name,
            "input": messages,
            "reasoning": {"effort": "high"},
            "text_format": response_model,
        }

        if any(model_name.startswith(base) for base in OPENAI_MODELS_LIST):
            client = OpenAI()
            if model_name.startswith("gpt-"):
                params.pop("reasoning", None)
        else:
            client = OpenAI(
                base_url="http://localhost:8000/v1",
                api_key="EMPTY"
            )
            assert client.models.list().data[0].id == model_name, "Model name does not match the vLLM server model."

        response = client.responses.parse(**params)
        result = response.output_parsed.model_dump()
        raw_response = json.dumps(result, indent=2)

    return result, raw_response


# =============================================================
# Prompt wrappers
# =============================================================

def llm_to_pddl(
    model_name: str,
    brief_obs: str,
    valid_actions: List[str],
    goal: str,
    prev_df: str = "",
    prev_pf: str = "",
    prev_err: str = "",
    have_duplicate: bool = False,
    overall_memory: Optional[str] = None,
    large_loop_error_message: Optional[str] = None,
    goal_type: str = "detailed",
):
    prompt = prompt_format.strip() + "\n\n"

    base_kwargs = {
        "goal": goal,
        "brief_obs": brief_obs,
        "valid_actions": valid_actions,
        "pddl_action_specs": PDDL_ACTION_SPECS.strip(),
    }

    if goal_type == "detailed":
        # prompt += prompt_obs_action_detailed.format(**base_kwargs)
        prompt += prompt_pddl.format(goal=goal, brief_obs=brief_obs)
    elif goal_type == "subgoal":
        prompt += prompt_obs_action_subgoal.format(**base_kwargs)
    elif goal_type == "without_hint":
        prompt += prompt_obs_action_without_hint.format(**base_kwargs)
    elif goal_type == "without_detailed_goal":
        prompt += prompt_obs_action_without_detailed_goal.format(**base_kwargs)
    else:
        prompt += prompt_obs_action_general_goal.format(**base_kwargs)

    if prev_df or prev_pf:
        prompt += "\n\n" + prompt_prev_files.format(
            prev_df=prev_df or "N/A",
            prev_pf=prev_pf or "N/A",
            overall_memory=overall_memory or "N/A",
        )

        # any_feedback = bool(large_loop_error_message or prev_err or have_duplicate)
        # if any_feedback:
        #     if large_loop_error_message:
        #         prompt += "\n\n" + prompt_simulation_error.format(
        #             large_loop_error_message=large_loop_error_message
        #         )
        #     if prev_err:
        #         prompt += "\n\n" + prompt_error_parser.format(prev_err=prev_err)
        #     if have_duplicate:
        #         prompt += "\n\n" + prompt_duplicate_note
        #     prompt += "\n\nNow rewrite both the domain and problem files with the minimal fix.\n"
        # else:
        #     prompt += "\n\n" + prompt_new_obs
        if large_loop_error_message:
            prompt += "\n\n" + prompt_simulation_error.format(
                large_loop_error_message=large_loop_error_message
            )
        if prev_err:
            prompt += "\n\n" + prompt_error_parser.format(prev_err=prev_err)
        if have_duplicate:
            prompt += "\n\n" + prompt_duplicate_note
        
        if not (large_loop_error_message or prev_err or have_duplicate):
            prompt += "\n\n" + prompt_new_obs
        
        prompt += "\n\nNow rewrite both the domain and problem files with the minimal fix.\n"
        

    resp, raw_response = run_llm(prompt, model_name, system_prompt=SYS_PROMPT_PDDL, response_model=PDDLResponse)
    return resp["df"], resp["pf"], prompt, raw_response


def llm_to_pddl_with_initial_df(
    model_name: str,
    brief_obs: str,
    valid_actions: List[str],
    goal: str,
    initial_df: str
):
    prompt = prompt_format.strip() + "\n\n"

    # prompt += prompt_obs_action_initial_df.format(
    #     initial_df=initial_df,
    #     goal=goal,
    #     brief_obs=brief_obs,
    #     valid_actions=valid_actions,
    #     pddl_action_specs=PDDL_ACTION_SPECS.strip(),
    # )
    prompt += prompt_pddl_initDF.format(
        initial_df=initial_df,
        goal=goal,
        brief_obs=brief_obs,
    )

    resp, raw_response = run_llm(prompt, model_name, system_prompt=SYS_PROMPT_PDDL, response_model=PDDLResponse)
    return resp["df"], resp["pf"], prompt, raw_response


def llm_to_actions_baseline(
    model_name: str,
    brief_obs: str,
    valid_actions: List[str],
    goal: str,
    overall_memory: Optional[str] = None,
    large_loop_error_message: Optional[str] = None,
):

    # prompt = prompt_baseline.format(
    #     goal=goal or "Explore and interact meaningfully based on available observations.",
    #     brief_obs=brief_obs,
    #     valid_actions=valid_actions,
    #     overall_memory=overall_memory or "N/A",
    #     large_loop_error_message=large_loop_error_message or "N/A",
    # )
    prompt = prompt_baseline2.format(
        goal=goal or "Explore and interact meaningfully based on available observations.",
        brief_obs=brief_obs,
        overall_memory=overall_memory or "No additional memory available.",
        large_loop_error_message=large_loop_error_message or "No errors or obstacles mentioned.",
    )
    resp, raw_response = run_llm(prompt, model_name, system_prompt=SYS_PROMPT_PLAN, response_model=ActionResponse)
    return resp["actions"], prompt, raw_response

def _pick_problem_id() -> int:
    lo, hi = DEFAULT_PROBLEM_ID_RANGE
    return random.randint(lo, hi)


# =============================================================
# Main runners 
# =============================================================

def run_iterative_model_full(
    model_name: str,
    problem_type: str = None,
    start_index: int = 0,
    num_problems: int = None,
    start_trial: int = None,
    end_trial: int = None,
    split: str = "valid_train",
    folder_name: str = "alfworld_runs",
    result_name: str = "alfworld_results",
    goal_type: str = "detailed",
    initial_df: str = None,
    mode: str = "PDDL",
    use_enhanced_errors: bool = True,
):
    """Unified iterative model runner with all features.

    Args:
        model_name: LLM model to use
        problem_type: Type of problems to run (for typed selection)
        start_index: Starting index for typed selection
        num_problems: Number of problems for typed selection
        start_trial/end_trial: For random selection (legacy)
        split: Data split for typed selection
        folder_name: Output folder name
        result_name: CSV result file name
        goal_type: Goal description type
        initial_df: Initial domain file (for initDF mode)
        mode: Execution mode identifier
        use_enhanced_errors: Whether to use enhanced error messages
    """
    # Determine which problems to run
    if problem_type and ProblemSelector:
        # Type-based selection
        selector = ProblemSelector(split=split)
        problems = selector.get_problems_by_type(
            problem_type,
            max_count=num_problems or 10,
            offset=start_index
        )
        if not problems:
            print(f"No problems found for type '{problem_type}' in split '{split}'")
            if selector:
                print("Available types:")
                for ptype, count in selector.get_available_types().items():
                    print(f"  {ptype}: {count} problems")
            return
        print(f"Running {len(problems)} {problem_type} problems from {split}")
        trials_iter = [(start_index + i, prob) for i, prob in enumerate(problems)]
    else:
        # Random selection (legacy)
        if start_trial is None:
            start_trial = start_index
        if end_trial is None:
            end_trial = start_trial + (num_problems or 10)
        trials_iter = [(t, None) for t in range(start_trial, end_trial)]

    # Load initial DF if provided
    initial_df_text = None
    if initial_df:
        if os.path.isfile(initial_df):
            with open(initial_df, "r", encoding="utf-8") as f:
                initial_df_text = f.read()
        else:
            initial_df_text = initial_df

    # Main trial loop
    for trial_idx, problem_info in trials_iter:
        retry = 0
        while retry < 2:
            env = None
            try:
                succeed = False
                today = date.today()

                # Prepare problem
                try:
                    if problem_info:
                        # Typed selection
                        problem_path, prob_type, game_type = problem_info
                        env, expert, meta, init_scene, goal, valid_actions, infos = prepare_problem_flexible(
                            problem_path=problem_path
                        )
                    else:
                        # Random selection
                        problem_id = _pick_problem_id()
                        env, expert, meta, init_scene, goal, valid_actions, infos = prepare_problem_flexible(
                            problem_id=problem_id
                        )
                        game_type = meta.get("game_type", "unknown")
                except KeyError as e:
                    if str(e) == "'val1'":
                        print(f"Skipping trial {trial_idx} due to val1 error in PDDL parsing")
                        # Log the skip to errors file
                        error_log_path = f"output/{folder_name}/errors.txt"
                        os.makedirs(f"output/{folder_name}", exist_ok=True)
                        with open(error_log_path, "a", encoding="utf-8") as f:
                            f.write(f"[{mode}] Trial {trial_idx} | Model: {model_name} | Type: {problem_type or 'random'} | Skipped: val1 PDDL parsing error\n")
                        break  # Skip this trial completely
                    else:
                        raise

                # Create output file
                tag = f"{mode}_{goal_type}"
                if problem_type:
                    tag += f"_{game_type}"
                file_name = make_output_file(
                    folder_name, today, model_name, tag, trial_idx,
                    goal_type=goal_type, game_type=game_type if problem_type else None
                )

                if os.path.exists(file_name):
                    open(file_name, "w").close()

                # Initialize trial
                trial_record = []
                with open(file_name, "a", encoding="utf-8") as f:
                    f.write(f"Problem: {meta.get('problem_name', 'N/A')}\n")
                    f.write(f"Type: {meta.get('problem_type', 'N/A')} -> {game_type}\n")
                    if problem_info:
                        f.write(f"Path: {problem_path}\n")
                    else:
                        f.write(f"Problem ID: {meta.get('problem_id', 'N/A')}\n")
                    f.write(f"Observations: {init_scene}\n")
                    f.write(f"Valid Actions: {valid_actions}\n")
                    f.write(f"Goal: {goal}\n")
                    if initial_df_text:
                        f.write("\n[Seed DF]\n")
                        f.write(initial_df_text + "\n")

                brief_obs = "Action: look around\n" + summarize_obs(init_scene) + "\n"
                overall_memory = brief_obs

                df = ""
                pf = ""
                all_actions: List[str] = []
                successful_actions: List[str] = []
                obs_queue: List[str] = []
                end_game = False

                # Main step loop
                for step_id in range(MAX_STEPS):
                    with open(file_name, "a", encoding="utf-8") as f:
                        f.write(f"\n\n====Step {step_id}====\n")

                    trial_step_record = []
                    within_step_tries = 0
                    action_passed = False
                    large_loop_error_message = ""

                    # Larger loop (environment retry)
                    while within_step_tries < 5 and not action_passed:
                        with open(file_name, "a", encoding="utf-8") as f:
                            f.write(f"\n----Larger Loop No. {within_step_tries}----\n")
                            f.write(f"successful_actions: {successful_actions}\n")

                        within_step_tries += 1

                        if within_step_tries > 1:
                            if env is not None:
                                try:
                                    env.close()
                                except Exception:
                                    pass
                            env, obs, infos = reset_env_with_prefix(
                                meta["gamefile"], successful_actions, expert
                            )
                            _scene, goal = parse_alfworld_obs(obs)
                            valid_actions = sanitize_valid_actions(infos)

                        action_queue: List[str] = []
                        tem_action_queue: List[str] = []
                        tem_memory = ""
                        start_checkpoint = True

                        # Small loop (action execution)
                        while start_checkpoint or action_queue:
                            with open(file_name, "a", encoding="utf-8") as f:
                                f.write(f"Small Loop, action_queue: {action_queue}\n")
                            start_checkpoint = False

                            if not action_queue:
                                if obs_queue:
                                    brief_obs = "\n".join(obs_queue)
                                    obs_queue = []

                                current_valid_actions = sanitize_valid_actions(infos)

                                # Generate or regenerate DF/PF
                                if not df and not pf:
                                    num_tries = 0
                                    if initial_df_text:
                                        # Use seed DF for first generation
                                        first_gen_tries = 0
                                        while (not pf) and first_gen_tries < 3:
                                            df, pf, prompt, raw_response = llm_to_pddl_with_initial_df(
                                                model_name,
                                                brief_obs,
                                                current_valid_actions,
                                                goal,
                                                initial_df_text
                                            )

                                            first_gen_tries += 1

                                    else:
                                        df, pf, prompt, raw_response = llm_to_pddl(
                                            model_name,
                                            brief_obs,
                                            current_valid_actions,
                                            goal,
                                            goal_type=goal_type,
                                        )
                                    action, err_2, plan_text = get_action_from_pddl(df, pf)

                                    with open(file_name, "a", encoding="utf-8") as f:
                                        f.write(f"--Small Loop--: {num_tries}\n")
                                        f.write(f"=== PROMPT SENT TO LLM ===\n{prompt}\n=== END PROMPT ===\n")
                                        f.write(f"=== RAW LLM RESPONSE ===\n{raw_response}\n=== END RESPONSE ===\n")
                                        f.write(f"Generated df and pf:\n{df or 'None'}\n{pf or 'None'}\n")
                                        f.write(f"Actions: {action}\n")

                                    while (not action) and (num_tries < 5):
                                        num_tries += 1
                                        df, pf, prompt, raw_response = llm_to_pddl(
                                            model_name,
                                            brief_obs,
                                            current_valid_actions,
                                            goal,
                                            prev_df=df,
                                            prev_pf=pf,
                                            prev_err=err_2,
                                            overall_memory=overall_memory,
                                            large_loop_error_message=large_loop_error_message,
                                            goal_type=goal_type,
                                        )
                                        action, err_2, plan_text = get_action_from_pddl(df, pf)

                                        with open(file_name, "a", encoding="utf-8") as f:
                                            f.write(f"--Small Loop--: {num_tries}\n")
                                            f.write(f"=== PROMPT SENT TO LLM ===\n{prompt}\n=== END PROMPT ===\n")
                                            f.write(f"=== RAW LLM RESPONSE ===\n{raw_response}\n=== END RESPONSE ===\n")
                                            f.write(f"Generated df and pf:\n{df or 'None'}\n{pf or 'None'}\n")
                                            f.write(f"Actions: {action}\n")
                                            f.write(f"Solver stderr: {err_2}\n")
                                            f.write(f"Plan text:\n{plan_text}\n")
                                    
                                    if not action:
                                        with open(file_name, "a", encoding="utf-8") as f:
                                            f.write(f"Critical: No actions after small-loop retries. Ending trial.\n")
                                        end_game = True
                                  


                                else:
                                    # Regeneration with feedback
                                    num_tries = 0
                                    err_2 = ""
                                    while num_tries < 5:
                                        df, pf, prompt, raw_response = llm_to_pddl(
                                            model_name,
                                            brief_obs,
                                            current_valid_actions,
                                            goal,
                                            prev_df=df,
                                            prev_pf=pf,
                                            prev_err=err_2 if num_tries > 0 else "",
                                            have_duplicate=detect_duplicates(all_actions, 3) if num_tries == 0 else False,
                                            overall_memory=overall_memory,
                                            large_loop_error_message=large_loop_error_message,
                                            goal_type=goal_type,
                                        )
                                        action, err_2, plan_text = get_action_from_pddl(df, pf)

                                        with open(file_name, "a", encoding="utf-8") as f:
                                            f.write(f"--Small Loop (regen)--: {num_tries}\n")
                                            f.write(f"=== PROMPT SENT TO LLM ===\n{prompt}\n=== END PROMPT ===\n")
                                            f.write(f"=== RAW LLM RESPONSE ===\n{raw_response}\n=== END RESPONSE ===\n")
                                            f.write(f"Generated df and pf:\n{df or 'None'}\n{pf or 'None'}\n")
                                            f.write(f"Actions: {action}\n")
                                            f.write(f"Solver stderr: {err_2}\n")
                                            f.write(f"Plan text:\n{plan_text}\n")
                                        
                                        if action:
                                            break

                                        num_tries += 1
                                    
                                    if not action:
                                        with open(file_name, "a", encoding="utf-8") as f:
                                            f.write(f"Critical: No actions after small-loop retries. Ending trial.\n")
                                        end_game = True
                                        

                                # trial_step_record.append([within_step_tries, num_tries])
                                trial_step_record.append([within_step_tries, num_tries])
                                if not action:
                                    end_game = True
                                    break

                                # If we reach here, action exists
                                action_queue.extend(action)
                                tem_action_queue.extend(action)
                                all_actions.extend(action)
                                # else:
                                #     # No actions generated - this is a critical failure
                                #     # Let the large loop retry if we haven't exhausted attempts
                                #     with open(file_name, "a", encoding="utf-8") as f:
                                #         f.write(f"Warning: No actions generated from PDDL (attempt {within_step_tries}/5)\n")
                                #     end_game = True
                                #     break

                            # Execute action from queue
                            with open(file_name, "a", encoding="utf-8") as f:
                                f.write(f"Current action_queue: {action_queue}\n")

                            if not action_queue:
                                continue

                            taken_action = action_queue.pop(0)
                            taken_action = " ".join(taken_action.strip().split())
                            obs, _, _, infos = env.step(taken_action)

                            if infos.get("won"):
                                succeed = True
                                end_game = True
                                with open(file_name, "a", encoding="utf-8") as f:
                                    f.write("Done! Task completed successfully.\n")
                                break

                            action_text = "Action: " + taken_action + "\n"
                            obs_text = summarize_obs(obs) + "\n"
                            brief_obs = action_text + obs_text
                            obs_queue.append(brief_obs)

                            with open(file_name, "a", encoding="utf-8") as f:
                                f.write(f"> {brief_obs}\n")
                                f.write(f"After taking action '{taken_action}', valid actions: {sanitize_valid_actions(infos)}\n")

                            if "Nothing happens." in brief_obs:
                                step_trace = "".join(obs_queue)
                                if use_enhanced_errors:
                                    msg, handling = build_enhanced_error_message(step_trace, taken_action)
                                else:
                                    msg, handling = build_basic_error_message(step_trace, taken_action)
                                large_loop_error_message = msg
                                if handling == "ignore":
                                    continue
                                break

                            tem_memory += brief_obs

                            if not action_queue:
                                action_passed = True
                                successful_actions.extend(tem_action_queue)
                                overall_memory += tem_memory

                        if (within_step_tries == 5 and not action_passed) or end_game:
                            end_game = True
                            break

                    trial_record.append(trial_step_record)
                    if end_game:
                        break

                # Write results to CSV
                with open(f"output/{result_name}.csv", "a", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    data_row = [
                        today,
                        model_name,
                        mode,
                        game_type,
                        goal_type,
                        trial_idx,
                        succeed,
                        len(trial_record) - 1 if trial_record else -1,
                        trial_record[-1][-1] if trial_record and trial_record[-1] else None,
                        trial_record,
                    ]
                    writer.writerow(data_row)

                break  # Success, no retry needed

            except Exception as e:
                error_log_path = f"output/{folder_name}/errors.txt"
                os.makedirs(f"output/{folder_name}", exist_ok=True)
                with open(error_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"[{mode}] Trial {trial_idx} (Attempt {retry+1}) | "
                        f"Model: {model_name} | Type: {problem_type or 'random'} | "
                        f"Failed: {str(e)}\n"
                    )
                retry += 1
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass


def run_baseline_model_full(
    model_name: str,
    problem_type: str = None,
    start_index: int = 0,
    num_problems: int = None,
    start_trial: int = None,
    end_trial: int = None,
    split: str = "valid_train",
    folder_name: str = "alfworld_baseline",
    result_name: str = "alfworld_results",
    use_enhanced_errors: bool = True,
):
    """Unified baseline runner (direct action generation without PDDL)."""
    # Determine which problems to run (same logic as iterative)
    if problem_type and ProblemSelector:
        selector = ProblemSelector(split=split)
        problems = selector.get_problems_by_type(
            problem_type,
            max_count=num_problems or 10,
            offset=start_index
        )
        if not problems:
            print(f"No problems found for type '{problem_type}' in split '{split}'")
            return
        print(f"Running baseline on {len(problems)} {problem_type} problems from {split}")
        trials_iter = [(start_index + i, prob) for i, prob in enumerate(problems)]
    else:
        if start_trial is None:
            start_trial = start_index
        if end_trial is None:
            end_trial = start_trial + (num_problems or 10)
        trials_iter = [(t, None) for t in range(start_trial, end_trial)]

    for trial_idx, problem_info in trials_iter:
        retry = 0
        while retry < 2:
            env = None
            try:
                succeed = False
                today = date.today()

                # Prepare problem
                try:
                    if problem_info:
                        problem_path, prob_type, game_type = problem_info
                        env, expert, meta, init_scene, goal, valid_actions, infos = prepare_problem_flexible(
                            problem_path=problem_path
                        )
                    else:
                        problem_id = _pick_problem_id()
                        env, expert, meta, init_scene, goal, valid_actions, infos = prepare_problem_flexible(
                            problem_id=problem_id
                        )
                        game_type = meta.get("game_type", "unknown")
                except KeyError as e:
                    if str(e) == "'val1'":
                        print(f"Skipping trial {trial_idx} due to val1 error in PDDL parsing")
                        # Log the skip to errors file
                        error_log_path = f"output/{folder_name}/errors.txt"
                        os.makedirs(f"output/{folder_name}", exist_ok=True)
                        with open(error_log_path, "a", encoding="utf-8") as f:
                            f.write(f"[Baseline] Trial {trial_idx} | Model: {model_name} | Type: {problem_type or 'random'} | Skipped: val1 PDDL parsing error\n")
                        break  # Skip this trial completely
                    else:
                        raise

                tag = f"baseline"
                if problem_type:
                    tag += f"_{game_type}"
                file_name = make_output_file(
                    folder_name, today, model_name, tag, trial_idx,
                    game_type=game_type if problem_type else None
                )

                if os.path.exists(file_name):
                    open(file_name, "w").close()

                trial_record = []
                with open(file_name, "a", encoding="utf-8") as f:
                    f.write(f"Trial {trial_idx} - {model_name} (Baseline)\n")
                    f.write(f"Problem: {meta.get('problem_name', 'N/A')}\n")
                    f.write(f"Type: {meta.get('problem_type', 'N/A')} -> {game_type}\n")
                    f.write(f"Initial Observation: {init_scene}\n")
                    f.write(f"Goal: {goal}\n")
                    f.write(f"Valid Actions: {valid_actions}\n")

                brief_obs = "Action: look around\n" + summarize_obs(init_scene) + "\n"
                overall_memory = brief_obs
                obs_queue: List[str] = []
                successful_actions: List[str] = []
                end_game = False
                max_steps = min(20, MAX_STEPS)

                for step_id in range(max_steps):
                    with open(file_name, "a", encoding="utf-8") as f:
                        f.write(f"\n==== Step {step_id} ====\n")

                    trial_step_record = []
                    within_step_tries = 0
                    action_passed = False
                    large_loop_error_message = ""

                    while within_step_tries < 5 and not action_passed:
                        with open(file_name, "a", encoding="utf-8") as f:
                            f.write(f"\n---- Larger Loop No. {within_step_tries} ----\n")
                            f.write(f"successful_actions: {successful_actions}\n")

                        within_step_tries += 1

                        if within_step_tries > 1:
                            if env is not None:
                                try:
                                    env.close()
                                except Exception:
                                    pass
                            env, obs, infos = reset_env_with_prefix(meta["gamefile"], successful_actions, expert)
                            _scene, goal = parse_alfworld_obs(obs)

                        current_valid_actions = sanitize_valid_actions(infos)
                        if obs_queue:
                            brief_obs = "\n".join(obs_queue)
                            obs_queue = []

                        actions, prompt, raw_response = llm_to_actions_baseline(
                            model_name,
                            brief_obs,
                            current_valid_actions,
                            goal,
                            overall_memory=overall_memory,
                            large_loop_error_message=large_loop_error_message,
                        )

                        with open(file_name, "a", encoding="utf-8") as f:
                            f.write(f"=== PROMPT SENT TO LLM ===\n{prompt}\n=== END PROMPT ===\n")
                            f.write(f"=== RAW LLM RESPONSE ===\n{raw_response}\n=== END RESPONSE ===\n")
                            f.write(f"Parsed Actions: {actions}\n")

                        if not actions:
                            end_game = True
                            break

                        action_queue = list(actions)
                        tem_action_queue: List[str] = []
                        tem_memory = ""
                        tem_action_queue.extend(action_queue)

                        while action_queue:
                            taken_action = action_queue.pop(0)
                            obs, _, _, infos = env.step(taken_action)

                            if infos.get("won"):
                                succeed = True
                                end_game = True
                                with open(file_name, "a", encoding="utf-8") as f:
                                    f.write("Success! Task completed.\n")
                                break

                            action_text = "Action: " + taken_action + "\n"
                            obs_text = summarize_obs(obs) + "\n"
                            brief_obs = action_text + obs_text
                            obs_queue.append(brief_obs)

                            with open(file_name, "a", encoding="utf-8") as f:
                                f.write(f"> {brief_obs}\n")

                            if "Nothing happens." in brief_obs:
                                step_trace = "".join(obs_queue)
                                if use_enhanced_errors:
                                    msg, handling = build_enhanced_error_message(step_trace, taken_action)
                                else:
                                    msg, handling = build_basic_error_message(step_trace, taken_action)
                                large_loop_error_message = msg
                                if handling == "ignore":
                                    continue
                                break

                            tem_memory += brief_obs

                        if succeed:
                            break

                        if not action_queue and not end_game:
                            action_passed = True
                            successful_actions.extend(tem_action_queue)
                            overall_memory += tem_memory

                        trial_step_record.append(within_step_tries)
                        trial_record.append(trial_step_record)

                        if within_step_tries == 5 or end_game:
                            end_game = True
                            break

                    if end_game:
                        break

                with open(f"output/{result_name}.csv", "a", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    model_type = "baseline"
                    goal_type = "detailed"
                    data_row = [
                        today,
                        model_name,
                        model_type,
                        game_type,
                        goal_type,
                        trial_idx,
                        succeed,
                        len(trial_record) - 1 if trial_record else -1,
                        trial_record[-1][-1] if trial_record and trial_record[-1] else None,
                        trial_record,
                    ]
                    writer.writerow(data_row)

                break

            except Exception as e:
                error_log_path = f"output/{folder_name}/errors.txt"
                os.makedirs(f"output/{folder_name}", exist_ok=True)
                with open(error_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"[Baseline] Trial {trial_idx} (Attempt {retry+1}) | "
                        f"Model: {model_name} | Type: {problem_type or 'random'} | "
                        f"Failed: {str(e)}\n"
                    )
                retry += 1
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass


# =============================================================
# Backward compatibility wrappers
# =============================================================

def run_iterative_model(
    model_name: str,
    start_trial: int = 0,
    end_trial: int = 11,
    folder_name: str = "alfworld_runs",
    result_name: str = "alfworld_results",
    goal_type: str = "detailed",
):
    """Legacy wrapper for random problem selection."""
    return run_iterative_model_full(
        model_name=model_name,
        start_trial=start_trial,
        end_trial=end_trial,
        folder_name=folder_name,
        result_name=result_name,
        goal_type=goal_type,
        mode="PDDL",
        use_enhanced_errors=False,  # Keep legacy behavior
    )


def run_baseline_model(
    model_name: str,
    start_trial: int = 0,
    end_trial: int = 5,
    folder_name: str = "alfworld_runs",
    result_name: str = "alfworld_results",
):
    """Legacy wrapper for baseline with random selection."""
    return run_baseline_model_full(
        model_name=model_name,
        start_trial=start_trial,
        end_trial=end_trial,
        folder_name=folder_name,
        result_name=result_name,
        use_enhanced_errors=False,
    )


def run_iterative_model_initDF(
    model_name: str,
    start_trial: int,
    end_trial: int,
    folder_name: str,
    result_name: str,
    initial_df: str,
    goal_type: str = "detailed",
    mode: str = "initDF",
):
    """Legacy wrapper for initDF mode."""
    return run_iterative_model_full(
        model_name=model_name,
        start_trial=start_trial,
        end_trial=end_trial,
        folder_name=folder_name,
        result_name=result_name,
        goal_type=goal_type,
        initial_df=initial_df,
        mode=f"PDDL_{mode}",
        use_enhanced_errors=False,
    )


def run_iterative_model_typed(
    model_name: str,
    problem_type: str = "all",
    start_index: int = 0,
    num_problems: int = 50,
    split: str = "valid_train",
    folder_name: str = "alfworld_typed",
    result_name: str = "alfworld_results",
    goal_type: str = "detailed",
):
    """Enhanced wrapper for typed problem selection."""
    return run_iterative_model_full(
        model_name=model_name,
        problem_type=problem_type,
        start_index=start_index,
        num_problems=num_problems,
        split=split,
        folder_name=folder_name,
        result_name=result_name,
        goal_type=goal_type,
        mode="PDDL",
        use_enhanced_errors=True,
    )


def run_baseline_model_typed(
    model_name: str,
    problem_type: str = "all",
    start_index: int = 0,
    num_problems: int = 50,
    split: str = "valid_train",
    folder_name: str = "alfworld_baseline_typed",
    result_name: str = "alfworld_results",
):
    """Enhanced wrapper for baseline with typed selection."""
    return run_baseline_model_full(
        model_name=model_name,
        problem_type=problem_type,
        start_index=start_index,
        num_problems=num_problems,
        split=split,
        folder_name=folder_name,
        result_name=result_name,
        use_enhanced_errors=True,
    )


def run_iterative_model_initDF_typed(
    model_name: str,
    initial_df: str,
    problem_type: str = "all",
    start_index: int = 0,
    num_problems: int = 50,
    split: str = "valid_train",
    folder_name: str = "alfworld_initDF_typed",
    result_name: str = "alfworld_results",
    goal_type: str = "detailed",
    mode: str = "initDF",
):
    """Enhanced wrapper for initDF with typed selection."""
    return run_iterative_model_full(
        model_name=model_name,
        problem_type=problem_type,
        start_index=start_index,
        num_problems=num_problems,
        split=split,
        folder_name=folder_name,
        result_name=result_name,
        goal_type=goal_type,
        initial_df=initial_df,
        mode=f"PDDL_{mode}",
        use_enhanced_errors=True,
    )


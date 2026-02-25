import os
import time
from datetime import date
import csv
import json
import re
import json

from dotenv import load_dotenv
load_dotenv()

from textworld_express import TextWorldExpressEnv

from openai import OpenAI
from .config import ENV_PARAMS, MAX_STEPS, OPENAI_MODELS_LIST
from .utils import (detect_duplicates, get_action_from_pddl, map_env_feedback_to_large_loop_error, extract_json, sanitize_valid_actions, summarize_obs)
from .prompts import *

from pydantic import BaseModel
from typing import List

class PDDLResponse(BaseModel):
    df: str
    pf: str

class ActionResponse(BaseModel):
    actions: List[str]
    

def run_llm(prompt: str, model_name: str, system_prompt, response_model = None) -> tuple:
    messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]

    raw_response = None 

    if "deepseek" in model_name.lower():
        api_key = open(f'../../../_private/key_deepseek.txt').read()
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages,
            response_format={'type': 'json_object'}
        )
        response_content = response.choices[0].message.content
        raw_response = response_content  # Capture raw response

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
        params = {"model": model_name,
                "input": messages,
                "reasoning": {"effort": "high"},
                "text_format": response_model,}

        if any(model_name.startswith(base) for base in OPENAI_MODELS_LIST):
            api_key = open(f'../../../_private/key_pddlego.txt').read()
            client = OpenAI(api_key=api_key)
            if model_name.startswith("gpt-"):
                params.pop("reasoning", None)
        elif "gemini" in model_name:
            api_key = open(f'../../../_private/key_gemini.txt').read()
            client = OpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
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



def llm_to_pddl(model_name, brief_obs, valid_actions, prev_df="", prev_pf="", prev_err="", have_duplicate=False, overall_memory=None, large_loop_error_message = None, goal_type='detailed', unit=False, expected_action=""):
    prompt = prompt_format.format()

    if goal_type == 'detailed':
        prompt += prompt_obs_action_detailed.format(brief_obs=brief_obs, valid_actions=valid_actions)
    elif goal_type == 'subgoal':
        prompt += prompt_obs_action_subgoal.format(brief_obs=brief_obs, valid_actions=valid_actions)

    if prev_df or prev_pf:
        prompt += prompt_prev_files.format(prev_df=prev_df or "N/A", prev_pf=prev_pf or "N/A", overall_memory=overall_memory or "N/A")

        if large_loop_error_message or prev_err or have_duplicate:
            if large_loop_error_message:
                prompt += prompt_simulation_error.format(large_loop_error_message=large_loop_error_message)
            if prev_err:
                prompt += prompt_error_parser.format(prev_err=prev_err)
            if have_duplicate:
                prompt += prompt_duplicate_note.format()

            prompt += "\nNow update the files based on the new observations and feedback. Make minimal but effective changes to fix the issues.\n"
        
        else:
            prompt += "\nNow generate updated files based on the new observations.\n"
    
    if unit and expected_action:
        prompt += unit_goal_reminder.format(expected_action=expected_action)


    resp, raw_response = run_llm(prompt, model_name, system_prompt=SYS_PROMPT_PDDL, response_model=PDDLResponse)
    df = resp['df']
    pf = resp['pf']
    return df, pf, prompt, raw_response


def llm_to_actions_baseline(model_name, brief_obs, valid_actions, overall_memory=None, large_loop_error_message=None):
    prompt = prompt_baseline.format(
        brief_obs=brief_obs,
        valid_actions=valid_actions,
        overall_memory=overall_memory or "N/A",
        large_loop_error_message=large_loop_error_message or "N/A",
    )

    resp, raw_response = run_llm(prompt, model_name, system_prompt=SYS_PROMPT_PLAN, response_model=ActionResponse)
    actions = resp['actions']
    return actions, prompt, raw_response


def run_iterative_model(model_name, start_trial = 0, end_trial = 11, folder_name="3_0421_CC", result_name="CC_results"):

    env = TextWorldExpressEnv(envStepLimit=100)
    env.load(**ENV_PARAMS)

    adjusted_end = end_trial
    trial = start_trial
    while trial < adjusted_end:
        start_time = time.time()
        retry = 0
        while retry < 2: 
            try:
                coin_found = False
                today = date.today()

                fixed_model_name = model_name.replace("/","_")

                folder_path = f"output/{folder_name}"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                file_name = f"{folder_path}/{today}_{fixed_model_name}_PDDL_{trial}.txt"
                
                if os.path.exists(file_name): 
                    open(file_name, 'w').close()  
                    print(f"[Trial {trial}] Retrying: cleared file and retrying...")

                trial_record = []
                
                obs, infos = env.reset(seed=trial, gameFold="train", generateGoldPath=True)
                valid_actions = sanitize_valid_actions(infos)

                if "coin" in obs.lower():
                    print(f"[Seed {trial}] Coin found at the beginning — skipping this seed and extending the trial window.")
                    adjusted_end += 1
                    break 
 
                with open(file_name, "a") as f:  
                    f.write(f"Observations: {obs} \n") 
                    f.write(f"Gold path: {env.getGoldActionSequence()} \n")
                    f.write(f"Valid Actions: {valid_actions} \n")
                    f.write(f"taskDescription: {infos['taskDescription']} \n")


                brief_obs = "Action: look around\n" + summarize_obs(obs)+'\n' 
                with open(file_name, "a") as f:
                    f.write(f"brief_obs: {brief_obs} \n") 
                # print(brief_obs)

                action_queue = []
                obs_queue = []
                df = ""
                pf = ""
                all_actions = []
                successful_actions = []
                end_game = False

                overall_memory = brief_obs

                for step_id in range(MAX_STEPS):
                    with open(file_name, "a") as f:
                        f.write(f"\n\n====Step {step_id}==== \n")

                    trial_step_record = []
                    within_step_tries = 0
                    action_passed = False
                    large_loop_error_message = ""


                    while within_step_tries < 5 and not action_passed:
                        with open(file_name, "a") as f:
                            f.write(f'\n----Larger Loop No. {within_step_tries}---- \n') 
                            f.write(f'successful_actions: {successful_actions} \n')

                        within_step_tries += 1

                        if within_step_tries > 1:
                            obs, infos = env.reset(seed=trial, gameFold="train", generateGoldPath=True) # obs, infos
                            for successful_action in successful_actions: # ,
                                obs, reward, done, infos = env.step(successful_action) # <-... action
                            valid_actions = sanitize_valid_actions(infos) # . . -> no, . .

                        action_queue = [] # reset action_queue ()
                        # tem_action_queue = []
                        # tem_memory = ""

                        start_checkpoint = True

                        # executed_actions = [] 
                        # executed_memory = ""

                        while start_checkpoint or action_queue:
                            with open(file_name, "a") as f:
                                f.write(f'Small Loop, action_queue: {action_queue} \n')
                            start_checkpoint = False

                            if not action_queue: 
                                if obs_queue:
                                    brief_obs = "\n".join(obs_queue)
                                    obs_queue = []
                                action = ""
                                
                                if not df and not pf: # First step no need duplicates detection
                                    num_tries = 0
                                    df, pf, prompt, raw_response = llm_to_pddl(model_name, brief_obs, valid_actions) # error 1 here
                                    action, err, plan_text = get_action_from_pddl(df, pf) # error 2 here
                                    with open(file_name, "a") as f:
                                        f.write(f"--Small Loop--: {num_tries} \n")
                                        f.write(f"=== PROMPT SENT TO LLM ===\n{prompt}\n=== END PROMPT ===\n")
                                        f.write(f"=== RAW LLM RESPONSE ===\n{raw_response}\n=== END RESPONSE ===\n")
                                        f.write(f"Parsed df and pf: \n {df} \n {pf} \n")
                                        f.write(f"Actions from solver(df, pf): {action} \n")
                                        f.write(f"Raw plan text: {plan_text} \n")

                                    while not action and num_tries < 5:
                                        df, pf, prompt, raw_response = llm_to_pddl(model_name, brief_obs, valid_actions, df, pf, err, False)
                                        action, err, plan_text = get_action_from_pddl(df, pf)
                                        num_tries += 1

                                        with open(file_name, "a") as f:
                                            f.write(f"--Small Loop--: {num_tries} \n")
                                            f.write(f"=== PROMPT SENT TO LLM ===\n{prompt}\n=== END PROMPT ===\n")
                                            f.write(f"=== RAW LLM RESPONSE ===\n{raw_response}\n=== END RESPONSE ===\n")
                                            f.write(f"Parsed df and pf: \n {df} \n {pf} \n")
                                            f.write(f"Actions from solver(df, pf): {action} \n")
                                            f.write(f"Raw plan text: {plan_text} \n")
                                else:
                                    num_tries = 0
                                    df, pf, prompt, raw_response = llm_to_pddl(model_name, brief_obs, valid_actions, df, pf, "", detect_duplicates(all_actions, 3), overall_memory, large_loop_error_message) # need to add new error message
                                    action, err, plan_text = get_action_from_pddl(df, pf)

                                    with open(file_name, "a") as f:
                                        f.write(f"--Small Loop--: {num_tries} \n")
                                        f.write(f"=== PROMPT SENT TO LLM ===\n{prompt}\n=== END PROMPT ===\n")
                                        f.write(f"=== RAW LLM RESPONSE ===\n{raw_response}\n=== END RESPONSE ===\n")
                                        f.write(f"Parsed df and pf: \n {df} \n {pf} \n")
                                        f.write(f"Actions from solver(df, pf): {action} \n")
                                        f.write(f"Raw plan text: {plan_text} \n")

                                    while not action and num_tries < 5:
                                        df, pf, prompt, raw_response = llm_to_pddl(model_name, brief_obs, valid_actions, df, pf, err, detect_duplicates(all_actions, 3), overall_memory, large_loop_error_message)
                                        action, err, plan_text = get_action_from_pddl(df, pf)
                                        num_tries += 1

                                        with open(file_name, "a") as f:
                                            f.write(f"--Small Loop--: {num_tries} \n")
                                            f.write(f"=== PROMPT SENT TO LLM ===\n{prompt}\n=== END PROMPT ===\n")
                                            f.write(f"=== RAW LLM RESPONSE ===\n{raw_response}\n=== END RESPONSE ===\n")
                                            f.write(f"Parsed df and pf: \n {df} \n {pf} \n")
                                            f.write(f"Actions from solver(df, pf): {action} \n")
                                            f.write(f"Raw plan text: {plan_text} \n")

                      
                                trial_step_record.append([within_step_tries, num_tries])

                                if action:
                                    action_queue.extend(action)
                                    # tem_action_queue.extend(action) 
                                    all_actions.extend(action) # to detect duplicated
                                else:
                                    end_game = True
                                    break

                            with open(file_name, "a") as f:
                                f.write(f"Current action_queue: {action_queue} \n")
                            
                            taken_action = action_queue.pop(0)

                            obs, reward, done, infos = env.step(taken_action)
                            valid_actions = sanitize_valid_actions(infos)

                            if "coin" in obs:
                                taken_action = "take coin"
                                obs, reward, done, infos = env.step(taken_action)
                                valid_actions = sanitize_valid_actions(infos)
                                end_game = True
                                with open(file_name, "a") as f:
                                    f.write('Coin found!\n')
                                    f.write(f"Final obs: {obs} \n")
                                    coin_found = True
                                break
                            
                            action_text = "Action: " + taken_action + "\n"
                            obs_text = summarize_obs(obs) + "\n"

                            brief_obs = action_text + obs_text

                            obs_queue.append(brief_obs)
                            with open(file_name, "a") as f:
                                f.write(f"> {taken_action} \n {obs} \n")

                            large_loop_error_message, _code = map_env_feedback_to_large_loop_error(brief_obs, taken_action)
            
                            if _code:
                                action_passed = False
                                break

                            # # append into overall memory and dictionary format
                            # tem_memory += brief_obs

                            # # It should be the last step and passed all actions
                            # if not action_queue:
                            #     action_passed = True
                            #     successful_actions.extend(tem_action_queue)
                            #     overall_memory += tem_memory
                            else:
                                successful_actions.append(taken_action)
                                overall_memory += brief_obs
                                if not action_queue:
                                    action_passed = True

                        if (within_step_tries == 5 and not action_passed) or end_game:
                            end_game = True
                            break

                    trial_record.append(trial_step_record)

                    if end_game:
                        break
                duration = time.time() - start_time
                with open(f"output/{result_name}.csv", "a", newline="") as csvfile:
                    # date, model_name, trial, failed at step #, [large loop, small loop], detailed loop info
                    model_type = "PDDL"
                    data_row = [today, model_name, model_type, trial, coin_found, duration, len(trial_record)-1,trial_record[-1][-1], trial_record]
                    # [today, model_name, trial, coin_found, len(trial_record)-1,trial_record[-1][-1], trial_record]
                    writer = csv.writer(csvfile)
                    writer.writerow(data_row)
                break

            except Exception as e:
                error_log_path = f"output/{folder_name}/errors.txt"
                with open(error_log_path, "a") as f:
                    log_message = (
                        f"[PDDLego+] Trial {trial} (Attempt {retry+1}) | "
                        f"Model: {model_name} | "
                        f"Failed: {str(e)}\n"
                    )
                    f.write(log_message)
                retry += 1
        trial += 1


def run_baseline_model(model_name, start_trials, end_trials, folder_name="08_031825_alfworld", result_name="alfworld_results"):
    env = TextWorldExpressEnv(envStepLimit=100)
    env.load(**ENV_PARAMS)

    adjusted_end = end_trials
    trial = start_trials
    while trial < adjusted_end:
        start_time = time.time()
        retry = 0
        while retry < 2:  
            try:
                coin_found = False
                today = date.today()

                fixed_model_name = model_name.replace("/","_")

                folder_path = f"output/{folder_name}"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                file_name = f"{folder_path}/{today}_{fixed_model_name}_baseline_{trial}.txt"
                if os.path.exists(file_name): 
                    open(file_name, 'w').close() 
                    print(f"[Trial {trial}] Retrying: cleared file and retrying...")

                trial_record = []

                obs, infos = env.reset(seed=trial, gameFold="train", generateGoldPath=True)
                valid_actions = sanitize_valid_actions(infos)

                if "coin" in obs.lower():
                    print(f"[Seed {trial}] Coin found at the beginning — skipping this seed and extending the trial window.")
                    adjusted_end += 1
                    break 
                with open(file_name, "a") as f:
                    f.write(f"Observations: {obs} \n")
                    f.write(f"Gold path: {env.getGoldActionSequence()} \n")
                    f.write(f"Valid Actions: {valid_actions} \n")
                    f.write(f"taskDescription: {infos['taskDescription']} \n")

                brief_obs = "Action: look around\n" + summarize_obs(obs) + "\n"
                with open(file_name, "a") as f:
                    f.write(f"brief_obs: {brief_obs} \n")

                action_queue = []
                obs_queue = []
                all_actions = []
                successful_actions = []
                overall_memory = brief_obs
                overall_memory_dic = []  
                end_game = False

                for step_id in range(MAX_STEPS):
                    with open(file_name, "a") as f:
                        f.write(f"\n\n====Step {step_id}==== \n")
                    trial_step_record = [] 
                    within_step_tries = 0
                    action_passed = False
                    large_loop_error_message = ""

                    while within_step_tries < 5 and not action_passed:
                        with open(file_name, "a") as f:
                            f.write(f"\n----Larger Loop No. {within_step_tries}---- \n")
                            f.write(f"successful_actions: {successful_actions} \n")
                        within_step_tries += 1

                        if within_step_tries > 1:  
                            obs, infos = env.reset(seed=trial, gameFold="train", generateGoldPath=True)
                            for act in successful_actions:
                                obs, reward, done, infos = env.step(act)
                            valid_actions = sanitize_valid_actions(infos)

            
                        action_queue = []
                        tem_action_queue = []
                        tem_memory = ""

                        start_checkpoint = True
                        while start_checkpoint or action_queue:
                            with open(file_name, "a") as f:
                                f.write(f"Small Loop, action_queue: {action_queue} \n")
                            start_checkpoint = False

                            if not action_queue:
                                if obs_queue:
                                    brief_obs = "\n".join(obs_queue)
                                    obs_queue = []
                                # Generate actions using the baseline LLM function.
                                actions, prompt, raw_response = llm_to_actions_baseline(model_name, brief_obs, valid_actions, overall_memory, large_loop_error_message)
                                with open(file_name, "a") as f:
                                    f.write(f"=== PROMPT SENT TO LLM ===\n{prompt}\n=== END PROMPT ===\n")
                                    f.write(f"=== RAW LLM RESPONSE ===\n{raw_response}\n=== END RESPONSE ===\n")
                                    f.write(f"Parsed actions: {actions} \n")

                                if actions:
                                    action_queue.extend(actions)
                                    tem_action_queue.extend(actions)
                                    all_actions.extend(actions)
                                else:
                                    end_game = True
                                    break

                            with open(file_name, "a") as f:
                                f.write(f"Current action_queue: {action_queue} \n")
                            taken_action = action_queue.pop(0)
                            obs, reward, done, infos = env.step(taken_action)
                            valid_actions = sanitize_valid_actions(infos)

                            if "coin" in obs:
                                taken_action = "take coin"
                                obs, reward, done, infos = env.step(taken_action)
                                valid_actions = sanitize_valid_actions(infos)
                                end_game = True
                                with open(file_name, "a") as f:
                                    f.write("Coin found!\n")
                                    f.write(f"Final obs: {obs} \n")
                                coin_found = True
                                break

                            action_text = "Action: " + taken_action + "\n"
                            obs_text = summarize_obs(obs) + "\n"
                            brief_obs = action_text + obs_text
                            obs_queue.append(brief_obs)
                            with open(file_name, "a") as f:
                                f.write(f"> {taken_action} \n {obs} \n")
                            
                            msg, _code = map_env_feedback_to_large_loop_error(brief_obs, taken_action)
                            if msg:
                                large_loop_error_message = msg
                                with open(file_name, "a") as f:
                                    f.write(f"Large loop error message: {large_loop_error_message} \n")
                                break

                            tem_memory += brief_obs
                            overall_memory_dic.append({"type": "action", "content": taken_action})
                            overall_memory_dic.append({"type": "observation", "content": summarize_obs(obs)})

                            if not action_queue:
                                action_passed = True
                                successful_actions.extend(tem_action_queue)
                                overall_memory += tem_memory

                        trial_step_record.append(within_step_tries)
                        if (within_step_tries == 5 and not action_passed) or end_game:
                            end_game = True
                            break

                    trial_record.append(trial_step_record)
                    if end_game:
                        break

                duration = time.time() - start_time
                with open(f"output/{result_name}.csv", "a", newline="") as csvfile:
                    # Write out: date, model_name, trial, coin_found, last step index, last large-loop iteration, and the full trial record.
                    # data_row = [today, model_name, trial, coin_found, len(trial_record)-1, trial_record[-1] if trial_record else None, trial_record]
                    model_type = 'baseline' # PDDL
                    data_row = [today, model_name, model_type, trial, coin_found, duration, len(trial_record)-1,trial_record[-1][-1], trial_record]

                    writer = csv.writer(csvfile)
                    writer.writerow(data_row)
                break

            except Exception as e:
                error_log_path = f"output/{folder_name}/errors.txt"
                with open(error_log_path, "a") as f:
                    log_message = (
                        f"[Baseline] Trial {trial} (Attempt {retry+1}) | "
                        f"Model: {model_name} | "
                        f"Failed: {str(e)}\n"
                    )
                    f.write(log_message)
                retry += 1
        trial += 1


def run_iterative_model_initDF(model_name, start_trial, end_trial, folder_name, result_name, initial_df, goal_type="detailed", mode="unknown"):
    
    env = TextWorldExpressEnv(envStepLimit=100)
    env.load(**ENV_PARAMS)

    adjusted_end = end_trial
    trial = start_trial
    while trial < adjusted_end:
        start_time = time.time()
        retry = 0
        while retry < 2:
            try:
                coin_found = False
                today = date.today()

                fixed_model_name = model_name.replace("/","_")

                folder_path = f"output/{folder_name}"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                file_name = f"{folder_path}/{today}_{fixed_model_name}_PDDL_Unit_{mode}_{goal_type}_{trial}.txt"

                if os.path.exists(file_name): 
                    open(file_name, 'w').close()  
                    print(f"[Trial {trial}] Retrying: cleared file and retrying...")

                trial_record = []

      
                obs, infos = env.reset(seed=trial, gameFold="train", generateGoldPath=True)
                valid_actions = sanitize_valid_actions(infos)

                if "coin" in obs.lower():
                    print(f"[Seed {trial}] Coin found at the beginning — skipping this seed and extending the trial window.")
                    adjusted_end += 1
                    break  

                with open(file_name, "a") as f:  
                    f.write(f"Observations: {obs} \n")
                    f.write(f"Gold path: {env.getGoldActionSequence()} \n")
                    f.write(f"Valid Actions: {valid_actions} \n")
                    f.write(f"taskDescription: {infos['taskDescription']} \n")


                brief_obs = "Action: look around\n" + summarize_obs(obs)+'\n' 
                with open(file_name, "a") as f:
                    f.write(f"brief_obs: {brief_obs} \n")
                # print(brief_obs)

                action_queue = []
                obs_queue = []
                df = ""
                pf = ""
                all_actions = []
                successful_actions = []
                end_game = False

                overall_memory = brief_obs

                for step_id in range(MAX_STEPS):
                    with open(file_name, "a") as f:
                        f.write(f"\n\n====Step {step_id}==== \n")

                    trial_step_record = []
                    within_step_tries = 0
                    action_passed = False
                    large_loop_error_message = ""

                    while within_step_tries < 5 and not action_passed:
                        with open(file_name, "a") as f:
                            f.write(f'\n----Larger Loop No. {within_step_tries}---- \n')
                            f.write(f'successful_actions: {successful_actions} \n')

                        within_step_tries += 1

                        if within_step_tries > 1: 
                            obs, infos = env.reset(seed=trial, gameFold="train", generateGoldPath=True)
                            for successful_action in successful_actions:
                                obs, reward, done, infos = env.step(successful_action)
                            valid_actions = sanitize_valid_actions(infos)

                        action_queue = [] # reset action_queue ()


                        start_checkpoint = True
                        while start_checkpoint or action_queue:
                            with open(file_name, "a") as f:
                                f.write(f'Small Loop, action_queue: {action_queue} \n')
                            start_checkpoint = False

                            if not action_queue:
                                if obs_queue:
                                    brief_obs = "\n".join(obs_queue)
                                    obs_queue = []
                                action = ""

                                if not df and not pf: # First step no need duplicates detection
                                    num_tries = 0

                                    first_generation_tries = 0
                                    while (not pf) and first_generation_tries < 3:
                                        prompt_first = prompt_format.format() + prompt_obs_action_initial_df.format(brief_obs=brief_obs, valid_actions=valid_actions, initial_df=initial_df)
                                        resp1, raw_response = run_llm(prompt_first, model_name, system_prompt=SYS_PROMPT_PDDL, response_model=PDDLResponse)
                                        df = resp1['df']
                                        pf = resp1['pf']

                                        with open(file_name, "a") as f:
                                            f.write(f"--First Generation Try--: {first_generation_tries} \n")
                                            f.write(f"Initial df: {initial_df} \n")
                                            f.write(f"=== PROMPT SENT TO LLM ===\n{prompt_first}\n=== END PROMPT ===\n")
                                            f.write(f"=== RAW LLM RESPONSE ===\n{raw_response}\n=== END RESPONSE ===\n")
                                            f.write(f"Generated df: \n {df} \n")
                                            f.write(f"Generated pf: \n {pf} \n")

                                        first_generation_tries += 1

                                    action, err, plan_text = get_action_from_pddl(df, pf) # error 2 here

                                    with open(file_name, "a") as f:
                                        f.write(f"--Small Loop--: {num_tries} \n")
                                        f.write(f"Actions from solver(df, pf): {action} \n")
                                        f.write(f"Raw plan text: {plan_text} \n")

                                    while not action and num_tries < 5:
                                        df, pf, prompt, raw_response = llm_to_pddl(model_name, brief_obs, valid_actions, df, pf, err, False, goal_type=goal_type)
                                        action, err, plan_text = get_action_from_pddl(df, pf)
                                        num_tries += 1

                                        with open(file_name, "a") as f:
                                            f.write(f"--Small Loop--: {num_tries} \n")
                                            f.write(f"=== PROMPT SENT TO LLM ===\n{prompt}\n=== END PROMPT ===\n")
                                            f.write(f"=== RAW LLM RESPONSE ===\n{raw_response}\n=== END RESPONSE ===\n")
                                            f.write(f"Parsed df and pf: \n {df} \n {pf} \n")
                                            f.write(f"Actions from solver(df, pf): {action} \n")
                                            f.write(f"Raw plan text: {plan_text} \n")
                                else:
                                    num_tries = 0
                                    df, pf, prompt, raw_response = llm_to_pddl(model_name, brief_obs, valid_actions, df, pf, "", detect_duplicates(all_actions, 3), overall_memory, large_loop_error_message, goal_type=goal_type) # need to add new error message
                                    action, err, plan_text = get_action_from_pddl(df, pf)

                                    with open(file_name, "a") as f:
                                        f.write(f"--Small Loop--: {num_tries} \n")
                                        f.write(f"=== PROMPT SENT TO LLM ===\n{prompt}\n=== END PROMPT ===\n")
                                        f.write(f"=== RAW LLM RESPONSE ===\n{raw_response}\n=== END RESPONSE ===\n")
                                        f.write(f"Parsed df and pf: \n {df} \n {pf} \n")
                                        f.write(f"Actions from solver(df, pf): {action} \n")
                                        f.write(f"Raw plan text: {plan_text} \n")

                                    while not action and num_tries < 5:
                                        df, pf, prompt, raw_response = llm_to_pddl(model_name, brief_obs, valid_actions, df, pf, err, detect_duplicates(all_actions, 3), overall_memory, large_loop_error_message, goal_type=goal_type)
                                        action, err, plan_text = get_action_from_pddl(df, pf)
                                        num_tries += 1

                                        with open(file_name, "a") as f:
                                            f.write(f"--Small Loop--: {num_tries} \n")
                                            f.write(f"=== PROMPT SENT TO LLM ===\n{prompt}\n=== END PROMPT ===\n")
                                            f.write(f"=== RAW LLM RESPONSE ===\n{raw_response}\n=== END RESPONSE ===\n")
                                            f.write(f"Parsed df and pf: \n {df} \n {pf} \n")
                                            f.write(f"Actions from solver(df, pf): {action} \n")
                                            f.write(f"Raw plan text: {plan_text} \n")

                                trial_step_record.append([within_step_tries, num_tries])

                                if action:
                                    action_queue.extend(action)
                                    # tem_action_queue.extend(action) # temporary action queue to put in successful_actions
                                    all_actions.extend(action) # to detect duplicated
                                else:
                                    end_game = True
                                    break

                            with open(file_name, "a") as f:
                                f.write(f"Current action_queue: {action_queue} \n")

                            taken_action = action_queue.pop(0)

                            obs, reward, done, infos = env.step(taken_action)
                            valid_actions = sanitize_valid_actions(infos)

                            if "coin" in obs:
                                taken_action = "take coin"
                                obs, reward, done, infos = env.step(taken_action)
                                valid_actions = sanitize_valid_actions(infos)
                                end_game = True
                                with open(file_name, "a") as f:
                                    f.write('Coin found!\n')
                                    f.write(f"Final obs: {obs} \n")
                                    coin_found = True
                                break

                            action_text = "Action: " + taken_action + "\n"
                            obs_text = summarize_obs(obs) + "\n"

                            brief_obs = action_text + obs_text

                            obs_queue.append(brief_obs)
                            with open(file_name, "a") as f:
                                f.write(f"> {taken_action} \n {obs} \n")

                            large_loop_error_message, _code = map_env_feedback_to_large_loop_error(brief_obs, taken_action)

                            if _code:
                                action_passed = False
                                break

                            else:
                                successful_actions.append(taken_action)
                                overall_memory += brief_obs
                                if not action_queue:
                                    action_passed = True

                        if (within_step_tries == 5 and not action_passed) or end_game:
                            end_game = True
                            break

                    trial_record.append(trial_step_record)

                    if end_game:
                        break

                duration = time.time() - start_time
                with open(f"output/{result_name}.csv", "a", newline="") as csvfile:
                    model_type = f"PDDL_Unit_{mode}"
                    data_row = [today, model_name, model_type, trial, coin_found, duration, len(trial_record)-1,trial_record[-1][-1], trial_record]
                    writer = csv.writer(csvfile)
                    writer.writerow(data_row)
                break

            except Exception as e:
                error_log_path = f"output/{folder_name}/errors.txt"
                with open(error_log_path, "a") as f:
                    log_message = (
                        f"[PDDL_Unit_{mode}] Trial {trial} (Attempt {retry+1}) | "
                        f"Model: {model_name} | Mode: {mode} | Goal Type: {goal_type} | "
                        f"Failed: {str(e)}\n"
                    )
                    f.write(log_message)
                retry += 1
        trial += 1
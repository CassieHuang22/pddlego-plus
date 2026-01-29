# =========================
# System prompts
# =========================

SYS_PROMPT_PLAN = """
You will be given a naturalistic domain description and problem description.
Your task is to generate a plan (a series of actions).
"""
SYS_PROMPT_PDDL = """
You will be given a naturalistic domain description and problem description. 
Your task is to generate domain file and problem file in Planning Domain Definition Language (PDDL) with appropriate tags.
"""


# =========================
# Common formatting blocks
# =========================

prompt_format = """
Please provide the output in strict JSON format, without any additional text or explanation, including a PDDL domain file as 'df' and a PDDL problem file as 'pf'. 
The format should strictly be:
    {{
    "df": "...",
    "pf": "..."
    }}
"""


prompt_prev_files = """
This is all the memory you have in this game including each action and its corresponding observations: 
{overall_memory}

You have already generated df and pf files according to the observations.
This is previous domain file: 
{prev_df}

This is previous problem file: 
{prev_pf}

"""

prompt_new_obs = """
Now modify those domain and problem files according to the new observations and notes. Fix any errors you made in the previous setting according to the new observation.
Generate updated files based on your new observation.

"""


prompt_error_parser = """
You made some mistakes when generating those files. Here is the error message: 
{prev_err}

"""



# error from simulation environment
prompt_simulation_error = """
Based on the df and pf that you generated, the external solver could generate a plan but after simulating in the game environment, it caused those errors: 
{large_loop_error_message}

"""

prompt_duplicate_note = """
You are repeating the same sequence of actions for at least three times. You may stuck in one location or have the wrong goal.
You should revise your problem file to avoid the repeat.
Remember your goal is always to keep exploration and go to a location you have not visited yet, i.e. your goal should be go to other not visited location but shouldn't be at one fixed location.

"""


# =========================
# PDDL generation (ALFWorld)
# =========================

# NOTE: The solver->env mapper expects the plan to contain these action names.
#       The LLM must define and use them.

# PDDL_ACTION_SPECS = """
# Your DF MUST define actions with these exact names (case-insensitive is okay, but keep spelling):

# 1) GOTOLOCATION
#    :parameters (?from - receptacle ?to - receptacle)
#    - Precondition: agent is at ?from
#    - Effect: agent is at ?to (and not at ?from)

# 2) OPENOBJECT
#    :parameters (?r - receptacle)
#    - Precondition: receptacle is closed
#    - Effect: receptacle is open

# 3) CLOSEOBJECT
#    :parameters (?r - receptacle)
#    - Precondition: receptacle is open
#    - Effect: receptacle is closed

# 4) PICKUPOBJECT
#    :parameters (?o - object ?r - receptacle)
#    - Precondition: object is in receptacle, agent is at receptacle
#    - Effect: agent is holding object (object not in receptacle)

# 5) PUTOBJECT
#    :parameters (?o - object ?r - receptacle)
#    - Precondition: agent is holding object, agent is at receptacle
#    - Effect: object is in receptacle (agent not holding object)

# 6) USEOBJECT
#    :parameters (?o - object)
#    - For turning objects on/off (e.g., lamps)
#    - Precondition: agent is at object location or holding object

# 7) HEATOBJECT
#    :parameters (?o - object ?r - microwaveReceptacle)
#    - Precondition: agent holding object, agent at microwave
#    - Effect: object is heated

# 8) CLEANOBJECT
#    :parameters (?o - object ?r - sinkbasinReceptacle)
#    - Precondition: agent holding object, agent at sinkbasin
#    - Effect: object is clean

# 9) COOLOBJECT
#    :parameters (?o - object ?r - fridgeReceptacle)
#    - Precondition: agent holding object, agent at fridge
#    - Effect: object is cooled

# 10) SLICEOBJECT
#     :parameters (?r - receptacle ?co - object ?sharp_o - sharpObject)
#     - Precondition: agent holding sharp object, object is at receptacle
#     - Effect: object is sliced

# IMPORTANT USAGE NOTES:
# - You must go to a receptacle first before you can use/open it or take/put objects
# - For heat/clean/cool: Go to appliance then apply action directly (don't put object in)
# - For slicing: Pick up sharp object ONLY, then slice (don't pick up target object)
# - Some receptacles cannot be opened (directly visible after going to them)

# Keep DF compact. Use typing and predicates such as:
#   (at ?agent ?receptacle), (open ?r), (closed ?r), (in ?o ?r),
#   (holding ?agent ?o), (heated ?o), (clean ?o), (cooled ?o), (sliced ?o)

# PF should contain ONLY grounded objects and facts that are supported by observations.
# Goal examples: (at cabinet3), (open drawer2), (holding cloth1), (clean fork1)
# """

PDDL_ACTION_SPECS = """
The following actions are allowed: 
1. go to a receptacle
    :action GotoLocation
    :parameters (?from - receptacle ?to - receptacle)
2. open a receptacle if it is closed
    :action OpenObject
    :parameters (?r - receptacle)
3. close a receptacle
    :action CloseObject
    :parameters (?r - receptacle)
4. take an object from another receptacle
    :action PickupObject
    :parameters (?o - object ?r - receptacle)
5. put object into/on/in another receptacle
    :action PutObject
    :parameters (?o - object ?r - receptacle)
6. using an object/receptacle by turning it on/off with a switch
    :action useObject
    :parameters (?o - object)
7. heat an object using a receptacle
    :action HeatObject
    :parameters (?o - object ?r - microwaveReceptacle)
8. clean an object using a receptacle
    :action CleanObject
    :parameters (?o - object ?r - receptacle)
9. cool an object using a receptacle
    :action CoolObject
    :parameters (?o - object ?r - fridgeReceptacle)
10. slice an object using a sharp object
    :action SliceObject
    :parameters (?r - receptacle ?co - object ?sharp_o - object)

"""
prompt_obs_action_detailed = """
You are in an environment that you must explore step by step. Your task is to build and update PDDL files for the environment using only your direct observations. Do not create or assume any objects, relationships, or details that have not been observed, and ensure you include all observations.

The environment is a room containing various objects. Some of these objects are on, in, or contained within other objects and receptacles. You will initially be located as init_receptacle. You can assume all receptacles are freely reachable.

Now, {goal}
Here are your current observations: {brief_obs}

Valid actions you are currently allowed to take:
{valid_actions}

{pddl_action_specs}

Guidelines:
- Do NOT invent objects or receptacles that were not observed.
- If the task requires finding an object, your near-term goal can be "be at" an unvisited receptacle.
- PF goal MUST be grounded (no variables, no quantifiers).
- PF init should not use "(not ...)" unless strictly necessary.
"""


prompt_obs_action_subgoal = """
Same as the detailed setting, but choose a single grounded *subgoal* that makes progress toward the final goal.
The subgoal should be immediately solvable given the current knowledge.

Task goal:
{goal}

Latest observation summary:
{brief_obs}

Valid actions:
{valid_actions}

{pddl_action_specs}
"""


prompt_obs_action_without_hint = """
Same as detailed, but do NOT rely on the valid actions list as a hint.
Use it only as a constraint check after you model the world.

Task goal:
{goal}

Latest observation summary:
{brief_obs}

Valid actions (constraint only):
{valid_actions}

{pddl_action_specs}
"""


prompt_obs_action_without_detailed_goal = """
Same as detailed, but treat the goal text as incomplete.
Model what you can from the observation and take a reasonable grounded exploration goal.

Goal (may be incomplete):
{goal}

Latest observation summary:
{brief_obs}

Valid actions:
{valid_actions}

{pddl_action_specs}
"""


prompt_obs_action_general_goal = """
You are in ALFWorld.
Model the world conservatively and set a grounded goal that encourages exploration without loops.

Task goal:
{goal}

Latest observation summary:
{brief_obs}

Valid actions:
{valid_actions}

{pddl_action_specs}
"""

prompt_pddl = """
You are in an environment that you must explore step by step. Your task is to build and update PDDL files for the environment using only your direct observations. Do not create or assume any objects, relationships, or details that have not been observed, and ensure you include all observations.

The environment is a room containing various objects. Some of these objects are on, in, or contained within other objects and receptacles. You will initially be located as init_receptacle. You can assume all receptacles are freely reachable.

Now, {goal}
Here are your current observations: {brief_obs}

Only the following actions are allowed: (There are only two types: object and receptacle)
1. go to a receptacle
    :action GotoLocation
    :parameters (?from - receptacle ?to - receptacle)
2. open a receptacle if it is closed
    :action OpenObject
    :parameters (?r - receptacle)
3. close a receptacle
    :action CloseObject
    :parameters (?r - receptacle)
4. take an object from another receptacle
    :action PickupObject
    :parameters (?o - object ?r - receptacle)
5. put object into/on/in another receptacle
    :action PutObject
    :parameters (?o - object ?r - receptacle)
6. using an object/receptacle by turning it on/off with a switch
    :action useObject
    :parameters (?o - object)
7. heat an object using a receptacle
    :action HeatObject
    :parameters (?o - object ?r - microwaveReceptacle)
8. clean an object using a receptacle
    :action CleanObject
    :parameters (?o - object ?r - sinkbasinReceptacle)
9. cool an object using a receptacle
    :action CoolObject
    :parameters (?o - object ?r - fridgeReceptacle)
10. slice an object using a sharp object
    :action SliceObject
    :parameters (?r - receptacle ?co - object ?sharp_o - sharpObject)

You must go to a receptacle first in order to use/open it or take/put objects from/on it.

The process involves two main stages:

1. Always searching for the aim Object first!!!
    In this stage, your goal is to go to and may need to open new, unvisited recepatacles until you find the object mentioned in the task. Some receptacles cannot be opened so you can directly see what objects after you go to that receptacle.

    You can only use the GotoLocation action to travel to a new location and the OpenObject action (if the receptacle is closed) to verify whether it contains the target object.

    Goal 1.1: Reach a location that has not been visited (the location should be a receptacle) using the GotoLocation action. 
        You goal should look like this:
        (:goal 
            (at ?recepatacle)
        ) where recepatacle should be somewhere or some recepatacles not visited.

    Goal 1.2: If you already go to the recepatacle and found the recepatacle is closed, use the OpenObject action to open it and inspect the contents. 
        Your goal should look like this:
        (:goal 
            (opened ?recepatacle)
        ) where recepatacle should be the recepatacle you want to open.

2. After you seeing the aim object in any receptacle, using the Object to Complete the Task:
    After you have located the object (the object may have some numbers added), you should always first pick up the object from that receptacle and update your goal to focus on how the object is used to complete the task. Remember your goal is {goal}. Based on different adjectives, you may need to perform different actions for the object in different ways.

    This may involve more than simply transferring it from one place to another.
    For example: You might examine the object or a nearby receptacle to gather information. You may need to use another tool or device (like a lamp or a switch). Some tasks require you to slice, heat, cool, or clean the object using an appropriate receptacle (e.g., microwave, sink, fridge).

    If necessary, use the PickupObject action to retrieve the item, and the GotoLocation action to move to the correct place.
    Then, apply the object in a purposeful way — not just move it — but interact with the environment to fulfill the task’s actual goal.

    Hint: 
    1. If you want to heat, clean, and cool an object, after you go to that aim receptacle, do not put the object in the receptacle but do the action directly. For example, go to fridge, then cool the object with receptacle.
    2. If you want to slice an object, you should first go to the receptacle where both the sharp object and the aim object are located and ONLY pick up the sharp object then do the slice action. Don't forget to put the sharp object back to the receptacle after you finish slicing.
    3. If you want to examine or look at an object with a lamp, you should first go to the receptacle where the object is located and then pick it up and take the USE action of the lamp. You don't need to take the lamp but directly use it.
    4. If there are multiple actions needed to complete the task, you can break them down into smaller subgoals. For example, if you need to slice and then heat an object, first focus on slicing it, and then move on to heating it.

In summary, the first stage is all about finding the object—this might involve going to an unvisited receptacle and opening it if necessary.

Note: 
1. some receptacles have numbers in their names. Always keep them as they are. For example, "towelholder1" should not be changed to "towelholder".
2. Your initial goal should always be to go to a new location instead of put something into somewhere.
3. Do not enter stage 2 when not finishing stage 1.

Note: Always include :negative-preconditions in your :requirements whenever you use (not …) or delete effects, and never leave an :precondition or :effect block empty—either omit it or include at least one literal.
""" 

# =========================
# Baseline (LLM as planner)
# =========================

prompt_baseline = """
You are in an environment that you explore step by step. Based on your observations, generate one valid action at a time to progress in the environment.
Your task is to interact with objects and receptacles to complete a goal step by step.

Task goal:
{goal}

Here are your current observations:
{brief_obs}

Valid actions you can take now:
{valid_actions}

Memory:
{overall_memory}

If the previous attempt caused an environment error, here is the message:
{large_loop_error_message}

Provide the output in strict JSON format like this while you should only generate one action at a time:
{{
    "actions": ["action1"]
}}
"""

prompt_baseline2 = """
You are in an environment that you explore step by step. Based on your observations, generate one valid action at a time to progress in the environment.
Your task is to interact with objects and receptacles to complete a goal step by step.

Your specific task goal: {goal}

Here are your current observations: {brief_obs}

Valid actions you can take (follow exactly this format, replacing the parts in brackets with actual object and location names. Do not include any brackets in your output):
    - go to [towelholder 1]
    - open [cabinet 2]
    - take [cloth 1] from [cabinet 3]
    - move [soap bar 1] to [sink basin 2]
    - use [desk lamp 1]
    - heat [bread 1] with [microwave 1]
    - clean [fork 1] with [sink basin 1]
    - cool [wine bottle 1] with [fridge 1]
    - slice [bread 1] with [knife 1]
Only replace the object and receptacle names (the words that were inside the brackets) with the actual values from the game environment. Do not change the structure. Do not include brackets.

Your actions should exactly follow this phrasing — do not invent new formats. Every action must correspond to a valid command from the environment.

You must go to a receptacle first in order to use/open it or take/put objects from/on it. You can assume all receptacles are freely reachable.

The process involves two main stages:

1. Always searching for the aim Object first!!!
    In this stage, your goal is to go to and may need to open new, unvisited recepatacles until you find the object mentioned in the task. Some receptacles cannot be opened so you can directly see what objects after you go to that receptacle.

    You can only use the GotoLocation action to travel to a new location and the OpenObject action (if the receptacle is closed) to verify whether it contains the target object.

2. Using the Object to Complete the Task:
    Once you have located and picked up the object, update your goal to focus on how the object is used to complete the task. This may involve more than simply transferring it from one place to another.
    For example: You might examine the object or a nearby receptacle to gather information. You may need to use another tool or device (like a lamp or a switch). Some tasks require you to slice, heat, cool, or clean the object using an appropriate receptacle (e.g., microwave, sink, fridge).

    If necessary, use the PickupObject action to retrieve the item, and the GotoLocation action to move to the correct place.
    Then, apply the object in a purposeful way — not just move it — but interact with the environment to fulfill the task’s actual goal.

    Note: if you want to heat, clean, and cool an object, you can go to that receptacle then do the action directly without put the object into that receptacle.
        But if you want to slice an object, you should first go to the receptacle and pick up the sharp object then do the slice action.

In summary, the first stage is all about finding the object—this might involve going to an unvisited receptacle and opening it if necessary.

Note: 
1. some receptacles have numbers in their names. Always keep them as they are. For example, "towelholder1" should not be changed to "towelholder".
2. Your initial goal should always be to go to a new location instead of put something into somewhere.
3. Do not enter stage 2 when not finishing stage 1.

Memory of past steps:
{overall_memory}

If there are errors or obstacles, here is the message:
{large_loop_error_message}

Provide the output in strict JSON format like this while you should only generate one action at a time:
{{
    "actions": ["action1"]
}}
"""
# =========================
# Seed DF mode (initDF)
# =========================

prompt_obs_action_initial_df = """
You are in a partially observable ALFWorld environment.

You are provided with an initial PDDL domain file (seed). Do NOT discard it.
Start from this seed and keep changes minimal unless required for solvability or consistency.

Initial domain file (seed DF):
{initial_df}

Task goal:
{goal}

Latest observation summary:
{brief_obs}

Valid actions:
{valid_actions}

{pddl_action_specs}

Output updated DF and a PF consistent with the observations.
"""
prompt_pddl_initDF = """
You are in an environment that you must explore step by step. Your task is to build and update PDDL files for the environment using only your direct observations. Do not create or assume any objects, relationships, or details that have not been observed, and ensure you include all observations.

The environment is a room containing various objects. Some of these objects are on, in, or contained within other objects and receptacles. You will initially be located as init_receptacle. You can assume all receptacles are freely reachable.

You are provided with an initial PDDL domain file (seed). Do not discard it. Start from this seed and keep changes minimal unless required for solvability or consistency with the observations and the valid action schemas. If the seed conflicts with the observations or the action schemas, you may modify it accordingly. Ensure the DF and PF remain consistent with each other.

Here is the initial domain file (seed):
|initial domain file START|
{initial_df}
|initial domain file END|

Now, {goal}
Here are your current observations: {brief_obs}

Only the following actions are allowed: (There are only two types: object and receptacle)
1. go to a receptacle
    :action GotoLocation
    :parameters (?from - receptacle ?to - receptacle)
2. open a receptacle if it is closed
    :action OpenObject
    :parameters (?r - receptacle)
3. close a receptacle
    :action CloseObject
    :parameters (?r - receptacle)
4. take an object from another receptacle
    :action PickupObject
    :parameters (?o - object ?r - receptacle)
5. put object into/on/in another receptacle
    :action PutObject
    :parameters (?o - object ?r - receptacle)
6. using an object/receptacle by turning it on/off with a switch
    :action useObject
    :parameters (?o - object)
7. heat an object using a receptacle
    :action HeatObject
    :parameters (?o - object ?r - microwaveReceptacle)
8. clean an object using a receptacle
    :action CleanObject
    :parameters (?o - object ?r - sinkbasinReceptacle)
9. cool an object using a receptacle
    :action CoolObject
    :parameters (?o - object ?r - fridgeReceptacle)
10. slice an object using a sharp object
    :action SliceObject
    :parameters (?r - receptacle ?co - object ?sharp_o - sharpObject)

You must go to a receptacle first in order to use/open it or take/put objects from/on it.

The process involves two main stages:

1. Always searching for the aim Object first!!!
    In this stage, your goal is to go to and may need to open new, unvisited recepatacles until you find the object mentioned in the task. Some receptacles cannot be opened so you can directly see what objects after you go to that receptacle.

    You can only use the GotoLocation action to travel to a new location and the OpenObject action (if the receptacle is closed) to verify whether it contains the target object.

    Goal 1.1: Reach a location that has not been visited (the location should be a receptacle) using the GotoLocation action. 
        You goal should look like this:
        (:goal 
            (at ?recepatacle)
        ) where recepatacle should be somewhere or some recepatacles not visited.

    Goal 1.2: If you already go to the recepatacle and found the recepatacle is closed, use the OpenObject action to open it and inspect the contents. 
        Your goal should look like this:
        (:goal 
            (opened ?recepatacle)
        ) where recepatacle should be the recepatacle you want to open.

2. After you seeing the aim object in any receptacle, using the Object to Complete the Task:
    After you have located the object (the object may have some numbers added), you should always first pick up the object from that receptacle and update your goal to focus on how the object is used to complete the task. Remember your goal is {goal}. Based on different adjectives, you may need to perform different actions for the object in different ways.

    This may involve more than simply transferring it from one place to another.
    For example: You might examine the object or a nearby receptacle to gather information. You may need to use another tool or device (like a lamp or a switch). Some tasks require you to slice, heat, cool, or clean the object using an appropriate receptacle (e.g., microwave, sink, fridge).

    If necessary, use the PickupObject action to retrieve the item, and the GotoLocation action to move to the correct place.
    Then, apply the object in a purposeful way — not just move it — but interact with the environment to fulfill the task’s actual goal.

    Hint: 
    1. If you want to heat, clean, and cool an object, after you go to that aim receptacle, do not put the object in the receptacle but do the action directly. For example, go to fridge, then cool the object with receptacle.
    2. If you want to slice an object, you should first go to the receptacle where both the sharp object and the aim object are located and ONLY pick up the sharp object then do the slice action. Don't forget to put the sharp object back to the receptacle after you finish slicing.
    3. If you want to examine or look at an object with a lamp, you should first go to the receptacle where the object is located and then pick it up and take the USE action of the lamp. You don't need to take the lamp but directly use it.
    4. If there are multiple actions needed to complete the task, you can break them down into smaller subgoals. For example, if you need to slice and then heat an object, first focus on slicing it, and then move on to heating it.

In summary, the first stage is all about finding the object—this might involve going to an unvisited receptacle and opening it if necessary.

Note: 
1. some receptacles have numbers in their names. Always keep them as they are. For example, "towelholder1" should not be changed to "towelholder".
2. Your initial goal should always be to go to a new location instead of put something into somewhere.
3. Do not enter stage 2 when not finishing stage 1.

Note: Always include :negative-preconditions in your :requirements whenever you use (not …) or delete effects, and never leave an :precondition or :effect block empty—either omit it or include at least one literal.
""" 
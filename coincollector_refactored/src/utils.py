import json
import re
from .solver import run_solver

def extract_json(content):

    content = _remove_think_tags(content)
    content = _extract_json_from_codeblock(content)
    content = _fix_triple_quoted_strings(content)
    content = _extract_json_from_plain_text(content)
    content = _fix_unescaped_characters(content)
    content = _strip_formatting(content)

    return content

def _remove_think_tags(text):
    think_end = text.find('</think>')
    if think_end != -1:
        return text[think_end + 8:].strip()  
    return text

def _extract_json_from_codeblock(text):
    patterns = [
        r"(?s)```json\s*(.*?)\s*```",  
        r"(?s)```\s*(.*?)\s*```"       
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    
    return text

def _extract_json_from_plain_text(text: str) -> str:
    last_brace_pos = text.rfind('{')
    if last_brace_pos != -1:
        return text[last_brace_pos:]

    return text

def _fix_triple_quoted_strings(text):
    pattern = r'"(\w+)":\s*"""(.*?)"""'
    
    def escape_for_json(match):
        key = match.group(1)
        content = match.group(2)
        
        replacements = [
            ('\\', '\\\\'),  
            ('"', '\\"'),    
            ('\n', '\\n'),  
            ('\r', '\\r'),   
            ('\t', '\\t'),   
        ]
        
        for old, new in replacements:
            content = content.replace(old, new)
        
        return f'"{key}": "{content}"'
    
    return re.sub(pattern, escape_for_json, text, flags=re.DOTALL)

def _fix_unescaped_characters(text: str) -> str:

    text = text.replace('\\\n', '\\n') # llama

    max_attempts = 200
    
    for i in range(max_attempts):
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError as e:
            

            if "Invalid control character" in e.msg:
                char_to_escape = text[e.pos]
                if char_to_escape == '\n':
                    escaped_char = '\\n'
                elif char_to_escape == '\r':
                    escaped_char = '\\r'
                elif char_to_escape == '\t':
                    escaped_char = '\\t'
                else:
                    escaped_char = ''
                
                text = text[:e.pos] + escaped_char + text[e.pos + 1:]

            elif "Unterminated string" in e.msg:
                text = text[:e.pos - 1] + '\\' + text[e.pos - 1:]

            else:
                break
    return text

def _strip_formatting(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:] 
    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()
    if text.startswith("'") and text.endswith("'") and len(text) > 1:
        text = text[1:-1]

    return text.strip()


def sanitize_valid_actions(infos):
    # vals = sorted(infos['validActions'])
    # return [v for v in vals if v not in ("look around", "inventory") and not v.startswith("close door to")]
    return ['move east', 'move north', 'move south', 'move west', 'open door to east', 'open door to north', 'open door to south', 'open door to west']

def summarize_obs(obs):
    if len(obs.split('\n')) == 1:
        return obs
    else:
        return obs.split('\n')[0].split(". ")[0] + ". " + obs.split('\n')[1]
    

def map_actions(action):
    actions = action.lower().replace("(", "").replace(")", "").split('\n')
    action_lst = []
    for act in actions:
        if "open" in act and "door" in act:
            direction = act.split(' ')[-1]
            action_lst.append(f'open door to {direction}')
        elif "move" in act:
            action_lst.append(f"move {act.split(' ')[-1]}")
    if len(action_lst) == 0:
        return None
    return action_lst


def get_action_from_pddl(df, pf):
    result = run_solver(df, pf, "dual-bfws-ffparser", validate_with_val=True)
    plan_text = result["output"].get("plan") or ""
    mapped = map_actions(plan_text) if plan_text else None
    return mapped, result["stderr"], plan_text


def map_env_feedback_to_large_loop_error(brief_obs: str, taken_action: str):
    if "You can't move there, the door is closed." in brief_obs:
        msg = (
            f"This is the action you take: {taken_action}. "
            "The door that you are moving to is closed. "
            "You should first open door to that direction then move there!"
        )
        return msg, "door_closed"

    if "That is already open." in brief_obs:
        msg = (
            f"This is the action you take: {taken_action}. "
            "You try to open a door that is already open. You already visited here. "
            "Make sure the status of door is correct."
        )
        return msg, "already_open"

    if "I'm not sure what you mean." in brief_obs:
        if "open door" in taken_action:
            msg = (
                f"This is the action you take: {taken_action}. "
                "When you try to open door, there is no door here or there is nothing in this direction. "
            )
            return msg, "invalid_open"
        elif "move" in taken_action:
            msg = (
                f"This is the action you take: {taken_action}. "
                "You cannot move to that direction."
            )
            return msg, "invalid_move"
        else:
            msg = (
                f"This is the action you take: {taken_action}. "
                "You got the environment error!"
            )
            return msg, "env_error"

    return None, None


def detect_duplicates(action_lst, threshold):
    n = len(action_lst)
    
    for seq_len in range(1, n // 2 + 1):
        sequence = action_lst[-seq_len:]
        
        count = 1
        for i in range(2, threshold + 1):
            if action_lst[-i * seq_len: - (i - 1) * seq_len] == sequence:
                count += 1
            else:
                break
     
        if count >= threshold:
            return True

    return False
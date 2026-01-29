from __future__ import annotations

import json
import re
from typing import Iterable, List, Optional, Tuple

from .solver import run_solver

def extract_json(content: str) -> str:

    content = _remove_think_tags(content)
    content = _extract_json_from_codeblock(content)
    content = _fix_triple_quoted_strings(content)
    content = _extract_json_from_plain_text(content)
    content = _fix_unescaped_characters(content)
    content = _strip_formatting(content)
    return content


def _remove_think_tags(text: str) -> str:
    think_end = text.find("</think>")
    if think_end != -1:
        return text[think_end + 8 :].strip()  # len('</think>') == 8
    return text


def _extract_json_from_codeblock(text: str) -> str:
    patterns = [
        r"(?s)```json\s*(.*?)\s*```",
        r"(?s)```\s*(.*?)\s*```",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()
    return text


def _extract_json_from_plain_text(text: str) -> str:
    last_brace_pos = text.rfind("{")
    if last_brace_pos != -1:
        return text[last_brace_pos:]
    return text


def _fix_triple_quoted_strings(text: str) -> str:
    pattern = r'"(\w+)":\s*"""(.*?)"""'

    def escape_for_json(match: re.Match) -> str:
        key = match.group(1)
        content = match.group(2)
        replacements = [
            ("\\", "\\\\"),
            ('"', '\\"'),
            ("\n", "\\n"),
            ("\r", "\\r"),
            ("\t", "\\t"),
        ]
        for old, new in replacements:
            content = content.replace(old, new)
        return f'"{key}": "{content}"'

    return re.sub(pattern, escape_for_json, text, flags=re.DOTALL)


def _fix_unescaped_characters(text: str) -> str:
    text = text.replace("\\\n", "\\n")
    max_attempts = 200
    for _ in range(max_attempts):
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError as e:
            if "Invalid control character" in e.msg:
                ch = text[e.pos]
                if ch == "\n":
                    repl = "\\n"
                elif ch == "\r":
                    repl = "\\r"
                elif ch == "\t":
                    repl = "\\t"
                else:
                    repl = ""
                text = text[: e.pos] + repl + text[e.pos + 1 :]
            elif "Unterminated string" in e.msg and e.pos - 1 >= 0:
                text = text[: e.pos - 1] + "\\" + text[e.pos - 1 :]
            else:
                break
    return text


def _strip_formatting(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if text.startswith("'") and text.endswith("'") and len(text) > 1:
        text = text[1:-1]
    return text.strip()


# ========================================
# ALFWorld helpers
# ========================================

def filter_valid_actions(cmds: Iterable[str]) -> List[str]:
    s = set(cmds)
    s.discard("look")
    s.discard("inventory")
    s.discard("help")
    return sorted(s)


def sanitize_valid_actions(infos: dict) -> List[str]:
    cmds = infos.get("admissible_commands")
    if cmds is None:
        cmds = infos.get("admissibleCommands")
    if cmds is None:
        return []
    return filter_valid_actions(cmds)


def summarize_obs(obs: str) -> str:
    lines = obs.split("\n")
    if len(lines) <= 1:
        return obs
    # Only keep where you are and location information
    first = lines[0].split(". ")[0] + "."
    second = lines[1]
    return f"{first} {second}"


def parse_alfworld_obs(obs: str) -> Tuple[str, str]:
    lines = [l for l in obs.split("\n") if l.strip()]
    if not lines:
        return "", ""
    goal = lines[-1]
    scene = "\n".join(lines[:-1]) if len(lines) > 1 else lines[0]
    return scene, goal


# ========================================
# Solver <-> env action mapping
# ========================================

_LETTER_DIGIT_RE = re.compile(r"(\D+)(\d+)$")


def _format_obj_token(tok: str) -> str:
    tok = tok.strip().lower()
    tok = tok.replace("_", " ")
    m = _LETTER_DIGIT_RE.match(tok)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    return tok


def _normalize_plan_line(line: str) -> Optional[List[str]]:
    line = line.strip()
    if not line:
        return None
    if line.startswith(";"):
        return None
    line = re.sub(r"^\s*\d+\s*[:.]\s*", "", line)
    line = line.replace("(", " ").replace(")", " ")
    line = re.sub(r"\s+", " ", line).strip().lower()
    if not line:
        return None
    return line.split(" ")


def map_actions(plan_text: str) -> Optional[List[str]]:
    action_lst: List[str] = []
    for raw in plan_text.split("\n"):
        parts = _normalize_plan_line(raw)
        if not parts:
            continue

        act = parts[0]

        start_idx = 2 if len(parts) >= 2 and parts[1] in {"agent1", "agent"} else 1

        if "gotolocation" in act:
            dest = parts[-1]
            action_lst.append(f"go to {_format_obj_token(dest)}")

        elif "openobject" in act:
            obj = parts[-1]
            action_lst.append(f"open {_format_obj_token(obj)}")

        elif "pickupobject" in act:
            if len(parts) >= start_idx + 2:
                obj = parts[start_idx]
                container = parts[start_idx + 1]
            elif len(parts) >= 3:
                obj, container = parts[-2], parts[-1]
            else:
                continue
            action_lst.append(f"take {_format_obj_token(obj)} from {_format_obj_token(container)}")

        elif "putobject" in act:
            if len(parts) >= start_idx + 2:
                obj = parts[start_idx]
                container = parts[start_idx + 1]
            elif len(parts) >= 3:
                obj, container = parts[-2], parts[-1]
            else:
                continue
            action_lst.append(f"move {_format_obj_token(obj)} to {_format_obj_token(container)}")

        elif "useobject" in act:
            if len(parts) >= start_idx + 1:
                obj = parts[start_idx]
            else:
                obj = parts[-1]
            action_lst.append(f"use {_format_obj_token(obj)}")

        elif "heatobject" in act:
            if len(parts) >= start_idx + 2:
                obj = parts[start_idx]
                receptacle = parts[start_idx + 1]
            else:
                continue
            action_lst.append(f"heat {_format_obj_token(obj)} with {_format_obj_token(receptacle)}")

        elif "cleanobject" in act:
            if len(parts) >= start_idx + 2:
                obj = parts[start_idx]
                receptacle = parts[start_idx + 1]
            else:
                continue
            action_lst.append(f"clean {_format_obj_token(obj)} with {_format_obj_token(receptacle)}")

        elif "coolobject" in act:
            if len(parts) >= start_idx + 2:
                obj = parts[start_idx]
                receptacle = parts[start_idx + 1]
            else:
                continue
            action_lst.append(f"cool {_format_obj_token(obj)} with {_format_obj_token(receptacle)}")

        elif "sliceobject" in act:
            if len(parts) >= start_idx + 3:
                obj = parts[start_idx + 1]
                tool = parts[start_idx + 2]
            elif len(parts) >= 4:
                obj = parts[-2]
                tool = parts[-1]
            else:
                continue
            action_lst.append(f"slice {_format_obj_token(obj)} with {_format_obj_token(tool)}")

        else:
            # unknown action => ignore
            continue

    return action_lst or None


def get_action_from_pddl(df: str, pf: str):
    result = run_solver(df, pf, "dual-bfws-ffparser", validate_with_val=True)
    plan_text = result["output"].get("plan") or ""
    mapped = map_actions(plan_text) if plan_text else None
    return mapped, result["stderr"], plan_text


# ========================================
# Feedback processing ("Nothing happens")
# ========================================

def build_large_loop_error_message(step_trace: str, taken_action: str) -> Tuple[str, str]:
    msg = (
        "In this step, you took the following actions and observations:\n"
        f"{step_trace}\n\n"
        f"This is the action that caused an issue: {taken_action}\n"
    )
    a = taken_action.lower()

    if a.startswith("go to "):
        msg += (
            "You tried to go to a receptacle but nothing happened. "
            "You may already be there, or that receptacle does not exist. "
            "Avoid repeating the same go-to target; pick an unvisited receptacle instead."
        )
        return msg, "ignore"

    if a.startswith("open "):
        msg += (
            "You tried to open a receptacle but nothing happened. "
            "You may need to go to it first. If you are already there and still see this, "
            "it might be non-openable; do not keep trying to open it."
        )
        return msg, "retry"

    if a.startswith("take "):
        msg += (
            "You tried to take an object from a receptacle but nothing happened. "
            "You likely are not at that receptacle, or that object is not inside it. "
            "Model the object locations conservatively and search other receptacles."
        )
        return msg, "retry"

    if a.startswith("move "):
        msg += (
            "You tried to move an object to a receptacle but nothing happened. "
            "Make sure you are holding the object first, then go to the target receptacle and move it."
        )
        return msg, "retry"

    if a.startswith("slice "):
        msg += (
            "You tried to slice an object but nothing happened. "
            "You usually need to pick up the sharp tool first (knife, etc.), then slice."
        )
        return msg, "retry"

    if a.startswith("cool "):
        msg += (
            "You tried to cool an object but nothing happened. "
            "You usually must pick up the object, then go to the fridge and use the cool action (not move-to-fridge)."
        )
        return msg, "retry"

    if a.startswith("heat "):
        msg += (
            "You tried to heat an object but nothing happened. "
            "You usually must pick up the object, then go to the microwave and heat it."
        )
        return msg, "retry"

    if a.startswith("clean "):
        msg += (
            "You tried to clean an object but nothing happened. "
            "You usually must pick up the object, then go to the sinkbasin and clean it."
        )
        return msg, "retry"

    if a.startswith("use "):
        msg += (
            "You tried to use an object but nothing happened. "
            "In many ALFWorld tasks, 'use' is mainly for toggling/using devices. "
            "Ensure you are using a valid usable object." 
        )
        return msg, "retry"

    if any(k in a for k in ("fridge", "sinkbasin", "microwave")) and (
        a.startswith("move ") or a.startswith("take ")
    ):
        msg += (
            "You attempted to move/take involving fridge/sinkbasin/microwave. "
            "Often you should directly use cool/clean/heat actions instead of moving/taking to/from them."
        )
        return msg, "ignore"

    msg += "This action produced no effect. Revise your DF/PF to better match the environment constraints."
    return msg, "retry"


def detect_duplicates(action_lst: List[str], threshold: int) -> bool:
    n = len(action_lst)
    for seq_len in range(1, n // 2 + 1):
        seq = action_lst[-seq_len:]
        count = 1
        for i in range(2, threshold + 1):
            if action_lst[-i * seq_len : -(i - 1) * seq_len] == seq:
                count += 1
            else:
                break
        if count >= threshold:
            return True
    return False

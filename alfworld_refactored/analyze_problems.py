"""Analyze ALFWorld problems to understand types and overlaps."""

import os
import glob
import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple

ALFWORLD_DATA = "/home/yl3427/.cache/alfworld/json_2.1.1"

def analyze_problems():
    """Analyze all problem directories."""

    results = {
        'train': defaultdict(list),
        'valid_train': defaultdict(list),
        'valid_seen': defaultdict(list),
        'valid_unseen': defaultdict(list)
    }

    all_problems_by_type = defaultdict(list)
    problem_paths = {}

    for split in results.keys():
        split_path = os.path.join(ALFWORLD_DATA, split)
        if not os.path.exists(split_path):
            print(f"Skipping {split} - path doesn't exist")
            continue

        problems = os.listdir(split_path)
        print(f"\n{split}: {len(problems)} problems")

        for problem_dir in problems:
            # Extract problem type (everything before first hyphen)
            problem_type = problem_dir.split('-')[0]
            results[split][problem_type].append(problem_dir)

            full_path = os.path.join(split_path, problem_dir)

            trials = glob.glob(os.path.join(full_path, "trial_*"))
            if trials:
                pddl_path = os.path.join(trials[0], "initial_state.pddl")
                if os.path.exists(pddl_path):
                    problem_key = f"{split}/{problem_dir}"
                    problem_paths[problem_key] = pddl_path
                    all_problems_by_type[problem_type].append((split, problem_dir, pddl_path))

        print(f"  Problem types in {split}:")
        for ptype, probs in sorted(results[split].items()):
            print(f"    {ptype}: {len(probs)}")

    print("\n=== Checking for overlaps ===")
    all_names = set()
    overlaps = []

    for split, type_dict in results.items():
        for ptype, prob_list in type_dict.items():
            for prob in prob_list:
                if prob in all_names:
                    overlaps.append((split, prob))
                all_names.add(prob)

    if overlaps:
        print(f"Found {len(overlaps)} overlapping problem names:")
        for split, prob in overlaps[:10]:  # Show first 10
            print(f"  {split}: {prob}")
    else:
        print("No overlapping problem names found across splits!")

    print("\n=== All Problem Types ===")
    all_types = set()
    for split_types in results.values():
        all_types.update(split_types.keys())

    for ptype in sorted(all_types):
        total = sum(len(results[split].get(ptype, [])) for split in results.keys())
        print(f"  {ptype}: {total} total problems")
        for split in results.keys():
            count = len(results[split].get(ptype, []))
            if count > 0:
                print(f"    {split}: {count}")

    return results, all_problems_by_type, problem_paths


def create_problem_type_mapping(all_problems_by_type: Dict) -> Dict[str, str]:

    type_mapping = {
        'look_at_obj_in_light': 'use',  # Use a lamp to examine objects
        'pick_and_place_simple': 'basic',  # Basic pick and place
        'pick_two_obj_and_place': 'basic',  # Pick two objects
        'pick_clean_then_place_in_recep': 'clean',  # Clean object then place
        'pick_cool_then_place_in_recep': 'cool',  # Cool object then place
        'pick_heat_then_place_in_recep': 'heat',  # Heat object then place
        'puttwo': 'basic',  # Put two objects
        'GotoLocation': 'basic',  # Simple navigation
        'pick_and_place_with_movable_recep': 'basic',  # Pick and place with movable receptacles
    }

    found_types = set(all_problems_by_type.keys())
    mapped_types = set(type_mapping.keys())
    unmapped = found_types - mapped_types

    if unmapped:
        print(f"\nWarning: Unmapped problem types: {unmapped}")
        for t in unmapped:
            type_mapping[t] = 'unknown'

    return type_mapping


def save_problem_index(results, all_problems_by_type, problem_paths):

    index = {
        'by_split': {},
        'by_type': {},
        'type_mapping': create_problem_type_mapping(all_problems_by_type),
        'paths': {}
    }

    for split, type_dict in results.items():
        index['by_split'][split] = dict(type_dict)

    for ptype, prob_list in all_problems_by_type.items():
        index['by_type'][ptype] = []
        for split, name, path in prob_list:
            index['by_type'][ptype].append({
                'split': split,
                'name': name,
                'path': path
            })

    index['paths'] = problem_paths

    output_file = '/local-ssd/yl3427/pddlego-plus/alfworld_refactored3/problem_index.json'
    with open(output_file, 'w') as f:
        json.dump(index, f, indent=2)

    print(f"\nSaved problem index to {output_file}")
    return index


if __name__ == "__main__":
    results, all_problems_by_type, problem_paths = analyze_problems()
    index = save_problem_index(results, all_problems_by_type, problem_paths)

    print("\n=== Final Summary ===")
    print(f"Total unique problem types: {len(index['type_mapping'])}")
    print(f"Total problems: {len(index['paths'])}")
    print("\nProblem type mapping:")
    for orig, mapped in sorted(index['type_mapping'].items()):
        count = len(index['by_type'].get(orig, []))
        print(f"  {orig} -> {mapped} ({count} problems)")
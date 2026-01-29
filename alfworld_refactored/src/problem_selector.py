import os
import json
import glob
from typing import List, Dict, Optional, Tuple
from os.path import join as pjoin

from alfworld.info import ALFWORLD_DATA


PROBLEM_INDEX_PATH = '/local-ssd/yl3427/pddlego-plus/alfworld_refactored3/problem_index.json'


class ProblemSelector:

    # Mapping from detailed problem types to simplified categories
    PROBLEM_TYPE_MAPPING = {
        'look_at_obj_in_light': 'use',  # Use a lamp to examine objects
        'pick_and_place_simple': 'basic',  # Basic pick and place
        'pick_two_obj_and_place': 'basic',  # Pick two objects
        'pick_and_place_with_movable_recep': 'basic',  # Pick/place with movable receptacles
        'pick_clean_then_place_in_recep': 'clean',  # Clean object then place
        'pick_cool_then_place_in_recep': 'cool',  # Cool object then place
        'pick_heat_then_place_in_recep': 'heat',  # Heat object then place
    }

    # Reverse mapping
    CATEGORY_TO_TYPES = {
        'use': ['look_at_obj_in_light'],
        'basic': ['pick_and_place_simple', 'pick_two_obj_and_place', 'pick_and_place_with_movable_recep'],
        'clean': ['pick_clean_then_place_in_recep'],
        'cool': ['pick_cool_then_place_in_recep'],
        'heat': ['pick_heat_then_place_in_recep'],
    }

    def __init__(self, split: str = 'valid_train'):
        self.split = split
        self.problems_by_type = {}
        self.all_problems = []
        self._load_problems()

    def _load_problems(self):

        if os.path.exists(PROBLEM_INDEX_PATH):
            with open(PROBLEM_INDEX_PATH, 'r') as f:
                index = json.load(f)

            for prob_type, problems in index.get('by_type', {}).items():
                # Filter by split
                self.problems_by_type[prob_type] = [
                    p for p in problems if p['split'] == self.split
                ]
        else:
            # Fall back to scanning directories
            self._scan_problems()

        for prob_type, problems in self.problems_by_type.items():
            for prob in problems:
                if isinstance(prob, dict):
                    # From index
                    path = prob['path']
                else:
                    # From scan
                    path = prob
                self.all_problems.append((prob_type, path))

    def _scan_problems(self):
        """Scan problem directories directly."""
        base_path = pjoin(ALFWORLD_DATA, self.split)

        if not os.path.exists(base_path):
            all_problems = glob.glob(
                pjoin(ALFWORLD_DATA, "**", "initial_state.pddl"),
                recursive=True
            )
            all_problems = [p for p in all_problems if "movable_recep" not in p]
        else:
            # Scan specific split
            all_problems = glob.glob(
                pjoin(base_path, "*", "*", "initial_state.pddl")
            )

        # Organize by type
        for path in all_problems:
            # Extract problem type from path
            parts = path.split(os.sep)
            for part in parts:
                if '-' in part:
                    prob_type = part.split('-')[0]
                    if prob_type not in self.problems_by_type:
                        self.problems_by_type[prob_type] = []
                    self.problems_by_type[prob_type].append(path)
                    break

    def get_problems_by_type(
        self,
        problem_type: str,
        max_count: int = 50,
        offset: int = 0
    ) -> List[Tuple[str, str, str]]:

        if problem_type in self.CATEGORY_TO_TYPES:
            detailed_types = self.CATEGORY_TO_TYPES[problem_type]
        elif problem_type in self.PROBLEM_TYPE_MAPPING:
            detailed_types = [problem_type]
        elif problem_type == 'all':
            detailed_types = list(self.problems_by_type.keys())
        else:
            print(f"Warning: Unknown problem type '{problem_type}', using all problems")
            detailed_types = list(self.problems_by_type.keys())

        results = []
        for dtype in detailed_types:
            if dtype in self.problems_by_type:
                category = self.PROBLEM_TYPE_MAPPING.get(dtype, 'unknown')
                for prob in self.problems_by_type[dtype]:
                    if isinstance(prob, dict):
                        path = prob['path']
                    else:
                        path = prob
                    results.append((path, dtype, category))

        return results[offset:offset + max_count]

    def get_problem_by_index(self, index: int) -> Optional[Tuple[str, str, str]]:

        if 0 <= index < len(self.all_problems):
            prob_type, path = self.all_problems[index]
            category = self.PROBLEM_TYPE_MAPPING.get(prob_type, 'unknown')
            return (path, prob_type, category)
        return None

    def get_available_types(self) -> Dict[str, int]:
        result = {}

        # Categories
        for cat, types in self.CATEGORY_TO_TYPES.items():
            count = sum(
                len(self.problems_by_type.get(t, []))
                for t in types
            )
            if count > 0:
                result[f"Category: {cat}"] = count

        # Detailed types
        for prob_type, problems in self.problems_by_type.items():
            count = len(problems)
            if count > 0:
                result[f"Type: {prob_type}"] = count

        return result

    def get_problem_info(self, problem_path: str) -> Dict[str, str]:
        if "initial_state.pddl" in problem_path:
            problem_dir = os.path.dirname(os.path.dirname(problem_path))
        else:
            problem_dir = problem_path

        parts = problem_dir.split(os.sep)
        problem_name = None
        for part in parts:
            if '-' in part:
                problem_name = part
                break

        if problem_name:
            problem_type = problem_name.split('-')[0]
            category = self.PROBLEM_TYPE_MAPPING.get(problem_type, 'unknown')
        else:
            problem_type = 'unknown'
            category = 'unknown'

        return {
            'path': problem_path,
            'dir': problem_dir,
            'name': problem_name or 'unknown',
            'type': problem_type,
            'category': category,
            'split': self.split
        }
"""Analyze ALFWorld problem distribution to determine appropriate sample sizes for evaluation."""

import json
import os
from collections import defaultdict
from src.problem_selector import ProblemSelector

def analyze_problem_distribution():
    """Analyze problem distribution across different splits and types."""

    splits = ['train', 'valid_train', 'valid_seen', 'valid_unseen']

    # Track overall statistics
    all_stats = {}
    category_stats = defaultdict(lambda: defaultdict(int))
    detailed_stats = defaultdict(lambda: defaultdict(int))

    for split in splits:
        print(f"\n{'='*60}")
        print(f"Split: {split}")
        print(f"{'='*60}")

        try:
            selector = ProblemSelector(split=split)
            available_types = selector.get_available_types()

            if not available_types:
                print(f"  No problems found for split '{split}'")
                continue

            # Separate categories and types
            categories = {}
            types = {}

            for key, count in available_types.items():
                if key.startswith("Category:"):
                    cat_name = key.replace("Category: ", "")
                    categories[cat_name] = count
                    category_stats[split][cat_name] = count
                elif key.startswith("Type:"):
                    type_name = key.replace("Type: ", "")
                    types[type_name] = count
                    detailed_stats[split][type_name] = count

            # Display categories
            print("\nCategories (Simplified):")
            print(f"{'Category':<15} {'Count':<10}")
            print("-" * 25)
            total_cat = 0
            for cat, count in sorted(categories.items()):
                print(f"{cat:<15} {count:<10}")
                total_cat += count
            print("-" * 25)
            print(f"{'Total':<15} {total_cat:<10}")

            # Display detailed types
            print("\nDetailed Types:")
            print(f"{'Type':<40} {'Count':<10} {'Category':<15}")
            print("-" * 65)
            total_type = 0
            for prob_type, count in sorted(types.items()):
                category = ProblemSelector.PROBLEM_TYPE_MAPPING.get(prob_type, 'unknown')
                print(f"{prob_type:<40} {count:<10} {category:<15}")
                total_type += count
            print("-" * 65)
            print(f"{'Total':<40} {total_type:<10}")

            all_stats[split] = {
                'categories': categories,
                'types': types,
                'total': total_type
            }

        except Exception as e:
            print(f"  Error processing split '{split}': {e}")

    output_file = '/local-ssd/yl3427/pddlego-plus/alfworld_refactored3/problem_distribution_analysis.json'
    with open(output_file, 'w') as f:
        json.dump({
            'by_split': all_stats,
            'category_summary': dict(category_stats),
            'type_summary': dict(detailed_stats)
        }, f, indent=2)

    print(f"\nFull analysis saved to: {output_file}")

if __name__ == "__main__":
    analyze_problem_distribution()
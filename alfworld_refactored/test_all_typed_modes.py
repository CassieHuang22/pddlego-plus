import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.pddl_engine import (
    run_iterative_model_typed,
    run_baseline_model_typed,
    run_iterative_model_initDF_typed,
    ProblemSelector
)


def run_comparison(
    model_name: str = "Qwen/Qwen3-32B",
    problem_type: str = "basic",
    num_problems: int = 5,
    split: str = "valid_train",
    initial_df_path: str = "/home/yl3427/.cache/alfworld/logic/alfred.pddl",
    base_folder: str = None,
    base_result: str = None
):
    """Run all three approaches on the same problem type for comparison.

    Args:
        model_name: Model to use
        problem_type: Problem type ('clean', 'heat', 'cool', 'basic', 'use', 'all')
        num_problems: Number of problems to run
        split: Data split to use
        initial_df_path: Path to initial domain file for initDF mode
        base_folder: Base output folder name (default: compare_{problem_type})
        base_result: Base result CSV name (default: compare_{problem_type}_results)
    """

    # Use defaults if not provided
    if base_folder is None:
        base_folder = f"compare_{problem_type}"
    if base_result is None:
        base_result = f"compare_{problem_type}_results"

    print(f"\n{'='*60}")
    print(f"Running Comparison: {problem_type} problems")
    print(f"Model: {model_name}")
    print(f"Split: {split}")
    print(f"Count: {num_problems}")
    print(f"Output folder: output/{base_folder}_*")
    print(f"Result file: output/{base_result}.csv")
    print(f"{'='*60}\n")

    # 1. Baseline (LLM as direct planner)
    print("\n[1/3] Running BASELINE (LLM as direct planner)...")
    run_baseline_model_typed(
        model_name=model_name,
        problem_type=problem_type,
        start_index=0,
        num_problems=num_problems,
        split=split,
        folder_name=f"{base_folder}_baseline",
        result_name=base_result,
    )

    # 2. PDDLego+ (LLM as translator)
    print("\n[2/3] Running PDDLego+ (LLM as translator)...")
    run_iterative_model_typed(
        model_name=model_name,
        problem_type=problem_type,
        start_index=0,
        num_problems=num_problems,
        split=split,
        folder_name=f"{base_folder}_translator",
        result_name=base_result,
        goal_type='detailed'
    )

    # 3. PDDLego+ with initial DF
    if os.path.exists(initial_df_path):
        print("\n[3/3] Running PDDLego+ with initial DF...")
        run_iterative_model_initDF_typed(
            model_name=model_name,
            initial_df=initial_df_path,
            problem_type=problem_type,
            start_index=0,
            num_problems=num_problems,
            split=split,
            folder_name=f"{base_folder}_initDF",
            result_name=base_result,
            goal_type='detailed',
            mode='initDF'
        )
    else:
        print(f"\n[3/3] Skipping initDF mode - file not found: {initial_df_path}")

    print(f"\n{'='*60}")
    print(f"Comparison complete! Check output/{base_result}.csv for results")
    print(f"{'='*60}\n")


def show_problem_distribution():
    """Show distribution of problem types across splits."""

    print("\n" + "="*60)
    print("Problem Type Distribution")
    print("="*60)

    splits = ['valid_train', 'valid_seen', 'valid_unseen']

    for split in splits:
        print(f"\n{split.upper()}:")
        try:
            selector = ProblemSelector(split=split)
            types = selector.get_available_types()

            # Separate categories and types
            categories = {k: v for k, v in types.items() if k.startswith("Category:")}
            detailed = {k: v for k, v in types.items() if k.startswith("Type:")}

            print("\n  Categories:")
            for cat, count in sorted(categories.items()):
                cat_name = cat.replace("Category: ", "")
                print(f"    {cat_name:10s}: {count:3d} problems")

            print("\n  Detailed Types:")
            for typ, count in sorted(detailed.items()):
                typ_name = typ.replace("Type: ", "")
                print(f"    {typ_name:35s}: {count:3d} problems")
        except Exception as e:
            print(f"  Error: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Run ALFWorld experiments with all three approaches'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show problem distribution'
    )
    parser.add_argument(
        '--type',
        default='basic',
        choices=['clean', 'heat', 'cool', 'basic', 'use', 'all'],
        help='Problem type to run'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=100,
        help='Number of problems to run'
    )
    parser.add_argument(
        '--split',
        default='train',
        choices=['train', 'valid_train', 'valid_seen', 'valid_unseen'],
        help='Data split to use'
    )
    parser.add_argument(
        '--model',
        default='Qwen/Qwen3-32B', # Qwen/Qwen3-32B, deepseek, o3-mini
        help='Model name'
    )
    parser.add_argument(
        '--init-df',
        # default='/home/yl3427/.cache/alfworld/logic/alfred.pddl',
        default='/local-ssd/yl3427/pddlego-plus/alfworld_refactored3/output/seeddf_all_20260122_184108/seed_domain.pddl',
        help='Path to initial domain file for initDF mode'
    )
    parser.add_argument(
        '--mode',
        choices=['baseline', 'translator', 'initdf', 'all'],
        default='all',
        help='Which mode(s) to run'
    )
    parser.add_argument(
        '--output-folder',
        default="0127_train_qwen_basic",
        help='Base output folder name (default: auto-generated based on mode/type)'
    )
    parser.add_argument(
        '--result-file',
        default="0127_train_qwen_basic",
        help='Result CSV file name (default: auto-generated based on mode/type)'
    )

    args = parser.parse_args()

    if args.show:
        show_problem_distribution()
    else:
        if args.mode == 'all':
            # Run full comparison
            run_comparison(
                model_name=args.model,
                problem_type=args.type,
                num_problems=args.count,
                split=args.split,
                initial_df_path=args.init_df,
                base_folder=args.output_folder,
                base_result=args.result_file
            )
        else:
            if args.output_folder:
                folder_name = args.output_folder
            else:
                folder_name = f"{args.mode}_{args.type}_typed"

            if args.result_file:
                result_name = args.result_file
            else:
                result_name = f"{args.mode}_{args.type}_results"

            if args.mode == 'baseline':
                print(f"Running baseline only on {args.count} '{args.type}' problems")
                print(f"Output folder: output/{folder_name}")
                print(f"Result file: output/{result_name}.csv")
                run_baseline_model_typed(
                    model_name=args.model,
                    problem_type=args.type,
                    start_index=0,
                    num_problems=args.count,
                    split=args.split,
                    folder_name=folder_name,
                    result_name=result_name,
                )
            elif args.mode == 'translator':
                print(f"Running translator only on {args.count} '{args.type}' problems")
                print(f"Output folder: output/{folder_name}")
                print(f"Result file: output/{result_name}.csv")
                run_iterative_model_typed(
                    model_name=args.model,
                    problem_type=args.type,
                    start_index=0,
                    num_problems=args.count,
                    split=args.split,
                    folder_name=folder_name,
                    result_name=result_name,
                    goal_type='detailed'
                )
            elif args.mode == 'initdf':
                print(f"Running initDF only on {args.count} '{args.type}' problems")
                print(f"Output folder: output/{folder_name}")
                print(f"Result file: output/{result_name}.csv")
                run_iterative_model_initDF_typed(
                    model_name=args.model,
                    initial_df=args.init_df,
                    problem_type=args.type,
                    start_index=0,
                    num_problems=args.count,
                    split=args.split,
                    folder_name=folder_name,
                    result_name=result_name,
                    goal_type='detailed',
                    mode='initDF'
                )


if __name__ == "__main__":
    main()
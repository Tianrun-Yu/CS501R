"""
run_all.py
----------
Convenience script to run evaluation for all model/dataset combinations
and print a final summary table.

Usage:
    python run_all.py
    python run_all.py --max_problems 10   # quick test run
"""

import argparse
from evaluate import evaluate_model
from utils import print_summary_table


MODELS   = ["qwen2.5-7b", "llama-3.2-3b", "qwen2.5-1.5b"]
DATASETS = ["aime", "amc"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples",    type=int, default=5,
                        help="Number of samples per problem (default: 5).")
    parser.add_argument("--judge",        type=str, default="qwen2.5-7b",
                        help="Judge model (default: qwen2.5-7b).")
    parser.add_argument("--max_problems", type=int, default=None,
                        help="Limit number of problems per run (for quick testing).")
    parser.add_argument("--output_dir",   type=str, default="results",
                        help="Directory to save JSON results.")
    args = parser.parse_args()

    all_results = []
    for model in MODELS:
        for dataset in DATASETS:
            result = evaluate_model(
                model_name       = model,
                dataset_name     = dataset,
                n_samples        = args.n_samples,
                judge_model_name = args.judge,
                output_dir       = args.output_dir,
                max_problems     = args.max_problems,
            )
            all_results.append(result)

    print_summary_table(all_results)


if __name__ == "__main__":
    main()

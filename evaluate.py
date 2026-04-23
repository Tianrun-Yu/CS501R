"""
evaluate.py
-----------
Main evaluation script for LLM math reasoning evaluation.
Computes Acc@5 and LLM-as-a-Judge scores (D1: Rigor, D2: Clarity).

Usage:
    python evaluate.py --model qwen2.5-7b --dataset aime --n_samples 5
"""

import argparse
import json
import re
import os
from pathlib import Path
from tqdm import tqdm

from models import load_model, generate_responses
from judge import judge_response
from datasets import load_dataset_problems
from metrics import compute_acc_at_k, compute_judge_scores
from utils import save_results, load_results


def extract_boxed_answer(text: str) -> str | None:
    """Extract the final \\boxed{...} answer from a model response."""
    pattern = r"\\boxed\{([^}]*)\}"
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else None


def check_answer_correct(predicted: str | None, ground_truth: str) -> bool:
    """Check if the predicted answer matches the ground truth."""
    if predicted is None:
        return False
    # Normalize: strip whitespace, lowercase
    pred = predicted.strip().lower().replace(" ", "")
    gt   = ground_truth.strip().lower().replace(" ", "")
    return pred == gt


def evaluate_model(
    model_name: str,
    dataset_name: str,
    n_samples: int = 5,
    judge_model_name: str = "qwen2.5-7b",
    output_dir: str = "results",
    max_problems: int | None = None,
) -> dict:
    """
    Run the full evaluation pipeline for one model on one dataset.

    Returns a dict with per-problem results and aggregate scores.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result_file = Path(output_dir) / f"{model_name}_{dataset_name}.json"

    # Load problems
    problems = load_dataset_problems(dataset_name)
    if max_problems:
        problems = problems[:max_problems]
    print(f"\n=== Evaluating {model_name} on {dataset_name} ({len(problems)} problems) ===")

    # Load generator model
    model, tokenizer = load_model(model_name)

    # Load judge model (may be same as generator)
    if judge_model_name == model_name:
        judge_model, judge_tokenizer = model, tokenizer
    else:
        judge_model, judge_tokenizer = load_model(judge_model_name)

    all_results = []

    for problem in tqdm(problems, desc="Problems"):
        problem_id   = problem["id"]
        problem_text = problem["problem"]
        ground_truth = str(problem["answer"])

        # --- Step 1: Generate n_samples responses ---
        responses = generate_responses(
            model=model,
            tokenizer=tokenizer,
            problem=problem_text,
            n_samples=n_samples,
        )

        # --- Step 2: Check correctness for each sample ---
        correctness = []
        for resp in responses:
            pred = extract_boxed_answer(resp)
            correctness.append(check_answer_correct(pred, ground_truth))

        acc_at_k = sum(correctness) / len(correctness)  # fraction correct out of k

        # --- Step 3: LLM-as-a-Judge on each sample ---
        d1_scores, d2_scores = [], []
        for resp in responses:
            scores = judge_response(
                model=judge_model,
                tokenizer=judge_tokenizer,
                problem=problem_text,
                solution=resp,
            )
            d1_scores.append(scores.get("D1_rigor", 0))
            d2_scores.append(scores.get("D2_clarity", 0))

        problem_result = {
            "problem_id":   problem_id,
            "model":        model_name,
            "dataset":      dataset_name,
            "responses":    responses,
            "correctness":  correctness,
            "acc_at_k":     acc_at_k,
            "d1_scores":    d1_scores,
            "d2_scores":    d2_scores,
            "d1_mean":      sum(d1_scores) / len(d1_scores),
            "d2_mean":      sum(d2_scores) / len(d2_scores),
        }
        all_results.append(problem_result)

    # --- Step 4: Aggregate ---
    aggregated = compute_acc_at_k(all_results)
    aggregated.update(compute_judge_scores(all_results))
    aggregated["model"]   = model_name
    aggregated["dataset"] = dataset_name

    output = {"aggregated": aggregated, "per_problem": all_results}
    save_results(output, result_file)
    print(f"\n[Results saved to {result_file}]")
    print(f"  Acc@{n_samples}   : {aggregated['acc_at_k']:.4f}")
    print(f"  D1 Rigor  : {aggregated['d1_mean']:.2f}/10")
    print(f"  D2 Clarity: {aggregated['d2_mean']:.2f}/10")
    return output


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLMs on math reasoning.")
    parser.add_argument("--model",       type=str, required=True,
                        choices=["qwen2.5-7b", "qwen2.5-1.5b", "llama-3.2-3b"],
                        help="Generator model to evaluate.")
    parser.add_argument("--dataset",     type=str, required=True,
                        choices=["aime", "amc"],
                        help="Dataset to evaluate on.")
    parser.add_argument("--judge",       type=str, default="qwen2.5-7b",
                        help="Judge model (default: qwen2.5-7b).")
    parser.add_argument("--n_samples",   type=int, default=5,
                        help="Number of samples per problem for Acc@k (default: 5).")
    parser.add_argument("--max_problems",type=int, default=None,
                        help="Limit number of problems (for quick testing).")
    parser.add_argument("--output_dir",  type=str, default="results",
                        help="Directory to save results JSON.")
    args = parser.parse_args()

    evaluate_model(
        model_name      = args.model,
        dataset_name    = args.dataset,
        n_samples       = args.n_samples,
        judge_model_name= args.judge,
        output_dir      = args.output_dir,
        max_problems    = args.max_problems,
    )


if __name__ == "__main__":
    main()

"""
metrics.py
----------
Metric computation functions for the evaluation framework.
"""

from __future__ import annotations
import statistics


def compute_acc_at_k(results: list[dict]) -> dict:
    """
    Compute Acc@k from per-problem results.

    Acc@k = average over problems of (fraction of k samples that are correct).
    This is equivalent to Pass@k when we average fraction-correct across problems.

    Parameters
    ----------
    results : list of per-problem result dicts (each must have 'acc_at_k' key).

    Returns
    -------
    dict with:
        acc_at_k        - mean Acc@k across all problems
        acc_std         - std deviation
        n_problems      - number of problems
        n_fully_correct - problems where all k samples were correct
        n_any_correct   - problems where at least 1 sample was correct
    """
    acc_values = [r["acc_at_k"] for r in results]
    correctness_lists = [r["correctness"] for r in results]

    return {
        "acc_at_k":        statistics.mean(acc_values),
        "acc_std":         statistics.stdev(acc_values) if len(acc_values) > 1 else 0.0,
        "n_problems":      len(results),
        "n_fully_correct": sum(1 for c in correctness_lists if all(c)),
        "n_any_correct":   sum(1 for c in correctness_lists if any(c)),
    }


def compute_judge_scores(results: list[dict]) -> dict:
    """
    Aggregate LLM-as-a-Judge D1/D2 scores across all problems and samples.

    Parameters
    ----------
    results : list of per-problem result dicts (each must have 'd1_mean', 'd2_mean').

    Returns
    -------
    dict with:
        d1_mean, d1_std - mean and std of D1 (rigor) scores
        d2_mean, d2_std - mean and std of D2 (clarity) scores
    """
    d1_values = [r["d1_mean"] for r in results]
    d2_values = [r["d2_mean"] for r in results]

    return {
        "d1_mean": statistics.mean(d1_values),
        "d1_std":  statistics.stdev(d1_values) if len(d1_values) > 1 else 0.0,
        "d2_mean": statistics.mean(d2_values),
        "d2_std":  statistics.stdev(d2_values) if len(d2_values) > 1 else 0.0,
    }

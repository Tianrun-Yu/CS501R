"""
utils.py
--------
Utility functions for saving/loading evaluation results.
"""

from __future__ import annotations
import json
from pathlib import Path


def save_results(data: dict, path: str | Path) -> None:
    """Save results dict to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_results(path: str | Path) -> dict:
    """Load a previously saved results JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_summary_table(all_results: list[dict]) -> None:
    """
    Print a formatted summary table for all evaluated models and datasets.

    Parameters
    ----------
    all_results : list of aggregated result dicts, each with keys:
                  model, dataset, acc_at_k, d1_mean, d2_mean
    """
    print("\n" + "=" * 72)
    print(f"{'Model':<20} {'Dataset':<8} {'Acc@5':>8} {'D1 Rigor':>10} {'D2 Clarity':>12}")
    print("-" * 72)
    for r in all_results:
        agg = r.get("aggregated", r)
        print(
            f"{agg['model']:<20} {agg['dataset']:<8} "
            f"{agg['acc_at_k']:>8.4f} {agg['d1_mean']:>10.2f} {agg['d2_mean']:>12.2f}"
        )
    print("=" * 72)

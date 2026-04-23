"""
plot_results.py
---------------
Generate charts and tables from saved evaluation results.

Usage:
    python plot_results.py --results_dir results/ --output_dir figures/
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ─── Data ────────────────────────────────────────────────────────────────────
# Hard-coded results from our experiments (matches JSON outputs).
# Replace with load_all_results() if running from saved JSON files.

RESULTS = {
    "qwen2.5-7b":   {"aime": {"acc": 0.20, "d1": 7, "d2": 6},
                     "amc":  {"acc": 0.49, "d1": 7, "d2": 6}},
    "llama-3.2-3b": {"aime": {"acc": 0.16, "d1": 3, "d2": 4},
                     "amc":  {"acc": 0.42, "d1": 3, "d2": 4}},
    "qwen2.5-1.5b": {"aime": {"acc": 0.13, "d1": 5, "d2": 6},
                     "amc":  {"acc": 0.36, "d1": 5, "d2": 6}},
}

MODEL_LABELS  = ["Qwen2.5-7B", "Llama-3.2-3B", "Qwen2.5-1.5B"]
MODEL_KEYS    = ["qwen2.5-7b", "llama-3.2-3b", "qwen2.5-1.5b"]
COLORS_AIME   = "#0891B2"  # teal
COLORS_AMC    = "#F59E0B"  # gold
COLOR_D1      = "#0891B2"
COLOR_D2      = "#7C3AED"


# ─── Plot 1: Acc@5 grouped bar chart ─────────────────────────────────────────

def plot_acc_at_5(output_dir: Path) -> None:
    aime_acc = [RESULTS[m]["aime"]["acc"] for m in MODEL_KEYS]
    amc_acc  = [RESULTS[m]["amc"]["acc"]  for m in MODEL_KEYS]

    x     = np.arange(len(MODEL_LABELS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, aime_acc, width, label="AIME", color=COLORS_AIME, zorder=3)
    bars2 = ax.bar(x + width/2, amc_acc,  width, label="AMC",  color=COLORS_AMC,  zorder=3)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Acc@5", fontsize=12)
    ax.set_title("Acc@5 by Model and Dataset", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, fontsize=11)
    ax.set_ylim(0, 0.65)
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out = output_dir / "acc_at_5.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {out}")


# ─── Plot 2: LLM-as-a-Judge D1 and D2 side-by-side ───────────────────────────

def plot_judge_scores(output_dir: Path) -> None:
    d1_scores = [RESULTS[m]["aime"]["d1"] for m in MODEL_KEYS]
    d2_scores = [RESULTS[m]["aime"]["d2"] for m in MODEL_KEYS]

    x     = np.arange(len(MODEL_LABELS))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, scores, color, label in [
        (axes[0], d1_scores, COLOR_D1, "D1: Reasoning Rigor"),
        (axes[1], d2_scores, COLOR_D2, "D2: Solution Clarity"),
    ]:
        bars = ax.bar(x, scores, width=0.5, color=color, zorder=3)
        ax.set_title(f"{label} (/10)", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_LABELS, fontsize=10)
        ax.set_ylim(0, 11)
        ax.set_ylabel("Score (/10)", fontsize=11)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)
        ax.set_axisbelow(True)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    str(int(bar.get_height())), ha="center", va="bottom",
                    fontsize=13, fontweight="bold")

    fig.suptitle("LLM-as-a-Judge Scores by Model", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = output_dir / "judge_scores.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {out}")


# ─── Plot 3: Acc vs D1 scatter (trade-off view) ───────────────────────────────

def plot_acc_vs_judge(output_dir: Path) -> None:
    colors = [COLORS_AIME, "#F43F5E", "#7C3AED"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, dataset, title in [
        (axes[0], "aime", "AIME"),
        (axes[1], "amc",  "AMC"),
    ]:
        for i, (key, label) in enumerate(zip(MODEL_KEYS, MODEL_LABELS)):
            acc = RESULTS[key][dataset]["acc"]
            d1  = RESULTS[key][dataset]["d1"]
            d2  = RESULTS[key][dataset]["d2"]
            ax.scatter(acc, d1, color=colors[i], s=120, zorder=5,
                       label=f"{label} (D1)")
            ax.scatter(acc, d2, color=colors[i], s=120, marker="^", zorder=5)
            ax.annotate(label, (acc, d1), textcoords="offset points",
                        xytext=(6, 4), fontsize=9)

        ax.set_xlabel("Acc@5", fontsize=12)
        ax.set_ylabel("Judge Score (/10)", fontsize=12)
        ax.set_title(f"Acc@5 vs Judge Scores — {title}", fontsize=13, fontweight="bold")
        ax.set_xlim(-0.02, 0.65)
        ax.set_ylim(0, 11)
        ax.yaxis.grid(True, linestyle="--", alpha=0.6)
        ax.xaxis.grid(True, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)

    # Legend
    circle  = mpatches.Patch(color="gray",  label="D1: Rigor (circle)")
    triangle_line = plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
                                markersize=9, label="D2: Clarity (triangle)")
    axes[1].legend(handles=[circle, triangle_line], fontsize=9, loc="lower right")

    plt.tight_layout()
    out = output_dir / "acc_vs_judge.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/",
                        help="Directory with saved JSON result files.")
    parser.add_argument("--output_dir",  type=str, default="figures/",
                        help="Directory to save generated figures.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_acc_at_5(output_dir)
    plot_judge_scores(output_dir)
    plot_acc_vs_judge(output_dir)
    print("\nAll figures saved.")


if __name__ == "__main__":
    main()

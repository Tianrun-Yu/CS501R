"""
datasets.py
-----------
Dataset loaders for AIME and AMC math problems.
Problems are loaded from local JSON files in the data/ directory.

Each problem is a dict with keys:
    id      (str)  - unique identifier
    problem (str)  - the problem statement
    answer  (str)  - the ground-truth answer
    source  (str)  - "aime" or "amc"
    year    (int)  - year of competition (optional)
"""

from __future__ import annotations
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

DATASET_FILES = {
    "aime": DATA_DIR / "aime_problems.json",
    "amc":  DATA_DIR / "amc_problems.json",
}


def load_dataset_problems(dataset_name: str) -> list[dict]:
    """
    Load problems for the given dataset name ("aime" or "amc").

    Returns a list of problem dicts.
    Falls back to built-in sample problems if data file not found,
    so you can run the code without downloading a full dataset.
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_FILES:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose from: {list(DATASET_FILES)}")

    data_file = DATASET_FILES[dataset_name]
    if data_file.exists():
        with open(data_file, "r", encoding="utf-8") as f:
            problems = json.load(f)
        print(f"[Loaded {len(problems)} problems from {data_file}]")
        return problems

    # Fallback: built-in sample problems for quick testing
    print(f"[WARNING] Data file {data_file} not found. Using sample problems.")
    return _get_sample_problems(dataset_name)


def _get_sample_problems(dataset_name: str) -> list[dict]:
    """Return a small set of sample problems for offline testing."""
    aime_samples = [
        {
            "id": "aime_2024_i_1",
            "source": "aime",
            "year": 2024,
            "problem": (
                "Every morning Aya goes for a 9-kilometer-long walk and stops at a coffee shop "
                "afterwards. When she walks at a constant speed of s kilometers per hour, the walk "
                "takes 4 hours, including t minutes spent in the coffee shop. When she walks at "
                "s+2 kilometers per hour, the walk takes 2 hours and 24 minutes, including t minutes "
                "spent in the coffee shop. Suppose Aya decides to walk at s+1/2 kilometers per hour. "
                "Find the number of minutes the walk takes, including the t minutes spent in the coffee shop."
            ),
            "answer": "204",
        },
        {
            "id": "aime_2024_i_2",
            "source": "aime",
            "year": 2024,
            "problem": (
                "There exist real numbers x and y, both greater than 1, such that "
                "log_x(y^x) = log_y(x^(4y)) = 10. Find xy."
            ),
            "answer": "25",
        },
        {
            "id": "aime_2024_i_3",
            "source": "aime",
            "year": 2024,
            "problem": (
                "Find the largest prime factor of 2^15 + 2^10 + 1."
            ),
            "answer": "331",
        },
    ]

    amc_samples = [
        {
            "id": "amc_2024_10a_1",
            "source": "amc",
            "year": 2024,
            "problem": (
                "What is the value of 9901 * 101 - 99 * 10001? "
                "(A) 2 (B) 20 (C) 200 (D) 2000 (E) 20000"
            ),
            "answer": "200",
        },
        {
            "id": "amc_2024_10a_2",
            "source": "amc",
            "year": 2024,
            "problem": (
                "A bowl of fruit contains 14 apples and 23 oranges. "
                "How many oranges must be removed so that 70% of the pieces of fruit in the bowl will be apples? "
                "(A) 3 (B) 6 (C) 9 (D) 14 (E) 17"
            ),
            "answer": "6",
        },
        {
            "id": "amc_2024_10a_3",
            "source": "amc",
            "year": 2024,
            "problem": (
                "What is the sum of the digits of the smallest prime that can be written as "
                "a sum of 5 distinct primes? "
                "(A) 5 (B) 7 (C) 9 (D) 10 (E) 11"
            ),
            "answer": "10",
        },
    ]

    return aime_samples if dataset_name == "aime" else amc_samples

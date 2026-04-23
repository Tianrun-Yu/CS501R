"""
judge.py
--------
LLM-as-a-Judge scoring module.
Evaluates model solutions on two dimensions:
  D1 - Rigor:   Is the reasoning logically sound, correct, and free from errors?
  D2 - Clarity: Is the solution detailed and easy for a student to follow?
"""

from __future__ import annotations
import json
import re


# ─── Judge Prompt ────────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """You are an expert math evaluator. \
Score the solution on two dimensions (each 0-10, integer only)."""

JUDGE_USER_TEMPLATE = """Problem:
{problem}

Model Solution:
{solution}

Evaluate the solution on the following two dimensions:

D1 - Rigor (0-10):
Is the reasoning logically sound, correct, and free from errors?
- 0-3: Major logical errors or completely wrong approach
- 4-6: Partially correct but with notable gaps or mistakes
- 7-9: Mostly correct with minor issues
- 10: Perfectly rigorous and correct reasoning

D2 - Clarity (0-10):
Is the solution detailed and easy for a student to follow?
- 0-3: Hard to follow, missing key steps
- 4-6: Somewhat clear but skips important explanations
- 7-9: Clear and well-structured with good explanations
- 10: Exceptionally clear, every step explained

Return ONLY valid JSON with no additional text:
{{"D1_rigor": <int>, "D2_clarity": <int>}}"""


# ─── Helper ──────────────────────────────────────────────────────────────────

def _parse_judge_output(text: str) -> dict:
    """
    Extract D1/D2 scores from judge output.
    Tries strict JSON parse first, then regex fallback.
    """
    # 1. Try direct JSON parse
    text = text.strip()
    try:
        data = json.loads(text)
        return {
            "D1_rigor":   int(data.get("D1_rigor", 0)),
            "D2_clarity": int(data.get("D2_clarity", 0)),
        }
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Regex fallback — handle wrapped JSON or extra text
    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return {
                "D1_rigor":   int(data.get("D1_rigor", 0)),
                "D2_clarity": int(data.get("D2_clarity", 0)),
            }
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. Direct number extraction fallback
    d1 = re.search(r"D1[_\s]*rigor[\":\s]+(\d+)", text, re.IGNORECASE)
    d2 = re.search(r"D2[_\s]*clarity[\":\s]+(\d+)", text, re.IGNORECASE)
    return {
        "D1_rigor":   int(d1.group(1)) if d1 else 0,
        "D2_clarity": int(d2.group(1)) if d2 else 0,
    }


# ─── Main judge function ──────────────────────────────────────────────────────

def judge_response(
    model,
    tokenizer,
    problem: str,
    solution: str,
    max_new_tokens: int = 128,
) -> dict:
    """
    Use the judge model to score a single (problem, solution) pair.

    Parameters
    ----------
    model, tokenizer : HuggingFace model and tokenizer for the judge.
    problem          : The math problem text.
    solution         : The model's generated solution to evaluate.
    max_new_tokens   : Max tokens for judge output (JSON is short).

    Returns
    -------
    dict with keys "D1_rigor" and "D2_clarity", each an int in [0, 10].
    """
    import torch

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user",   "content": JUDGE_USER_TEMPLATE.format(
            problem=problem, solution=solution
        )},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,   # near-deterministic for consistent scoring
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    scores = _parse_judge_output(raw_output)

    # Clamp to valid range
    scores["D1_rigor"]   = max(0, min(10, scores["D1_rigor"]))
    scores["D2_clarity"] = max(0, min(10, scores["D2_clarity"]))
    return scores

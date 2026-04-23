# LLM Math Reasoning Evaluation

**A dual-metric framework for evaluating open-source LLMs on math reasoning tasks, combining Acc@5 accuracy with LLM-as-a-Judge quality scoring.**

> Final Project — LLM Evaluation Course | Tianrun Yu | April 2026

---

## Overview

This project evaluates open-source LLMs on math reasoning benchmarks (AIME and AMC) using two complementary metrics:

| Metric | Description |
|--------|-------------|
| **Acc@5** | Fraction of 5 independent samples that contain the correct final answer (`\boxed{answer}`) |
| **D1 — Rigor** | LLM-as-a-Judge score (0–10) for logical soundness and correctness of the reasoning chain |
| **D2 — Clarity** | LLM-as-a-Judge score (0–10) for how detailed and student-friendly the solution is |

### Models Evaluated

| Model | Params | Role |
|-------|--------|------|
| Qwen2.5-7B-Instruct | 7B | Generator + Judge |
| Llama-3.2-3B-Instruct | 3B | Generator |
| Qwen2.5-1.5B-Instruct | 1.5B | Generator |

### Datasets

- **AIME** — American Invitational Mathematics Examination (competition-level, integer answers)
- **AMC** — American Mathematics Competition (exam-level, multiple choice)

---

## Results

| Model | AIME Acc@5 | AMC Acc@5 | D1 Rigor | D2 Clarity |
|-------|-----------|----------|----------|------------|
| Qwen2.5-7B | **0.20** | **0.49** | **7/10** | 6/10 |
| Llama-3.2-3B | 0.16 | 0.42 | 3/10 | 4/10 |
| Qwen2.5-1.5B | 0.13 | 0.36 | 5/10 | **6/10** |

**Key finding:** Qwen2.5-1.5B matches the 7B model on D2 Clarity, showing that solution expression quality does not always scale with model size.

---

## Project Structure

```
llm-eval-project/
├── evaluate.py       # Main evaluation pipeline
├── models.py         # Model loading & response generation
├── judge.py          # LLM-as-a-Judge scoring (with prompt)
├── datasets.py       # Dataset loaders for AIME and AMC
├── metrics.py        # Acc@5 and Judge score aggregation
├── utils.py          # Save/load results, print summary table
├── plot_results.py   # Generate all figures
├── run_all.py        # Run all model×dataset combinations
├── data/
│   ├── aime_problems.json   # AIME problem set
│   └── amc_problems.json    # AMC problem set
├── results/          # JSON result files (auto-generated)
├── figures/          # Output figures (auto-generated)
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/<your-username>/llm-eval-project.git
cd llm-eval-project
pip install -r requirements.txt
```

You will need HuggingFace access to the Llama-3.2 model:
```bash
huggingface-cli login
```

---

## Usage

**Evaluate a single model:**
```bash
python evaluate.py --model qwen2.5-7b --dataset aime
python evaluate.py --model llama-3.2-3b --dataset amc
```

**Run all combinations:**
```bash
python run_all.py
```

**Quick test with limited problems:**
```bash
python run_all.py --max_problems 5
```

**Generate figures:**
```bash
python plot_results.py
```

---

## Judge Prompt

The LLM-as-a-Judge uses Qwen2.5-7B with the following prompt structure (see `judge.py` for full details):

```
System:
"You are an expert math evaluator.
Score the solution on two dimensions (each 0-10, integer only)."

User:
"Problem: {problem}
Model Solution: {solution}

D1 - Rigor (0-10): Is the reasoning logically sound, correct, and free from errors?
D2 - Clarity (0-10): Is the solution detailed and easy for a student to follow?

Return ONLY valid JSON: {"D1_rigor": <int>, "D2_clarity": <int>}"
```

---

## Citation / Related Work

- [AIME Problems](https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions)
- [AMC Problems](https://artofproblemsolving.com/wiki/index.php/AMC_Problems_and_Solutions)
- Zheng et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS.
- Chen et al. (2021). *Evaluating Large Language Models Trained on Code (Pass@k).* arXiv.

"""
models.py
---------
Model loading and response generation utilities.
Supports Qwen2.5 and Llama-3.2 families via HuggingFace Transformers.
"""

from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Map short names to HuggingFace model IDs
MODEL_MAP = {
    "qwen2.5-7b":   "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
}

# Fixed generation prompt (system + user template)
GENERATION_SYSTEM_PROMPT = (
    "You are a math expert. Solve the problem step by step. "
    "Show ALL reasoning clearly. Box your final answer as \\boxed{answer}."
)

_loaded_models: dict = {}  # cache to avoid reloading


def load_model(model_name: str) -> tuple:
    """
    Load a HuggingFace model and tokenizer by short name.
    Returns (model, tokenizer). Caches models in memory.
    """
    if model_name in _loaded_models:
        return _loaded_models[model_name]

    hf_id = MODEL_MAP.get(model_name)
    if hf_id is None:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(MODEL_MAP)}")

    print(f"[Loading model: {hf_id}]")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    _loaded_models[model_name] = (model, tokenizer)
    return model, tokenizer


def build_messages(problem: str) -> list[dict]:
    """Build the chat messages list for a math problem."""
    return [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {"role": "user",   "content": problem},
    ]


def generate_single(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate a single response from a chat-formatted prompt."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def generate_responses(
    model,
    tokenizer,
    problem: str,
    n_samples: int = 5,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> list[str]:
    """
    Generate n_samples independent responses for a problem.
    Each sample is generated independently (no batching) to ensure diversity.
    """
    messages = build_messages(problem)
    responses = []
    for _ in range(n_samples):
        resp = generate_single(
            model, tokenizer, messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        responses.append(resp)
    return responses

"""
Base vs. fine-tuned model comparison, evaluated on the held-out
validation set (30 examples, 5 categories) plus a single-case sanity
check. Determines whether LoRA fine-tuning measurably improved model
behavior beyond what prompt engineering alone achieves.
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = "model/checkpoints/healthcare_slm_lora/final_adapter"

PERMISSIVE_PROMPT = """You are a clinical reference assistant supporting an IoT-based patient monitoring system. You must answer using ONLY the reference material provided below. Follow these rules strictly:

1. Base your answer only on facts and thresholds stated in the provided reference documents.
2. You ARE permitted, and expected, to compare a specific numeric value given in the question against a range or threshold stated in the reference material. This is retrieval-grounded reasoning, not guessing, and you must perform it when the reference material supports it.
3. Only refuse to answer if the reference material genuinely does not contain a relevant threshold, range, or fact needed to address the question.
4. When you state a fact or conclusion, mention which document title it is grounded in.
5. Do not introduce any medical fact, threshold, or number that is not present in the reference material.
6. Be concise and clinically precise."""

STRICT_PROMPT = """You are a clinical reference assistant. Answer the question using only the reference material provided below. If the reference material does not directly answer the question, say so."""


def load_model(use_adapter: bool):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=quant_config, device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH) if use_adapter else base_model
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, system_prompt: str, question: str, context: str, max_new_tokens: int = 200) -> str:
    user_content = f"Reference material:\n\n{context}\n\nQuestion: {question}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=0.3,
            do_sample=True, pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def run_full_validation_evaluation():
    """
    Evaluates base vs. fine-tuned model against the full held-out
    val.jsonl set (30 examples, all 5 categories), scored per-category
    since different categories test different fine-tuned behaviors.
    """
    val_examples = []
    with open("data/training/val.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            val_examples.append(json.loads(line))
    print(f"Loaded {len(val_examples)} held-out validation examples")

    results_by_condition = {}

    for use_adapter, model_label in [(False, "BASE"), (True, "FINE-TUNED")]:
        print(f"\n{'='*70}\nLoading {model_label} model...\n{'='*70}")
        model, tokenizer = load_model(use_adapter)

        category_correct, category_total = {}, {}

        for ex in val_examples:
            category = ex["category"]
            category_total[category] = category_total.get(category, 0) + 1

            answer = generate(model, tokenizer, PERMISSIVE_PROMPT, ex["question"], ex["context"])

            if category == "calibrated_refusal":
                correct = ("insufficient information" in answer.lower()
                           or "does not contain" in answer.lower()
                           or "qualified clinician" in answer.lower())
            elif category == "threshold_comparison":
                expected_normal = "falls within this normal range" in ex["response"]
                model_says_normal = ("within" in answer.lower() and "normal" in answer.lower()
                                      and "exceed" not in answer.lower() and "below" not in answer.lower())
                model_says_abnormal = ("exceed" in answer.lower() or "falls below" in answer.lower()
                                       or "warrant" in answer.lower())
                correct = (expected_normal and model_says_normal) or (not expected_normal and model_says_abnormal)
            else:
                correct = not ("insufficient information" in answer.lower()
                               or "does not contain" in answer.lower())

            category_correct[category] = category_correct.get(category, 0) + (1 if correct else 0)

        results_by_condition[model_label] = (category_correct, category_total)
        del model
        torch.cuda.empty_cache()

    print(f"\n\n{'='*70}\nFULL VALIDATION SET RESULTS (n={len(val_examples)})\n{'='*70}")
    print(f"{'Category':<24}{'BASE':<15}{'FINE-TUNED':<15}")
    all_categories = set()
    for _, (cat_correct, cat_total) in results_by_condition.items():
        all_categories.update(cat_total.keys())

    for category in sorted(all_categories):
        base_correct, base_total = results_by_condition["BASE"]
        ft_correct, ft_total = results_by_condition["FINE-TUNED"]
        base_score = f"{base_correct.get(category,0)}/{base_total.get(category,0)}"
        ft_score = f"{ft_correct.get(category,0)}/{ft_total.get(category,0)}"
        print(f"{category:<24}{base_score:<15}{ft_score:<15}")

    return results_by_condition


if __name__ == "__main__":
    run_full_validation_evaluation()

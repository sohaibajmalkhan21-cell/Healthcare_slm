import argparse
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

SYSTEM_PROMPT = "You are a clinical reference assistant. Based on the described vital sign trends, predict whether the patient is likely to develop sepsis within the next 6 hours. Answer with 'yes' or 'no' only."

def load_model(base_model_name, adapter_path=None):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=quant_config, device_map="auto")
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer

def get_yes_prob(model, tokenizer, context, question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{context}\n\nQuestion: {question}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("no", add_special_tokens=False)[0]
    yes_p = probs[yes_id].item()
    no_p = probs[no_id].item()
    return yes_p / (yes_p + no_p + 1e-9)

def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            scores.append(metric_fn(np.array(y_true)[idx], np.array(y_pred)[idx]))
        except Exception:
            continue
    return np.mean(scores), np.percentile(scores, 2.5), np.percentile(scores, 97.5)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--adapter_path", default=None)
    p.add_argument("--test_path", default="../outputs/test.jsonl")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    model, tokenizer = load_model(args.model_name, args.adapter_path)
    examples = [json.loads(l) for l in open(args.test_path)]
    if args.limit:
        examples = examples[:args.limit]

    y_true, y_score = [], []
    for ex in examples:
        prob = get_yes_prob(model, tokenizer, ex["context"], ex["question"])
        y_true.append(ex["label"])
        y_score.append(prob)

    y_pred = [1 if s >= args.threshold else 0 for s in y_score]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    sens = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan")
    avg = np.mean([sens, spec, f1, auc])

    print(f"Model: {args.model_name} | Adapter: {args.adapter_path}")
    print(f"N: {len(y_true)} | Positives: {sum(y_true)}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Avg: {avg:.4f}")

if __name__ == "__main__":
    main()

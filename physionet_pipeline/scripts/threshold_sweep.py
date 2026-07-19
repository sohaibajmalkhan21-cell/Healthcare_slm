import argparse
import json
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from evaluate import load_model, get_yes_prob

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--test_path", default="../outputs/test_balanced.jsonl")
    args = p.parse_args()

    model, tokenizer = load_model(args.model_name, args.adapter_path)
    examples = [json.loads(l) for l in open(args.test_path)]

    y_true, y_score = [], []
    for ex in examples:
        prob = get_yes_prob(model, tokenizer, ex["context"], ex["question"])
        y_true.append(ex["label"])
        y_score.append(prob)

    print(f"\nScore distribution: min={min(y_score):.3f} max={max(y_score):.3f} mean={np.mean(y_score):.3f}")
    print(f"\n{'Threshold':<10}{'Sens':<10}{'Spec':<10}{'F1':<10}")
    best_f1, best_t = 0, 0.5
    for t in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]:
        y_pred = [1 if s >= t else 0 for s in y_score]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"{t:<10.2f}{sens:<10.4f}{spec:<10.4f}{f1:<10.4f}")
        if f1 > best_f1:
            best_f1, best_t = f1, t

    auc = roc_auc_score(y_true, y_score)
    print(f"\nAUC (threshold-independent): {auc:.4f}")
    print(f"Best F1 at threshold={best_t}: {best_f1:.4f}")

if __name__ == "__main__":
    main()

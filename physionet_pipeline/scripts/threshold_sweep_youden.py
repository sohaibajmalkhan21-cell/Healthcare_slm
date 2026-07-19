import argparse
import json
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
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

    best_j, best_t, best_sens, best_spec = -1, 0.5, 0, 0
    for t in np.arange(0.05, 0.95, 0.01):
        y_pred = [1 if s >= t else 0 for s in y_score]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        j = sens + spec - 1
        if j > best_j:
            best_j, best_t, best_sens, best_spec = j, t, sens, spec

    auc = roc_auc_score(y_true, y_score)
    print(f"Youden-optimal threshold: {best_t:.2f}")
    print(f"Sensitivity: {best_sens:.4f}")
    print(f"Specificity: {best_spec:.4f}")
    print(f"Youden's J: {best_j:.4f}")
    print(f"AUC: {auc:.4f}")

if __name__ == "__main__":
    main()

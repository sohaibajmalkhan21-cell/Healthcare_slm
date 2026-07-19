import argparse
import json
import random
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from evaluate import load_model, get_yes_prob

def bootstrap_ci(y_true, y_pred_or_score, metric_fn, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)
    y_true = np.array(y_true)
    y_pred_or_score = np.array(y_pred_or_score)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_pred_or_score[idx]))
    return np.mean(scores), np.percentile(scores, 2.5), np.percentile(scores, 97.5)

def sens_fn(t, y): 
    tn, fp, fn, tp = confusion_matrix(t, y, labels=[0,1]).ravel()
    return tp / (tp + fn + 1e-9)

def spec_fn(t, y):
    tn, fp, fn, tp = confusion_matrix(t, y, labels=[0,1]).ravel()
    return tn / (tn + fp + 1e-9)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--natural_test_path", default="../outputs/test_natural.jsonl")
    args = p.parse_args()

    model, tokenizer = load_model(args.model_name, args.adapter_path)
    examples = [json.loads(l) for l in open(args.natural_test_path)]

    # Split by patient into tuning (40%) and final-report (60%) sets
    patient_ids = list({e["patient_id"] for e in examples})
    random.seed(123)
    random.shuffle(patient_ids)
    n_tune = int(len(patient_ids) * 0.4)
    tune_ids = set(patient_ids[:n_tune])
    report_ids = set(patient_ids[n_tune:])

    tune_examples = [e for e in examples if e["patient_id"] in tune_ids]
    report_examples = [e for e in examples if e["patient_id"] in report_ids]
    print(f"Tuning set: {len(tune_examples)} ({sum(e['label'] for e in tune_examples)} pos)")
    print(f"Report set: {len(report_examples)} ({sum(e['label'] for e in report_examples)} pos)")

    # Get scores on tuning set, find Youden-optimal threshold
    tune_true, tune_score = [], []
    for ex in tune_examples:
        prob = get_yes_prob(model, tokenizer, ex["context"], ex["question"])
        tune_true.append(ex["label"])
        tune_score.append(prob)

    best_j, best_t = -1, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        y_pred = [1 if s >= t else 0 for s in tune_score]
        j = sens_fn(tune_true, y_pred) + spec_fn(tune_true, y_pred) - 1
        if j > best_j:
            best_j, best_t = j, t
    print(f"\nThreshold selected on TUNING set: {best_t:.2f} (Youden J={best_j:.4f})")

    # Apply fixed threshold to REPORT set (never seen during tuning)
    report_true, report_score = [], []
    for ex in report_examples:
        prob = get_yes_prob(model, tokenizer, ex["context"], ex["question"])
        report_true.append(ex["label"])
        report_score.append(prob)

    report_pred = [1 if s >= best_t else 0 for s in report_score]

    sens = sens_fn(report_true, report_pred)
    spec = spec_fn(report_true, report_pred)
    f1 = f1_score(report_true, report_pred, zero_division=0)
    auc = roc_auc_score(report_true, report_score)

    sens_ci = bootstrap_ci(report_true, report_pred, sens_fn)
    spec_ci = bootstrap_ci(report_true, report_pred, spec_fn)
    f1_ci = bootstrap_ci(report_true, report_pred, lambda t,p: f1_score(t,p,zero_division=0))
    auc_ci = bootstrap_ci(report_true, report_score, roc_auc_score)
    avg = np.mean([sens, spec, f1, auc])

    print(f"\n=== FINAL RESULTS (natural ratio, held-out from threshold tuning) ===")
    print(f"N: {len(report_true)} | Positives: {sum(report_true)}")
    print(f"Sensitivity: {sens:.4f} ({sens_ci[1]:.4f}-{sens_ci[2]:.4f})")
    print(f"Specificity: {spec:.4f} ({spec_ci[1]:.4f}-{spec_ci[2]:.4f})")
    print(f"F1: {f1:.4f} ({f1_ci[1]:.4f}-{f1_ci[2]:.4f})")
    print(f"AUC: {auc:.4f} ({auc_ci[1]:.4f}-{auc_ci[2]:.4f})")
    print(f"Avg: {avg:.4f}")

if __name__ == "__main__":
    main()

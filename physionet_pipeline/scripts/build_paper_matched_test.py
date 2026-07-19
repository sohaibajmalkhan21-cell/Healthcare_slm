import json
import random
import pandas as pd
from sare import build_summary_for_patient_hour

IN_PATH = "../outputs/patients_combined.parquet"
OUT_PATH = "../outputs/test_paper_matched.jsonl"
TRAIN_OUT = "../outputs/train_paper_matched.jsonl"
SYSTEM_PROMPT = "You are a clinical reference assistant. Based on the described vital sign trends, predict whether the patient is likely to develop sepsis within the next 6 hours. Answer with 'yes' or 'no' only."

def one_window_per_patient(df, min_hour=6):
    """
    Matches paper's per-encounter sampling: one summary window per
    patient, not every hour. For septic patients, take the window at
    onset-6h (the actual prediction point). For non-septic patients,
    take a representative mid-stay window.
    """
    examples = []
    for pid, pdf in df.groupby("patient_id"):
        pdf = pdf.sort_values("hour").reset_index(drop=True)
        sepsis_hours = pdf[pdf["SepsisLabel"] == 1]
        if len(sepsis_hours) > 0:
            hour = sepsis_hours["hour"].min()
            label = 1
        else:
            if len(pdf) <= min_hour:
                continue
            hour = len(pdf) // 2
            label = 0
        if hour < min_hour:
            continue
        summary = build_summary_for_patient_hour(pdf, hour)
        if summary is None:
            continue
        examples.append({
            "system": SYSTEM_PROMPT, "context": summary,
            "question": "Is sepsis likely within the next 6 hours?",
            "response": "yes" if label == 1 else "no", "label": label,
            "patient_id": pid, "hour": int(hour),
        })
    return examples

if __name__ == "__main__":
    df = pd.read_parquet(IN_PATH)
    examples = one_window_per_patient(df)
    positives = [e for e in examples if e["label"] == 1]
    negatives = [e for e in examples if e["label"] == 0]
    print(f"Total patients with usable window: {len(examples)} ({len(positives)} pos, {len(negatives)} neg)")

    random.seed(42)
    random.shuffle(positives)
    random.shuffle(negatives)
    # Match paper's 20% positive test ratio (2346 neg : 586 pos in their Table 1)
    n_test_pos = min(586, len(positives) // 4)
    n_test_neg = n_test_pos * 4
    test = positives[:n_test_pos] + negatives[:n_test_neg]
    remaining_pos = positives[n_test_pos:]
    remaining_neg = negatives[n_test_neg:n_test_neg + len(remaining_pos) * 4]
    train = remaining_pos + remaining_neg
    random.shuffle(test)
    random.shuffle(train)

    print(f"Test: {len(test)} ({sum(e['label'] for e in test)} pos, {100*sum(e['label'] for e in test)/len(test):.1f}%)")
    print(f"Train: {len(train)} ({sum(e['label'] for e in train)} pos)")

    with open(TRAIN_OUT, "w") as f:
        for e in train:
            f.write(json.dumps(e) + "\n")
    with open(OUT_PATH, "w") as f:
        for e in test:
            f.write(json.dumps(e) + "\n")

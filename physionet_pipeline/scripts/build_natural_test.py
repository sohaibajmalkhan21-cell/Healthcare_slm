import json
import random
import pandas as pd
from sare import build_summary_for_patient_hour

IN_PATH = "../outputs/patients_combined.parquet"
OUT_PATH = "../outputs/test_natural.jsonl"
SYSTEM_PROMPT = "You are a clinical reference assistant. Based on the described vital sign trends, predict whether the patient is likely to develop sepsis within the next 6 hours. Answer with 'yes' or 'no' only."

def get_test_patient_ids():
    train_ex = [json.loads(l) for l in open("../outputs/train.jsonl")]
    val_ex = [json.loads(l) for l in open("../outputs/val.jsonl")]
    excluded = {e["patient_id"] for e in train_ex} | {e["patient_id"] for e in val_ex}
    return excluded

def build_natural_examples(df, excluded_ids, min_hour=6, max_patients=5000):
    random.seed(42)
    test_patients = [pid for pid in df["patient_id"].unique() if pid not in excluded_ids]
    random.shuffle(test_patients)
    test_patients = test_patients[:max_patients]
    examples = []
    for pid in test_patients:
        pdf = df[df["patient_id"] == pid].sort_values("hour").reset_index(drop=True)
        for hour in range(min_hour, len(pdf)):
            summary = build_summary_for_patient_hour(pdf, hour)
            if summary is None:
                continue
            label = int(pdf.loc[hour, "SepsisLabel"])
            examples.append({
                "system": SYSTEM_PROMPT, "context": summary,
                "question": "Is sepsis likely within the next 6 hours?",
                "response": "yes" if label == 1 else "no", "label": label,
                "patient_id": pid, "hour": int(hour),
            })
    return examples

if __name__ == "__main__":
    df = pd.read_parquet(IN_PATH)
    excluded = get_test_patient_ids()
    examples = build_natural_examples(df, excluded)
    print(f"Natural test set: {len(examples)} examples, {sum(e['label'] for e in examples)} positive ({100*sum(e['label'] for e in examples)/len(examples):.2f}%)")
    with open(OUT_PATH, "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")
